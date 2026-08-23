from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Iterable

import numpy as np
import pandas as pd
from scipy.stats import kurtosis, skew

from .metrics import entropy, softmax


@dataclass(frozen=True)
class SignatureBatch:
    service_id: str
    structure: str
    known: bool
    victim_seed: int
    probe_seed: int
    batch_id: int
    features: dict[str, float]


def _safe_stat(fn: Any, values: np.ndarray, default: float = 0.0) -> float:
    try:
        value = float(fn(values))
    except Exception:
        value = default
    return value if math.isfinite(value) else default


def per_query_features(victim_logits: np.ndarray, base_logits: np.ndarray) -> dict[str, float]:
    victim_logits = np.asarray(victim_logits, dtype=np.float64)
    base_logits = np.asarray(base_logits, dtype=np.float64)
    if victim_logits.shape != base_logits.shape:
        raise ValueError(f"Logit shape mismatch: {victim_logits.shape} vs {base_logits.shape}")
    residual = victim_logits - base_logits
    vp = softmax(victim_logits)
    bp = softmax(base_logits)
    topv = np.argsort(victim_logits)[::-1]
    topb = np.argsort(base_logits)[::-1]
    k = min(5, len(topv))
    overlap = len(set(topv[:k]).intersection(set(topb[:k]))) / max(k, 1)
    v_margin = (
        victim_logits[topv[0]] - victim_logits[topv[1]] if len(topv) > 1 else victim_logits[topv[0]]
    )
    b_margin = (
        base_logits[topb[0]] - base_logits[topb[1]] if len(topb) > 1 else base_logits[topb[0]]
    )
    denom = np.linalg.norm(victim_logits) * np.linalg.norm(base_logits)
    cosine = float(np.dot(victim_logits, base_logits) / denom) if denom > 0 else 0.0
    return {
        "res_l1": float(np.abs(residual).sum()),
        "res_l2": float(np.linalg.norm(residual)),
        "res_linf": float(np.abs(residual).max(initial=0.0)),
        "res_mean": float(residual.mean()),
        "res_std": float(residual.std()),
        "res_skew": _safe_stat(skew, residual),
        "res_kurtosis": _safe_stat(kurtosis, residual),
        "victim_entropy": float(entropy(vp)),
        "base_entropy": float(entropy(bp)),
        "entropy_delta": float(entropy(vp) - entropy(bp)),
        "top1_agree": float(topv[0] == topb[0]),
        "topk_overlap": float(overlap),
        "margin_delta": float(v_margin - b_margin),
        "logit_cosine": cosine,
        "max_abs_index": float(np.argmax(np.abs(residual))),
    }


def spectral_features(residual_matrix: np.ndarray, prefix: str = "spec") -> dict[str, float]:
    x = np.asarray(residual_matrix, dtype=np.float64)
    if x.ndim != 2 or min(x.shape) == 0:
        return {f"{prefix}_sv{i}": 0.0 for i in range(1, 6)} | {
            f"{prefix}_effective_rank": 0.0,
            f"{prefix}_entropy": 0.0,
            f"{prefix}_energy90": 0.0,
        }
    x = x - x.mean(axis=0, keepdims=True)
    s = np.linalg.svd(x, full_matrices=False, compute_uv=False)
    energy = np.square(s)
    total = float(energy.sum())
    p = energy / total if total > 0 else np.zeros_like(energy)
    effective_rank = float(np.exp(-(p[p > 0] * np.log(p[p > 0])).sum())) if np.any(p > 0) else 0.0
    cumulative = np.cumsum(p)
    energy90 = int(np.searchsorted(cumulative, 0.9) + 1) if total > 0 else 0
    result = {f"{prefix}_sv{i + 1}": float(s[i]) if i < len(s) else 0.0 for i in range(5)}
    result.update(
        {
            f"{prefix}_effective_rank": effective_rank,
            f"{prefix}_entropy": float(-(p[p > 0] * np.log(p[p > 0])).sum())
            if np.any(p > 0)
            else 0.0,
            f"{prefix}_energy90": float(energy90),
        }
    )
    return result


def aggregate_signature(rows: list[dict[str, Any]]) -> dict[str, float]:
    if not rows:
        raise ValueError("Cannot aggregate an empty signature batch")
    query_features: list[dict[str, float]] = []
    residuals: list[np.ndarray] = []
    lengths: list[float] = []
    base_lengths: list[float] = []
    for row in rows:
        victim = np.asarray(row["victim_logits"], dtype=float)
        base = np.asarray(row["base_logits"], dtype=float)
        query_features.append(per_query_features(victim, base))
        residuals.append(victim - base)
        lengths.append(float(row.get("input_length", 1)))
        base_lengths.append(float(row.get("base_input_length", row.get("input_length", 1))))
    frame = pd.DataFrame(query_features)
    signature: dict[str, float] = {}
    for column in frame.columns:
        values = frame[column].to_numpy(dtype=float)
        for stat_name, fn in (
            ("mean", np.mean),
            ("std", np.std),
            ("q10", lambda a: np.quantile(a, 0.10)),
            ("q25", lambda a: np.quantile(a, 0.25)),
            ("q50", lambda a: np.quantile(a, 0.50)),
            ("q75", lambda a: np.quantile(a, 0.75)),
            ("q90", lambda a: np.quantile(a, 0.90)),
        ):
            signature[f"{column}_{stat_name}"] = _safe_stat(fn, values)
    residual_matrix = np.stack(residuals)
    signature.update(spectral_features(residual_matrix))
    lens = np.asarray(lengths, dtype=float)
    residual_norm = np.linalg.norm(residual_matrix, axis=1)
    base_lens = np.asarray(base_lengths, dtype=float)
    signature["length_mean"] = float(lens.mean())
    signature["length_std"] = float(lens.std())
    signature["base_length_mean"] = float(base_lens.mean())
    signature["base_length_std"] = float(base_lens.std())
    signature["tokenization_length_gap_mean"] = float(np.mean(np.abs(lens - base_lens)))
    signature["length_normalized_residual"] = float(
        np.mean(residual_norm / np.clip(lens, 1.0, None))
    )
    signature.update(_probe_sensitivity_features(rows, query_features))
    return signature


def _probe_sensitivity_features(
    rows: list[dict[str, Any]],
    query_features: list[dict[str, float]],
) -> dict[str, float]:
    """Sensitivity of residuals to template / layer / head probes.

    Computed only from queries present in the batch. Unknown PEFT families never
    enter probe construction; these stats just describe how the observed residual
    moves when the same source example is rewritten.
    """
    by_source: dict[int, dict[str, dict[str, float]]] = {}
    for row, features in zip(rows, query_features, strict=True):
        source_id = int(row.get("source_query_id", row.get("query_id", -1)))
        family = str(row.get("probe_family", "canonical"))
        by_source.setdefault(source_id, {})[family] = features
    template_delta: list[float] = []
    layer_delta: list[float] = []
    head_delta: list[float] = []
    lexical_delta: list[float] = []
    family_std: list[float] = []
    for families in by_source.values():
        l2 = {name: float(stats["res_l2"]) for name, stats in families.items()}
        if len(l2) >= 2:
            family_std.append(float(np.std(list(l2.values()))))
        canonical = l2.get("canonical")
        for name, values in (
            ("template_nli", template_delta),
            ("template_contradict", template_delta),
            ("layer_early", layer_delta),
            ("layer_late", layer_delta),
            ("module_head", head_delta),
            ("lexical_local", lexical_delta),
        ):
            if name in l2 and canonical is not None:
                values.append(abs(l2[name] - canonical))
            elif name in l2 and "canonical" not in l2:
                values.append(l2[name])
    def _mean(values: list[float]) -> float:
        return float(np.mean(values)) if values else 0.0

    return {
        "sens_family_std_mean": _mean(family_std),
        "sens_template_delta_mean": _mean(template_delta),
        "sens_layer_delta_mean": _mean(layer_delta),
        "sens_head_delta_mean": _mean(head_delta),
        "sens_lexical_delta_mean": _mean(lexical_delta),
    }


def build_signature_batches(
    rows: Iterable[dict[str, Any]],
    batch_size: int,
    shuffle_seed: int,
) -> list[dict[str, Any]]:
    grouped: dict[tuple[str, int], list[dict[str, Any]]] = {}
    for row in rows:
        key = (str(row["service_id"]), int(row.get("probe_seed", 0)))
        grouped.setdefault(key, []).append(row)
    rng = np.random.default_rng(shuffle_seed)
    output: list[dict[str, Any]] = []
    for (service_id, probe_seed), service_rows in sorted(grouped.items()):
        indices = np.arange(len(service_rows))
        rng.shuffle(indices)
        for batch_id, start in enumerate(range(0, len(indices), batch_size)):
            selected = [service_rows[i] for i in indices[start : start + batch_size]]
            if len(selected) < max(4, batch_size // 2):
                continue
            first = selected[0]
            metadata: dict[str, Any] = {}
            excluded = {
                "victim_logits",
                "base_logits",
                "query_id",
                "label",
                "input_length",
                "base_input_length",
                "interface",
                "attacker_base",
            }
            for key, value in first.items():
                if key in excluded or key in {
                    "service_id",
                    "structure",
                    "known",
                    "victim_seed",
                    "probe_seed",
                }:
                    continue
                if value is None or isinstance(value, (str, int, float, bool)):
                    metadata[key] = value
            output.append(
                {
                    "service_id": service_id,
                    "structure": first["structure"],
                    "known": bool(first.get("known", True)),
                    "victim_seed": int(first.get("victim_seed", -1)),
                    "probe_seed": probe_seed,
                    "batch_id": batch_id,
                    **metadata,
                    **aggregate_signature(selected),
                }
            )
    return output


def feature_columns(frame: pd.DataFrame) -> list[str]:
    # Whitelist generated residual-signature families. Numeric experiment
    # metadata (template IDs, mismatch condition IDs, seeds, etc.) must never
    # become classifier input, otherwise the detector can learn the protocol
    # instead of the victim/base residual.
    prefixes = (
        "res_",
        "victim_entropy_",
        "base_entropy_",
        "entropy_delta_",
        "top1_agree_",
        "topk_overlap_",
        "margin_delta_",
        "logit_cosine_",
        "max_abs_index_",
        "spec_",
        "length_",
        "base_length_",
        "tokenization_length_gap_",
        "sens_",
    )
    columns = [
        column
        for column in frame.columns
        if column.startswith(prefixes) and np.issubdtype(frame[column].dtype, np.number)
    ]
    if not columns:
        raise ValueError("No numeric signature features found")
    return columns


def aggregate_service_views(frame: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
    """Aggregate query-batch signatures into service/probe views.

    Structural inference is a service-level decision made after a probe batch,
    not an IID classification of every internal mini-batch. Averaging all
    signature mini-batches from one independently selected probe view prevents
    a few extreme batches from dominating the unknown threshold while keeping
    probe views disjoint for evaluation.
    """
    keys = ["service_id", "structure", "known", "victim_seed", "probe_seed"]
    missing = [column for column in keys + columns if column not in frame.columns]
    if missing:
        raise KeyError(f"Cannot aggregate service views; missing columns: {missing}")
    aggregated = frame.groupby(keys, as_index=False, dropna=False)[columns].mean()
    counts = (
        frame.groupby(keys, as_index=False, dropna=False)
        .size()
        .rename(columns={"size": "signature_batch_count"})
    )
    return aggregated.merge(counts, on=keys, how="left", validate="one_to_one")


def attach_sensitivity_to_views(
    views: pd.DataFrame,
    rows: list[dict[str, Any]],
) -> pd.DataFrame:
    """Overwrite mini-batch sensitivity with a full-view estimate."""
    grouped: dict[tuple[str, int], list[dict[str, Any]]] = {}
    for row in rows:
        grouped.setdefault((str(row["service_id"]), int(row.get("probe_seed", 0))), []).append(row)
    records = []
    for (service_id, probe_seed), group in grouped.items():
        query_features = [
            per_query_features(
                np.asarray(item["victim_logits"], dtype=float),
                np.asarray(item["base_logits"], dtype=float),
            )
            for item in group
        ]
        stats = _probe_sensitivity_features(group, query_features)
        stats["service_id"] = service_id
        stats["probe_seed"] = probe_seed
        records.append(stats)
    if not records:
        return views
    extra = pd.DataFrame(records)
    drop_cols = [
        column
        for column in extra.columns
        if column not in {"service_id", "probe_seed"} and column in views.columns
    ]
    merged = views.drop(columns=drop_cols)
    out = merged.merge(extra, on=["service_id", "probe_seed"], how="left")
    for column in extra.columns:
        if column.startswith("sens_"):
            out[column] = out[column].fillna(0.0)
    return out
