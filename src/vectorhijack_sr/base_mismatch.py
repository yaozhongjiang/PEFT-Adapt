from __future__ import annotations

import copy
import re
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

from .collect import (
    _load_checkpoint_model,
    collect_causal_label_logits,
    collect_sequence_classification,
    collect_sequence_models,
)
from .config import get
from .drift import continued_pretrain_backbone
from .features import build_signature_batches, feature_columns
from .hf import dynamic_int8_quantize, load_model, load_tokenizer
from .io import read_json, read_jsonl, write_csv, write_json, write_jsonl
from .metrics import safe_roc_auc
from .reproducibility import environment_snapshot, git_commit, sha256_file
from .structural import StructuralClassifier


def _mean_residual_norm(rows: list[dict[str, Any]]) -> float:
    norms = [
        np.linalg.norm(np.asarray(row["victim_logits"]) - np.asarray(row["base_logits"]))
        for row in rows
    ]
    return float(np.mean(norms)) if norms else float("nan")


def _task_accuracy(rows: list[dict[str, Any]]) -> float:
    if not rows:
        return float("nan")
    correct = [int(np.argmax(row["victim_logits"])) == int(row["label"]) for row in rows]
    return float(np.mean(correct))


def _condition_attacker_cfg(cfg: dict[str, Any], condition: dict[str, Any]) -> dict[str, Any]:
    attacker = copy.deepcopy(cfg["model"])
    for key in (
        "model_id",
        "revision",
        "dtype",
        "quantization",
        "tokenizer_id",
        "tokenizer_revision",
        "device_map",
    ):
        if key in condition:
            attacker[key] = condition[key]
    attacker.update(condition.get("model_overrides", {}))
    return attacker


def _structure_family(name: str) -> str:
    lowered = name.lower()
    for family in ("dora", "ia3", "lora", "prefix", "bitfit", "head", "adapter", "full"):
        if family in lowered:
            return family
    return lowered.split("_")[0]


def _rank_bucket(name: str) -> str:
    match = re.search(r"(?:^|[_-])r(\d+)", name.lower())
    if not match:
        return "non_low_rank"
    rank = int(match.group(1))
    if rank <= 16:
        return "low"
    if rank <= 128:
        return "mid"
    return "high"


def _collect_condition(
    cfg: dict[str, Any],
    victim_entry: dict[str, Any],
    condition: dict[str, Any],
    drift_cache: Path,
) -> list[dict[str, Any]]:
    task_type = cfg["model"].get("task_type", "sequence_classification")
    attacker_cfg = _condition_attacker_cfg(cfg, condition)
    tokenizer_override = condition.get("tokenizer_override")
    kind = condition["kind"]
    if kind == "continued_pretraining":
        drift_dir = drift_cache / condition["name"]
        # The local drifted base is a property of the mismatch condition, not of
        # the private adapter. Reuse the immutable checkpoint across all services
        # so that only the victim adapter changes between service runs.
        model_markers = (
            drift_dir / "config.json",
            drift_dir / "model.safetensors",
            drift_dir / "pytorch_model.bin",
        )
        if (drift_dir / "config.json").exists() and any(
            marker.exists() for marker in model_markers[1:]
        ):
            checkpoint = str(drift_dir)
        else:
            checkpoint = continued_pretrain_backbone(
                cfg["model"],
                cfg["data"],
                drift_dir,
                seed=int(condition.get("seed", 991)),
                max_steps=int(condition.get("max_steps", 50)),
                learning_rate=float(condition.get("learning_rate", 5e-5)),
                max_samples=int(condition.get("max_samples", 2000)),
            )
        attacker_cfg["model_id"] = checkpoint
        attacker_cfg["revision"] = None
    if task_type == "causal_lm":
        rows = collect_causal_label_logits(
            cfg,
            victim_entry,
            attacker_model_cfg=attacker_cfg,
            system_prompt_override=condition.get("hidden_system_prompt"),
        )
    elif kind == "dynamic_int8":
        num_labels = int(cfg["data"]["num_labels"])
        tokenizer = load_tokenizer(cfg["model"], tokenizer_override=tokenizer_override)
        victim = _load_checkpoint_model(cfg["model"], victim_entry["checkpoint"], num_labels)
        exact_base = load_model(cfg["model"], num_labels=num_labels)
        mismatch_base = dynamic_int8_quantize(exact_base)
        cpu_cfg = copy.deepcopy(cfg)
        cpu_cfg.setdefault("inference", {})["device"] = "cpu"
        rows = collect_sequence_models(
            cpu_cfg,
            victim,
            mismatch_base,
            tokenizer,
            {
                "service_id": victim_entry["service_id"],
                "structure": victim_entry["structure"],
                "known": bool(victim_entry.get("known", True)),
                "victim_seed": int(victim_entry["victim_seed"]),
            },
        )
    else:
        rows = collect_sequence_classification(
            cfg,
            victim_entry,
            attacker_model_cfg=attacker_cfg,
            tokenizer_override=tokenizer_override,
        )
    for row in rows:
        row["mismatch_condition"] = condition["name"]
        row["mismatch_kind"] = kind
    return rows


def _fit_structural_classifier(
    cfg: dict[str, Any],
) -> tuple[StructuralClassifier, list[str]] | None:
    path = cfg["inputs"].get("structural_calibration_jsonl")
    if not path:
        return None
    rows = list(read_jsonl(path))
    signatures = pd.DataFrame(
        build_signature_batches(
            rows,
            batch_size=int(get(cfg, "features.signature_batch_size", 100)),
            shuffle_seed=int(get(cfg, "experiment.seed", 7)),
        )
    )
    calibration_seeds = set(int(v) for v in cfg["splits"]["calibration_victim_seeds"])
    signatures = signatures[
        signatures["known"] & signatures["victim_seed"].isin(calibration_seeds)
    ].copy()
    columns = feature_columns(signatures)
    classifier = StructuralClassifier(random_state=int(get(cfg, "experiment.seed", 7))).fit(
        signatures[columns].to_numpy(), signatures["structure"].to_numpy()
    )
    return classifier, columns


def run_base_mismatch(cfg: dict[str, Any]) -> dict[str, Any]:
    root = Path(get(cfg, "experiment.output_root", "artifact"))
    name = cfg["experiment"]["name"]
    grid = read_json(cfg["inputs"]["victim_grid_json"])
    target_structure = cfg["inputs"].get("victim_structure")
    candidates = [
        entry
        for entry in grid
        if target_structure is None or entry["structure"] == target_structure
    ]
    calibration_seeds = set(int(v) for v in cfg["splits"]["calibration_victim_seeds"])
    detector_fit_seeds = set(
        int(v)
        for v in cfg["splits"].get("detector_fit_victim_seeds", sorted(calibration_seeds)[:-1])
    )
    threshold_seeds = set(
        int(v) for v in cfg["splits"].get("threshold_victim_seeds", [max(calibration_seeds)])
    )
    test_seeds = set(int(v) for v in cfg["splits"]["test_victim_seeds"])
    if detector_fit_seeds.intersection(threshold_seeds):
        raise ValueError("Mismatch detector-fit and threshold seeds must be disjoint")
    if (detector_fit_seeds | threshold_seeds) != calibration_seeds:
        raise ValueError("calibration_victim_seeds must equal detector-fit union threshold seeds")
    if calibration_seeds.intersection(test_seeds):
        raise ValueError("Base-mismatch calibration and test victim seeds must be disjoint")
    calibration_victims = [
        entry for entry in candidates if int(entry["victim_seed"]) in calibration_seeds
    ]
    test_victims = [entry for entry in candidates if int(entry["victim_seed"]) in test_seeds]
    if not calibration_victims or not test_victims:
        raise ValueError(
            "Base-mismatch experiment requires calibration and test victim checkpoints"
        )
    conditions = list(cfg["base_mismatch"]["conditions"])
    matched = next((condition for condition in conditions if condition["kind"] == "matched"), None)
    if matched is None:
        raise ValueError("A matched reference condition is required")

    all_rows: list[dict[str, Any]] = []
    skipped: list[dict[str, Any]] = []
    drift_cache = root / "cache" / "base_mismatch"
    for victim in calibration_victims + test_victims:
        for condition in conditions:
            try:
                rows = _collect_condition(cfg, victim, condition, drift_cache)
                all_rows.extend(rows)
            except Exception as exc:
                if condition.get("optional", False):
                    skipped.append(
                        {
                            "service_id": victim["service_id"],
                            "condition": condition["name"],
                            "reason": repr(exc),
                        }
                    )
                    continue
                raise
    signatures = pd.DataFrame(
        build_signature_batches(
            all_rows,
            batch_size=int(get(cfg, "features.signature_batch_size", 100)),
            shuffle_seed=int(get(cfg, "experiment.seed", 7)),
        )
    )
    columns = feature_columns(signatures)
    feature_mode = str(get(cfg, "base_mismatch.feature_mode", "all")).lower()
    if feature_mode in {"spectral", "spectral_core"}:
        spectral_cols = [
            c
            for c in columns
            if c.startswith("spec_")
            or c.startswith("logit_cosine_")
            or c.startswith("topk_overlap_")
        ]
        if spectral_cols:
            columns = spectral_cols
    elif feature_mode in {"magnitude", "magnitude_only"}:
        columns = [
            c
            for c in columns
            if c.startswith(("res_l1_", "res_l2_", "res_linf_", "length_normalized"))
        ]
    detector_fit = signatures[signatures["victim_seed"].isin(detector_fit_seeds)].copy()
    threshold_frame = signatures[signatures["victim_seed"].isin(threshold_seeds)].copy()
    test = signatures[signatures["victim_seed"].isin(test_seeds)].copy()
    for split in (detector_fit, threshold_frame, test):
        split["is_mismatch"] = (split["mismatch_kind"] != "matched").astype(int)
    if detector_fit["is_mismatch"].nunique() < 2:
        raise ValueError("Mismatch detector fit split needs matched and mismatched conditions")
    scaler = StandardScaler().fit(detector_fit[columns].to_numpy())
    mismatch_classifier = LogisticRegression(
        class_weight="balanced",
        max_iter=3000,
        random_state=int(get(cfg, "experiment.seed", 7)),
    ).fit(
        scaler.transform(detector_fit[columns].to_numpy()),
        detector_fit["is_mismatch"],
    )
    threshold_scores = mismatch_classifier.predict_proba(
        scaler.transform(threshold_frame[columns].to_numpy())
    )[:, 1]
    matched_threshold_scores = threshold_scores[threshold_frame["is_mismatch"].to_numpy() == 0]
    if not len(matched_threshold_scores):
        raise ValueError("Threshold split contains no matched reference rows")
    threshold = float(
        np.quantile(
            matched_threshold_scores,
            1.0 - float(get(cfg, "base_mismatch.target_matched_fpr", 0.05)),
            method="higher",
        )
    )
    test["mismatch_score"] = mismatch_classifier.predict_proba(
        scaler.transform(test[columns].to_numpy())
    )[:, 1]
    test["abstain"] = test["mismatch_score"] > threshold
    overall_mismatch_auc = safe_roc_auc(test["is_mismatch"], test["mismatch_score"])

    structural = _fit_structural_classifier(cfg)
    if structural is not None:
        structure_classifier, structure_columns = structural
        missing = [column for column in structure_columns if column not in test.columns]
        if missing:
            raise ValueError(f"Base-mismatch signatures miss structural features: {missing[:5]}")
        predictions = structure_classifier.predict_details(test[structure_columns].to_numpy())
        test["predicted_structure"] = predictions.labels
        test["max_posterior"] = predictions.probabilities.max(axis=1)
    else:
        test["predicted_structure"] = "unavailable"
        test["max_posterior"] = float("nan")
    test["true_family"] = test["structure"].map(_structure_family)
    test["predicted_family"] = test["predicted_structure"].map(_structure_family)
    test["true_rank_bucket"] = test["structure"].map(_rank_bucket)
    test["predicted_rank_bucket"] = test["predicted_structure"].map(_rank_bucket)

    matched_norm_by_service = {
        service: _mean_residual_norm(
            [
                row
                for row in all_rows
                if row["service_id"] == service and row["mismatch_condition"] == matched["name"]
            ]
        )
        for service in test["service_id"].unique()
    }
    summary: list[dict[str, Any]] = []
    for condition, group in test.groupby("mismatch_condition"):
        raw_condition = [
            row
            for row in all_rows
            if row["mismatch_condition"] == condition and int(row["victim_seed"]) in test_seeds
        ]
        residual_ratios = []
        for service, service_group in group.groupby("service_id"):
            service_rows = [row for row in raw_condition if row["service_id"] == service]
            residual_ratios.append(
                _mean_residual_norm(service_rows) / max(matched_norm_by_service[service], 1e-12)
            )
        accepted = group[~group["abstain"]]
        summary.append(
            {
                "condition": condition,
                "kind": group["mismatch_kind"].iloc[0],
                "utility": _task_accuracy(raw_condition),
                "residual_ratio": float(np.mean(residual_ratios)),
                "mismatch_score": float(group["mismatch_score"].mean()),
                "mismatch_detection_rate": float(group["abstain"].mean()),
                "family_accuracy": float(
                    np.mean(group["true_family"] == group["predicted_family"])
                ),
                "rank_accuracy": float(
                    np.mean(group["true_rank_bucket"] == group["predicted_rank_bucket"])
                ),
                "coverage_after_reject": float(len(accepted) / max(len(group), 1)),
                "family_accuracy_after_reject": float(
                    np.mean(accepted["true_family"] == accepted["predicted_family"])
                    if len(accepted)
                    else float("nan")
                ),
                "rank_accuracy_after_reject": float(
                    np.mean(accepted["true_rank_bucket"] == accepted["predicted_rank_bucket"])
                    if len(accepted)
                    else float("nan")
                ),
                "service_count": int(group["service_id"].nunique()),
                "signature_batches": int(len(group)),
            }
        )

    write_jsonl(root / "raw" / f"{name}_base_mismatch.jsonl", all_rows)
    write_csv(
        root / "summaries" / f"{name}_base_mismatch_signatures.csv", test.to_dict(orient="records")
    )
    write_csv(root / "summaries" / f"{name}_base_mismatch.csv", summary)
    report = {
        "run_id": name,
        "fixed_private_adapter_within_each_service": True,
        "calibration_victim_seeds": sorted(calibration_seeds),
        "detector_fit_victim_seeds": sorted(detector_fit_seeds),
        "threshold_victim_seeds": sorted(threshold_seeds),
        "test_victim_seeds": sorted(test_seeds),
        "threshold_selected_without_test_services": True,
        "peft_family_labels_used_by_mismatch_detector": False,
        "mismatch_detector_feature_mode": feature_mode,
        "mismatch_detector_feature_count": len(columns),
        "mismatch_detector_auroc": overall_mismatch_auc,
        "locked_threshold": threshold,
        "conditions": summary,
        "sensitivity_by_residual_ratio": sorted(
            [
                {
                    "condition": row["condition"],
                    "residual_ratio": row["residual_ratio"],
                    "family_accuracy": row["family_accuracy"],
                    "rank_accuracy": row["rank_accuracy"],
                    "mismatch_detection_rate": row["mismatch_detection_rate"],
                }
                for row in summary
            ],
            key=lambda row: row["residual_ratio"],
        ),
        "family_accuracy_min_across_conditions": float(
            min(row["family_accuracy"] for row in summary)
        )
        if summary
        else float("nan"),
        "skipped_optional_conditions": skipped,
    }
    write_json(root / "reports" / f"{name}_base_mismatch.json", report)
    write_json(
        root / "manifests" / f"{name}_base_mismatch.json",
        {
            "run_id": name,
            "experiment": cfg["experiment"],
            "config": cfg,
            "victim_grid_sha256": sha256_file(cfg["inputs"]["victim_grid_json"]),
            "structural_calibration_sha256": sha256_file(
                cfg["inputs"]["structural_calibration_jsonl"]
            ),
            "environment": environment_snapshot(),
            "git_commit": git_commit(),
            "report": report,
        },
    )
    return report
