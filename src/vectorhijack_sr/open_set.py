from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd

from .active_probes import select_query_ids
from .bootstrap import cluster_bootstrap
from .config import get
from .features import (
    aggregate_service_views,
    attach_sensitivity_to_views,
    build_signature_batches,
    feature_columns,
)
from .io import read_jsonl, write_csv, write_json
from .metrics import safe_roc_auc, wilson_ci
from .reproducibility import environment_snapshot, git_commit, sha256_file
from .unknown import HierarchicalManifoldUnknownDetector, HybridUnknownDetector


def _load_rows(paths: list[str | Path]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for path in paths:
        rows.extend(read_jsonl(path))
    return rows


def _build_detector(cfg: dict[str, Any]):
    rejector = str(get(cfg, "open_set.rejector", "hierarchical_manifold")).lower()
    seed = int(get(cfg, "experiment.seed", 7))
    fpr = float(get(cfg, "open_set.target_known_fpr", 0.05))
    if rejector in {"hybrid", "distance", "hybrid_unknown"}:
        return HybridUnknownDetector(random_state=seed, target_known_fpr=fpr), rejector
    if rejector in {"hierarchical_manifold", "manifold", "hierarchical"}:
        mode = str(get(cfg, "open_set.rejection_mode", "manifold")).lower()
        return (
            HierarchicalManifoldUnknownDetector(
                random_state=seed,
                target_known_fpr=fpr,
                rejection_mode=mode,
            ),
            "hierarchical_manifold",
        )
    raise ValueError(f"Unsupported open_set.rejector={rejector!r}")


def _load_service_frame(
    cfg: dict[str, Any],
) -> tuple[pd.DataFrame, list[str], list[str], dict[str, Any] | None]:
    """Return (service_views, feature_columns, input_paths, active_probe_meta)."""
    views_path = get(cfg, "inputs.service_views_csv", None)
    if views_path:
        frame = pd.read_csv(views_path)
        if "known" in frame.columns and frame["known"].dtype == object:
            frame["known"] = frame["known"].map(
                lambda v: str(v).lower() in {"1", "true", "yes"}
            )
        columns = feature_columns(frame)
        return frame, columns, [str(views_path)], None

    raw_paths = list(cfg["inputs"]["raw_jsonl"])
    rows = _load_rows(raw_paths)
    active_meta: dict[str, Any] | None = None
    if bool(get(cfg, "open_set.active_probe.enabled", False)):
        train_seeds = set(int(v) for v in cfg["splits"]["calibration_victim_seeds"])
        threshold_seeds = set(
            int(v)
            for v in cfg["splits"].get("threshold_victim_seeds", [max(train_seeds)])
        )
        fit_seeds = train_seeds - threshold_seeds
        active_meta = select_query_ids(
            rows,
            fit_seeds,
            keep_fraction=float(get(cfg, "open_set.active_probe.keep_fraction", 0.25)),
            min_queries=int(get(cfg, "open_set.active_probe.min_queries", 80)),
            max_queries=int(get(cfg, "open_set.active_probe.max_queries", 400)),
        )
        selected = set(active_meta["selected_query_ids"])
        rows = [row for row in rows if int(row["query_id"]) in selected]
        if not rows:
            raise ValueError("Active probe selection removed every query")
    signatures = build_signature_batches(
        rows,
        batch_size=int(get(cfg, "features.signature_batch_size", 100)),
        shuffle_seed=int(get(cfg, "experiment.seed", 7)),
    )
    batch_frame = pd.DataFrame(signatures)
    columns = feature_columns(batch_frame)
    frame = aggregate_service_views(batch_frame, columns)
    if any(row.get("probe_family") for row in rows):
        frame = attach_sensitivity_to_views(frame, rows)
        columns = feature_columns(frame)
    structural_meta = {
        "n_raw_queries": len(rows),
        "n_structural_queries": sum(1 for row in rows if row.get("probe_family")),
        "n_canonical_queries": sum(1 for row in rows if not row.get("probe_family")),
        "probe_families": sorted(
            {str(row.get("probe_family")) for row in rows if row.get("probe_family")}
        ),
    }
    # Persist intermediate batches when building from raw for auditability.
    root = Path(get(cfg, "experiment.output_root", "artifact"))
    name = cfg["experiment"]["name"]
    write_csv(
        root / "summaries" / f"{name}_signature_batches.csv",
        batch_frame.to_dict(orient="records"),
    )
    probe_meta = {"subset_selection": active_meta, "structural_probes": structural_meta}
    return frame, columns, [str(p) for p in raw_paths], probe_meta


def run_open_set(cfg: dict[str, Any]) -> dict[str, Any]:
    frame, columns, input_paths, active_meta = _load_service_frame(cfg)
    train_seeds = set(int(v) for v in cfg["splits"]["calibration_victim_seeds"])
    test_seeds = set(int(v) for v in cfg["splits"]["test_victim_seeds"])
    threshold_seeds = set(
        int(v)
        for v in cfg["splits"].get(
            "threshold_victim_seeds",
            [max(train_seeds)],
        )
    )
    detector_fit_seeds = train_seeds - threshold_seeds
    if not detector_fit_seeds or not threshold_seeds:
        raise ValueError("Open-set calibration needs disjoint fit and threshold victim seeds")
    if threshold_seeds.intersection(test_seeds) or detector_fit_seeds.intersection(test_seeds):
        raise ValueError("Open-set calibration and test victim seeds must be disjoint")
    train = frame[(frame["known"]) & frame["victim_seed"].isin(train_seeds)].copy()
    detector_fit = train[train["victim_seed"].isin(detector_fit_seeds)].copy()
    threshold_known = train[train["victim_seed"].isin(threshold_seeds)].copy()
    test_known = frame[(frame["known"]) & frame["victim_seed"].isin(test_seeds)].copy()
    test_unknown = frame[(~frame["known"]) & frame["victim_seed"].isin(test_seeds)].copy()
    if (
        train.empty
        or detector_fit.empty
        or threshold_known.empty
        or test_known.empty
        or test_unknown.empty
    ):
        raise ValueError(
            "Open-set split is empty. Check known flags and calibration/test victim-training seeds."
        )

    detector, rejector_name = _build_detector(cfg)
    fit_kwargs: dict[str, Any] = {
        "x_validation_known": threshold_known[columns].to_numpy(),
    }
    if isinstance(detector, HierarchicalManifoldUnknownDetector):
        fit_kwargs["feature_names"] = columns
    detector.fit(
        detector_fit[columns].to_numpy(),
        detector_fit["structure"].to_numpy(),
        detector_fit["service_id"].to_numpy(),
        **fit_kwargs,
    )
    metrics = detector.evaluate(
        test_known[columns].to_numpy(),
        test_known["structure"].to_numpy(),
        test_unknown[columns].to_numpy(),
        posterior_concentration=float(get(cfg, "open_set.posterior_concentration", 0.8)),
    )
    metrics["rejector"] = rejector_name

    known_pred = detector.predict(test_known[columns].to_numpy())
    unknown_pred = detector.predict(test_unknown[columns].to_numpy())
    _pred_cols = ["service_id", "structure", "victim_seed", "probe_seed"]
    known_out = test_known[_pred_cols].copy()
    known_out["true_known"] = True
    known_out["closed_prediction"] = known_pred.closed_labels
    known_out["final_prediction"] = known_pred.final_labels
    known_out["unknown_score"] = known_pred.unknown_scores
    known_out["rejected"] = known_pred.rejected
    known_out["max_posterior"] = known_pred.closed_probabilities.max(axis=1)
    unknown_out = test_unknown[_pred_cols].copy()
    unknown_out["true_known"] = False
    unknown_out["closed_prediction"] = unknown_pred.closed_labels
    unknown_out["final_prediction"] = unknown_pred.final_labels
    unknown_out["unknown_score"] = unknown_pred.unknown_scores
    unknown_out["rejected"] = unknown_pred.rejected
    unknown_out["max_posterior"] = unknown_pred.closed_probabilities.max(axis=1)
    if isinstance(detector, HierarchicalManifoldUnknownDetector):
        known_comp = detector.score_components(test_known[columns].to_numpy())
        unknown_comp = detector.score_components(test_unknown[columns].to_numpy())
        known_out["hybrid_score"] = known_comp["hybrid"]
        known_out["mahal_ecdf"] = known_comp["mahal_ecdf"]
        unknown_out["hybrid_score"] = unknown_comp["hybrid"]
        unknown_out["mahal_ecdf"] = unknown_comp["mahal_ecdf"]
    predictions = pd.concat([known_out, unknown_out], ignore_index=True)

    conc = float(get(cfg, "open_set.posterior_concentration", 0.8))
    metrics["by_unknown_structure"] = []
    for structure, group in unknown_out.groupby("structure"):
        n = int(len(group))
        rejected = int(group["rejected"].sum())
        false_conc = int((group["max_posterior"] >= conc).sum())
        _, tpr_lo, tpr_hi = wilson_ci(rejected, n)
        _, fc_lo, fc_hi = wilson_ci(false_conc, n)
        metrics["by_unknown_structure"].append(
            {
                "structure": str(structure),
                "service_views": n,
                "unknown_recall_locked": float(group["rejected"].mean()),
                "unknown_recall_locked_ci": [tpr_lo, tpr_hi],
                "false_concentration_raw": float((group["max_posterior"] >= conc).mean()),
                "false_concentration_raw_ci": [fc_lo, fc_hi],
                "false_concentration_accepted": float(
                    ((group["max_posterior"] >= conc) & (~group["rejected"])).mean()
                ),
            }
        )
    metrics["by_known_structure"] = [
        {
            "structure": str(structure),
            "service_views": int(len(group)),
            "closed_accuracy": float(
                (group["closed_prediction"].astype(str) == group["structure"].astype(str)).mean()
            ),
            "reject_rate": float(group["rejected"].mean()),
        }
        for structure, group in known_out.groupby("structure")
    ]

    def auc_stat(sample: pd.DataFrame) -> float:
        return safe_roc_auc((~sample["true_known"]).astype(int), sample["unknown_score"])

    boot = cluster_bootstrap(
        predictions,
        cluster_col="service_id",
        statistic=auc_stat,
        n_bootstrap=int(get(cfg, "open_set.n_bootstrap", 2000)),
        seed=int(get(cfg, "experiment.seed", 7)),
    )
    metrics.update(
        {
            "auroc_ci_lower": boot["ci_lower"],
            "auroc_ci_upper": boot["ci_upper"],
            "bootstrap_services": boot["cluster_count"],
            "feature_count": len(columns),
            "calibration_services": int(train["service_id"].nunique()),
            "known_test_services": int(test_known["service_id"].nunique()),
            "unknown_test_services": int(test_unknown["service_id"].nunique()),
            "calibration_victim_seeds": sorted(train_seeds),
            "detector_fit_victim_seeds": sorted(detector_fit_seeds),
            "threshold_victim_seeds": sorted(threshold_seeds),
            "test_victim_seeds": sorted(test_seeds),
            "real_unknown_test_used_for_fit": False,
            "active_probe": None if active_meta is None else active_meta,
        }
    )

    root = Path(get(cfg, "experiment.output_root", "artifact"))
    name = cfg["experiment"]["name"]
    write_csv(
        root / "summaries" / f"{name}_service_views.csv",
        frame.to_dict(orient="records"),
    )
    write_csv(root / "summaries" / f"{name}_predictions.csv", predictions.to_dict(orient="records"))
    write_json(root / "reports" / f"{name}_open_set_metrics.json", metrics)
    by_service: dict[str, dict[str, Any]] = {}
    for service_id, group in known_out.groupby("service_id"):
        counts = group["closed_prediction"].astype(str).value_counts()
        top1 = str(counts.index[0])
        by_service[str(service_id)] = {
            "top1": top1,
            "structure": top1,
            "vote_fraction": float(counts.iloc[0] / len(group)),
            "mean_max_posterior": float(group["max_posterior"].mean()),
            "reject_fraction": float(group["rejected"].mean()),
            "victim_seed": int(group["victim_seed"].iloc[0]),
            "derived_only_from_held_out_test_views": True,
        }
    write_json(
        root / "reports" / f"{name}_service_predictions.json",
        {
            "run_id": name,
            "by_service": by_service,
            "test_victim_seeds": sorted(test_seeds),
            "calibration_or_threshold_services_used_for_prediction": False,
        },
    )
    input_sha = {}
    for p in input_paths:
        path = Path(p)
        if path.is_file():
            input_sha[str(p)] = sha256_file(p)
    write_json(
        root / "manifests" / f"{name}_open_set.json",
        {
            "run_id": name,
            "experiment": cfg["experiment"],
            "config": cfg,
            "input_paths": input_paths,
            "input_sha256": input_sha,
            "environment": environment_snapshot(),
            "git_commit": git_commit(),
            "feature_columns": columns,
            "inference_unit": "service_probe_view",
            "split": cfg["splits"],
            "open_set": cfg.get("open_set", {}),
            "metrics": metrics,
        },
    )
    return {"metrics": metrics, "predictions": predictions, "signatures": frame}
