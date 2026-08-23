from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd

from .config import get
from .features import build_signature_batches, feature_columns
from .fingerprint import FingerprintExperiment
from .io import read_jsonl, write_csv, write_json
from .reproducibility import environment_snapshot, git_commit, sha256_file


def run_lineage(cfg: dict[str, Any]) -> dict[str, Any]:
    rows: list[dict[str, Any]] = []
    for path in cfg["inputs"]["raw_jsonl"]:
        rows.extend(read_jsonl(path))
    signatures = pd.DataFrame(
        build_signature_batches(
            rows,
            batch_size=int(get(cfg, "features.signature_batch_size", 100)),
            shuffle_seed=int(get(cfg, "experiment.seed", 7)),
        )
    )
    allowed = cfg.get("fingerprint", {}).get("structures")
    if allowed:
        allowed_set = {str(v) for v in allowed}
        signatures = signatures[signatures["structure"].astype(str).isin(allowed_set)].copy()
        if signatures.empty:
            raise ValueError(f"No signatures remain after structure filter {sorted(allowed_set)}")
    columns = feature_columns(signatures)
    calibration_seeds = [int(v) for v in cfg["splits"]["calibration_victim_seeds"]]
    validation_seeds = [int(v) for v in cfg["splits"]["validation_victim_seeds"]]
    test_seeds = [int(v) for v in cfg["splits"]["test_victim_seeds"]]
    partitions = [set(calibration_seeds), set(validation_seeds), set(test_seeds)]
    if any(partitions[i].intersection(partitions[j]) for i in range(3) for j in range(i + 1, 3)):
        raise ValueError(
            "Victim-training seeds must be disjoint across train, validation, and test"
        )
    experiment = FingerprintExperiment(random_state=int(get(cfg, "experiment.seed", 7)))
    result = experiment.run_from_signatures(
        signatures,
        feature_cols=columns,
        calibration_victim_seeds=calibration_seeds,
        validation_victim_seeds=validation_seeds,
        test_victim_seeds=test_seeds,
        n_bootstrap=int(get(cfg, "fingerprint.n_bootstrap", 2000)),
        target_tmr=float(get(cfg, "fingerprint.target_tmr", 0.95)),
        primary_structure=get(cfg, "fingerprint.primary_structure", None),
    )
    root = Path(get(cfg, "experiment.output_root", "artifact"))
    name = cfg["experiment"]["name"]
    write_csv(
        root / "summaries" / f"{name}_lineage_signatures.csv", signatures.to_dict(orient="records")
    )
    write_csv(
        root / "summaries" / f"{name}_lineage_validation_pairs.csv",
        result["validation_pairs"].drop(columns=["pair_features"]).to_dict(orient="records"),
    )
    write_csv(
        root / "summaries" / f"{name}_lineage_pairs.csv",
        result["pairs"].drop(columns=["pair_features"]).to_dict(orient="records"),
    )
    write_json(root / "reports" / f"{name}_lineage_metrics.json", result["metrics"])
    write_json(
        root / "manifests" / f"{name}_lineage.json",
        {
            "run_id": name,
            "experiment": cfg["experiment"],
            "config": cfg,
            "input_sha256": {str(path): sha256_file(path) for path in cfg["inputs"]["raw_jsonl"]},
            "environment": environment_snapshot(),
            "git_commit": git_commit(),
            "calibration_victim_seeds": calibration_seeds,
            "validation_victim_seeds": validation_seeds,
            "test_victim_seeds": test_seeds,
            "threshold_selected_on_validation_services": True,
            "probe_seed_is_not_victim_seed": True,
            "service_level_bootstrap": True,
            "feature_columns": columns,
            "metrics": result["metrics"],
        },
    )
    return result
