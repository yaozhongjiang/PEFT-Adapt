from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from .config import get
from .features import aggregate_service_views, build_signature_batches, feature_columns
from .fingerprint import FingerprintExperiment
from .io import ensure_dir, write_csv, write_json, write_jsonl
from .unknown import HybridUnknownDetector
from .reproducibility import environment_snapshot, git_commit


def generate_synthetic_raw(cfg: dict[str, Any]) -> list[dict[str, Any]]:
    seed = int(get(cfg, "experiment.seed", 7))
    rng = np.random.default_rng(seed)
    known_structures = list(
        get(cfg, "synthetic.known_structures", ["lora_r8", "lora_r64", "lora_r256"])
    )
    unknown_structures = list(
        get(cfg, "synthetic.unknown_structures", ["dora_r64", "ia3", "lora_r32"])
    )
    victim_seeds = list(get(cfg, "synthetic.victim_seeds", [11, 22, 33, 44, 55]))
    probe_seeds = list(get(cfg, "synthetic.probe_seeds", [7, 42, 123]))
    probes = int(get(cfg, "synthetic.probes_per_service", 120))
    num_labels = int(get(cfg, "synthetic.num_labels", 5))
    rows: list[dict[str, Any]] = []
    structure_offsets: dict[str, np.ndarray] = {}
    for index, structure in enumerate(known_structures + unknown_structures):
        base = rng.normal(0, 0.35, size=num_labels)
        base[index % num_labels] += 0.9 + 0.15 * index
        structure_offsets[structure] = base
    for known, structures in ((True, known_structures), (False, unknown_structures)):
        for structure in structures:
            for victim_seed in victim_seeds:
                service_rng = np.random.default_rng(
                    seed * 10_000 + victim_seed * 101 + sum(map(ord, structure))
                )
                lineage = service_rng.normal(0, 0.08, size=num_labels)
                for probe_seed in probe_seeds:
                    probe_rng = np.random.default_rng(probe_seed + victim_seed * 1_000)
                    service_id = f"{structure}-v{victim_seed}"
                    for query_id in range(probes):
                        base_logits = probe_rng.normal(0, 1.0, size=num_labels)
                        x_effect = probe_rng.normal(0, 0.05, size=num_labels)
                        victim_logits = (
                            base_logits + structure_offsets[structure] + lineage + x_effect
                        )
                        rows.append(
                            {
                                "run_id": "synthetic-smoke",
                                "service_id": service_id,
                                "structure": structure,
                                "known": known,
                                "victim_seed": victim_seed,
                                "probe_seed": probe_seed,
                                "query_id": query_id,
                                "input_length": int(probe_rng.integers(8, 64)),
                                "label": int(np.argmax(victim_logits)),
                                "victim_logits": victim_logits.tolist(),
                                "base_logits": base_logits.tolist(),
                                "lineage_id": service_id,
                                "view_id": f"probe-{probe_seed}",
                            }
                        )
    return rows


def run_synthetic_smoke(cfg: dict[str, Any]) -> dict[str, Any]:
    root = Path(get(cfg, "experiment.output_root", "artifact"))
    raw_dir = ensure_dir(root / "raw")
    summary_dir = ensure_dir(root / "summaries")
    manifest_dir = ensure_dir(root / "manifests")
    report_dir = ensure_dir(root / "reports")
    rows = generate_synthetic_raw(cfg)
    raw_path = raw_dir / "synthetic_queries.jsonl"
    write_jsonl(raw_path, rows)
    signatures = build_signature_batches(
        rows,
        batch_size=int(get(cfg, "features.signature_batch_size", 40)),
        shuffle_seed=int(get(cfg, "experiment.seed", 7)),
    )
    signature_batch_frame = pd.DataFrame(signatures)
    signature_path = summary_dir / "synthetic_signature_batches.csv"
    write_csv(signature_path, signature_batch_frame.to_dict(orient="records"))

    columns = feature_columns(signature_batch_frame)
    signature_frame = aggregate_service_views(signature_batch_frame, columns)
    service_view_path = summary_dir / "synthetic_service_views.csv"
    write_csv(service_view_path, signature_frame.to_dict(orient="records"))
    train_seed_set = set(int(v) for v in get(cfg, "splits.train_victim_seeds", [11, 22, 33]))
    test_seed_set = set(int(v) for v in get(cfg, "splits.test_victim_seeds", [44, 55]))
    threshold_seed_set = set(
        int(v) for v in get(cfg, "splits.threshold_victim_seeds", [max(train_seed_set)])
    )
    fit_seed_set = train_seed_set - threshold_seed_set
    known = signature_frame[signature_frame["known"]].copy()
    unknown = signature_frame[~signature_frame["known"]].copy()
    train = known[known["victim_seed"].isin(train_seed_set)]
    detector_fit = train[train["victim_seed"].isin(fit_seed_set)]
    threshold_known = train[train["victim_seed"].isin(threshold_seed_set)]
    test_known = known[known["victim_seed"].isin(test_seed_set)]
    test_unknown = unknown[unknown["victim_seed"].isin(test_seed_set)]
    detector = HybridUnknownDetector(random_state=int(get(cfg, "experiment.seed", 7)))
    detector.fit(
        detector_fit[columns].to_numpy(),
        detector_fit["structure"].to_numpy(),
        detector_fit["service_id"].to_numpy(),
        x_validation_known=threshold_known[columns].to_numpy(),
    )
    open_metrics = detector.evaluate(
        test_known[columns].to_numpy(),
        test_known["structure"].to_numpy(),
        test_unknown[columns].to_numpy(),
    )
    write_json(summary_dir / "synthetic_open_set.json", open_metrics)

    fingerprint = FingerprintExperiment(random_state=int(get(cfg, "experiment.seed", 7)))
    fp_calibration = [int(v) for v in get(cfg, "fingerprint.calibration_victim_seeds", [11, 22])]
    fp_validation = [int(v) for v in get(cfg, "fingerprint.validation_victim_seeds", [33, 44])]
    fp_test = [int(v) for v in get(cfg, "fingerprint.test_victim_seeds", [55, 66])]
    fp_result = fingerprint.run_from_signatures(
        signature_frame[signature_frame["known"]].copy(),
        feature_cols=columns,
        calibration_victim_seeds=fp_calibration,
        validation_victim_seeds=fp_validation,
        test_victim_seeds=fp_test,
        n_bootstrap=int(get(cfg, "fingerprint.n_bootstrap", 200)),
    )
    write_json(summary_dir / "synthetic_fingerprint.json", fp_result["metrics"])
    write_csv(
        summary_dir / "synthetic_fingerprint_pairs.csv",
        fp_result["pairs"].to_dict(orient="records"),
    )

    manifest = {
        "run_id": "synthetic-smoke",
        "experiment": {
            "name": str(get(cfg, "experiment.name", "synthetic_smoke")),
            "config_sha256": str(get(cfg, "experiment.config_sha256", "")),
        },
        "mode": "offline_synthetic",
        "raw_jsonl": str(raw_path),
        "signature_batch_csv": str(signature_path),
        "service_view_csv": str(service_view_path),
        "inference_unit": "service_probe_view",
        "counts": {
            "raw_rows": len(rows),
            "signature_batches": len(signature_batch_frame),
            "service_views": len(signature_frame),
            "known_test": len(test_known),
            "unknown_test": len(test_unknown),
        },
        "open_set_metrics": open_metrics,
        "fingerprint_metrics": fp_result["metrics"],
        "config": cfg,
        "environment": environment_snapshot(),
        "git_commit": git_commit(),
    }
    write_json(manifest_dir / "synthetic_smoke.json", manifest)
    write_json(report_dir / "synthetic_smoke_report.json", manifest)
    return manifest
