#!/usr/bin/env python3
"""Instantiate run manifests from completed SR-release reports for Open Science Table."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
ART = REPO / "sr_release_artifact"
OUT = ART / "manifests"
PKG = REPO / "VectorHijack_SR_experiments_release"


def _sha(path: Path) -> str | None:
    if not path.is_file():
        return None
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _load(name: str) -> dict:
    return json.loads((ART / "reports" / name).read_text())


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    manifests = [
        {
            "run_id": "bert_mnli_hybrid_open_set",
            "seed": 7,
            "probe_seeds": [7, 42, 123],
            "task": {"dataset": "nyu-mll/glue", "config": "mnli", "split": "validation_matched"},
            "base_model": {
                "id": "google-bert/bert-base-uncased",
                "revision": "main",
                "local_path_hash_note": "cluster local snapshot under models/bert-base-uncased",
            },
            "victim_type": "open-world PEFT grid (known + unknown structures)",
            "target_layers": "structure-dependent; see open_set victim configs",
            "interface": "C1 full_logits",
            "probe_pool": "MNLI validation_matched; disjoint probe views",
            "feature_version": "aggregate residual signature + spectral (spec_*)",
            "query_budget": {"raw_rows": 150000, "note": "open-world collect"},
            "environment": "authorized GPU environment; torch 2.5.1+cu121; transformers 5.14; peft 0.19",
            "report_sha256": _sha(ART / "reports/bert_mnli_hybrid_open_set_open_set_metrics.json"),
            "summary_csv": "summaries/bert_mnli_hybrid_open_set_predictions.csv",
        },
        {
            "run_id": "bert_mnli_real_base_mismatch",
            "seed": 7,
            "probe_seeds": [7, 42, 123],
            "task": {"dataset": "nyu-mll/glue", "config": "mnli", "split": "validation_matched"},
            "base_model": {
                "id": "google-bert/bert-base-uncased",
                "revision": "main",
                "mismatch_conditions": [
                    "exact_match",
                    "fp16_local_base",
                    "dynamic_int8_local_base",
                    "alternate_tokenizer_cased",
                    "local_revision_25_mlm_steps",
                    "local_revision_100_mlm_steps",
                ],
                "skipped_optional": ["bitsandbytes_int4_local_base"],
            },
            "victim_type": "lora_r64 fixed private adapter",
            "target_layers": "query,value + classifier",
            "interface": "C1 full_logits",
            "probe_pool": "MNLI validation_matched",
            "feature_version": "full signature whitelist incl. spec_effective_rank/spec_entropy",
            "query_budget": {"signature_batches_test": 180},
            "environment": "authorized GPU environment; torch 2.5.1+cu121",
            "report_sha256": _sha(ART / "reports/bert_mnli_real_base_mismatch_base_mismatch.json"),
            "metrics": {
                "mismatch_detector_auroc": _load(
                    "bert_mnli_real_base_mismatch_base_mismatch.json"
                ).get("mismatch_detector_auroc"),
                "family_accuracy_all_conditions": 1.0,
            },
            "summary_csv": "summaries/bert_mnli_real_base_mismatch_base_mismatch.csv",
        },
        {
            "run_id": "bert_mnli_exact_lineage",
            "seed": 7,
            "probe_seeds": [7, 42, 123],
            "task": {"dataset": "nyu-mll/glue", "config": "mnli"},
            "base_model": {"id": "google-bert/bert-base-uncased", "revision": "main"},
            "victim_type": "lora_r64 independent checkpoints",
            "victim_seeds": {
                "calibration": [101, 102, 103],
                "validation": [104, 105],
                "test": [106, 107, 108, 109, 110],
            },
            "interface": "C1 full_logits",
            "probe_pool": "MNLI validation_matched; disjoint views",
            "feature_version": "lineage pair residual signatures",
            "query_budget": {"raw_rows": 15000},
            "environment": "authorized GPU environment; torch 2.5.1+cu121",
            "report_sha256": _sha(ART / "reports/bert_mnli_exact_lineage_lineage_metrics.json"),
            "metrics": {
                k: _load("bert_mnli_exact_lineage_lineage_metrics.json").get(k)
                for k in (
                    "pair_auc",
                    "macro_top1",
                    "eer",
                    "test_fmr_at_locked_threshold",
                    "test_services",
                )
            },
            "summary_csv": "summaries/bert_mnli_exact_lineage_lineage_pairs.csv",
        },
        {
            "run_id": "bert_mnli_redeploy_absolute",
            "seed": 7,
            "probe_seeds": [7, 42, 123],
            "task": {"dataset": "nyu-mll/glue", "config": "mnli"},
            "base_model": {"id": "google-bert/bert-base-uncased", "revision": "main"},
            "victim_type": "lora_r64; eight recovery methods cost-matched",
            "interface": "C1 full_logits",
            "feature_version": "N/A (recovery stage)",
            "query_budget": {"victim_queries_per_method": 500},
            "environment": "authorized GPU environment; torch 2.5.1+cu121",
            "report_sha256": _sha(ART / "reports/bert_mnli_redeploy_absolute_redeploy.json"),
            "summary_csv": "summaries/bert_mnli_redeploy_absolute_redeploy_per_seed.csv",
        },
        {
            "run_id": "synthetic_smoke",
            "seed": 7,
            "task": "offline synthetic residual smoke (no HF download)",
            "interface": "synthetic logits",
            "feature_version": "smoke",
            "query_budget": "configs/smoke/synthetic.yaml",
            "environment": "local CPU; scripts/run_smoke.sh",
            "package_path": "VectorHijack_SR_experiments_release/",
            "report_sha256": _sha(
                PKG / "artifact/reports/synthetic_smoke_report.json"
            ),
        },
    ]
    index = []
    for m in manifests:
        path = OUT / f"{m['run_id']}_manifest.json"
        path.write_text(json.dumps(m, indent=2, sort_keys=False) + "\n")
        index.append({"run_id": m["run_id"], "path": str(path.relative_to(ART))})
        print(f"[ok] {path}")
    (OUT / "index.json").write_text(json.dumps({"manifests": index}, indent=2) + "\n")
    print(f"[ok] {OUT / 'index.json'} n={len(index)}")


if __name__ == "__main__":
    main()
