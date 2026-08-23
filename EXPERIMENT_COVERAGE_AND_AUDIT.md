# Experiment coverage and three-pass audit

## Scope implemented

| Priority | Executable entry point | Locked configuration | Primary outputs |
|---|---|---|---|
| Real fixed-victim base mismatch | `vectorhijack-sr base-mismatch` | `configs/base_mismatch/bert_mnli.yaml` | per-query JSONL, residual signatures, mismatch/structure metrics, skipped-condition reasons |
| Hybrid open-set rejection | `vectorhijack-sr open-set` | `configs/open_set/bert_mnli_detector.yaml` | service predictions, AUROC/AUPR/OSCR, TPR@5% FPR, raw/accepted false concentration |
| Independent exact-lineage fingerprinting | `vectorhijack-sr lineage` | `configs/lineage/bert_mnli.yaml` | held-out pair scores, attribution, locked TMR/FMR, service-checkpoint bootstrap CIs |
| Cost-matched redeployability | `vectorhijack-sr redeploy` | `configs/redeploy/bert_mnli.yaml` | per-seed absolute utility/cost rows, merge/composition results, paired bootstrap deltas |
| Reproducible artifact | `scripts/run_all.sh`, `scripts/validate_artifact.py` | `configs/full_plan.yaml`, schemas under `schemas/` | raw JSONL, CSV, manifests, reports, inventory, SHA-256 hashes |
| Llama trajectory mechanism | `scripts/run_llama_mechanism.sh` | `configs/llama_plan.yaml` | aligned/free-running diagnostics with `mechanism_only=true` and no general API claim |

## Three independent audit passes

1. **Static and scientific protocol:** formatting/lint, byte-code compilation, shell syntax, imports, plan closure, seed disjointness, real mismatch conditions, exact-lineage controls, posterior-to-redeploy wiring, secret and endpoint scan.
2. **Functional:** six unit-test groups, complete synthetic end-to-end execution, and an offline tiny Transformers/PEFT integration covering a real Trainer step, independent adapters, save/reload, safe merge, and weighted body-delta composition with task-specific heads.
3. **Artifact and statistics:** JSONL schemas, finite and shape-consistent logits, unique query identities, CSV/manifests/reports, split checks, service-cluster bootstrap, package inventory, and SHA-256 generation.

Machine-readable status is stored in `artifact/reports/audit_report.json`; detailed logs are in `artifact/reports/audit_round*.log`.

## Direct execution

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -e '.[dev]'
make audit                 # repeat all three audit rounds
bash scripts/run_all.sh    # full BERT/MNLI paper suite
bash scripts/run_llama_mechanism.sh  # separately scoped Llama mechanism suite
```

Optional real 4-bit mismatch requires `python -m pip install -e '.[quant]'` and compatible GPU/bitsandbytes support. Unsupported optional conditions are written as skipped records rather than silently replaced by proxies.

## Verification boundary

The included synthetic end-to-end run and tiny offline Transformers/PEFT integration were actually executed and passed. Full paper-scale BERT/Llama runs were not executed in the packaging environment because they require model/data downloads, gated-model authorization for Llama, and substantial GPU time. The locked full configurations generate the required raw evidence when run in an authorized environment.
