# Locked experiment matrix

| Priority | Command | Independent unit | Required outputs |
|---|---|---|---|
| Real base mismatch | `base-mismatch` | victim checkpoint | residual ratio, mismatch AUROC, family/rank accuracy, coverage after reject |
| Unknown detection | `open-set` | victim checkpoint | AUROC, AUPR, OSCR, TPR@5% FPR, false concentration |
| Exact lineage | `lineage` | service/checkpoint | pair AUC, top-1, EER, FMR@95% TMR, service bootstrap CI |
| Redeployability | `redeploy` | victim-training seed | AGR, task accuracy, merge, two-task utility, bytes, memory, FLOPs, paired CI |
| Artifact | validation scripts | file/run | raw JSONL, CSV, manifest, report, schema pass |
| Llama mechanism | `llama-mechanism` | checkpoint and probe seed | TF/FR KL, one-token KL, overlap, divergence step |

Main open-set/base-mismatch/redeploy results use victim seeds 11/22/33/44/55/66 and probe seeds 7/42/123. The strict lineage grid uses victim seeds 101--108 split into train 101--103, validation 104--105, and test 106--108.
