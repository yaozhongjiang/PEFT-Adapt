# Manuscript table mapping

- Open-set structural inference table: `artifact/summaries/*open_set*predictions.csv` and `artifact/reports/*open_set_metrics.json`.
- Public-base mismatch table: `artifact/summaries/*base_mismatch.csv`.
- Held-out lineage table: `artifact/summaries/*lineage_pairs.csv` and `artifact/reports/*lineage_metrics.json`.
- Absolute operational comparison: `artifact/summaries/*redeploy_per_seed.csv`.
- Paired operational deltas: `artifact/summaries/*redeploy_delta_summary.csv`.
- Generation mechanism figure/table: `artifact/summaries/*llama_mechanism.csv`.

Do not manually transcribe aggregate values before running `scripts/validate_artifact.py`.
