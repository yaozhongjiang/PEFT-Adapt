# Optional Slurm helpers

These templates are generic. They do not contain site paths, accounts, or tokens.

Set your own environment before submitting:

```bash
export HF_HOME=/path/to/hf-cache
export HF_DATASETS_CACHE=/path/to/datasets-cache
export SR_ARTIFACT_ROOT="$PWD/artifact"
# Optional, if you already have a Hugging Face token in the environment:
# export HF_TOKEN=...
```

Submit one locked command:

```bash
sbatch --export=ALL,SR_COMMAND=train-victims,SR_CONFIG=configs/open_set/bert_mnli_victims.yaml \
  slurm/submit_example.sbatch
```

Do not commit credentials, cluster usernames, or absolute scratch paths into this repository.
