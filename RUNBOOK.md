# Execution runbook

## Fast verification

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -e '.[dev]'
make audit
```

`make audit` performs three independent review rounds: static/protocol checks, unit plus offline Transformers/PEFT integration, and artifact/schema/package checks.

## Full BERT experiment suite

```bash
bash scripts/run_all.sh
```

This trains all known and open-set victim checkpoints, collects disjoint probe views, runs hybrid rejection, executes real local-base mismatch conditions, trains a separate fixed-structure lineage grid with train/validation/test checkpoint separation, evaluates the validation-locked attribution operating point, and runs per-seed cost-matched redeployability.

## Llama mechanism-only suite

After obtaining authorized access to `meta-llama/Llama-3.1-8B-Instruct`:

```bash
bash scripts/run_llama_mechanism.sh
```

The Llama outputs are hard-labeled `mechanism_only: true`. They compare one-token/teacher-forced alignment with free-running trajectory divergence and must not be generalized to arbitrary generative APIs.

## Resume and storage

All checkpoints and outputs use deterministic paths. Existing continued-pretraining base revisions are reused. Point Hugging Face caches outside the repository for large runs:

```bash
export HF_HOME=/data/hf-cache
export HF_DATASETS_CACHE=/data/hf-datasets
export TOKENIZERS_PARALLELISM=false
```
