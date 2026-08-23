# Artifact status

This public release ships complete executable code, locked configurations, and audit hooks
for the PEFT-Adapt experiments. A bundled synthetic smoke artifact and a tiny local
Transformers/PEFT integration test are included for offline verification.

## What is executable immediately

```bash
python -m pip install -e '.[dev]'
make smoke
make test
make audit
```

`make audit` does not claim that paper-scale BERT or Llama numbers have already been
generated on this machine. Those runs require authorized model/data access and a GPU.

## Locked experiment sequence

Use the generic configs under `configs/` (not a site-specific cluster overlay):

1. Train and collect open-set victims: `configs/open_set/`
2. Fit the hybrid unknown detector: `configs/open_set/bert_mnli_detector.yaml`
3. Real public-base mismatch: `configs/base_mismatch/bert_mnli.yaml`
4. Exact-lineage fingerprinting: `configs/lineage/`
5. Cost-matched redeployability: `configs/redeploy/bert_mnli.yaml`
6. Llama mechanism diagnostic: `configs/llama_mechanism/` (gated checkpoint)

See `README.md` and `RUNBOOK.md` for the command sequence.

## Verification boundary

- Synthetic smoke and the tiny Transformers/PEFT integration are included and should pass.
- Full BERT/Llama numerical artifacts are produced by running the locked configs in an
  authorized environment. Checkpoints and raw query logs are not part of this git release.
- Optional conditions that cannot run (for example INT4 without bitsandbytes) must be
  recorded as skipped with a reason, not converted into a fabricated number.
