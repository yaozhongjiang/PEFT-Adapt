# PEFT-Adapt

> **AdapterTrace artifact.** The code, configurations, per-example scores and case-study timestamp tokens for black-box run provenance of PEFT adapters are in [`AdapterTrace/`](AdapterTrace/). That directory is self-contained; see its README.

PEFT-Adapt measures black-box structural leakage from parameter-efficient adaptations served on a known public base. It sends the same probes to a local victim and the locally executed base, turns the aligned residuals into service-level signatures, and estimates PEFT family, layer locality, and a coarse rank bucket. The same pipeline can reject structures outside the calibrated known set, link two services to the same private checkpoint, and compare PEFT-format recovery against cost-matched distillation. Runs are limited to controlled local victims; the package does not include a production-API scanner.

Supported measurements:

1. real fixed-victim public-base mismatch;
2. cross-fitted hybrid unknown detection;
3. checkpoint- and victim-seed-disjoint exact-lineage fingerprinting;
4. per-seed, query-matched, parameter-/compute-matched redeployability;
5. raw JSONL, summary CSV, manifests, service-level bootstrap, and one-command validation;
6. Llama one-token results explicitly scoped as a trajectory-alignment mechanism diagnostic.

The locked unknown-detector decision unit is a disjoint service/probe view rather than an internal mini-batch, and weighted adapter composition combines only body deltas while restoring each task-specific head for evaluation.

## What is executable immediately

The offline smoke test uses synthetic logits and requires no model download:

```bash
python -m pip install -e . --no-deps   # when numpy/pandas/sklearn/torch are already present
make smoke
make test
make audit
```

For a clean environment with full experiments:

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -e '.[dev]'
make audit
```

The optional 4-bit/8-bit GPU condition additionally needs:

```bash
python -m pip install -e '.[quant]'
```

## Full experiment sequence

```bash
# 1. Train known and unknown victim checkpoints with independent victim-training seeds.
vectorhijack-sr train-victims --config configs/open_set/bert_mnli_victims.yaml

# 2. Query every controlled victim and the exact public base with disjoint probe seeds.
vectorhijack-sr collect --config configs/open_set/bert_mnli_victims.yaml

# 3. Fit the cross-fitted conformal/distance/energy rejector.
vectorhijack-sr open-set --config configs/open_set/bert_mnli_detector.yaml

# 4. Keep each private adapter fixed while changing only the local attacker base.
vectorhijack-sr base-mismatch --config configs/base_mismatch/bert_mnli.yaml

# 5. Train a dedicated same-task/same-family/same-rank lineage grid.
vectorhijack-sr train-victims --config configs/lineage/bert_mnli_victims.yaml
vectorhijack-sr collect --config configs/lineage/bert_mnli_victims.yaml

# 6. Train on checkpoints 101--103, lock the operating threshold on
#    checkpoints 104--105, and test only on checkpoints 106--108.
vectorhijack-sr lineage --config configs/lineage/bert_mnli.yaml

# 7. Run per-seed recovery and all matched operational baselines.
vectorhijack-sr redeploy --config configs/redeploy/bert_mnli.yaml

# 8. Validate the complete BERT result bundle.
python scripts/aggregate_results.py
python scripts/validate_artifact.py --root artifact --profile paper
```

The same sequence is encoded in `configs/full_plan.yaml` and `scripts/run_all.sh`. The gated, mechanism-only Llama training and diagnostic sequence is separately encoded in `configs/llama_plan.yaml` and `scripts/run_llama_mechanism.sh`.

## Experiment 1: real public-base mismatch

`base_mismatch.py` never substitutes a feature perturbation for a model mismatch. For a
fixed private adapter checkpoint it executes the same probes against:

- the exact public base;
- an actual FP16 local base when supported;
- a dynamically quantized INT8 local base;
- an optional bitsandbytes INT4 local base;
- an alternate tokenizer;
- two real local revisions produced by 25 and 100 steps of continued MLM training.

For sequence classifiers, continued pretraining updates the public backbone and copies it
back into a classifier while preserving the original public task head. Calibration and
test victim-training seeds are disjoint. The binary mismatch detector is trained on matched
versus mismatched residual signatures without PEFT-family labels. A separate structural
classifier measures family/rank degradation before and after abstention.

## Experiment 2: hybrid unknown detector

The detector combines:

- calibrated closed-set posterior confidence and entropy;
- class-conditional Ledoit-Wolf Mahalanobis distance;
- conformal p-values;
- discriminative energy;
- nearest-class distance ratio;
- a logistic meta-rejector trained with leave-one-structure-out pseudo-unknowns.

Real unknown test ranks/families are never used for fitting or threshold selection. The
threshold is locked at a configured known false-reject rate. Outputs include AUROC, AUPR,
OSCR, TPR at 5% FPR, known accuracy, and both raw and accepted false-posterior concentration.

## Experiment 3: exact-lineage fingerprinting

Each service ID is an independently trained adapter checkpoint. Positive pairs are
different probe/interface views of the same checkpoint. The strict negative is the same
task, same PEFT family, same rank, but a different victim-training seed/checkpoint.
The strict grid holds task, family, rank, target modules, and base model fixed. Training
checkpoints `[101,102,103]`, threshold-selection checkpoints `[104,105]`, and final test
checkpoints `[106,107,108]` are physically independent and seed-disjoint. The FMR operating
threshold is selected only on validation services. Confidence intervals use a two-way service-
checkpoint bootstrap, not probe-level or pair-row resampling.

Reported metrics are pair ROC-AUC, macro top-1 checkpoint attribution, EER, locked test TMR,
and locked test FMR at the validation-selected 95% TMR operating point, with strict
same-structure/new-checkpoint hard-negative breakdowns.

## Experiment 4: absolute redeployability

Every attack method uses the same unique victim-query pool. Reusing cached teacher outputs
across epochs does not consume additional victim calls. The runner includes:

- original victim adapter upper bound;
- posterior-selected PEFT;
- oracle-structure PEFT;
- wrong-structure PEFT;
- plain distilled student;
- parameter-matched student;
- compute-matched student;
- genuine two-stage student distillation followed by student-to-PEFT conversion.

The conversion baseline charges both stages. PEFT methods are tested for load and safe merge.
A second adapter is trained on the independent SNLI task. PEFT methods use a prespecified weighted composition of body deltas; their independently trained classifier heads are never averaged and are restored task-by-task for absolute evaluation. Unstructured students receive the same additional Task-B training budget so that two-task utility is not compared against an untrained student. Per-seed deltas and victim-seed bootstrap intervals are written to CSV.

Training FLOPs are recorded as an explicit estimate from `floating_point_ops` when available,
with a documented parameter-token fallback. Wall time, checkpoint bytes, peak serving memory,
trainable parameters, full deployed parameters, merge accuracy, and both task utilities are
also reported.

## Experiment 5: artifact outputs

Every command writes into:

```text
artifact/
  raw/          # per-query JSONL
  summaries/    # signatures, predictions, per-seed rows, paired deltas
  manifests/    # exact configuration, seeds, model/data metadata, evidence boundary
  reports/      # machine-readable aggregate metrics
  models/       # locally trained victims and substitutes
  cache/        # generated local base revisions
```

Schemas are in `schemas/`. `scripts/validate_artifact.py` checks JSONL syntax, required
fields, logit-shape agreement, CSV presence, manifests, reports, and the expected experiment
families. `scripts/bootstrap_ci.py` performs cluster-level confidence intervals.

## Experiment 6: Llama mechanism diagnostic

`llama_mechanism.py` measures four aligned quantities:

- next-token logits restricted to fixed one-token labels;
- two-token answer-only teacher forcing;
- free-running victim/base generation under each model's own history;
- the first trajectory-divergence step.

Every raw row and report contains `mechanism_only: true` and states that the result is not a
general claim about unconstrained free-running APIs. The default Llama checkpoint is gated;
users must have authorized access or replace the model ID with a compatible local checkpoint.

## Reproducibility rules enforced by code

- Victim-training seed and probe seed are separate fields.
- Detector-fit, threshold/validation, and final test victim checkpoints cannot share seeds.
- Stage-I and Stage-II victim calls are counted in the configured budget.
- Unknown-test structures do not enter detector fitting.
- Fingerprint confidence intervals resample services/checkpoints.
- Base mismatch changes the executable local base, not a post-hoc feature vector.
- Distill-to-PEFT includes the source-student cost.
- Task-B adaptation/composition cost is reported for every method.
- No missing optional condition is silently converted into a numerical result; it is recorded
  as skipped with an exception reason.

## Hardware and storage

The default full BERT grid trains many checkpoints and is intentionally expensive. Reduce
`max_train_samples`, victim seeds, or structures only for debugging; main-paper results should
use the full locked configuration. Llama-3.1-8B generally requires a capable GPU or explicit
quantization/offload configuration. Checkpoints are not included in this ZIP.

Set caches outside the repository when needed:

```bash
export HF_HOME=/path/to/hf-cache
export HF_DATASETS_CACHE=/path/to/datasets-cache
export TOKENIZERS_PARALLELISM=false
```

## Authorized use

Run only on local models and services you own or are authorized to test. See `ETHICS.md` and
`SECURITY.md`.

## Verification boundary

The supplied audit executes the offline synthetic pipeline and a tiny local Transformers/PEFT model. It does not claim that the expensive BERT/Llama numerical results have already been generated. `ARTIFACT_STATUS.md`, `RUNBOOK.md`, and `AUDIT_REPORT.md` make this boundary explicit.
