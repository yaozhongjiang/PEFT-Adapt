#!/bin/bash
#SBATCH --job-name=peft_adapt
#SBATCH --output=slurm/logs/%x_%j.out
#SBATCH --error=slurm/logs/%x_%j.err
#SBATCH --time=24:00:00
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G

# Generic one-command runner. Override with sbatch --export:
#   SR_COMMAND=train-victims
#   SR_CONFIG=configs/open_set/bert_mnli_victims.yaml

set -euo pipefail
REPO_ROOT="${REPO_ROOT:-${SLURM_SUBMIT_DIR:-$PWD}}"
cd "$REPO_ROOT"
mkdir -p slurm/logs "${SR_ARTIFACT_ROOT:-$REPO_ROOT/artifact}"

: "${SR_COMMAND:?SR_COMMAND required}"
: "${SR_CONFIG:?SR_CONFIG required}"
if [[ ! -f "$SR_CONFIG" ]]; then
  echo "missing config $SR_CONFIG" >&2
  exit 1
fi

if [[ -n "${SR_RELEASE_VENV:-}" && -f "${SR_RELEASE_VENV}/bin/activate" ]]; then
  # shellcheck disable=SC1091
  source "${SR_RELEASE_VENV}/bin/activate"
fi

export PYTHONPATH="$REPO_ROOT/src:${PYTHONPATH:-}"
export TOKENIZERS_PARALLELISM=false
export PYTHONUNBUFFERED=1

case "$SR_COMMAND" in
  train-victims|open-set|base-mismatch|lineage|redeploy|llama-mechanism)
    python -m vectorhijack_sr "$SR_COMMAND" --config "$SR_CONFIG"
    ;;
  collect)
    if [[ -n "${SR_GRID:-}" ]]; then
      python -m vectorhijack_sr collect --config "$SR_CONFIG" --grid "$SR_GRID"
    else
      python -m vectorhijack_sr collect --config "$SR_CONFIG"
    fi
    ;;
  *)
    echo "unknown SR_COMMAND=$SR_COMMAND" >&2
    exit 1
    ;;
esac
