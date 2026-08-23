#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"
export TOKENIZERS_PARALLELISM="${TOKENIZERS_PARALLELISM:-false}"
python -m vectorhijack_sr run-all --plan configs/llama_plan.yaml
python scripts/validate_artifact.py --root artifact --expected llama
