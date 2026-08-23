#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"
bash scripts/run_verified_child.sh \
  artifact/reports/synthetic_smoke_report.json run_id synthetic-smoke 300 -- \
  env PYTHONPATH=src OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 \
  python -u -m vectorhijack_sr smoke --config configs/smoke/synthetic.yaml
PYTHONPATH=src python scripts/validate_artifact.py --root artifact --profile smoke
