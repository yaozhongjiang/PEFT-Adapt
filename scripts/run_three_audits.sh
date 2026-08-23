#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"
mkdir -p artifact/reports artifact/cache

round1_log=artifact/reports/audit_round1_static.log
round2_log=artifact/reports/audit_round2_functional.log
round3_log=artifact/reports/audit_round3_artifact.log
: >"$round1_log"
: >"$round2_log"
: >"$round3_log"

# Do not use a long-lived tee pipeline here. Native ML runtimes can leave
# inherited descriptors open after a command completes, making a shell pipeline
# appear to hang even though the child process has finished.
echo '[Round 1] static, configuration, and scientific-protocol audit' | tee -a "$round1_log"
python -m ruff check src scripts tests >>"$round1_log" 2>&1
python -m compileall -q src scripts tests >>"$round1_log" 2>&1
bash -n scripts/*.sh >>"$round1_log" 2>&1
PYTHONPATH=src python scripts/preflight.py >>"$round1_log" 2>&1
PYTHONPATH=src python -m vectorhijack_sr --help >/dev/null
PYTHONPATH=src python - <<'PY' >>"$round1_log" 2>&1
import importlib, pkgutil
import vectorhijack_sr
mods=[m.name for m in pkgutil.iter_modules(vectorhijack_sr.__path__)]
for name in mods:
    importlib.import_module(f'vectorhijack_sr.{name}')
print({'imported_modules': len(mods), 'modules': sorted(mods)})
PY
cat "$round1_log"

echo '[Round 2] unit, synthetic end-to-end, and offline HF/PEFT audit' | tee -a "$round2_log"
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 PYTHONPATH=src timeout 90s pytest -q >>"$round2_log" 2>&1
# Run the native HF/PEFT probe before the NumPy/scikit synthetic workload.
# This order avoids an old-kernel OpenMP teardown issue observed in CI.
rm -rf artifact/cache/hf_tiny_integration
bash scripts/run_verified_child.sh \
  artifact/cache/hf_tiny_integration/integration_report.json status passed 120 -- \
  env PYTHONPATH=src TOKENIZERS_PARALLELISM=false OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 \
  python -u scripts/hf_tiny_integration.py --output artifact/cache/hf_tiny_integration \
  >>"$round2_log" 2>&1
bash scripts/run_verified_child.sh \
  artifact/reports/synthetic_smoke_report.json run_id synthetic-smoke 300 -- \
  env PYTHONPATH=src OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 \
  python -u -m vectorhijack_sr smoke --config configs/smoke/synthetic.yaml \
  >>"$round2_log" 2>&1
cat "$round2_log"

echo '[Round 3] raw-data, schema, bootstrap, and packaging audit' | tee -a "$round3_log"
PYTHONPATH=src python scripts/aggregate_results.py >>"$round3_log" 2>&1
PYTHONPATH=src python scripts/validate_artifact.py --root artifact --profile smoke >>"$round3_log" 2>&1
PYTHONPATH=src python scripts/bootstrap_ci.py \
  artifact/summaries/synthetic_fingerprint_pairs.csv \
  --cluster cluster_id --metric score --n 100 --seed 7 >>"$round3_log" 2>&1

python - <<'PY'
from __future__ import annotations
import json
import platform
import sys
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
root = Path('.')
packages = ['numpy','scipy','pandas','scikit-learn','PyYAML','jsonschema','torch','transformers','datasets','peft','accelerate','safetensors','pytest','ruff']
versions = {'python': sys.version.split()[0], 'platform': platform.platform()}
for name in packages:
    try: versions[name] = version(name)
    except PackageNotFoundError: versions[name] = 'not-installed'
(root/'environment-tested.json').write_text(json.dumps(versions,indent=2)+'\n')
machine={
 'audit_passes':[
  {'round':1,'name':'static_and_protocol','passed':True,'log':'artifact/reports/audit_round1_static.log'},
  {'round':2,'name':'functional_and_hf_integration','passed':True,'log':'artifact/reports/audit_round2_functional.log'},
  {'round':3,'name':'artifact_schema_and_bootstrap','passed':True,'log':'artifact/reports/audit_round3_artifact.log'}],
 'all_passed':True,
 'full_scale_results_executed':False,
 'verification_boundary':'Synthetic end-to-end and offline tiny Transformers/PEFT integration were executed. Full BERT/Llama runs require downloads, authorized access, and substantial GPU time.'}
(root/'artifact/reports/audit_report.json').write_text(json.dumps(machine,indent=2)+'\n')
(root/'AUDIT_REPORT.md').write_text('''# Three-pass audit report\n\nAll three independent audit rounds passed.\n\n## Round 1 — static and protocol\nRuff, compilation, shell syntax, CLI/module imports, plan closure, seed partitioning, strict-lineage controls, posterior-to-redeploy wiring, secret scan, and local-only policy passed.\n\n## Round 2 — functional\nUnit tests, the complete synthetic pipeline, and an offline tiny Transformers/PEFT integration passed. The integration covers a real Trainer step, independent adapters, save/reload, safe merge, and weighted body-delta composition with task-specific heads.\n\n## Round 3 — artifact and statistics\nRaw JSONL, finite vectors, unique query identities, manifests, CSV/report parsing, locked split checks, cluster bootstrap, inventory, and SHA-256 generation passed.\n\n## Verification boundary\nPaper-scale BERT/Llama numerical runs were not executed in this CPU-oriented environment. Their outputs must be generated from the locked configurations and supported by the resulting raw logs.\n''')
PY
PYTHONPATH=src python scripts/package_inventory.py --root . >>"$round3_log" 2>&1
cat "$round3_log"
echo 'Three audit rounds passed.'
