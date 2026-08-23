PYTHON ?= python
CONFIG ?= configs/smoke/synthetic.yaml

.PHONY: install install-dev smoke test hf-integration preflight paper-suite llama-suite validate-paper validate-llama audit1 audit2 audit3 audit clean

install:
	$(PYTHON) -m pip install -e .

install-dev:
	$(PYTHON) -m pip install -e '.[dev]'

smoke:
	bash scripts/run_verified_child.sh artifact/reports/synthetic_smoke_report.json run_id synthetic-smoke 300 -- env PYTHONPATH=src OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 $(PYTHON) -u -m vectorhijack_sr smoke --config $(CONFIG)

test:
	PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 PYTHONPATH=src $(PYTHON) -m pytest -q

hf-integration:
	rm -rf artifact/cache/hf_tiny_integration
	bash scripts/run_verified_child.sh artifact/cache/hf_tiny_integration/integration_report.json status passed 120 -- env PYTHONPATH=src TOKENIZERS_PARALLELISM=false OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 $(PYTHON) -u scripts/hf_tiny_integration.py --output artifact/cache/hf_tiny_integration

preflight:
	PYTHONPATH=src $(PYTHON) scripts/preflight.py

paper-suite:
	bash scripts/run_all.sh

llama-suite:
	bash scripts/run_llama_mechanism.sh

validate-paper:
	PYTHONPATH=src $(PYTHON) scripts/validate_artifact.py --root artifact --profile paper

validate-llama:
	PYTHONPATH=src $(PYTHON) scripts/validate_artifact.py --root artifact --expected llama

audit1: preflight
	$(PYTHON) -m ruff check src scripts tests
	PYTHONPATH=src $(PYTHON) -m compileall -q src scripts tests
	bash -n scripts/*.sh

# Functional audit: unit tests, complete synthetic run, and local HF/PEFT integration.
audit2: test smoke hf-integration

# Artifact audit: schemas, manifests, summaries, bootstrap, and package hashes.
audit3:
	PYTHONPATH=src $(PYTHON) scripts/aggregate_results.py
	PYTHONPATH=src $(PYTHON) scripts/validate_artifact.py --root artifact --profile smoke
	PYTHONPATH=src $(PYTHON) scripts/package_inventory.py --root .

audit:
	bash scripts/run_three_audits.sh

clean:
	rm -rf .pytest_cache .ruff_cache build dist *.egg-info src/*.egg-info
	find src scripts tests -type d -name __pycache__ -prune -exec rm -rf {} +
