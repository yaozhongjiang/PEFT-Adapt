# Three-pass audit checklist

## Pass 1 — static and scientific protocol

- run Ruff on `src/`, `scripts/`, and `tests/`;
- compile every Python module;
- import every package module;
- parse every YAML configuration and close every plan reference;
- verify calibration/threshold/test victim-training seed disjointness;
- verify the real base-mismatch matrix, unknown-family grid, strict lineage contrast, complete redeployability baselines, and Llama claim boundary;
- scan executable files for placeholders, credentials, private keys, and direct production-endpoint code.

## Pass 2 — functional

- run unit tests;
- execute the complete offline synthetic pipeline;
- execute a local tiny Transformers/PEFT integration with one Trainer step;
- save/reload an adapter and perform safe merge;
- execute weighted body-delta composition while restoring independent task heads;
- validate service/probe-view unknown rejection and service-pair lineage logic.

## Pass 3 — artifact and packaging

- validate raw JSONL and run manifests against JSON Schema;
- verify victim/base logit dimensions;
- aggregate summary CSVs;
- generate package inventory and SHA-256 checksums;
- verify ZIP integrity after packaging;
- record the exact tested software environment and distinguish smoke evidence from unexecuted full experiments.
