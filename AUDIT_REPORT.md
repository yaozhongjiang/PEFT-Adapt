# Three-pass audit report

All three independent audit rounds passed.

## Round 1 — static and protocol
Ruff, compilation, shell syntax, CLI/module imports, plan closure, seed partitioning, strict-lineage controls, posterior-to-redeploy wiring, secret scan, and local-only policy passed.

## Round 2 — functional
Unit tests, the complete synthetic pipeline, and an offline tiny Transformers/PEFT integration passed. The integration covers a real Trainer step, independent adapters, save/reload, safe merge, and weighted body-delta composition with task-specific heads.

## Round 3 — artifact and statistics
Raw JSONL, finite vectors, unique query identities, manifests, CSV/report parsing, locked split checks, cluster bootstrap, inventory, and SHA-256 generation passed.

## Verification boundary
Paper-scale BERT/Llama numerical runs were not executed in this CPU-oriented environment. Their outputs must be generated from the locked configurations and supported by the resulting raw logs.
