#!/usr/bin/env python3
"""Executable commit / verify protocol for run provenance.

COMMIT (owner, at training time, before any suspect exists)
  digest = SHA-256 over: tokenized training set, training configuration,
  shuffle seed, trainer software versions, SHA-256 of the owner's adapter, and
  the analysis plan (number of queried examples, seed of the query subset,
  shadow adapters or the seeds the arbiter will train them with, alpha,
  Monte-Carlo draws, and the maximum number of tests the owner may run).
  The digest is published with a timestamp (e.g. a timestamping authority or
  a public log); timestamping itself is outside this script.

VERIFY (arbiter, in a dispute)
  1. recompute every hash from the disclosed materials and compare;
  2. replay the owner's order from the committed seed with the public trainer;
  3. check the owner's own adapter carries the trace (sanity);
  4. query the suspect on the committed query subset, score the committed
     shadows, compute rho and its Monte-Carlo permutation p-value;
  5. reject H0^run if p < alpha / max_tests.
Anything not fixed by the commitment (choice of order, query subset, shadows,
number of tests) is exactly what the malicious-accuser experiments exploit.
"""
from __future__ import annotations

import hashlib
import json
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
from scipy import stats


def sha256_bytes(*parts: bytes) -> str:
    h = hashlib.sha256()
    for p in parts:
        h.update(hashlib.sha256(p).digest())
    return h.hexdigest()


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def canon(obj: Any) -> bytes:
    return json.dumps(obj, sort_keys=True, separators=(",", ":")).encode()


def dataset_hash(dataset) -> str:
    ids = [bytes(np.asarray(r, dtype=np.int32)) for r in dataset["input_ids"]]
    lab = [bytes(np.asarray(r, dtype=np.int32)) for r in dataset["labels"]]
    return sha256_bytes(*ids, *lab)


def commit(dataset, training_cfg: dict, seed: int, adapter: Path, plan: dict, versions: dict) -> dict:
    record = {
        "protocol": "adaptertrace-run-provenance/v1",
        "data_sha256": dataset_hash(dataset),
        "n_train": len(dataset),
        "training_config": training_cfg,
        "seed": int(seed),
        "software": versions,
        "adapter_sha256": sha256_file(adapter / "adapter_model.safetensors"),
        "plan": plan,
    }
    record["digest"] = hashlib.sha256(canon(record)).hexdigest()
    return record


def check(record: dict, dataset, adapter: Path, shadow_dirs: list[Path]) -> dict:
    body = {k: v for k, v in record.items() if k != "digest"}
    ok = {
        "digest": hashlib.sha256(canon(body)).hexdigest() == record["digest"],
        "data": dataset_hash(dataset) == record["data_sha256"],
        "adapter": sha256_file(adapter / "adapter_model.safetensors") == record["adapter_sha256"],
        "shadows": [sha256_file(d / "adapter_model.safetensors") for d in shadow_dirs] == record["plan"]["shadow_sha256"],
    }
    ok["all"] = all(v for v in ok.values())
    return ok


def query_subset(plan: dict, n_train: int) -> np.ndarray:
    return np.sort(np.random.default_rng(plan["query_seed"]).choice(n_train, plan["n_query"], replace=False))


def spearman(r: np.ndarray, w: np.ndarray) -> float:
    return float(stats.spearmanr(r, w).statistic)


def decide(rho: float, null_sorted: np.ndarray, plan: dict) -> dict:
    b = null_sorted.size
    p = (1 + (b - np.searchsorted(null_sorted, rho, side="left"))) / (b + 1)
    level = plan["alpha"] / plan["max_tests"]
    return {"rho": rho, "p_mc": float(p), "level": level, "reject_H0_run": bool(p < level),
            "p_t_descriptive": float(stats.t.sf(rho * np.sqrt((plan["n_query"] - 2) / (1 - rho ** 2)), plan["n_query"] - 2))}
