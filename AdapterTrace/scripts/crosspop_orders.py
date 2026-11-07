#!/usr/bin/env python3
"""Cross-population calibration with many independent *real* trainer orders.

For a suspect that does not derive from the owner, the owner enters the test
only through its training order w_O; the owner's adapter is never queried.
We can therefore obtain many independent owners without training: replay the
actual Trainer data loader for N fresh seeds spaced by more than the number of
epochs (so no two owners, and no owner and any shadow, share a permutation),
and test every null suspect against every replayed order. Owner orders are
now independent trials, so owner-level intervals are honest.

Suspect groups (all independent of the new orders): Dolly adapters, adapters
trained on other facts, distillation students, and the TOFU grid adapters
themselves (same-data siblings). The reference is the TOFU grid mean
(leave-one-out for grid suspects).
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
from scipy import stats

sys.path.insert(0, str(Path(__file__).resolve().parent))
from palimpsest_lineage import _one_sided_p, _train_dataset, independence_matrix, lr_weight, training_order  # noqa: E402

from adaptertrace.config import get, load_config  # noqa: E402

N_A = 2000


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, help="TOFU owner grid YAML")
    parser.add_argument("--n-owners", type=int, default=300)
    parser.add_argument("--seed-start", type=int, default=5001)
    parser.add_argument("--bootstrap", type=int, default=2000)
    args = parser.parse_args()
    cfg = load_config(args.config)
    root = Path(get(cfg, "experiment.output_root", "artifact"))
    name = cfg["experiment"]["name"]
    probe = np.load(root / "raw" / f"{name}_tofu_probe.npz")
    ll, labels, groups = probe["ll"][:, :N_A], probe["labels"].tolist(), probe["groups"].tolist()
    grid_orders = np.load(root / "raw" / f"{name}_palimpsest_loglik.npz")["positions"]

    dataset, _ = _train_dataset(cfg)
    epochs = max(1, int(np.ceil(float(cfg["training"].get("epochs", 1)))))
    seeds = [args.seed_start + epochs * i for i in range(args.n_owners)]
    orders = np.stack([training_order(dataset, cfg["training"], s) for s in seeds])
    print(f"[crosspop] replayed {len(seeds)} real trainer orders", flush=True)
    allpos = np.concatenate([orders, grid_orders], axis=0)
    indep = independence_matrix(allpos)
    n_new = len(seeds)
    shared_new = int((~indep[:n_new, :n_new])[~np.eye(n_new, dtype=bool)].sum())
    shared_grid = int((~indep[:n_new, n_new:]).sum())
    if shared_new or shared_grid:
        raise RuntimeError(f"replayed orders share permutations (new-new {shared_new}, new-grid {shared_grid})")
    np.savez_compressed(root / "raw" / f"{name}_replayed_orders.npz", seeds=np.asarray(seeds), positions=orders)
    W = np.stack([lr_weight(p, cfg["training"], N_A) for p in orders])

    grid_rows = [i for i, g in enumerate(groups) if g == "grid"]
    suspects = {g: [i for i, gg in enumerate(groups) if gg == g] for g in sorted(set(groups)) if g != "base"}
    rng = np.random.default_rng(0)
    result = {"n_owners": n_new, "seeds": [seeds[0], seeds[-1], epochs], "groups": {}}
    for group, rows in suspects.items():
        P = np.empty((n_new, len(rows)))
        for c, m in enumerate(rows):
            ref_rows = [r for r in grid_rows if r != m]
            r = ll[m] - ll[ref_rows].mean(axis=0)
            rr = stats.rankdata(r)
            for o in range(n_new):
                P[o, c] = _one_sided_p(float(stats.spearmanr(rr, W[o]).statistic), N_A)
        block = {"suspects": len(rows), "tests": int(P.size)}
        for a in (0.05, 0.01, 1e-3):
            hits = (P < a).mean(axis=1)  # per-owner FPR across suspects
            boot = [hits[rng.integers(0, n_new, n_new)].mean() for _ in range(args.bootstrap)]
            block[str(a)] = {"fpr": float((P < a).mean()),
                             "owner_bootstrap95": [float(np.percentile(boot, 2.5)), float(np.percentile(boot, 97.5))],
                             "max_per_suspect": float((P < a).mean(axis=0).max())}
        block["ks_uniform_p_first_suspect"] = float(stats.kstest(P[:, 0], "uniform").pvalue)
        result["groups"][group] = block
        print(f"== {group:<22} {json.dumps({k: v for k, v in block.items()})}", flush=True)
    out = root / "reports" / f"{name}_crosspop_orders.json"
    out.write_text(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
