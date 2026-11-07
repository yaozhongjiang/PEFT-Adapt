#!/usr/bin/env python3
"""Is the small-alpha tail of the cross-population test inflated?

Two independent checks on the same null suspects:

  real   additional real trainer orders (chunks replayed by a job array)
  exact  fresh uniform shuffles, the exact null of the statistic

Suspects in a group share a corpus, so their residuals correlate and a single
owner order can reject many of them at once. Pooled rates therefore stay
unbiased but are lumpy; the per-owner hit counts make that visible.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
from scipy import stats

sys.path.insert(0, str(Path(__file__).resolve().parent))
from palimpsest_lineage import _train_dataset, lr_weight, training_order  # noqa: E402
from permutation_calibration import _cp, _rank, synthetic_weight_chunks  # noqa: E402

from adaptertrace.config import get, load_config  # noqa: E402

N_A = 2000
ALPHAS = (0.05, 0.01, 1e-3)


def rho_crit(a: float, n: int) -> float:
    t = stats.t.isf(a, n - 2)
    return float(t / np.sqrt(n - 2 + t * t))


def replay(cfg, root: Path, name: str, chunk: int, size: int, start: int) -> None:
    dataset, _ = _train_dataset(cfg)
    epochs = max(1, int(np.ceil(float(cfg["training"].get("epochs", 1)))))
    seeds = [start + epochs * (chunk * size + i) for i in range(size)]
    orders = np.stack([training_order(dataset, cfg["training"], s) for s in seeds])
    np.savez_compressed(root / "raw" / f"{name}_replayed_extra_{chunk:03d}.npz", seeds=np.asarray(seeds), positions=orders)
    print(f"[tail] chunk {chunk}: seeds {seeds[0]}..{seeds[-1]}", flush=True)


def analyze(cfg, root: Path, name: str, draws: int) -> None:
    raw = root / "raw"
    probe = np.load(raw / f"{name}_tofu_probe.npz")
    ll, groups = probe["ll"][:, :N_A], probe["groups"].tolist()
    grid_rows = [i for i, g in enumerate(groups) if g == "grid"]
    tc = cfg["training"]
    parts = [np.load(raw / f"{name}_replayed_orders.npz")] + [np.load(f) for f in sorted(raw.glob(f"{name}_replayed_extra_*.npz"))]
    seeds = np.concatenate([p["seeds"] for p in parts])
    orders = np.concatenate([p["positions"] for p in parts])
    grid_orders = np.load(raw / f"{name}_palimpsest_loglik.npz")["positions"]
    # no epoch shuffle may be shared among replayed owners or with the grid
    # (grid adapters with nearby seeds do share shuffles with each other)
    def keys(block: np.ndarray) -> list[bytes]:
        return [(m[e] - m[e].min()).astype(np.int32).tobytes() for m in block for e in range(m.shape[0])]

    new_keys = keys(orders)
    if len(set(new_keys)) != len(new_keys) or set(new_keys) & set(keys(grid_orders)):
        raise RuntimeError("Replayed orders share an epoch shuffle")
    W = np.stack([_rank(lr_weight(p, tc, N_A)) for p in orders])
    print(f"[tail] {len(seeds)} real orders, all epoch shuffles distinct", flush=True)
    suspects = {g: [i for i, gg in enumerate(groups) if gg == g] for g in sorted(set(groups)) if g != "base"}
    R = {g: np.stack([_rank(ll[m] - ll[[r for r in grid_rows if r != m]].mean(axis=0)) for m in rows], axis=1)
         for g, rows in suspects.items()}
    crit = {a: rho_crit(a, N_A) for a in ALPHAS}
    exact_hits = {g: {a: 0 for a in ALPHAS} for g in R}
    exact_owner_any = {g: 0 for g in R}
    rng = np.random.default_rng(11)
    for chunk in synthetic_weight_chunks(tc, N_A, orders.shape[1], draws, rng):
        for g, r in R.items():
            rho = chunk @ r
            for a in ALPHAS:
                exact_hits[g][a] += int((rho >= crit[a]).sum())
            exact_owner_any[g] += int((rho >= crit[1e-3]).any(axis=1).sum())
    result: dict = {"n_real_orders": int(len(seeds)), "exact_draws": draws, "groups": {}}
    for g, r in R.items():
        rho = W @ r
        c = np.corrcoef(r.T) if r.shape[1] > 1 else np.ones((1, 1))
        block: dict = {"suspects": r.shape[1], "mean_residual_corr": float(c[np.triu_indices_from(c, 1)].mean()) if r.shape[1] > 1 else None}
        for a in ALPHAS:
            hits = rho >= crit[a]
            per_owner = hits.mean(axis=1)
            boot = [per_owner[rng.integers(0, per_owner.size, per_owner.size)].mean() for _ in range(2000)]
            total = draws * r.shape[1]
            block[str(a)] = {
                "real_fpr": float(hits.mean()),
                "real_owner_boot95": [float(np.percentile(boot, 2.5)), float(np.percentile(boot, 97.5))],
                "real_owners_any_hit": int(hits.any(axis=1).sum()),
                "exact_rate": exact_hits[g][a] / total,
                "exact_cp95": _cp(exact_hits[g][a], total),
            }
        block["exact_owner_any_hit_1e-3"] = exact_owner_any[g] / draws
        block["hit_owner_counts_1e-3"] = np.sort((rho >= crit[1e-3]).sum(axis=1))[-8:].tolist()
        result["groups"][g] = block
        print(f"== {g:<22} {json.dumps(block)}", flush=True)
    (root / "reports" / f"{name}_crosspop_tail.json").write_text(json.dumps(result, indent=2))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--stage", choices=["replay", "analyze"], required=True)
    parser.add_argument("--chunk", type=int, default=0)
    parser.add_argument("--chunk-size", type=int, default=250)
    parser.add_argument("--seed-start", type=int, default=7001)
    parser.add_argument("--draws", type=int, default=200000)
    args = parser.parse_args()
    cfg = load_config(args.config)
    root = Path(get(cfg, "experiment.output_root", "artifact"))
    name = cfg["experiment"]["name"]
    if args.stage == "replay":
        replay(cfg, root, name, args.chunk, args.chunk_size, args.seed_start)
    else:
        analyze(cfg, root, name, args.draws)


if __name__ == "__main__":
    main()
