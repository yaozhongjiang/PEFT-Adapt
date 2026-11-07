#!/usr/bin/env python3
"""Exact Monte-Carlo permutation null of the order statistic, per setting.

Under H0 the owner's epoch shuffles are uniform and independent of the
suspect and the reference, so the weight vector w_O is exchangeable over
examples. For a residual vector without ties, Spearman(r, w_O) then has the
same null distribution for every suspect: it depends only on the number of
queried examples, the schedule and the epoch structure. One large sample of
null correlations per (setting, n) therefore converts any reported rho into
its Monte-Carlo permutation p-value, p = (1 + #{rho* >= rho}) / (B + 1).

Writes raw/mc_null_<setting>_n<n>.npy (sorted null rho values).
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
from scipy import stats

SETTINGS = {
    # n_train, epochs, examples per optimizer step, warmup ratio
    "dolly": (4000, 1, 16, 0.03),
    "tofu": (2000, 5, 16, 0.03),
    "mnli": (8000, 1, 16, 0.03),
}


def lr_weight(pos: np.ndarray, n_train: int, epochs: int, per_step: int, warmup_ratio: float) -> np.ndarray:
    """Same schedule weight as scripts/palimpsest_lineage.lr_weight; pos has shape (..., epochs, n)."""
    steps = int(np.ceil(n_train / per_step)) * epochs
    warmup = int(np.ceil(steps * warmup_ratio))
    step = pos // per_step
    w = np.where(step < warmup, (step + 1) / max(warmup, 1), np.maximum(0.0, (steps - step) / max(steps - warmup, 1)))
    return w.sum(axis=-2)


def null_rho(setting: str, n: int, draws: int, seed: int, chunk: int = 2000) -> np.ndarray:
    n_train, epochs, per_step, warm = SETTINGS[setting]
    rng = np.random.default_rng(seed)
    out = np.empty(draws)
    r_rank = np.arange(n, dtype=float)
    r_c = r_rank - r_rank.mean()
    done = 0
    while done < draws:
        b = min(chunk, draws - done)
        # positions[e, i] = e * n_train + rank of example i in epoch e's shuffle
        perm = np.argsort(rng.random((b, epochs, n_train)), axis=-1)
        pos = np.empty_like(perm)
        np.put_along_axis(pos, perm, np.arange(n_train)[None, None, :], axis=-1)
        pos = pos + (np.arange(epochs) * n_train)[None, :, None]
        w = lr_weight(pos, n_train, epochs, per_step, warm)
        if n < n_train:
            # any subset chosen independently of the order is exchangeable; a
            # fresh random subset per draw is one such choice
            idx = np.argsort(rng.random((b, n_train)), axis=1)[:, :n]
            w = np.take_along_axis(w, idx, axis=1)
        wr = stats.rankdata(w, axis=1)
        wr = wr - wr.mean(axis=1, keepdims=True)
        out[done:done + b] = (wr @ r_c) / np.sqrt((wr ** 2).sum(axis=1) * (r_c ** 2).sum())
        done += b
    return np.sort(out)


def mc_p(rho: np.ndarray | float, null_sorted: np.ndarray) -> np.ndarray:
    b = null_sorted.size
    exceed = b - np.searchsorted(null_sorted, np.asarray(rho), side="left")
    return (1 + exceed) / (b + 1)


def t_p(rho, n):
    rho = np.clip(np.asarray(rho, float), -0.999999, 0.999999)
    return stats.t.sf(rho * np.sqrt((n - 2) / (1 - rho ** 2)), n - 2)


def rho_from_t_p(p, n):
    t = stats.t.isf(np.asarray(p, float), n - 2)
    return t / np.sqrt(n - 2 + t ** 2)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--setting", required=True, choices=SETTINGS)
    ap.add_argument("--n", type=int, required=True)
    ap.add_argument("--draws", type=int, default=1_000_000)
    ap.add_argument("--seed", type=int, default=20261106)
    args = ap.parse_args()
    out = Path(__file__).resolve().parents[1] / "raw" / f"mc_null_{args.setting}_n{args.n}.npy"
    null = null_rho(args.setting, args.n, args.draws, args.seed)
    np.save(out, null)
    for a in (0.05, 1e-3):
        crit = null[int(np.ceil((1 - a) * null.size)) - 1]
        print(f"{args.setting} n={args.n} alpha={a}: rho_crit={crit:.5f}  t-approx p at crit={float(t_p(crit, args.n)):.3g}")


if __name__ == "__main__":
    main()
