#!/usr/bin/env python3
"""Exact-null calibration of the order test.

Under H0 the only random object is the owner's shuffle. Drawing fresh owner
shuffles therefore samples the exact null distribution of the statistic for a
fixed suspect and reference. For each setting we

  1. estimate the true type-I error of the t-approximation p-value at each
     alpha (fraction of synthetic-owner draws with t-approx p < alpha), with a
     Clopper-Pearson interval, for several fixed null suspects;
  2. compute Monte-Carlo permutation p-values for every real owner (B draws;
     reported as an upper bound (1 + exceed) / (B + 1));
  3. recompute the real-owner/real-sibling FPR with an owner-cluster bootstrap
     interval, since those pairs share owners and suspects.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import yaml
from scipy import stats

sys.path.insert(0, str(Path(__file__).resolve().parent))
from palimpsest_lineage import _one_sided_p, independence_matrix, lr_weight, reference_ids  # noqa: E402


def _rank(x: np.ndarray) -> np.ndarray:
    r = stats.rankdata(x)
    return (r - r.mean()) / np.sqrt(((r - r.mean()) ** 2).sum())


def _cp(k: int, n: int, level: float = 0.95) -> list[float]:
    a = 1 - level
    lo = 0.0 if k == 0 else stats.beta.ppf(a / 2, k, n - k + 1)
    hi = 1.0 if k == n else stats.beta.ppf(1 - a / 2, k + 1, n - k)
    return [float(lo), float(hi)]


def synthetic_weight_chunks(training_cfg: dict, n: int, epochs: int, draws: int, rng: np.random.Generator, chunk: int = 5000):
    """Yield centred weight ranks for fresh owner shuffles, `chunk` draws at a time.

    Every owner's shuffle has the same distribution under H0, so one stream of
    draws serves all suspects and all real owners.
    """
    done = 0
    while done < draws:
        m = min(chunk, draws - done)
        out = np.empty((m, n))
        for d in range(m):
            pos = np.stack([e * n + rng.permutation(n) for e in range(epochs)])
            out[d] = _rank(lr_weight(pos if epochs > 1 else pos[0], training_cfg, n))
        done += m
        yield out


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--npz", required=True)
    parser.add_argument("--config", required=True)
    parser.add_argument("--label", required=True)
    parser.add_argument("--draws", type=int, default=100000)
    parser.add_argument("--suspects", type=int, default=8)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()
    rng = np.random.default_rng(0)
    cfg = yaml.safe_load(open(args.config))["training"]
    d = np.load(args.npz)
    A, P = d["adapter_loglik"], d["positions"]
    k, n = A.shape
    epochs = P.shape[1] if P.ndim == 3 else 1
    indep = independence_matrix(P)
    alphas = [0.05, 0.01, 1e-3]
    result: dict = {"label": args.label, "n_examples": int(n), "epochs": epochs, "alphas": alphas,
                    "draws": args.draws}

    # residual-rank vectors: null suspects (reference = all other adapters) and real owners
    null_ids = rng.choice(k, size=min(args.suspects, k), replace=False)
    R_null = np.stack([_rank(A[s] - A[[l for l in range(k) if l != s]].mean(axis=0)) for s in null_ids], axis=1)
    R_own = np.stack([_rank(A[i] - A[reference_ids(indep, i)].mean(axis=0)) for i in range(k)], axis=1)
    rho_obs = np.array([float(_rank(lr_weight(P[i], cfg, n)) @ R_own[:, i]) for i in range(k)])
    # t-approximation thresholds on rho for each alpha (one-sided)
    def rho_crit(a: float) -> float:
        t = stats.t.isf(a, n - 2)
        return float(t / np.sqrt(n - 2 + t * t))
    crit = {a: rho_crit(a) for a in alphas}
    null_hits = {a: 0 for a in alphas}
    exceed = np.zeros(k, dtype=np.int64)
    for W in synthetic_weight_chunks(cfg, n, epochs, args.draws, rng):
        rho_null = W @ R_null            # (chunk, suspects): exact null draws
        for a in alphas:
            null_hits[a] += int(np.sum(rho_null >= crit[a]))
        exceed += np.sum(W @ R_own >= rho_obs[None, :], axis=0)
    total = args.draws * R_null.shape[1]
    # 1) true type-I error of the t-approximation under the exact null
    result["t_approx_type1"] = {str(a): {"rate": null_hits[a] / total, "cp95": _cp(null_hits[a], total)} for a in alphas}
    result["t_approx_type1_draws"] = total
    # 2) Monte-Carlo permutation p-values for every real owner
    p_perm = (1 + exceed) / (1 + args.draws)
    result["owners"] = {
        "n": k,
        "p_perm_upper_max": float(p_perm.max()),
        "p_perm_below_1e-3": int(np.sum(p_perm < 1e-3)),
        "zero_exceedance_owners": int(np.sum(exceed == 0)),
        "rows": [{"owner": i, "rho": float(rho_obs[i]), "p_t": _one_sided_p(float(rho_obs[i]), n),
                  "p_perm_upper": float(p_perm[i]), "exceed": int(exceed[i])} for i in range(k)],
    }

    # 3) real sibling pairs: FPR with owner-cluster bootstrap
    pairs = []
    for i in range(k):
        w = lr_weight(P[i], cfg, n)
        for j in range(k):
            if i == j or not indep[i, j]:
                continue
            ref = A[reference_ids(indep, i, j)].mean(axis=0)
            pairs.append((i, _one_sided_p(float(stats.spearmanr(A[j] - ref, w).statistic), n)))
    pairs_arr = np.array(pairs)
    boot = {a: [] for a in alphas}
    owners = np.unique(pairs_arr[:, 0])
    for _ in range(2000):
        pick = rng.choice(owners, size=owners.size, replace=True)
        ps = np.concatenate([pairs_arr[pairs_arr[:, 0] == o, 1] for o in pick])
        for a in alphas:
            boot[a].append(np.mean(ps < a))
    result["sibling_pairs"] = {
        "n_pairs": int(len(pairs)),
        "n_owner_clusters": int(owners.size),
        **{str(a): {"fpr": float(np.mean(pairs_arr[:, 1] < a)),
                    "owner_bootstrap95": [float(np.percentile(boot[a], 2.5)), float(np.percentile(boot[a], 97.5))]}
           for a in alphas},
    }
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(result, indent=2))
    print(json.dumps({k2: v for k2, v in result.items() if k2 != "owners"} | {"owners_summary": {
        kk: vv for kk, vv in result["owners"].items() if kk != "rows"}}, indent=1))


if __name__ == "__main__":
    main()
