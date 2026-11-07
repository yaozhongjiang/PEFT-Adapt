#!/usr/bin/env python3
"""Malicious-accuser experiments: can an owner manufacture a false accusation?

Every suspect here is innocent: an independently trained same-recipe sibling
(H0^run is true). The accuser holds real per-example scores of all adapters
and replays real training orders. We measure the fraction of innocent
suspects that end up "detected" when the accuser adapts its analysis after
seeing the suspect, and whether pre-commitment or a multiplicity correction
restores the nominal level. Decisions use the Monte-Carlo permutation null
(raw/mc_null_<setting>_n<n>.npy); for a fixed n the p-value is monotone in
rho, so "best of M" equals "largest rho of M".

Attacks (A) and protocol checks (P):
  A1 forged order       : order chosen after seeing the suspect (no commitment)
  A2 seed grinding      : best of M real or fresh seeds (commitment after the fact)
  A3 query-set shopping : committed order, best of M random query subsets
  A4 cherry-picking     : query subset chosen with both r_S and w_O
  A5 reference shopping : committed order, best of M random shadow subsets
  P1 suspect-only choice: query subset / shadows chosen from suspect outputs
                          only (allowed by the proposition)
  P2 Bonferroni         : A3/A5 at alpha / M, and its cost for true owners
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
from scipy import stats

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
import mc_null as M  # noqa: E402

N_Q = 250  # queried examples
ALPHAS = (0.05, 1e-3)
rng = np.random.default_rng(20261106)


def ranks_c(x: np.ndarray) -> np.ndarray:
    r = stats.rankdata(x, axis=-1)
    return r - r.mean(axis=-1, keepdims=True)


def corr_rows(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Spearman between matching rows of already centred ranks."""
    return (a * b).sum(-1) / np.sqrt((a * a).sum(-1) * (b * b).sum(-1))


def crit(null: np.ndarray, alpha: float) -> float:
    """Smallest rho whose MC p-value is below alpha."""
    k = int(np.floor(alpha * (null.size + 1))) - 1  # need 1 + exceed <= alpha (B+1) - tiny
    return float(null[null.size - 1 - max(k, 0)]) if k >= 0 else np.inf


def rate(rho: np.ndarray, null: np.ndarray) -> dict:
    p = M.mc_p(rho, null)
    return {f"{a:g}": float(np.mean(p < a)) for a in ALPHAS} | {"n": int(np.size(rho))}


def dolly() -> dict:
    d = np.load(ROOT / "raw/llama31_dolly_grid_palimpsest_loglik.npz")
    ll, pos = d["adapter_loglik"], d["positions"]
    k, n = ll.shape
    nt, ep, ps, wr = M.SETTINGS["dolly"]
    W = M.lr_weight(pos[:, None, :], nt, ep, ps, wr)  # (k, n)
    null = np.load(ROOT / f"raw/mc_null_dolly_n{N_Q}.npy")
    null_full = np.load(ROOT / "raw/mc_null_dolly_n4000.npy")
    pairs = [(i, j) for i in range(k) for j in range(k) if i != j]
    q0 = np.sort(rng.choice(n, N_Q, replace=False))  # committed query subset
    out: dict = {"setting": "Llama/Dolly", "owners": k, "null_pairs": len(pairs), "n_query": N_Q}

    def resid(i, j, shadows=None):
        sh = [l for l in range(k) if l not in (i, j)] if shadows is None else shadows
        return ll[j] - ll[sh].mean(axis=0)

    Rfull = np.stack([resid(i, j) for i, j in pairs])  # (P, n)
    Wo = np.stack([W[i] for i, _ in pairs])

    # honest, pre-registered
    rho0 = corr_rows(ranks_c(Rfull[:, q0]), ranks_c(Wo[:, q0]))
    out["honest"] = rate(rho0, null)

    # A1 forged order: give the largest schedule weights to the examples with
    # the largest residual (the order a forger would write down)
    rho_f = corr_rows(ranks_c(Rfull[:, q0]), ranks_c(np.sort(W[0][q0])[np.argsort(np.argsort(Rfull[:, q0], axis=1), axis=1)]))
    out["A1_forged_order"] = rate(rho_f, null) | {"median_rho": float(np.median(rho_f))}

    Ms = (1, 10, 100, 1000)
    # A2 seed grinding over fresh uniform shuffles (the trainer's sampler is uniform)
    G = 3000
    perm = np.argsort(rng.random((G, nt)), axis=1)
    gpos = np.empty_like(perm)
    np.put_along_axis(gpos, perm, np.arange(nt)[None, :], axis=1)
    Wg = M.lr_weight(gpos[:, None, :], nt, ep, ps, wr)[:, q0]
    Rg = ranks_c(Rfull[:, q0])
    rho_g = (Rg @ ranks_c(Wg).T) / np.sqrt((Rg ** 2).sum(1)[:, None] * (ranks_c(Wg) ** 2).sum(1)[None, :])
    # each accusation grinds its own random set of M seeds from the pool
    order = np.argsort(rng.random(rho_g.shape), axis=1)
    rho_g = np.take_along_axis(rho_g, order, axis=1)
    out["A2_seed_grinding"] = {str(m): rate(rho_g[:, :m].max(1), null) for m in Ms}

    # A3 query-set shopping and A5 reference shopping (subsample of pairs for cost)
    sub = np.arange(len(pairs))
    a3 = {m: [] for m in Ms}
    a5 = {m: [] for m in Ms}
    for pi in sub:
        i, j = pairs[pi]
        qs = np.argsort(rng.random((max(Ms), n)), axis=1)[:, :N_Q]
        rho = corr_rows(ranks_c(Rfull[pi][qs]), ranks_c(W[i][qs]))
        pool = [l for l in range(k) if l not in (i, j)]
        sh = np.argsort(rng.random((max(Ms), len(pool))), axis=1)[:, :5]
        refs = ll[np.asarray(pool)[sh]].mean(axis=1)[:, q0]  # (M, N_Q)
        rho5 = corr_rows(ranks_c(ll[j][q0][None, :] - refs), ranks_c(W[i][q0])[None, :])
        for m in Ms:
            a3[m].append(rho[:m].max())
            a5[m].append(rho5[:m].max())
    out["A3_query_shopping"] = {str(m): rate(np.asarray(a3[m]), null) for m in Ms}
    out["A5_reference_shopping"] = {str(m): rate(np.asarray(a5[m]), null) for m in Ms}
    # Bonferroni at alpha/M
    bonf = {}
    for m in Ms:
        p3 = M.mc_p(np.asarray(a3[m]), null) * m
        p5 = M.mc_p(np.asarray(a5[m]), null) * m
        bonf[str(m)] = {"A3": {f"{a:g}": float(np.mean(p3 < a)) for a in ALPHAS},
                        "A5": {f"{a:g}": float(np.mean(p5 < a)) for a in ALPHAS}}
    out["P2_bonferroni"] = bonf

    # A4 cherry-picking with both r and w: keep the N_Q examples (of all 4000)
    # whose residual and weight agree most
    zr = (Rfull - Rfull.mean(1, keepdims=True)) / Rfull.std(1, keepdims=True)
    zw = (Wo - Wo.mean(1, keepdims=True)) / Wo.std(1, keepdims=True)
    pick = np.argsort(-(zr * zw), axis=1)[:, :N_Q]
    rho_c = corr_rows(ranks_c(np.take_along_axis(Rfull, pick, 1)), ranks_c(np.take_along_axis(Wo, pick, 1)))
    out["A4_cherry_pick"] = rate(rho_c, null)

    # P1 suspect-only adaptive choices (do not look at the order)
    pick_abs = np.argsort(-np.abs(Rfull), axis=1)[:, :N_Q]
    rho_p1 = corr_rows(ranks_c(np.take_along_axis(Rfull, pick_abs, 1)), ranks_c(np.take_along_axis(Wo, pick_abs, 1)))
    pick_hi = np.argsort(-np.stack([ll[j] for _, j in pairs]), axis=1)[:, :N_Q]
    rho_p1b = corr_rows(ranks_c(np.take_along_axis(Rfull, pick_hi, 1)), ranks_c(np.take_along_axis(Wo, pick_hi, 1)))
    rho_p1c = []
    for pi, (i, j) in enumerate(pairs):
        pool = [l for l in range(k) if l not in (i, j)]
        c = [np.corrcoef(ll[j][q0], ll[l][q0])[0, 1] for l in pool]
        sh = [pool[t] for t in np.argsort(c)[-5:]]  # 5 shadows most similar to the suspect
        rho_p1c.append(corr_rows(ranks_c(ll[j][q0] - ll[sh][:, q0].mean(0)), ranks_c(W[i][q0])))
    out["P1_suspect_only"] = {"queries_largest_abs_residual": rate(rho_p1, null),
                              "queries_highest_suspect_loglik": rate(rho_p1b, null),
                              "shadows_most_similar_to_suspect": rate(np.asarray(rho_p1c), null)}

    # power of the true owners at n = N_Q under Bonferroni
    own = np.array([corr_rows(ranks_c(resid(i, i)[q0]), ranks_c(W[i][q0])) for i in range(k)])
    p_own = M.mc_p(own, null)
    out["owners_power"] = {"min_mc_p_floor": 1 / (null.size + 1),
                           **{f"alpha/{m}": float(np.mean(p_own < 1e-3 / m)) for m in Ms},
                           "median_rho": float(np.median(own)),
                           "t_approx_median_p": float(np.median(M.t_p(own, N_Q)))}
    out["honest_full_n4000"] = rate(np.array([corr_rows(ranks_c(Rfull[p]), ranks_c(Wo[p])) for p in range(len(pairs))]), null_full)
    return out


def tofu() -> dict:
    """Seed grinding with real replayed trainer orders (TOFU, 5 epochs)."""
    d = np.load(ROOT / "raw/llama31_tofu_grid_palimpsest_loglik.npz")
    ll, pos = d["adapter_loglik"], d["positions"]
    k, n = ll.shape
    nt, ep, ps, wr = M.SETTINGS["tofu"]
    orders = [np.load(ROOT / "raw/llama31_tofu_grid_replayed_orders.npz")["positions"]]
    orders += [np.load(f)["positions"] for f in sorted((ROOT / "raw").glob("llama31_tofu_grid_replayed_extra_*.npz"))]
    O = np.concatenate(orders)  # (G, 5, 2000) real replayed orders, disjoint from the grid
    Wg = M.lr_weight(O, nt, ep, ps, wr)
    null = np.load(ROOT / "raw/mc_null_tofu_n2000.npy")
    # suspect j, reference = every other grid adapter independent of j
    sys.path.insert(0, str(ROOT / "scripts"))
    sigs = [{np.argsort(e).tobytes() for e in pos[i]} for i in range(k)]
    R = []
    for j in range(k):
        ref = [l for l in range(k) if l != j and not (sigs[l] & sigs[j])]
        R.append(ll[j] - ll[ref].mean(0))
    Rr = ranks_c(np.stack(R))
    Wr = ranks_c(Wg)
    rho = (Rr @ Wr.T) / np.sqrt((Rr ** 2).sum(1)[:, None] * (Wr ** 2).sum(1)[None, :])  # (k, G)
    G = rho.shape[1]
    out = {"setting": "Llama/TOFU", "suspects": k, "replayed_orders": int(G)}
    for m in (1, 10, 100, 1000):
        blocks = G // m
        best = rho[:, : blocks * m].reshape(k, blocks, m).max(axis=2).ravel()
        out[str(m)] = rate(best, null) | {"bonferroni": {f"{a:g}": float(np.mean(M.mc_p(best, null) * m < a)) for a in ALPHAS}}
    return out


def main() -> None:
    res = {"dolly": dolly(), "tofu_seed_grinding": tofu()}
    (ROOT / "reports").mkdir(exist_ok=True)
    (ROOT / "reports/r1_malicious_accuser.json").write_text(json.dumps(res, indent=2))
    print(json.dumps(res, indent=1))


if __name__ == "__main__":
    main()
