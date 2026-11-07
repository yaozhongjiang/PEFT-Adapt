#!/usr/bin/env python3
"""Recompute every detection decision in the paper with Monte-Carlo permutation p-values.

For a tie-free residual the permutation null of rho depends only on the
setting and the number of queried examples (see mc_null.py), so rows that
stored only rho (or only the t-approximate p, which is monotone in rho) are
converted exactly. Scores with ties (text-only metrics) are ranked with
random tie-breaking, which keeps the permutation test valid.
Prints, per table cell, the t-approximate decision count and the MC count.
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

RAW, REP = ROOT / "raw", ROOT / "reports"
NULL: dict = {}
rng = np.random.default_rng(11)


def null(setting: str, n: int) -> np.ndarray:
    if (setting, n) not in NULL:
        NULL[setting, n] = np.load(RAW / f"mc_null_{setting}_n{n}.npy")
    return NULL[setting, n]


def counts(rho, setting, n):
    rho = np.asarray(rho, float)
    pm = M.mc_p(rho, null(setting, n))
    pt = M.t_p(rho, n)
    return {"n_owners": int(rho.size),
            "t@1e-3": int((pt < 1e-3).sum()), "mc@1e-3": int((pm < 1e-3).sum()),
            "t@.05": int((pt < 0.05).sum()), "mc@.05": int((pm < 0.05).sum()),
            "min_mc_p": float(pm.min()) if rho.size else None}


def indep_matrix(P):
    pos = P if P.ndim == 3 else P[:, None, :]
    sig = [{np.argsort(e).tobytes() for e in pos[i]} for i in range(pos.shape[0])]
    return np.array([[i != j and not (sig[i] & sig[j]) for j in range(len(sig))] for i in range(len(sig))])


def ref_ids(ind, *m):
    return [l for l in range(ind.shape[0]) if l not in m and all(ind[l, x] for x in m)]


def weights(P, setting):
    nt, ep, ps, wr = M.SETTINGS[setting]
    return M.lr_weight(P if P.ndim == 3 else P[:, None, :], nt, ep, ps, wr)


def spear(r, w, tiebreak=False):
    if tiebreak:
        r = r + rng.random(r.shape) * 1e-9 * (np.abs(r).max() + 1)
    return float(stats.spearmanr(r, w).statistic)


def detection():
    out = {}
    for name, setting in (("llama31_dolly_grid", "dolly"), ("qwen25_dolly_grid", "dolly"),
                          ("llama31_tofu_grid", "tofu"), ("llama31_lineage_victims", "mnli")):
        d = np.load(RAW / f"{name}_palimpsest_loglik.npz")
        A, P = d["adapter_loglik"], d["positions"]
        k, n = A.shape
        W = weights(P, setting)
        ind = indep_matrix(P)
        own, nul = [], []
        for i in range(k):
            for j in range(k):
                if i != j and not ind[i, j]:
                    continue
                ref = A[ref_ids(ind, i, j) if i != j else ref_ids(ind, i)].mean(0)
                (own if i == j else nul).append(spear(A[j] - ref, W[i]))
        nm = M.mc_p(np.asarray(nul), null(setting, n))
        out[name] = {"owners": counts(own, setting, n),
                     "null_pairs": len(nul),
                     "null_fpr_t": [float(np.mean(M.t_p(np.asarray(nul), n) < a)) for a in (0.05, 1e-3)],
                     "null_fpr_mc": [float(np.mean(nm < a)) for a in (0.05, 1e-3)]}
    return out


def evasion():
    out = {}
    n_of = {"llama31_dolly_grid": ("dolly", 1000), "qwen25_dolly_grid": ("dolly", 1000),
            "llama31_tofu_grid": ("tofu", 2000), "llama31_lineage_victims": ("mnli", 8000)}
    for f in sorted(REP.glob("*_palimpsest_evasion*.json")):
        name = f.name.split("_palimpsest_evasion")[0]
        setting, n = n_of[name]
        d = json.loads(f.read_text())
        by = {}
        for r in d["rows"]:
            by.setdefault(r["variant"], []).append(r["rho"])
        for v, rhos in by.items():
            # sanity: the stored p is the t approximation at this n
            out[f"{f.stem}/{v}"] = counts(rhos, setting, n)
        chk = d["rows"][0]
        assert abs(M.t_p(chk["rho"], n) - chk["p"]) <= 1e-6 * max(chk["p"], 1e-300) + 1e-12, (f.name, chk)
    for name, setting in (("llama31_dolly_grid", "dolly"), ("llama31_tofu_grid", "tofu")):
        for src in ("ownerprompts", "otherdata"):
            d = json.loads((REP / f"{name}_distill_{src}.json").read_text())
            n = len(np.load(RAW / f"{name}_distill_{src}_scores.npz")["keep"])
            out[f"{name}_distill_{src}"] = counts([r["rho"] for r in d["rows"]], setting, n)
    return out


def contft():
    """Table contft from raw scores, t-approximate and MC."""
    d = np.load(RAW / "llama31_tofu_grid_palimpsest_loglik.npz")
    A, P = d["adapter_loglik"], d["positions"]
    n = A.shape[1]
    W = weights(P, "tofu")
    ind = indep_matrix(P)

    def merge(tag):
        parts = [np.load(f) for f in sorted(RAW.glob(f"llama31_tofu_grid_contft_{tag}*.npz"))]
        return np.concatenate([p["indices"] for p in parts]), np.concatenate([p["ll"] for p in parts]), np.concatenate([p["utility"] for p in parts])

    out = {}
    for block, thief_tag, same_tag in (("thief_101-200", "thief", "sim_same"), ("thief_101-150", "h_thief", "h_sim")):
        ti, th, tu = merge(thief_tag + "_")
        di, dl, _ = merge("sim_dolly")
        si, sl, _ = merge(same_tag + ("_" if same_tag == "sim_same" else ""))
        rows = {}
        for s_i, step in enumerate((50, 200, 800)):
            for label in ("unmodified", "dolly_agnostic", "dolly_matched", "same_agnostic", "same_matched"):
                rhos = []
                for c, o in enumerate(ti):
                    o = int(o)
                    if label == "unmodified":
                        ref = A[ref_ids(ind, o)].mean(0)
                    else:
                        idx, ll = (di, dl) if label.startswith("dolly") else (si, sl)
                        keep = [q for q, g in enumerate(idx) if ind[o, g]]
                        ref = ll[keep, s_i].mean(0) if label.endswith("matched") else ll[keep].mean((0, 1))
                    rhos.append(spear(th[c, s_i] - ref, W[o]))
                rows[f"{label}/step{step}"] = counts(rhos, "tofu", n)
        out[block] = {"owners": ti.tolist(), "value_utility": tu.mean(0).tolist(), "rows": rows}
    return out


def textonly():
    out = {}
    for name, setting, f in (("llama31_dolly_grid", "dolly", "llama31_dolly_grid_textonly.npz"),
                             ("llama31_dolly_grid", "dolly", "../../overflow_results/raw/llama31_dolly_grid_textonly_n4000.npz"),
                             ("llama31_tofu_grid", "tofu", "llama31_tofu_grid_textonly.npz")):
        d = np.load(RAW / f)
        orders = np.load(RAW / f"{name}_palimpsest_loglik.npz")
        k = d["ngram4"].shape[0]
        keep = d["keep"]
        assert orders["seeds"].tolist()[:k] == d["seeds"].tolist()
        Pk = orders["positions"][:k]
        W = weights(Pk, setting)[:, keep]
        ind = indep_matrix(Pk)
        n = keep.size
        for metric in ("ngram4", "lcp"):
            for m_use in sorted({1, 2, 4, d[metric].shape[2]}):
                S = d[metric][:, :, :m_use].mean(2)
                own, nul = [], []
                for i in range(k):
                    for j in range(k):
                        if i != j and not ind[i, j]:
                            continue
                        ref = S[ref_ids(ind, i, j) if i != j else ref_ids(ind, i)].mean(0)
                        (own if i == j else nul).append(spear(S[j] - ref, W[i], tiebreak=True))
                nm = M.mc_p(np.asarray(nul), null_for(setting, n))
                c = counts(own, setting, n) if (setting, n) in NULL or (RAW / f"mc_null_{setting}_n{n}.npy").exists() else {}
                c["null_fpr_mc"] = [float(np.mean(nm < a)) for a in (0.05, 1e-3)]
                c["null_fpr_t"] = [float(np.mean(M.t_p(np.asarray(nul), n) < a)) for a in (0.05, 1e-3)]
                out[f"{name}/n{n}/{metric}/m{m_use}"] = c
    return out


def prefix():
    out = {}
    for name, setting in (("llama31_dolly_grid", "dolly"), ("llama31_tofu_grid", "tofu")):
        d = np.load(RAW / f"{name}_prefixprompt.npz")
        o = np.load(RAW / f"{name}_palimpsest_loglik.npz")
        k = d["lcp"].shape[0]
        keep = d["keep"]
        Pk = o["positions"][:k]
        W = weights(Pk, setting)[:, keep]
        ind = indep_matrix(Pk)
        n = keep.size
        for metric in ("lcp", "recall"):
            per = d[metric].reshape(k, n, 2)
            for label, S in {"both": per.mean(2), "cut25": per[:, :, 0], "cut50": per[:, :, 1]}.items():
                own, nul = [], []
                for i in range(k):
                    for j in range(k):
                        if i != j and not ind[i, j]:
                            continue
                        ref = S[ref_ids(ind, i, j)].mean(0)
                        (own if i == j else nul).append(spear(S[j] - ref, W[i], tiebreak=True))
                nm = M.mc_p(np.asarray(nul), null_for(setting, n))
                c = counts(own, setting, n)
                c["null_fpr_mc"] = [float(np.mean(nm < a)) for a in (0.05, 1e-3)]
                out[f"{name}/{metric}/{label}"] = c
    return out


def interfaces():
    """Fig. interfaces: same subsamples as provenance_interfaces.analyze, t vs MC."""
    d = np.load(RAW / "llama31_dolly_grid_interfaces.npz")
    P = d["positions"]
    k = P.shape[0]
    n_train = int(d["n_train"])
    W = M.lr_weight(P[:, None, :], n_train, 1, 16, 0.03)
    ind = indep_matrix(P)
    lrng = np.random.default_rng(7)
    out = {}
    for metric in ["full", "top20", "top5", "top1", "top1match", "gen_lcs"]:
        s = d[metric][1:]
        cols = len(d["gen_keep"]) if metric == "gen_lcs" else s.shape[1]
        s, w = s[:, :cols], W[:, :cols]
        for i in range(k):  # consume nothing: the original consumed rng only in the curve loop
            pass
        curve = {}
        for m in (50, 100, 250, 500, 1000):
            if m > cols:
                continue
            tp_t, tp_m = [], []
            for _ in range(200):
                idx = lrng.choice(cols, size=m, replace=False)
                i = int(lrng.integers(k))
                j = int(lrng.choice(np.flatnonzero(ind[i])))
                ref_i = s[ref_ids(ind, i)][:, idx].mean(0)
                rho = spear(s[i, idx] - ref_i, w[i, idx], tiebreak=metric in ("top1match", "gen_lcs"))
                tp_t.append(float(M.t_p(rho, m)))
                tp_m.append(float(M.mc_p(rho, null_for("dolly", m))))
            curve[m] = {"t": float(np.mean(np.asarray(tp_t) < 1e-3)), "mc": float(np.mean(np.asarray(tp_m) < 1e-3))}
        out[metric] = curve
    return out


def null_for(setting, n):
    p = RAW / f"mc_null_{setting}_n{n}.npy"
    if not p.exists():
        np.save(p, M.null_rho(setting, n, 200_000, 99))
    return null(setting, n)


def main():
    res = {"detection": detection(), "evasion": evasion(), "contft": contft(), "textonly": textonly(), "prefix": prefix(), "interfaces": interfaces()}
    (REP / "r4_mc_decisions.json").write_text(json.dumps(res, indent=1))
    for sec, d in res.items():
        print("=====", sec)
        if sec == "interfaces":
            for k, c in d.items():
                print(k, json.dumps(c))
            continue
        if sec == "contft":
            for b, v in d.items():
                print(b, v["owners"])
                for k, c in v["rows"].items():
                    print(f"  {k:<26} t@1e-3 {c['t@1e-3']} mc@1e-3 {c['mc@1e-3']} | t@.05 {c['t@.05']} mc@.05 {c['mc@.05']}")
            continue
        for k, c in d.items():
            flag = "" if ("t@1e-3" not in c or (c.get("t@1e-3") == c.get("mc@1e-3") and c.get("t@.05") == c.get("mc@.05"))) else "   <-- differs"
            print(k, json.dumps(c), flag)


if __name__ == "__main__":
    main()
