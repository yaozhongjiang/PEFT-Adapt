#!/usr/bin/env python3
"""TOFU layered provenance: knowledge probe + training-order test, large null pool.

One scoring pass: every model scores all 4000 TOFU QA pairs (authors 1-100 =
the owner's training facts "A", authors 101-200 = sealed control facts "B" of
the same generator). Per-question answer log-likelihoods serve two tests:

Knowledge probe (H0: the suspect never learned from the owner's fact set).
  G = mean over A questions of (ll_suspect - ll_base), compared with the same
  gain for a null pool of models that never saw A (Dolly and other-fact
  adapters): empirical rank p-value (min 1/(pool+1)) and z-score.
  Catches functional copies (distillation) that carry no training-order trace.
  The A-minus-B author-permutation variant is reported only as a secondary
  statistic: TOFU's A/B split is contiguous rather than randomised by the
  owner, so unrelated models are not exchangeable across it (Dolly adapters
  all show a small positive A-B gap), and a suspect that merely degrades on B
  inflates A-B.

Training-order test (H0: the suspect is independent of the owner's training
  run). Shadow-referenced, learning-rate-weighted Spearman on the A examples
  (see palimpsest_lineage.py); distinguishes same-data siblings.

Null pool for both tests: adapters that never saw A (other-fact TOFU adapters
and the 40 Dolly adapters), giving hundreds of independent (owner, suspect)
pairs to check calibration.
"""

from __future__ import annotations

import argparse
import copy
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
from scipy import stats

sys.path.insert(0, str(Path(__file__).resolve().parent))
from palimpsest_lineage import (  # noqa: E402
    _answer_loglik,
    _one_sided_p,
    _train_dataset,
    discover_grid,
    independence_matrix,
    lr_weight,
    reference_ids,
)

from adaptertrace.config import get, load_config  # noqa: E402

N_A = 2000
NULL_GROUPS = ("dolly", "otherfacts")  # never trained on the owner's facts
PER_AUTHOR = 20


def _all_facts_cfg(cfg: dict[str, Any]) -> dict[str, Any]:
    full = copy.deepcopy(cfg)
    full["data"]["train_index_range"] = [0, 2 * N_A]
    full["data"]["max_train_samples"] = None
    return full


def collect(cfg: dict[str, Any], other_cfg: dict[str, Any], dolly_cfg: dict[str, Any], out: Path, batch: int) -> None:
    from peft import PeftModel

    from adaptertrace.hf import infer_device, load_model

    root = Path(get(cfg, "experiment.output_root", "artifact"))
    dataset, _ = _train_dataset(_all_facts_cfg(cfg))
    if len(dataset) != 2 * N_A:
        raise RuntimeError(f"Expected {2 * N_A} TOFU rows, got {len(dataset)}")
    models: list[tuple[str, str, str]] = []  # (label, group, path)
    models += [(f"grid_s{e['victim_seed']}", "grid", e["checkpoint"]) for e in discover_grid(cfg)]
    models += [(f"otherfacts_s{e['victim_seed']}", "otherfacts", e["checkpoint"]) for e in discover_grid(other_cfg)]
    students = root / "models" / f"{cfg['experiment']['name']}_distill_students"
    for path in sorted(students.glob("student_*_o*")):
        if (path / "adapter_config.json").exists():
            source = path.name.split("_")[1]
            models.append((path.name, f"student_{source}", str(path)))
    models += [(f"dolly_s{e['victim_seed']}", "dolly", e["checkpoint"]) for e in discover_grid(dolly_cfg)]
    device = infer_device(get(cfg, "inference.device"))
    base = load_model(cfg["model"]).to(device)
    scores = [_answer_loglik(base, dataset, batch, device)]
    labels, groups = ["base"], ["base"]
    print(f"[tofu-probe] scored base; {len(models)} adapters to go", flush=True)
    peft_model, previous = None, None
    for idx, (label, group, path) in enumerate(models):
        name = f"m{idx}"  # exactly one adapter resident at a time
        if peft_model is None:
            peft_model = PeftModel.from_pretrained(base, path, adapter_name=name)
        else:
            peft_model.load_adapter(path, adapter_name=name)
        peft_model.set_adapter(name)
        if previous is not None:
            peft_model.delete_adapter(previous)
        previous = name
        scores.append(_answer_loglik(peft_model, dataset, batch, device))
        labels.append(label)
        groups.append(group)
        print(f"[tofu-probe] scored {label}", flush=True)
    out.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(out, ll=np.stack(scores), labels=np.asarray(labels), groups=np.asarray(groups))
    print(f"[tofu-probe] wrote {out}")


def knowledge_probe(ll: np.ndarray, base: np.ndarray, n_perm: int, rng: np.random.Generator) -> tuple[float, float, float]:
    """Return (T, permutation p, Welch t-test p) for one suspect."""
    gain = (ll - base).reshape(-1, PER_AUTHOR).mean(axis=1)  # per author
    n_auth = gain.size
    labels = np.zeros(n_auth, dtype=bool)
    labels[: N_A // PER_AUTHOR] = True
    t_obs = gain[labels].mean() - gain[~labels].mean()
    perms = np.argsort(rng.random((n_perm, n_auth)), axis=1) < labels.sum()
    t_null = (perms * gain).sum(1) / labels.sum() - (~perms * gain).sum(1) / (~labels).sum()
    p_perm = (1 + np.sum(t_null >= t_obs)) / (1 + n_perm)
    p_t = stats.ttest_ind(gain[labels], gain[~labels], equal_var=False, alternative="greater").pvalue
    return float(t_obs), float(p_perm), float(p_t)


def analyze(cfg: dict[str, Any], npz: Path, order_npz: Path, report: Path, n_perm: int, n_worlds: int = 20) -> None:
    data = np.load(npz)
    ll, labels, groups = data["ll"], data["labels"].tolist(), data["groups"].tolist()
    base = ll[labels.index("base")]
    orders = np.load(order_npz)
    positions = orders["positions"]
    seeds = orders["seeds"].tolist()
    indep = independence_matrix(positions)
    grid_rows = [labels.index(f"grid_s{s}") for s in seeds]
    grid_ll = ll[grid_rows][:, :N_A]
    weights = np.stack([lr_weight(p, cfg["training"], N_A) for p in positions])
    rng = np.random.default_rng(7)

    gain_a = (ll[:, :N_A] - base[:N_A]).mean(axis=1)
    null_mask = np.isin(np.asarray(groups), NULL_GROUPS)
    null_idx = np.flatnonzero(null_mask)
    rows: list[dict[str, Any]] = []
    for m, (label, group) in enumerate(zip(labels, groups, strict=True)):
        if group == "base":
            continue
        pool = gain_a[[i for i in null_idx if i != m]]  # leave-one-out for null members
        p_emp = (1 + np.sum(pool >= gain_a[m])) / (1 + pool.size)
        z = (gain_a[m] - pool.mean()) / pool.std(ddof=1)
        t_obs, p_perm, p_t = knowledge_probe(ll[m], base, n_perm, rng)
        rows.append({
            "model": label, "group": group,
            "knowledge_gain_A": float(gain_a[m]), "knowledge_p": float(p_emp), "knowledge_z": float(z),
            "secondary_A_minus_B_T": t_obs, "secondary_A_minus_B_p_perm": p_perm,
        })

    # Training-order test of non-grid suspects against every grid owner.
    order_rows: list[dict[str, Any]] = []
    for m, (label, group) in enumerate(zip(labels, groups, strict=True)):
        if group in {"base", "grid"}:
            continue
        teacher = None
        if group.startswith("student_"):
            teacher = int(label.rsplit("_o", 1)[1])
        for oi, owner in enumerate(seeds):
            ref = grid_ll[reference_ids(indep, oi)].mean(axis=0)
            rho = stats.spearmanr(ll[m, :N_A] - ref, weights[oi]).statistic
            relation = "derived" if teacher == owner else (
                "shares_order_with_teacher" if teacher is not None and not indep[seeds.index(teacher), oi] else "independent"
            )
            order_rows.append(
                {"model": label, "group": group, "owner": owner, "relation": relation, "p": _one_sided_p(float(rho), N_A)}
            )

    def summarize(sel: list[dict[str, Any]], key: str) -> dict[str, Any]:
        ps = np.asarray([r[key] for r in sel])
        return {
            "n": int(ps.size),
            "rate_at_0.05": float(np.mean(ps < 0.05)) if ps.size else None,
            "rate_at_0.001": float(np.mean(ps < 1e-3)) if ps.size else None,
            "median_p": float(np.median(ps)) if ps.size else None,
        }

    summary: dict[str, Any] = {"knowledge_probe": {}, "order_test": {}, "layered": {}}
    for group in sorted({r["group"] for r in rows}):
        sel = [r for r in rows if r["group"] == group]
        summary["knowledge_probe"][group] = {
            **summarize(sel, "knowledge_p"),
            "median_gain_A": float(np.median([r["knowledge_gain_A"] for r in sel])),
            "median_z": float(np.median([r["knowledge_z"] for r in sel])),
            "secondary_A_minus_B_rate_at_0.05": float(np.mean([r["secondary_A_minus_B_p_perm"] < 0.05 for r in sel])),
        }
    for (group, relation) in sorted({(r["group"], r["relation"]) for r in order_rows}):
        sel = [r for r in order_rows if r["group"] == group and r["relation"] == relation]
        block = summarize(sel, "p")
        if relation == "independent":
            block["ks_uniform_p"] = float(stats.kstest([r["p"] for r in sel], "uniform").pvalue)
        summary["order_test"][f"{group}/{relation}"] = block
    # Layered decision per (suspect, owner): Bonferroni over the two tests.
    know = {r["model"]: r["knowledge_p"] for r in rows}
    for (group, relation) in sorted({(r["group"], r["relation"]) for r in order_rows}):
        sel = [r for r in order_rows if r["group"] == group and r["relation"] == relation]
        # Bonferroni over the two tests. The knowledge p-value is an empirical
        # rank (floor 1/(pool+1)), so only the alpha = 0.05 level is reported.
        flags = [min(r["p"], know[r["model"]]) < 0.025 for r in sel]
        summary["layered"][f"{group}/{relation}"] = {"n": len(sel), "flag_rate_alpha_0.05": float(np.mean(flags))}
    # Order-test calibration with synthetic owners: fresh random shuffles with
    # the same epoch structure. Pooled FPR over the 20 real owners is noisy
    # because all their weights derive from ~24 shared permutations.
    epochs = positions.shape[1] if positions.ndim == 3 else 1
    synth_fpr = []
    for world in range(n_worlds):
        ps = []
        for _ in range(len(seeds)):
            pos = np.stack([e * N_A + rng.permutation(N_A) for e in range(epochs)])
            w = lr_weight(pos if epochs > 1 else pos[0], cfg["training"], N_A)
            for m in null_idx:
                ref = grid_ll.mean(axis=0)
                ps.append(_one_sided_p(float(stats.spearmanr(ll[m, :N_A] - ref, w).statistic), N_A))
        synth_fpr.append(float(np.mean(np.asarray(ps) < 0.05)))
    summary["order_test_synthetic_owner_calibration"] = {
        "worlds": n_worlds,
        "mean_fpr_at_0.05": float(np.mean(synth_fpr)),
        "fpr_5th_95th_percentile": [float(v) for v in np.percentile(synth_fpr, [5, 95])],
    }
    report.parent.mkdir(parents=True, exist_ok=True)
    report.write_text(json.dumps({"summary": summary, "knowledge_rows": rows}, indent=2))
    for section, blocks in summary.items():
        for key, block in blocks.items():
            print(f"== {section:<16} {key:<40} {json.dumps(block)}")
    print(f"== calibration {json.dumps(summary['order_test_synthetic_owner_calibration'])}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, help="TOFU owner grid YAML")
    parser.add_argument("--other-config", required=True, help="TOFU other-facts grid YAML")
    parser.add_argument("--dolly-config", required=True, help="Dolly grid YAML (null pool)")
    parser.add_argument("--stage", choices=["collect", "analyze", "all"], default="all")
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--n-perm", type=int, default=100000)
    args = parser.parse_args()
    cfg = load_config(args.config)
    root = Path(get(cfg, "experiment.output_root", "artifact"))
    name = cfg["experiment"]["name"]
    npz = root / "raw" / f"{name}_tofu_probe.npz"
    if args.stage in {"collect", "all"}:
        collect(cfg, load_config(args.other_config), load_config(args.dolly_config), npz, args.batch_size)
    if args.stage in {"analyze", "all"}:
        analyze(cfg, npz, root / "raw" / f"{name}_palimpsest_loglik.npz", root / "reports" / f"{name}_tofu_probe.json", args.n_perm)


if __name__ == "__main__":
    main()
