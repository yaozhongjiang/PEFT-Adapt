#!/usr/bin/env python3
"""Continued fine-tuning: does a simulated-attack reference restore detection?

Continued training on other data causes forgetting and changes which examples
look hard, so unmodified shadows no longer cancel per-example difficulty. The
verifier can apply the same kind of modification to its own shadows and use
them as the reference. It does not know the thief's data, so we evaluate:

  thief        owner adapters continued on the thief's data (other TOFU facts)
  sim_dolly    shadows continued on a different corpus (Dolly): realistic
  sim_same     shadows continued on the thief's data: optimistic upper bound

Validity is unchanged: only the owner's shuffle is random, and neither the
simulated shadows nor the null suspects depend on it. Null suspects are the
sim_same shadows, tested against independently replayed real trainer orders.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
from scipy import stats

sys.path.insert(0, str(Path(__file__).resolve().parent))
from palimpsest_evasion import continue_finetune, task_utility  # noqa: E402
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

STEPS = [50, 200, 800]


def collect(cfg, data_cfg, indices: list[int], tag: str, out: Path, batch: int) -> None:
    from peft import PeftModel

    from adaptertrace.hf import infer_device, load_model

    dataset, tokenizer = _train_dataset(cfg)
    train_rows, _ = _train_dataset(data_cfg)
    grid = discover_grid(cfg)
    device = infer_device(get(cfg, "inference.device"))
    base = load_model(cfg["model"]).to(device)
    base.gradient_checkpointing_enable()
    base.enable_input_require_grads()
    scores, utility = [], []
    model, previous = None, None
    for gi in indices:
        name = f"g{gi}"
        if model is None:
            model = PeftModel.from_pretrained(base, grid[gi]["checkpoint"], adapter_name=name, is_trainable=True)
        else:
            model.load_adapter(grid[gi]["checkpoint"], adapter_name=name, is_trainable=True)
        model.set_adapter(name)
        if previous is not None:
            model.delete_adapter(previous)
        previous = name
        if hasattr(model, "_thief_opt"):
            del model._thief_opt
        done, per_step, util = 0, [], []
        for step in STEPS:
            done = continue_finetune(model, name, train_rows, step, done, cfg, 7000 + gi, device)
            per_step.append(_answer_loglik(model, dataset, batch, device))
            util.append(task_utility(model, cfg, tokenizer, device, batch))
            print(f"[contft] {tag} model {gi} step {step} utility {util[-1]:.4f}", flush=True)
        scores.append(np.stack(per_step))
        utility.append(util)
    out.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(out, indices=np.asarray(indices), steps=np.asarray(STEPS), ll=np.stack(scores), utility=np.asarray(utility))
    print(f"[contft] wrote {out}")


def _merge(paths: list[Path]) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    parts = [np.load(p) for p in paths]
    return (np.concatenate([p["indices"] for p in parts]),
            np.concatenate([p["ll"] for p in parts]),
            np.concatenate([p["utility"] for p in parts]))


def analyze(cfg, root: Path, name: str, disjoint: bool = False) -> None:
    """With `disjoint`, thief and simulator train on different facts of the same kind."""
    raw = root / "raw"
    thief_tag, same_tag = ("h_thief", "h_sim") if disjoint else ("thief", "sim_same")
    thief_idx, thief, thief_u = _merge(sorted(raw.glob(f"{name}_contft_{thief_tag}*.npz")))
    dolly_idx, dolly, _ = _merge(sorted(raw.glob(f"{name}_contft_sim_dolly*.npz")))
    same_idx, same, _ = _merge(sorted(raw.glob(f"{name}_contft_{same_tag}*.npz")))
    orders = np.load(raw / f"{name}_palimpsest_loglik.npz")
    A, P = orders["adapter_loglik"], orders["positions"]
    n = A.shape[1]
    indep = independence_matrix(P)
    tc = cfg["training"]
    replay = np.load(raw / f"{name}_replayed_orders.npz")["positions"]
    W_new = np.stack([lr_weight(p, tc, n) for p in replay])
    rng = np.random.default_rng(0)
    result: dict = {"steps": STEPS, "owners": thief_idx.tolist(), "utility_by_step": thief_u.mean(axis=0).tolist(), "reference": {}}

    def ok(shadow_idx: np.ndarray, owner: int) -> list[int]:
        return [c for c, g in enumerate(shadow_idx) if indep[owner, g]]

    for s_i, step in enumerate(STEPS):
        for label in ("unmodified", "sim_dolly", "sim_same"):
            ps = []
            for c, owner in enumerate(thief_idx):
                if label == "unmodified":
                    ref = A[reference_ids(indep, int(owner))].mean(axis=0)
                else:
                    idx, ll = (dolly_idx, dolly) if label == "sim_dolly" else (same_idx, same)
                    keep = ok(idx, int(owner))
                    if not keep:
                        raise RuntimeError(f"No independent simulated shadow for owner {owner}")
                    ref = ll[keep, s_i].mean(axis=0)
                w = lr_weight(P[owner], tc, n)
                ps.append(_one_sided_p(float(stats.spearmanr(thief[c, s_i] - ref, w).statistic), n))
            ps = np.asarray(ps)
            # null: shadows continued on the thief's data, against replayed orders
            if label == "unmodified":
                ref_null = A.mean(axis=0)
            elif label == "sim_dolly":
                ref_null = dolly[:, s_i].mean(axis=0)
            else:
                ref_null = None
            null = []
            for c in range(same.shape[0]):
                if label == "sim_same":
                    ref_n = np.delete(same[:, s_i], c, axis=0).mean(axis=0)
                else:
                    ref_n = ref_null
                rr = stats.rankdata(same[c, s_i] - ref_n)
                null.append([_one_sided_p(float(stats.spearmanr(rr, w).statistic), n) for w in W_new])
            null = np.asarray(null).T  # owners x suspects
            per_owner = (null < 0.05).mean(axis=1)
            boot = [per_owner[rng.integers(0, per_owner.size, per_owner.size)].mean() for _ in range(2000)]
            block = {
                "detected_0.05": float(np.mean(ps < 0.05)),
                "detected_1e-3": float(np.mean(ps < 1e-3)),
                "median_p": float(np.median(ps)),
                "p": ps.tolist(),
                "null_tests": int(null.size),
                "null_fpr_0.05": float((null < 0.05).mean()),
                "null_fpr_0.05_owner_boot95": [float(np.percentile(boot, 2.5)), float(np.percentile(boot, 97.5))],
                "null_fpr_1e-3": float((null < 1e-3).mean()),
            }
            result["reference"][f"{label}/step{step}"] = block
            print(f"== {label:<11} step {step:<4} " + json.dumps({k: v for k, v in block.items() if k != "p"}), flush=True)
    print("== utility " + json.dumps(result["utility_by_step"]))
    suffix = "_disjoint" if disjoint else ""
    (root / "reports" / f"{name}_contft_matched{suffix}.json").write_text(json.dumps(result, indent=2))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--stage", choices=["collect", "analyze", "analyze_disjoint"], required=True)
    parser.add_argument("--data-range", help="start,end: restrict continued-training data to these indices")
    parser.add_argument("--tag", help="thief_a, thief_b, sim_dolly_a, ...")
    parser.add_argument("--data-config", help="grid YAML whose training data is used for continued training")
    parser.add_argument("--indices", help="comma-separated grid indices")
    parser.add_argument("--batch-size", type=int, default=16)
    args = parser.parse_args()
    cfg = load_config(args.config)
    root = Path(get(cfg, "experiment.output_root", "artifact"))
    name = cfg["experiment"]["name"]
    if args.stage == "collect":
        data_cfg = load_config(args.data_config)
        if args.data_range:
            data_cfg["data"]["train_index_range"] = [int(v) for v in args.data_range.split(",")]
            data_cfg["data"].pop("max_train_samples", None)
        collect(cfg, data_cfg, [int(v) for v in args.indices.split(",")], args.tag,
                root / "raw" / f"{name}_contft_{args.tag}.npz", args.batch_size)
    else:
        analyze(cfg, root, name, disjoint=args.stage == "analyze_disjoint")


if __name__ == "__main__":
    main()
