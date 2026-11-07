#!/usr/bin/env python3
"""Step 0 / item 3: does the training-order test survive realistic API interfaces?

For each adapter of a grid (and the base), score a fixed subset of the owner
training examples under several observation models of a deployed endpoint:

  full       sum of teacher-forced token log-probs (needs echo/logprob scoring)
  top{k}     same, but the true token's log-prob is only visible if it is in
             the top-k; otherwise the k-th log-prob (an upper bound) is used.
             Obtainable from a top-k logprobs API with assistant prefill.
  top1match  fraction of response tokens equal to the greedy next token under
             prefill (text-only API with prefill, one token per call).
  gen_lcs    text-only API without prefill: greedy response to the prompt
             alone, scored by token LCS with the training response / its length.

Every score is turned into the shadow-referenced, learning-rate-weighted
Spearman test (owner x suspect matrix); the null keeps its exact validity for
any per-example score because only the owner's training order is random.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
from scipy import stats

sys.path.insert(0, str(Path(__file__).resolve().parent))
from palimpsest_lineage import (  # noqa: E402
    _one_sided_p,
    _train_dataset,
    discover_grid,
    independence_matrix,
    lr_weight,
    reference_ids,
)

from adaptertrace.config import get, load_config  # noqa: E402

TOPK = (20, 5, 1)


def teacher_forced_scores(model: Any, dataset: Any, batch_size: int, device: str) -> dict[str, np.ndarray]:
    import torch

    n = len(dataset)
    out = {"full": np.zeros(n), "top1match": np.zeros(n)}
    for k in TOPK:
        out[f"top{k}"] = np.zeros(n)
    model.eval()
    with torch.no_grad():
        for start in range(0, n, batch_size):
            rows = dataset[start : start + batch_size]
            ids = torch.tensor(rows["input_ids"])
            mask = torch.tensor(rows["attention_mask"])
            labels = torch.tensor(rows["labels"])
            width = int(mask.sum(dim=1).max())
            ids, mask, labels = (t[:, :width].to(device) for t in (ids, mask, labels))
            logp = torch.log_softmax(model(input_ids=ids, attention_mask=mask).logits.float()[:, :-1], -1)
            target = labels[:, 1:]
            valid = target != -100
            safe = target.clamp(min=0)
            true_lp = logp.gather(-1, safe.unsqueeze(-1)).squeeze(-1)
            top_vals = logp.topk(max(TOPK), dim=-1).values  # descending
            # rank of the true token = number of strictly larger log-probs
            rank = (logp > true_lp.unsqueeze(-1)).sum(-1)
            denom = valid.sum(1).clamp(min=1)
            sl = slice(start, start + len(ids))
            out["full"][sl] = (true_lp * valid).sum(1).cpu().numpy()
            out["top1match"][sl] = (((rank == 0) & valid).sum(1) / denom).cpu().numpy()
            for k in TOPK:
                visible = rank < k
                censored = torch.where(visible, true_lp, top_vals[..., k - 1])
                out[f"top{k}"][sl] = (censored * valid).sum(1).cpu().numpy()
    return out


def _lcs(a: list[int], b: list[int]) -> int:
    prev = [0] * (len(b) + 1)
    for x in a:
        cur = [0]
        for j, y in enumerate(b):
            cur.append(prev[j] + 1 if x == y else max(prev[j + 1], cur[j]))
        prev = cur
    return prev[-1]


def greedy_lcs(model: Any, dataset: Any, tokenizer: Any, batch_size: int, device: str, max_new: int) -> np.ndarray:
    """Text-only, no prefill: greedy answer to the prompt vs. the training response."""
    import torch

    scores = np.zeros(len(dataset))
    pad = int(tokenizer.pad_token_id)
    model.eval()
    for start in range(0, len(dataset), batch_size):
        rows = dataset[start : start + batch_size]
        prompts, refs = [], []
        for ids, labels, mask in zip(rows["input_ids"], rows["labels"], rows["attention_mask"], strict=True):
            length = sum(mask)
            cut = next(i for i, lab in enumerate(labels) if lab != -100)
            prompts.append(ids[:cut])
            refs.append([t for t in ids[cut:length] if t != tokenizer.eos_token_id][:max_new])
        width = max(len(p) for p in prompts)
        input_ids = torch.tensor([[pad] * (width - len(p)) + p for p in prompts], device=device)
        attn = torch.tensor([[0] * (width - len(p)) + [1] * len(p) for p in prompts], device=device)
        with torch.no_grad():
            gen = model.generate(
                input_ids=input_ids,
                attention_mask=attn,
                max_new_tokens=max_new,
                do_sample=False,
                pad_token_id=pad,
            )[:, width:].tolist()
        for i, (g, ref) in enumerate(zip(gen, refs, strict=True)):
            g = [t for t in g if t not in (pad, tokenizer.eos_token_id)]
            scores[start + i] = _lcs(g, ref) / max(len(ref), 1)
    return scores


def collect(cfg: dict[str, Any], out: Path, n_tf: int, n_gen: int, batch_size: int, max_new: int, npz_orders: Path) -> None:
    from peft import PeftModel

    from adaptertrace.hf import infer_device, load_model

    dataset, tokenizer = _train_dataset(cfg)
    stored = np.load(npz_orders)
    grid = discover_grid(cfg)
    seeds = [int(e["victim_seed"]) for e in grid]
    if seeds != stored["seeds"].tolist():
        raise RuntimeError("Grid differs from the stored training orders")
    rng = np.random.default_rng(0)
    keep = np.sort(rng.choice(len(dataset), size=n_tf, replace=False))
    gen_keep = keep[:n_gen]
    tf_ds = dataset.select(keep.tolist())
    gen_ds = dataset.select(gen_keep.tolist())
    device = infer_device(get(cfg, "inference.device"))
    base = load_model(cfg["model"]).to(device)
    tokenizer.padding_side = "left"

    names = ["base"] + [f"s{s}" for s in seeds]
    scores: dict[str, list[np.ndarray]] = {}

    def record(model: Any, label: str) -> None:
        tf = teacher_forced_scores(model, tf_ds, batch_size, device)
        tf["gen_lcs"] = greedy_lcs(model, gen_ds, tokenizer, batch_size, device, max_new)
        for key, val in tf.items():
            scores.setdefault(key, []).append(val)
        print(f"[interfaces] scored {label}", flush=True)

    record(base, "base")
    peft_model = PeftModel.from_pretrained(base, grid[0]["checkpoint"], adapter_name=names[1])
    for entry, name in zip(grid[1:], names[2:], strict=True):
        peft_model.load_adapter(entry["checkpoint"], adapter_name=name)
    for name in names[1:]:
        peft_model.set_adapter(name)
        record(peft_model, name)
    out.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        out,
        seeds=np.asarray(seeds),
        keep=keep,
        gen_keep=gen_keep,
        positions=stored["positions"][..., keep],
        n_train=len(dataset),
        **{k: np.stack(v) for k, v in scores.items()},
    )
    print(f"[interfaces] wrote {out}")


def analyze(npz: Path, report: Path, training_cfg: dict[str, Any], sizes: list[int], repeats: int) -> None:
    data = np.load(npz)
    n_train = int(data["n_train"])
    positions = data["positions"]
    k = positions.shape[0]
    n_gen = len(data["gen_keep"])
    weights = np.stack([lr_weight(p, training_cfg, n_train) for p in positions])
    indep = independence_matrix(positions)
    rng = np.random.default_rng(7)
    result: dict[str, Any] = {}
    for metric in ["full", *[f"top{t}" for t in TOPK], "top1match", "gen_lcs"]:
        s = data[metric][1:]  # drop base row
        cols = n_gen if metric == "gen_lcs" else s.shape[1]
        s = s[:, :cols]
        w = weights[:, :cols]
        pos_p, null_p = [], []
        for i in range(k):
            for j in range(k):
                if i != j and not indep[i, j]:
                    continue
                ref = s[reference_ids(indep, i, j) if i != j else reference_ids(indep, i)].mean(axis=0)
                rho = stats.spearmanr(s[j] - ref, w[i]).statistic
                (pos_p if i == j else null_p).append(_one_sided_p(rho, cols))
        pos_p, null_p = np.asarray(pos_p), np.asarray(null_p)
        curve = []
        for m in sizes:
            if m > cols:
                continue
            tp, fp = [], []
            for _ in range(repeats):
                idx = rng.choice(cols, size=m, replace=False)
                i = int(rng.integers(k))
                j = int(rng.choice(np.flatnonzero(indep[i])))
                ref_i = s[reference_ids(indep, i)][:, idx].mean(axis=0)
                ref_j = s[reference_ids(indep, i, j)][:, idx].mean(axis=0)
                tp.append(_one_sided_p(stats.spearmanr(s[i, idx] - ref_i, w[i, idx]).statistic, m))
                fp.append(_one_sided_p(stats.spearmanr(s[j, idx] - ref_j, w[i, idx]).statistic, m))
            tp, fp = np.asarray(tp), np.asarray(fp)
            curve.append(
                {
                    "n_examples": m,
                    "tpr_at_0.05": float(np.mean(tp < 0.05)),
                    "tpr_at_0.001": float(np.mean(tp < 1e-3)),
                    "fpr_at_0.05": float(np.mean(fp < 0.05)),
                    "fpr_at_0.001": float(np.mean(fp < 1e-3)),
                }
            )
        result[metric] = {
            "n_examples": cols,
            "detected_at_0.001": f"{int(np.sum(pos_p < 1e-3))}/{k}",
            "median_positive_p": float(np.median(pos_p)),
            "null_fpr_at_0.05": float(np.mean(null_p < 0.05)),
            "null_fpr_at_0.001": float(np.mean(null_p < 1e-3)),
            "null_ks_uniform_p": float(stats.kstest(null_p, "uniform").pvalue),
            "power_curve": curve,
        }
        print(f"== {metric:<10} {json.dumps({x: y for x, y in result[metric].items() if x != 'power_curve'})}")
        for row in curve:
            print(f"   {row}")
    report.parent.mkdir(parents=True, exist_ok=True)
    report.write_text(json.dumps(result, indent=2))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--stage", choices=["collect", "analyze", "all"], default="all")
    parser.add_argument("--n-tf", type=int, default=1000)
    parser.add_argument("--n-gen", type=int, default=300)
    parser.add_argument("--max-new", type=int, default=64)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--repeats", type=int, default=200)
    args = parser.parse_args()
    cfg = load_config(args.config)
    root = Path(get(cfg, "experiment.output_root", "artifact"))
    name = cfg["experiment"]["name"]
    npz = root / "raw" / f"{name}_interfaces.npz"
    if args.stage in {"collect", "all"}:
        collect(cfg, npz, args.n_tf, args.n_gen, args.batch_size, args.max_new, root / "raw" / f"{name}_palimpsest_loglik.npz")
    if args.stage in {"analyze", "all"}:
        analyze(npz, root / "reports" / f"{name}_interfaces.json", cfg["training"], [50, 100, 250, 500, 1000], args.repeats)


if __name__ == "__main__":
    main()
