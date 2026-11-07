#!/usr/bin/env python3
"""Text-only adaptation of the order test (no log-probabilities, no prefill).

Observational setting in the spirit of palimpsestic text attribution: the
verifier only sees sampled responses. For each queried training prompt the
suspect produces m samples (temperature 1); a per-example score measures how
much of the owner's training response the samples reproduce:

  ngram4  mean recall of the reference response's 4-grams in a sample
  lcp     mean longest common token prefix with the reference (capped)

Scores enter the same shadow-referenced, schedule-weighted Spearman test; the
null argument is unchanged because only the owner's shuffle is random.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
from scipy import stats

sys.path.insert(0, str(Path(__file__).resolve().parent))
from palimpsest_lineage import _one_sided_p, _train_dataset, discover_grid, independence_matrix, lr_weight, reference_ids  # noqa: E402

from adaptertrace.config import get, load_config  # noqa: E402


def _ngrams(seq: list[int], k: int) -> set[tuple[int, ...]]:
    return {tuple(seq[i : i + k]) for i in range(len(seq) - k + 1)}


def score_samples(samples: list[list[int]], ref: list[int]) -> tuple[float, float]:
    ref_ng = _ngrams(ref, 4)
    ng, lcp = [], []
    for s in samples:
        ng.append(len(ref_ng & _ngrams(s, 4)) / max(len(ref_ng), 1))
        m = 0
        for a, b in zip(s, ref):
            if a != b:
                break
            m += 1
        lcp.append(m / max(min(len(ref), 64), 1))
    return ng, lcp


def collect(cfg, out: Path, n_examples: int, m: int, max_new: int, n_models: int, batch: int) -> None:
    import torch
    from peft import PeftModel

    from adaptertrace.hf import infer_device, load_model

    dataset, tokenizer = _train_dataset(cfg)
    n_examples = min(n_examples, len(dataset))
    keep = np.sort(np.random.default_rng(0).choice(len(dataset), size=n_examples, replace=False))
    rows = dataset.select(keep.tolist())
    prompts, refs = [], []
    for ids, labels, mask in zip(rows["input_ids"], rows["labels"], rows["attention_mask"], strict=True):
        cut = next(i for i, lab in enumerate(labels) if lab != -100)
        length = sum(mask)
        prompts.append(list(ids[:cut]))
        refs.append([t for t in ids[cut:length] if t != tokenizer.eos_token_id][:max_new])
    grid = discover_grid(cfg)[:n_models]
    device = infer_device(get(cfg, "inference.device"))
    base = load_model(cfg["model"]).to(device)
    pad, eos = int(tokenizer.pad_token_id), int(tokenizer.eos_token_id)
    ng_all = np.zeros((len(grid), n_examples, m))
    lcp_all = np.zeros((len(grid), n_examples, m))
    model, previous = None, None
    for gi, entry in enumerate(grid):
        name = f"g{gi}"
        if model is None:
            model = PeftModel.from_pretrained(base, entry["checkpoint"], adapter_name=name)
        else:
            model.load_adapter(entry["checkpoint"], adapter_name=name)
        model.set_adapter(name)
        if previous:
            model.delete_adapter(previous)
        previous = name
        model.eval()
        torch.manual_seed(1234 + gi)
        for start in range(0, n_examples, batch):
            chunk = prompts[start : start + batch]
            width = max(len(p) for p in chunk)
            ids = torch.tensor([[pad] * (width - len(p)) + p for p in chunk], device=device)
            attn = torch.tensor([[0] * (width - len(p)) + [1] * len(p) for p in chunk], device=device)
            with torch.no_grad():
                gen = model.generate(input_ids=ids, attention_mask=attn, do_sample=True, temperature=1.0,
                                     top_p=1.0, max_new_tokens=max_new, num_return_sequences=m,
                                     pad_token_id=pad)[:, width:].tolist()
            for b in range(len(chunk)):
                samples = []
                for j in range(m):
                    seq = []
                    for t in gen[b * m + j]:
                        if t in (pad, eos):
                            break
                        seq.append(t)
                    samples.append(seq)
                ng, lcp = score_samples(samples, refs[start + b])
                ng_all[gi, start + b], lcp_all[gi, start + b] = ng, lcp
        print(f"[textonly] model {gi} ({entry['victim_seed']})", flush=True)
    out.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(out, keep=keep, ngram4=ng_all, lcp=lcp_all,
                        seeds=np.asarray([int(e["victim_seed"]) for e in grid]))


def analyze(cfg, npz: Path, order_npz: Path, report: Path) -> None:
    d = np.load(npz)
    orders = np.load(order_npz)
    n_train = orders["adapter_loglik"].shape[1]
    keep = d["keep"]
    k = d["ngram4"].shape[0]
    if orders["seeds"].tolist()[:k] != d["seeds"].tolist():
        raise RuntimeError("Order file and text-only grid disagree")
    W = np.stack([lr_weight(orders["positions"][i], cfg["training"], n_train)[keep] for i in range(k)])
    # Multi-epoch training: nearby seeds share shuffles, so they are neither null
    # pairs nor valid reference members.
    indep = independence_matrix(orders["positions"][:k])
    rng = np.random.default_rng(7)
    result = {}
    for metric in ("ngram4", "lcp"):
        for m_use in (1, 2, 4, d[metric].shape[2]):
            S = d[metric][:, :, :m_use].mean(axis=2)
            pos, neg = [], []
            for i in range(k):
                for j in range(k):
                    if i != j and not indep[i, j]:
                        continue
                    ref = S[reference_ids(indep, i, j)].mean(axis=0)
                    p = _one_sided_p(float(stats.spearmanr(S[j] - ref, W[i]).statistic), S.shape[1])
                    (pos if i == j else neg).append(p)
            pos, neg = np.array(pos), np.array(neg)
            curve = {}
            for n_use in sorted({min(100, S.shape[1]), min(250, S.shape[1]), S.shape[1]}):
                tp = []
                for _ in range(200):
                    i = int(rng.integers(k)); idx = rng.choice(S.shape[1], n_use, replace=False)
                    ref = S[reference_ids(indep, i)][:, idx].mean(axis=0)
                    tp.append(_one_sided_p(float(stats.spearmanr(S[i, idx] - ref, W[i, idx]).statistic), n_use))
                curve[str(n_use)] = {"tpr_0.05": float(np.mean(np.array(tp) < 0.05)),
                                     "tpr_1e-3": float(np.mean(np.array(tp) < 1e-3))}
            result[f"{metric}/m={m_use}"] = {
                "owners_detected_1e-3": f"{int(np.sum(pos < 1e-3))}/{k}",
                "owners_detected_0.05": f"{int(np.sum(pos < 0.05))}/{k}",
                "median_p": float(np.median(pos)),
                "null_fpr_0.05": float(np.mean(neg < 0.05)),
                "null_fpr_1e-3": float(np.mean(neg < 1e-3)),
                "power_by_n": curve,
            }
            print(f"== {metric} m={m_use}: {json.dumps(result[f'{metric}/m={m_use}'])}", flush=True)
    report.parent.mkdir(parents=True, exist_ok=True)
    report.write_text(json.dumps(result, indent=2))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--stage", choices=["collect", "analyze", "all"], default="all")
    parser.add_argument("--n-examples", type=int, default=500)
    parser.add_argument("--samples", type=int, default=8)
    parser.add_argument("--max-new", type=int, default=64)
    parser.add_argument("--n-models", type=int, default=20)
    parser.add_argument("--batch", type=int, default=8)
    parser.add_argument("--tag", default="", help="suffix that keeps runs with other budgets apart")
    args = parser.parse_args()
    cfg = load_config(args.config)
    root = Path(get(cfg, "experiment.output_root", "artifact"))
    name = cfg["experiment"]["name"]
    tag = f"_{args.tag}" if args.tag else ""
    npz = root / "raw" / f"{name}_textonly{tag}.npz"
    if args.stage in {"collect", "all"}:
        collect(cfg, npz, args.n_examples, args.samples, args.max_new, args.n_models, args.batch)
    if args.stage in {"analyze", "all"}:
        analyze(cfg, npz, root / "raw" / f"{name}_palimpsest_loglik.npz", root / "reports" / f"{name}_textonly{tag}.json")


if __name__ == "__main__":
    main()
