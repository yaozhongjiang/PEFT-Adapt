#!/usr/bin/env python3
"""Text-only endpoints without prefill: emulate the prefill inside the prompt.

The endpoint returns only text and the caller cannot start the assistant turn.
The verifier therefore places the training prompt and a prefix of the owner's
training response in the *user* message and asks the model to continue. The
greedy continuation is compared word by word with the rest of the response;
the per-example score enters the same shadow-referenced, schedule-weighted
test. Only examples long enough for every cut are queried, a selection that
does not depend on the owner's shuffle, so validity is unchanged.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

import numpy as np
from scipy import stats

sys.path.insert(0, str(Path(__file__).resolve().parent))
from palimpsest_lineage import _one_sided_p, _train_dataset, discover_grid, independence_matrix, lr_weight, reference_ids  # noqa: E402

from adaptertrace.config import get, load_config  # noqa: E402

CUTS = (0.25, 0.5)
MIN_RESPONSE_TOKENS = 32
TARGET_WORDS = 12
TEMPLATE = "{user}\n\nHere is the beginning of the answer. Continue it exactly from where it stops, without repeating it:\n\n{prefix}"


def _words(text: str) -> list[str]:
    return re.findall(r"[a-z0-9']+", text.lower())


def score_pair(generated: str, reference: list[str]) -> tuple[float, float]:
    """Longest common word prefix and word recall, over the target window."""
    target = reference[:TARGET_WORDS]
    gen = _words(generated)
    lcp = 0
    for a, b in zip(gen, target):
        if a != b:
            break
        lcp += 1
    recall = len(set(gen) & set(target)) / max(len(set(target)), 1)
    return lcp / max(len(target), 1), recall


def _raw_rows(cfg, dataset) -> list[dict]:
    from adaptertrace.hf import load_task_dataset, select_split

    d = cfg["data"]
    raw = select_split(load_task_dataset(d), d.get("train_split", "train"), d.get("max_train_samples"),
                       seed=int(d.get("selection_seed", cfg["seeds"]["victim"][0])), index_range=d.get("train_index_range"))
    raw = raw.filter(lambda r: bool(str(r[d["response_field"]]).strip()))
    if len(raw) != len(dataset):
        raise RuntimeError("Raw rows do not align with the tokenized training set")
    return [dict(raw[i]) for i in range(len(raw))]


def collect(cfg, out: Path, n_examples: int, n_models: int, batch: int, max_new: int) -> None:
    import torch
    from peft import PeftModel

    from adaptertrace.hf import infer_device, load_model

    d = cfg["data"]
    dataset, tokenizer = _train_dataset(cfg)
    raw = _raw_rows(cfg, dataset)
    long_enough = [i for i, r in enumerate(dataset["labels"]) if sum(1 for t in r if t != -100) >= MIN_RESPONSE_TOKENS]
    keep = np.sort(np.random.default_rng(0).choice(long_enough, size=min(n_examples, len(long_enough)), replace=False))
    print(f"[prefixprompt] {len(long_enough)} long examples; querying {len(keep)}", flush=True)
    prompts, refs = [], []
    for i in keep.tolist():
        row = raw[i]
        user = str(row[d["instruction_field"]]).strip()
        ctx = str(row.get(d.get("context_field") or "", "") or "").strip()
        user = f"{user}\n\n{ctx}" if ctx else user
        words = str(row[d["response_field"]]).split()
        for frac in CUTS:
            cut = max(1, int(len(words) * frac))
            content = TEMPLATE.format(user=user, prefix=" ".join(words[:cut]))
            prompts.append(tokenizer.apply_chat_template([{"role": "user", "content": content}], tokenize=False, add_generation_prompt=True))
            refs.append(_words(" ".join(words[cut:])))
    tokenizer.padding_side = "left"
    grid = discover_grid(cfg)[:n_models]
    device = infer_device(get(cfg, "inference.device"))
    base = load_model(cfg["model"]).to(device)
    n_q = len(prompts)
    lcp, rec = np.zeros((len(grid), n_q)), np.zeros((len(grid), n_q))
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
        for start in range(0, n_q, batch):
            enc = tokenizer(prompts[start : start + batch], return_tensors="pt", padding=True, add_special_tokens=False).to(device)
            with torch.no_grad():
                gen = model.generate(**enc, do_sample=False, max_new_tokens=max_new, pad_token_id=tokenizer.pad_token_id)
            texts = tokenizer.batch_decode(gen[:, enc["input_ids"].shape[1]:], skip_special_tokens=True)
            for j, text in enumerate(texts):
                lcp[gi, start + j], rec[gi, start + j] = score_pair(text, refs[start + j])
        print(f"[prefixprompt] model {gi} mean lcp {lcp[gi].mean():.4f} recall {rec[gi].mean():.4f}", flush=True)
    out.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(out, keep=keep, lcp=lcp, recall=rec, seeds=np.asarray([int(e["victim_seed"]) for e in grid]))


def analyze(cfg, npz: Path, order_npz: Path, report: Path) -> None:
    d = np.load(npz)
    orders = np.load(order_npz)
    k = d["lcp"].shape[0]
    if orders["seeds"].tolist()[:k] != d["seeds"].tolist():
        raise RuntimeError("Order file and grid disagree")
    n_train = orders["adapter_loglik"].shape[1]
    keep = d["keep"]
    W = np.stack([lr_weight(orders["positions"][i], cfg["training"], n_train)[keep] for i in range(k)])
    indep = independence_matrix(orders["positions"][:k])
    n_cuts = len(CUTS)
    rng = np.random.default_rng(7)
    result = {}
    for metric in ("lcp", "recall"):
        per_cut = d[metric].reshape(k, keep.size, n_cuts)
        for label, S in {"both cuts": per_cut.mean(axis=2), "cut 25%": per_cut[:, :, 0], "cut 50%": per_cut[:, :, 1]}.items():
            pos, neg = [], []
            for i in range(k):
                for j in range(k):
                    if i != j and not indep[i, j]:
                        continue
                    ref = S[reference_ids(indep, i, j)].mean(axis=0)
                    p = _one_sided_p(float(stats.spearmanr(S[j] - ref, W[i]).statistic), S.shape[1])
                    (pos if i == j else neg).append(p)
            pos, neg = np.asarray(pos), np.asarray(neg)
            power = {}
            for m in (100, 200, S.shape[1]):
                tp = []
                for _ in range(200):
                    i = int(rng.integers(k)); idx = rng.choice(S.shape[1], m, replace=False)
                    ref = S[reference_ids(indep, i)][:, idx].mean(axis=0)
                    tp.append(_one_sided_p(float(stats.spearmanr(S[i, idx] - ref, W[i, idx]).statistic), m))
                power[str(m)] = {"tpr_0.05": float(np.mean(np.asarray(tp) < 0.05)), "tpr_1e-3": float(np.mean(np.asarray(tp) < 1e-3))}
            block = {
                "detected_0.05": f"{int(np.sum(pos < 0.05))}/{k}",
                "detected_1e-3": f"{int(np.sum(pos < 1e-3))}/{k}",
                "median_p": float(np.median(pos)),
                "null_fpr_0.05": float(np.mean(neg < 0.05)),
                "null_fpr_1e-3": float(np.mean(neg < 1e-3)),
                "ks_uniform_p": float(stats.kstest(neg, "uniform").pvalue),
                "power": power,
            }
            result[f"{metric}/{label}"] = block
            print(f"== {metric:<6} {label:<9} {json.dumps(block)}", flush=True)
    report.parent.mkdir(parents=True, exist_ok=True)
    report.write_text(json.dumps(result, indent=2))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--stage", choices=["collect", "analyze", "all"], default="all")
    parser.add_argument("--n-examples", type=int, default=400)
    parser.add_argument("--n-models", type=int, default=20)
    parser.add_argument("--batch", type=int, default=16)
    parser.add_argument("--max-new", type=int, default=24)
    args = parser.parse_args()
    cfg = load_config(args.config)
    root = Path(get(cfg, "experiment.output_root", "artifact"))
    name = cfg["experiment"]["name"]
    npz = root / "raw" / f"{name}_prefixprompt.npz"
    if args.stage in {"collect", "all"}:
        collect(cfg, npz, args.n_examples, args.n_models, args.batch, args.max_new)
    if args.stage in {"analyze", "all"}:
        analyze(cfg, npz, root / "raw" / f"{name}_palimpsest_loglik.npz", root / "reports" / f"{name}_prefixprompt.json")


if __name__ == "__main__":
    main()
