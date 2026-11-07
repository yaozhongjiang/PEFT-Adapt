#!/usr/bin/env python3
"""Step 0 / item 2: distillation evasion (upper bound of functional copying).

The thief uses the stolen adapter as a teacher: greedy responses to N prompts,
then trains a *fresh* LoRA (new seed, same recipe) on the generated pairs and
deploys only the student. Prompt sources:

  otherdata     the thief's own prompts (disjoint Dolly half): realistic
  ownerprompts  the owner's training prompts without their responses: a
                stronger leak, upper bound for a distillation thief

Each student is scored on the owner's training examples and tested with the
shadow-referenced, learning-rate-weighted order test. Per-example student
scores are stored so students can also be tested against *other* owners'
orders (false-positive control).
"""

from __future__ import annotations

import argparse
import copy
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from palimpsest_evasion import response_loglik_per_token  # noqa: E402
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


def _prompt_ids(dataset: Any) -> list[list[int]]:
    prompts = []
    for ids, labels in zip(dataset["input_ids"], dataset["labels"], strict=True):
        cut = next(i for i, lab in enumerate(labels) if lab != -100)
        prompts.append(list(ids[:cut]))
    return prompts


def generate(model: Any, tokenizer: Any, prompts: list[list[int]], batch_size: int, max_new: int, device: str) -> list[list[int]]:
    import torch

    pad = int(tokenizer.pad_token_id)
    eos = int(tokenizer.eos_token_id)
    out: list[list[int]] = []
    model.eval()
    for start in range(0, len(prompts), batch_size):
        batch = prompts[start : start + batch_size]
        width = max(len(p) for p in batch)
        ids = torch.tensor([[pad] * (width - len(p)) + p for p in batch], device=device)
        attn = torch.tensor([[0] * (width - len(p)) + [1] * len(p) for p in batch], device=device)
        with torch.no_grad():
            gen = model.generate(
                input_ids=ids, attention_mask=attn, max_new_tokens=max_new, do_sample=False, pad_token_id=pad
            )[:, width:].tolist()
        for g in gen:
            cleaned = []
            for t in g:
                if t in (pad, eos):
                    break
                cleaned.append(t)
            out.append(cleaned)
    return out


def build_sft(prompts: list[list[int]], responses: list[list[int]], tokenizer: Any, max_length: int) -> Any:
    from datasets import Dataset

    rows = []
    for p, r in zip(prompts, responses, strict=True):
        if not r:
            continue
        answer = r + [int(tokenizer.eos_token_id)]
        p = p[-max(max_length // 2, max_length - len(answer)) :]
        answer = answer[: max_length - len(p)]
        ids = p + answer
        labels = [-100] * len(p) + answer
        pad = max_length - len(ids)
        rows.append(
            {
                "input_ids": ids + [int(tokenizer.pad_token_id)] * pad,
                "attention_mask": [1] * len(ids) + [0] * pad,
                "labels": labels + [-100] * pad,
            }
        )
    return Dataset.from_list(rows)


def run(cfg: dict[str, Any], thief_cfg: dict[str, Any], owners: list[int], source: str, n_prompts: int,
        n_score: int, batch_size: int, max_new: int, out_dir: Path, score_batch: int = 4) -> None:
    from peft import PeftModel

    from adaptertrace.hf import apply_peft, infer_device, load_model, train_with_trainer
    from adaptertrace.reproducibility import set_global_seed

    root = Path(get(cfg, "experiment.output_root", "artifact"))
    name = cfg["experiment"]["name"]
    stored = np.load(root / "raw" / f"{name}_palimpsest_loglik.npz")
    seeds = stored["seeds"].tolist()
    n_train = stored["adapter_loglik"].shape[1]
    keep = np.sort(np.random.default_rng(0).choice(n_train, size=n_score, replace=False))
    owner_ds, tokenizer = _train_dataset(cfg)
    score_ds = owner_ds.select(keep.tolist())
    shadow = stored["adapter_loglik"][:, keep]
    positions = stored["positions"][..., keep]
    indep = independence_matrix(stored["positions"])
    grid = {int(e["victim_seed"]): e["checkpoint"] for e in discover_grid(cfg)}
    if source == "otherdata":
        prompt_pool = _prompt_ids(_train_dataset(thief_cfg)[0])
    elif source == "ownerprompts":
        prompt_pool = _prompt_ids(owner_ds)
    else:
        raise ValueError(source)
    prompts = prompt_pool[:n_prompts]
    device = infer_device(get(cfg, "inference.device"))
    spec = copy.deepcopy(cfg["structures"][0])
    training_cfg = {**cfg["training"], "eval_strategy": "no"}
    max_length = int(cfg["data"].get("max_length", 768))

    base = load_model(cfg["model"]).to(device)
    rows: list[dict[str, Any]] = []
    student_scores: dict[int, np.ndarray] = {}
    out_dir.mkdir(parents=True, exist_ok=True)
    for oi in owners:
        owner_seed = seeds[oi]
        # 1) teacher outputs from the stolen adapter
        base.gradient_checkpointing_disable()
        base.config.use_cache = True
        teacher = PeftModel.from_pretrained(base, grid[owner_seed], adapter_name="teacher")
        responses = generate(teacher, tokenizer, prompts, batch_size, max_new, device)
        base = teacher.unload()
        sft = build_sft(prompts, responses, tokenizer, max_length)
        print(f"[distill] owner {owner_seed}: {len(sft)} teacher pairs from {source}", flush=True)
        # 2) fresh student, new seed, same recipe
        student_seed = 9000 + owner_seed
        set_global_seed(student_seed)
        student = apply_peft(base, spec, "causal_lm")
        train_with_trainer(
            student, tokenizer, sft, sft.select(range(min(32, len(sft)))),
            out_dir / f"student_{source}_o{owner_seed}", training_cfg, student_seed,
        )
        student.eval()
        # 3) owner test + utility. Full-vocabulary log-softmax over 768-token
        # sequences is large; release training memory and score in small batches.
        import gc

        import torch

        gc.collect()
        torch.cuda.empty_cache()
        ll = _answer_loglik(student, score_ds, score_batch, device)
        student_scores[owner_seed] = ll
        ref = shadow[reference_ids(indep, oi)].mean(axis=0)
        w = lr_weight(positions[oi], cfg["training"], n_train)
        from scipy import stats

        rho = float(stats.spearmanr(ll - ref, w).statistic)
        p = _one_sided_p(rho, len(ll))
        util = response_loglik_per_token(student, cfg, tokenizer, device, score_batch)
        rows.append({"owner_seed": owner_seed, "source": source, "rho": rho, "p": p, "utility": util, "n_pairs": len(sft)})
        print(f"[distill] owner {owner_seed} {source}: p={p:.2e} rho={rho:.3f} utility={util:.4f}", flush=True)
        base = student.unload()
    # 4) false-positive control: each student against every *other* owner's order
    null_p = []
    for oi in owners:
        for oj in owners:
            if oi == oj or not indep[oi, oj]:
                continue  # student of oi shares training order with owner oj
            ll = student_scores[seeds[oi]]
            ref = shadow[reference_ids(indep, oj)].mean(axis=0)
            from scipy import stats

            rho = stats.spearmanr(ll - ref, lr_weight(positions[oj], cfg["training"], n_train)).statistic
            null_p.append(_one_sided_p(float(rho), len(ll)))
    ps = np.asarray([r["p"] for r in rows])
    summary = {
        "source": source,
        "n_owners": len(rows),
        "n_prompts": n_prompts,
        "detected_at_0.05": float(np.mean(ps < 0.05)),
        "detected_at_0.001": float(np.mean(ps < 1e-3)),
        "median_p": float(np.median(ps)),
        "mean_utility": float(np.mean([r["utility"] for r in rows])),
        "cross_owner_null_fpr_at_0.05": float(np.mean(np.asarray(null_p) < 0.05)) if null_p else None,
    }
    report = root / "reports" / f"{name}_distill_{source}.json"
    report.parent.mkdir(parents=True, exist_ok=True)
    report.write_text(json.dumps({"summary": summary, "rows": rows, "null_p": null_p}, indent=2))
    np.savez_compressed(root / "raw" / f"{name}_distill_{source}_scores.npz",
                        owners=np.asarray([seeds[o] for o in owners]), keep=keep,
                        scores=np.stack([student_scores[seeds[o]] for o in owners]))
    print(f"== distill {json.dumps(summary)}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--thief-config", required=True)
    parser.add_argument("--owners", default="0,1,2,3,4,5,6,7")
    parser.add_argument("--source", choices=["otherdata", "ownerprompts"], default="otherdata")
    parser.add_argument("--n-prompts", type=int, default=2000)
    parser.add_argument("--n-score", type=int, default=1000)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--max-new", type=int, default=256)
    parser.add_argument("--score-batch", type=int, default=4, help="batch size for log-likelihood scoring")
    args = parser.parse_args()
    cfg = load_config(args.config)
    root = Path(get(cfg, "experiment.output_root", "artifact"))
    run(
        cfg,
        load_config(args.thief_config),
        [int(v) for v in args.owners.split(",")],
        args.source,
        args.n_prompts,
        args.n_score,
        args.batch_size,
        args.max_new,
        root / "models" / f"{cfg['experiment']['name']}_distill_students",
        args.score_batch,
    )


if __name__ == "__main__":
    main()
