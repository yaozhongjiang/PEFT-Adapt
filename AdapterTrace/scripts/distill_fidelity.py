#!/usr/bin/env python3
"""Did the distilled students actually copy their teacher?

Reference-answer likelihood penalises a student for adopting the teacher's
(greedy) style, so it can understate what a thief gained. This script measures
behavioural fidelity instead: the owner adapter answers held-out prompts
greedily, and every model is scored by its per-token log-likelihood of those
owner answers. A faithful copy scores close to the owner and well above the
base and an independent sibling adapter.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from palimpsest_lineage import _answer_loglik, _train_dataset, discover_grid  # noqa: E402
from provenance_distill import _prompt_ids, build_sft, generate  # noqa: E402

from adaptertrace.config import get, load_config  # noqa: E402


def per_token(model, ds, batch_size, device) -> float:
    total = _answer_loglik(model, ds, batch_size, device).sum()
    tokens = sum(sum(1 for t in row if t != -100) for row in ds["labels"])
    return float(total / max(tokens, 1))


def main() -> None:
    from peft import PeftModel

    from adaptertrace.hf import (
        infer_device,
        load_model,
        load_task_dataset,
        prepare_causal_lm_dataset,
        select_split,
    )

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--owners", default="0,1,2,3,4,5,6,7")
    parser.add_argument("--n-prompts", type=int, default=200)
    parser.add_argument("--max-new", type=int, default=128)
    parser.add_argument("--batch-size", type=int, default=8)
    args = parser.parse_args()
    cfg = load_config(args.config)
    root = Path(get(cfg, "experiment.output_root", "artifact"))
    name = cfg["experiment"]["name"]
    _, tokenizer = _train_dataset(cfg)
    d = cfg["data"]
    raw = select_split(
        load_task_dataset(d), d["eval_split"], d.get("max_eval_samples"),
        seed=int(d.get("selection_seed", 0)), index_range=d.get("eval_index_range"),
    )
    prompts = _prompt_ids(prepare_causal_lm_dataset(raw, tokenizer, d))[: args.n_prompts]
    grid = discover_grid(cfg)
    seeds = [int(e["victim_seed"]) for e in grid]
    paths = {int(e["victim_seed"]): e["checkpoint"] for e in grid}
    students = root / "models" / f"{name}_distill_students"
    device = infer_device(get(cfg, "inference.device"))
    base = load_model(cfg["model"]).to(device)
    rows = []
    for oi in [int(v) for v in args.owners.split(",")]:
        owner = seeds[oi]
        sibling = seeds[(oi + 1) % len(seeds)]
        model = PeftModel.from_pretrained(base, paths[owner], adapter_name="owner")
        answers = generate(model, tokenizer, prompts, args.batch_size, args.max_new, device)
        ds = build_sft(prompts, answers, tokenizer, int(d.get("max_length", 768)))
        scores = {"owner": per_token(model, ds, args.batch_size, device)}
        model.load_adapter(paths[sibling], adapter_name="sibling")
        model.set_adapter("sibling")
        scores["sibling"] = per_token(model, ds, args.batch_size, device)
        for source in ("otherdata", "ownerprompts"):
            path = students / f"student_{source}_o{owner}"
            if path.exists():
                model.load_adapter(str(path), adapter_name=source)
                model.set_adapter(source)
                scores[f"student_{source}"] = per_token(model, ds, args.batch_size, device)
        with model.disable_adapter():
            scores["base"] = per_token(model, ds, args.batch_size, device)
        base = model.unload()
        rows.append({"owner_seed": owner, **scores})
        print(f"[fidelity] owner {owner}: " + " ".join(f"{k}={v:.4f}" for k, v in scores.items()), flush=True)
    keys = [k for k in rows[0] if k != "owner_seed"]
    summary = {k: float(np.mean([r[k] for r in rows if k in r])) for k in keys}
    out = root / "reports" / f"{name}_distill_fidelity.json"
    out.write_text(json.dumps({"summary": summary, "rows": rows}, indent=2))
    print("== fidelity " + json.dumps(summary))


if __name__ == "__main__":
    main()
