#!/usr/bin/env python3
"""Baselines on the sibling-discrimination problem (Dolly grid).

Owner adapter o; derived suspects = modified copies of o (SVD rank-4, 50/50
merge with an other-data adapter); null suspects = independently trained
same-recipe siblings. Baselines:

  MPT        Model Provenance Testing (Nikolic et al., NeurIPS 2025), as
             published: is the suspect's first-word agreement with candidate
             parent o higher than with every unrelated control model
             (Phi-3.5-mini-instruct, Qwen2.5-7B-Instruct)? One-sided
             two-proportion z-tests, Holm-combined over controls.
  BASE-LINEAGE  LLMPrint-style proxy: the same test with the public base as the
             candidate parent. LLMPrint attributes a suspect to its *base*
             model, which every sibling shares; this proxy reproduces that
             decision level (it is not the LLMPrint prompt-optimisation code).
  SIM-RANK   sibling-aware similarity: mean KL(o || suspect) of the first
             response token over prompts, ranked against the KL of
             independent siblings (exact rank p-value, min 1/(K+1)).
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path
from typing import Any

import numpy as np
from scipy import stats

sys.path.insert(0, str(Path(__file__).resolve().parent))
from palimpsest_evasion import svd_truncate  # noqa: E402
from palimpsest_lineage import discover_grid  # noqa: E402

from adaptertrace.config import get, load_config  # noqa: E402

TOPK = 100


def _prompts(cfg: dict[str, Any], tokenizer: Any, n: int) -> list[str]:
    from adaptertrace.hf import load_task_dataset, select_split

    d = cfg["data"]
    raw = select_split(load_task_dataset(d), d["train_split"], None, 0, index_range=[7000, 14000])
    out = []
    for row in raw.select(range(n)):
        user = str(row[d["instruction_field"]]).strip()
        ctx = str(row.get(d.get("context_field") or "", "") or "").strip()
        out.append(f"{user}\n\n{ctx}" if ctx else user)
    return out


def _first_word(text: str) -> str:
    match = re.search(r"[A-Za-z0-9']+", text)
    return match.group(0).lower() if match else ""


def observe(model: Any, tokenizer: Any, prompts: list[str], device: str, batch: int, want_dist: bool) -> dict[str, Any]:
    import torch

    tokenizer.padding_side = "left"
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    words, top_idx, top_lp = [], [], []
    model.eval()
    for start in range(0, len(prompts), batch):
        chunk = [
            tokenizer.apply_chat_template([{"role": "user", "content": p}], tokenize=False, add_generation_prompt=True)
            for p in prompts[start : start + batch]
        ]
        enc = tokenizer(chunk, return_tensors="pt", padding=True, add_special_tokens=False).to(device)
        with torch.no_grad():
            if want_dist:
                logp = torch.log_softmax(model(**enc).logits[:, -1].float(), -1)
                vals, idx = logp.topk(TOPK, dim=-1)
                top_idx.append(idx.cpu().numpy())
                top_lp.append(vals.cpu().numpy())
            gen = model.generate(**enc, max_new_tokens=8, do_sample=False, pad_token_id=tokenizer.pad_token_id)
        words += [_first_word(t) for t in tokenizer.batch_decode(gen[:, enc["input_ids"].shape[1]:], skip_special_tokens=True)]
    out: dict[str, Any] = {"words": words}
    if want_dist:
        out["top_idx"] = np.concatenate(top_idx)
        out["top_lp"] = np.concatenate(top_lp)
    return out


def _kl(o: dict[str, Any], s: dict[str, Any]) -> float:
    """KL(o || s) on o's top-k support; s's missing mass floored at its k-th value."""
    total = 0.0
    for oi, ol, si, sl in zip(o["top_idx"], o["top_lp"], s["top_idx"], s["top_lp"], strict=True):
        lookup = dict(zip(si.tolist(), sl.tolist(), strict=True))
        floor = float(sl[-1])
        s_lp = np.array([lookup.get(t, floor) for t in oi.tolist()])
        p = np.exp(ol)
        total += float(np.sum(p * (ol - s_lp)) / max(p.sum(), 1e-12))
    return total / len(o["top_idx"])


def _match(a: list[str], b: list[str]) -> np.ndarray:
    return np.array([x == y and x != "" for x, y in zip(a, b, strict=True)], dtype=float)


def mpt_p(suspect: list[str], parent: list[str], controls: list[list[str]]) -> float:
    """H0: agreement(suspect, parent) <= agreement(suspect, control) for some control."""
    m_par = _match(suspect, parent)
    ps = []
    for ctrl in controls:
        m_c = _match(suspect, ctrl)
        n = len(m_par)
        p1, p2 = m_par.mean(), m_c.mean()
        pooled = (m_par.sum() + m_c.sum()) / (2 * n)
        se = np.sqrt(max(pooled * (1 - pooled) * 2 / n, 1e-12))
        ps.append(float(stats.norm.sf((p1 - p2) / se)))
    # must beat every control: intersection-union test -> max p
    return max(ps)


def main() -> None:
    from peft import PeftModel
    from transformers import AutoModelForCausalLM, AutoTokenizer

    from adaptertrace.hf import infer_device, load_model, load_tokenizer

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, help="Dolly grid YAML")
    parser.add_argument("--partner-config", required=True, help="Dolly other-data grid YAML (merge partners)")
    parser.add_argument("--controls", required=True, help="comma-separated local paths of unrelated control models")
    parser.add_argument("--owners", default="0,1,2,3,4,5,6,7")
    parser.add_argument("--n-prompts", type=int, default=1000)
    parser.add_argument("--batch-size", type=int, default=16)
    args = parser.parse_args()
    import torch

    cfg = load_config(args.config)
    root = Path(get(cfg, "experiment.output_root", "artifact"))
    device = infer_device(get(cfg, "inference.device"))
    tokenizer = load_tokenizer(cfg["model"])
    prompts = _prompts(cfg, tokenizer, args.n_prompts)

    controls = []
    for path in [p for p in args.controls.split(",") if p]:
        tok = AutoTokenizer.from_pretrained(path)
        mdl = AutoModelForCausalLM.from_pretrained(path, dtype=torch.bfloat16).to(device)
        controls.append(observe(mdl, tok, prompts, device, args.batch_size, want_dist=False)["words"])
        del mdl
        torch.cuda.empty_cache()
        print(f"[baselines] control {path}", flush=True)

    base = load_model(cfg["model"]).to(device)
    obs: dict[str, dict[str, Any]] = {"base": observe(base, tokenizer, prompts, device, args.batch_size, True)}
    grid = discover_grid(cfg)
    partners = [e["checkpoint"] for e in discover_grid(load_config(args.partner_config))]
    peft_model = None
    owners = [int(v) for v in args.owners.split(",")]

    counter = [0]
    resident: list[str] = []

    def run(name: str, path: str, modify=None) -> None:
        nonlocal peft_model
        counter[0] += 1
        main_name = f"a{counter[0]}"
        if peft_model is None:
            peft_model = PeftModel.from_pretrained(base, path, adapter_name=main_name)
        else:
            peft_model.load_adapter(path, adapter_name=main_name)
        names = [main_name]
        active = main_name
        if modify == "svd4":
            svd_truncate(peft_model, main_name, 4)
        elif isinstance(modify, str) and modify.startswith("merge:"):
            partner, active = f"{main_name}p", f"{main_name}m"
            peft_model.load_adapter(modify.split(":", 1)[1], adapter_name=partner)
            peft_model.add_weighted_adapter([main_name, partner], [0.5, 0.5], adapter_name=active, combination_type="cat")
            names += [partner, active]
        peft_model.set_adapter(active)
        for old in resident:  # only ever keep the current adapters loaded
            peft_model.delete_adapter(old)
        resident[:] = names
        obs[name] = observe(peft_model, tokenizer, prompts, device, args.batch_size, True)
        print(f"[baselines] observed {name}", flush=True)

    for i, entry in enumerate(grid):
        run(f"g{i}", entry["checkpoint"])
    for oi in owners:
        run(f"g{oi}_svd4", grid[oi]["checkpoint"], "svd4")
        run(f"g{oi}_merge", grid[oi]["checkpoint"], f"merge:{partners[oi % len(partners)]}")

    k = len(grid)
    results: dict[str, list[dict[str, Any]]] = {"MPT": [], "BASE-LINEAGE": [], "SIM-RANK": []}
    for oi in owners:
        o = obs[f"g{oi}"]
        suspects = [(f"g{oi}_svd4", "derived"), (f"g{oi}_merge", "derived")] + [
            (f"g{j}", "sibling") for j in range(k) if j != oi
        ]
        sib_kl = {j: _kl(o, obs[f"g{j}"]) for j in range(k) if j != oi}
        for name, relation in suspects:
            s = obs[name]
            results["MPT"].append({"owner": oi, "suspect": name, "relation": relation,
                                   "p": mpt_p(s["words"], o["words"], controls)})
            results["BASE-LINEAGE"].append({"owner": oi, "suspect": name, "relation": relation,
                                            "p": mpt_p(s["words"], obs["base"]["words"], controls)})
            kl_s = _kl(o, s)
            pool = [v for j, v in sib_kl.items() if f"g{j}" != name]
            results["SIM-RANK"].append({"owner": oi, "suspect": name, "relation": relation, "kl": kl_s,
                                        "p": (1 + sum(v <= kl_s for v in pool)) / (1 + len(pool))})
    summary = {}
    for method, rows in results.items():
        for relation in ("derived", "sibling"):
            ps = np.array([r["p"] for r in rows if r["relation"] == relation])
            summary[f"{method}/{relation}"] = {
                "n": int(ps.size),
                "flag_rate_at_0.05": float(np.mean(ps < 0.05)),
                "flag_rate_at_0.001": float(np.mean(ps < 1e-3)),
                "median_p": float(np.median(ps)),
            }
    agreement = {name: float(_match(obs[name]["words"], obs["g0"]["words"]).mean()) for name in ("g1", "g0_svd4", "g0_merge", "base")}
    out = root / "reports" / f"{cfg['experiment']['name']}_baselines_sibling.json"
    out.write_text(json.dumps({"summary": summary, "first_word_agreement_with_owner0": agreement, "rows": results}, indent=2))
    for key, block in summary.items():
        print(f"== {key:<22} {json.dumps(block)}")
    print(f"== agreement_with_owner0 {json.dumps(agreement)}")


if __name__ == "__main__":
    main()
