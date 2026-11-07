#!/usr/bin/env python3
"""Step 0 / arm A: does the schedule-weighted palimpsest test survive evasion?

Every adapter of an existing same-recipe grid plays the owner. A thief applies a
weight-space or deployment-side modification to the owner's adapter; the owner
then runs the shadow-referenced, learning-rate-weighted training-order test on
the modified adapter (reference = mean of the *unmodified* other adapters,
which are independent of the owner's order). Utility of every modified adapter
is measured as held-out label accuracy so detection can be plotted against the
thief's utility cost.

Requires the npz written by palimpsest_lineage.py (orders + shadow scores).
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


def _lora_modules(model: Any, name: str):
    for module in model.modules():
        if hasattr(module, "lora_A") and name in getattr(module, "lora_A", {}):
            yield module


def svd_truncate(model: Any, name: str, keep: int) -> None:
    """Keep the top-`keep` singular directions of every delta B@A (shape preserved)."""
    import torch

    with torch.no_grad():
        for m in _lora_modules(model, name):
            a = m.lora_A[name].weight.float()  # r x in
            b = m.lora_B[name].weight.float()  # out x r
            qb, rb = torch.linalg.qr(b)
            qa, ra = torch.linalg.qr(a.T)
            u, s, vh = torch.linalg.svd(rb @ ra.T)
            s[keep:] = 0.0
            m.lora_B[name].weight.copy_((qb @ u @ torch.diag(s)).to(m.lora_B[name].weight.dtype))
            m.lora_A[name].weight.copy_((vh @ qa.T).to(m.lora_A[name].weight.dtype))


def add_noise(model: Any, name: str, relative: float, seed: int) -> None:
    """Gaussian noise on both factors, scaled to each factor's RMS."""
    import torch

    gen = torch.Generator(device="cpu").manual_seed(seed)
    with torch.no_grad():
        for m in _lora_modules(model, name):
            for p in (m.lora_A[name].weight, m.lora_B[name].weight):
                rms = p.float().pow(2).mean().sqrt()
                noise = torch.randn(p.shape, generator=gen).to(p.device) * rms * relative
                p.add_(noise.to(p.dtype))


def label_accuracy(model: Any, cfg: dict[str, Any], tokenizer: Any, device: str, batch_size: int) -> float:
    """Held-out accuracy: argmax over label words of the answer log-likelihood."""
    from adaptertrace.hf import load_task_dataset, prepare_causal_lm_dataset, select_split

    data_cfg = copy.deepcopy(cfg["data"])
    raw = select_split(
        load_task_dataset(data_cfg),
        data_cfg.get("eval_split", "validation"),
        data_cfg.get("max_eval_samples"),
        seed=int(cfg["seeds"]["victim"][0]),
    )
    raw = raw.filter(lambda r: int(r[data_cfg.get("label_field", "label")]) >= 0)
    truth = np.asarray(raw[data_cfg.get("label_field", "label")])
    scores = []
    for label in range(int(data_cfg["num_labels"])):
        forced = raw.map(lambda r: {data_cfg.get("label_field", "label"): label})
        scores.append(_answer_loglik(model, prepare_causal_lm_dataset(forced, tokenizer, data_cfg), batch_size, device))
    return float(np.mean(np.argmax(np.stack(scores), axis=0) == truth))


def response_loglik_per_token(
    model: Any, cfg: dict[str, Any], tokenizer: Any, device: str, batch_size: int
) -> float:
    """Generative utility: mean log-likelihood per held-out response token."""
    from adaptertrace.hf import load_task_dataset, prepare_causal_lm_dataset, select_split

    data_cfg = cfg["data"]
    raw = select_split(
        load_task_dataset(data_cfg),
        data_cfg.get("eval_split", "validation"),
        data_cfg.get("max_eval_samples"),
        seed=int(data_cfg.get("selection_seed", cfg["seeds"]["victim"][0])),
        index_range=data_cfg.get("eval_index_range"),
    )
    eval_ds = prepare_causal_lm_dataset(raw, tokenizer, data_cfg)
    total = _answer_loglik(model, eval_ds, batch_size, device).sum()
    tokens = sum(sum(1 for t in row if t != -100) for row in eval_ds["labels"])
    return float(total / max(tokens, 1))


def task_utility(model: Any, cfg: dict[str, Any], tokenizer: Any, device: str, batch_size: int) -> float:
    if cfg["data"].get("format") == "instruction":
        return response_loglik_per_token(model, cfg, tokenizer, device, batch_size)
    return label_accuracy(model, cfg, tokenizer, device, batch_size)


def _test(scores: np.ndarray, reference: np.ndarray, weight: np.ndarray) -> tuple[float, float]:
    rho = stats.spearmanr(scores - reference, weight).statistic
    return float(rho), _one_sided_p(rho, len(scores))


SUITES = ("weight", "merge_external", "quant", "continue_ft")


def _variants(suite: str, partner_labels: list[str]) -> list[tuple[str, dict[str, Any]]]:
    if suite == "weight":
        out: list[tuple[str, dict[str, Any]]] = [("none", {})]
        out += [(f"svd_keep{k}", {"svd": k}) for k in (32, 16, 8, 4)]
        out += [(f"merge_lambda{lam}", {"merge": lam, "partner": "sibling"}) for lam in (0.75, 0.5, 0.25)]
        out += [(f"noise{r}", {"noise": r}) for r in (0.1, 0.3, 1.0)]
        return out
    if suite == "merge_external":
        return [
            (f"merge_{label}_lambda{lam}", {"merge": lam, "partner": label})
            for label in partner_labels
            # lambda 0 = the partner alone: a false-positive control (never saw
            # the owner data, independent of the owner order).
            for lam in (0.75, 0.5, 0.25, 0.1, 0.0)
        ]
    raise ValueError(suite)


def _freeze_except(model: Any, adapter: str) -> list[Any]:
    params = []
    for pname, param in model.named_parameters():
        trainable = "lora_" in pname and f".{adapter}." in pname
        param.requires_grad = trainable
        if trainable:
            params.append(param)
    if not params:
        raise RuntimeError(f"No trainable LoRA parameters for adapter {adapter}")
    return params


def continue_finetune(
    model: Any, adapter: str, train_rows: Any, steps: int, start_step: int, cfg: dict[str, Any], seed: int, device: str
) -> int:
    """Thief keeps training the stolen adapter on its own data (constant lr)."""
    import torch

    tcfg = cfg["training"]
    accum = int(tcfg.get("gradient_accumulation_steps", 16)) * int(tcfg.get("train_batch_size", 1))
    params = _freeze_except(model, adapter)
    if not hasattr(model, "_thief_opt") or model._thief_opt[0] != adapter:
        model._thief_opt = (adapter, torch.optim.AdamW(params, lr=float(tcfg.get("learning_rate", 1e-4))))
    opt = model._thief_opt[1]
    rng = np.random.default_rng(seed)
    order = rng.permutation(len(train_rows))
    model.train()
    step = start_step
    cursor = start_step * accum
    while step < steps:
        for _ in range(accum):
            row = train_rows[int(order[cursor % len(order)])]
            cursor += 1
            ids = torch.tensor([row["input_ids"]])
            mask = torch.tensor([row["attention_mask"]])
            labels = torch.tensor([row["labels"]])
            width = int(mask.sum())
            out = model(
                input_ids=ids[:, :width].to(device),
                attention_mask=mask[:, :width].to(device),
                labels=labels[:, :width].to(device),
            )
            (out.loss / accum).backward()
        opt.step()
        opt.zero_grad(set_to_none=True)
        step += 1
    model.eval()
    return step


def run(
    cfg: dict[str, Any],
    npz: Path,
    report: Path,
    owners: list[int],
    batch_size: int,
    suite: str,
    partner_cfgs: list[dict[str, Any]],
    quant: str | None,
    thief_cfg: dict[str, Any] | None,
    ft_steps: list[int],
    max_examples: int | None = None,
) -> None:
    from peft import PeftModel

    from adaptertrace.hf import infer_device, load_model, prepare_causal_lm_dataset

    data = np.load(npz)
    seeds = data["seeds"].tolist()
    positions = data["positions"]
    shadow_ll = data["adapter_loglik"]
    n = shadow_ll.shape[1]
    dataset, tokenizer = _train_dataset(cfg)
    if len(dataset) != n:
        raise RuntimeError("Training set does not match the stored log-likelihoods")
    if max_examples and max_examples < n:
        # Fixed random subset of the owner's training examples (same for every
        # variant); the shuffle-based null is unaffected by subsampling.
        keep = np.sort(np.random.default_rng(0).choice(n, size=max_examples, replace=False))
        dataset = dataset.select(keep.tolist())
        positions = positions[..., keep]
        shadow_ll = shadow_ll[:, keep]
    indep = independence_matrix(positions)
    grid = {int(e["victim_seed"]): e["checkpoint"] for e in discover_grid(cfg)}
    partners = {
        pc["experiment"]["name"].split("_partner_")[-1].replace("llama31_", ""): [e["checkpoint"] for e in discover_grid(pc)]
        for pc in partner_cfgs
    }
    device = infer_device(get(cfg, "inference.device"))

    model_cfg = copy.deepcopy(cfg["model"])
    if quant:
        model_cfg["quantization"] = quant
        variants: list[tuple[str, dict[str, Any]]] = [(f"quant_{quant}", {})]
    elif suite == "continue_ft":
        variants = [(f"ft_{s}steps", {"ft": s}) for s in ft_steps]
    else:
        variants = _variants(suite, sorted(partners))
    base = load_model(model_cfg)
    if not quant:
        base = base.to(device)
    if suite == "continue_ft":
        base.gradient_checkpointing_enable()
        base.enable_input_require_grads()
    thief_rows = None
    if thief_cfg is not None:
        thief_rows, _ = _train_dataset(thief_cfg)

    rows: list[dict[str, Any]] = []
    peft_model = None

    def score(vname: str, owner_seed: int, reference: np.ndarray, weight: np.ndarray) -> None:
        ll = _answer_loglik(peft_model, dataset, batch_size, device)
        rho, p = _test(ll, reference, weight)
        acc = task_utility(peft_model, cfg, tokenizer, device, batch_size)
        rows.append({"owner_seed": owner_seed, "variant": vname, "rho": rho, "p": p, "utility_acc": acc})
        print(f"[evasion] owner {owner_seed} {vname:<26} p={p:.2e} acc={acc:.3f}", flush=True)

    for oi in owners:
        owner_seed = seeds[oi]
        # Sibling partner and reference must not share any epoch shuffle with
        # the owner (nearby seeds do under multi-epoch training).
        sibling_idx = next(
            (oi + d) % len(seeds) for d in range(1, len(seeds)) if indep[oi, (oi + d) % len(seeds)]
        )
        reference = shadow_ll[reference_ids(indep, oi, sibling_idx)].mean(axis=0)
        weight = lr_weight(positions[oi], cfg["training"], n)  # n = full training-set size
        if suite == "continue_ft":
            name = f"o{owner_seed}_ft"
            if peft_model is None:
                peft_model = PeftModel.from_pretrained(base, grid[owner_seed], adapter_name=name, is_trainable=True)
            else:
                peft_model.load_adapter(grid[owner_seed], adapter_name=name, is_trainable=True)
            peft_model.set_adapter(name)
            done = 0
            for vname, spec in variants:
                done = continue_finetune(
                    peft_model, name, thief_rows, spec["ft"], done, cfg, owner_seed + 999, device
                )
                score(vname, owner_seed, reference, weight)
            if len(peft_model.peft_config) > 1:
                peft_model.delete_adapter(name)
            if hasattr(peft_model, "_thief_opt"):
                del peft_model._thief_opt
            continue
        for vname, spec in variants:
            # PEFT adapter names become module names and may not contain ".".
            name = f"o{owner_seed}_{vname}".replace(".", "p")
            if peft_model is None:
                peft_model = PeftModel.from_pretrained(base, grid[owner_seed], adapter_name=name)
            else:
                peft_model.load_adapter(grid[owner_seed], adapter_name=name)
            active, partner = name, None
            if "svd" in spec:
                svd_truncate(peft_model, name, spec["svd"])
            if "noise" in spec:
                add_noise(peft_model, name, spec["noise"], seed=owner_seed)
            if "merge" in spec:
                if spec["partner"] == "sibling":
                    partner_path = grid[seeds[sibling_idx]]
                else:
                    pool = partners[spec["partner"]]
                    partner_path = pool[oi % len(pool)]
                partner = f"{name}_partner"
                peft_model.load_adapter(partner_path, adapter_name=partner)
                active = f"{name}_merged"
                peft_model.add_weighted_adapter(
                    adapters=[name, partner],
                    weights=[spec["merge"], 1.0 - spec["merge"]],
                    adapter_name=active,
                    combination_type="cat",
                )
            peft_model.set_adapter(active)
            score(vname, owner_seed, reference, weight)
            for adapter in [a for a in (name, active, partner) if a]:
                if adapter in peft_model.peft_config and len(peft_model.peft_config) > 1:
                    peft_model.delete_adapter(adapter)

    report.parent.mkdir(parents=True, exist_ok=True)
    summary: dict[str, Any] = {}
    for vname, _ in variants:
        sel = [r for r in rows if r["variant"] == vname]
        ps = np.asarray([r["p"] for r in sel])
        summary[vname] = {
            "n_owners": len(sel),
            "detected_at_0.05": float(np.mean(ps < 0.05)),
            "detected_at_0.001": float(np.mean(ps < 1e-3)),
            "median_p": float(np.median(ps)),
            "mean_utility_acc": float(np.mean([r["utility_acc"] for r in sel])),
        }
    report.write_text(json.dumps({"summary": summary, "rows": rows}, indent=2))
    for vname, block in summary.items():
        print(f"== {vname:<26} {block}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--npz", default=None)
    parser.add_argument("--owners", default="all", help="comma-separated grid indices or 'all'")
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--suite", choices=SUITES, default="weight")
    parser.add_argument("--partner-configs", default="", help="comma-separated victim-grid YAMLs")
    parser.add_argument("--quant", choices=["int8", "int4"], default=None)
    parser.add_argument("--thief-config", default=None, help="grid YAML whose training data the thief uses")
    parser.add_argument("--ft-steps", default="50,200,800")
    parser.add_argument("--max-examples", type=int, default=None, help="score a fixed subset of training examples")
    args = parser.parse_args()
    cfg = load_config(args.config)
    root = Path(get(cfg, "experiment.output_root", "artifact"))
    name = cfg["experiment"]["name"]
    npz = Path(args.npz) if args.npz else root / "raw" / f"{name}_palimpsest_loglik.npz"
    k = len(np.load(npz)["seeds"])
    owners = list(range(k)) if args.owners == "all" else [int(v) for v in args.owners.split(",")]
    tag = args.suite if not args.quant else f"quant_{args.quant}"
    run(
        cfg,
        npz,
        root / "reports" / f"{name}_palimpsest_evasion_{tag}.json",
        owners,
        args.batch_size,
        "quant" if args.quant else args.suite,
        [load_config(p) for p in args.partner_configs.split(",") if p],
        args.quant,
        load_config(args.thief_config) if args.thief_config else None,
        [int(v) for v in args.ft_steps.split(",")],
        args.max_examples,
    )


if __name__ == "__main__":
    main()
