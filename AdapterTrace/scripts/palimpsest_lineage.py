#!/usr/bin/env python3
"""Step -1 feasibility probe: palimpsestic training-order test on existing LoRA grids.

For a grid of adapters trained on the *same* data with different seeds (hence
different shuffles), each adapter can play the owner. Under H0 (the suspect is
independent of the owner's training run) the Spearman correlation between the
suspect's per-example log-likelihood and the owner's training position has an
exact null under the shuffling assumption (Kuditipudi et al., NeurIPS 2025).
The public base model is a reference that is independent of every order, so the
reference-normalised statistic uses log p_suspect - log p_base.

Stages:
  collect: reconstruct each adapter's exact training order through the same
           Trainer dataloader used for training, and score every training
           example's answer log-likelihood under the base and every adapter.
  analyze: owner x suspect matrix of correlations and one-sided p-values, and
           power as a function of the number of queried training examples.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
from scipy import stats

from adaptertrace.config import get, load_config


def _train_dataset(cfg: dict[str, Any]) -> tuple[Any, Any]:
    from adaptertrace.hf import (
        load_task_dataset,
        load_tokenizer,
        prepare_causal_lm_dataset,
        select_split,
    )

    data_cfg = cfg["data"]
    tokenizer = load_tokenizer(cfg["model"])
    # Must mirror victims.train_victim_grid: the subset is drawn with the first
    # victim seed, so every adapter in the grid saw exactly the same examples.
    raw = select_split(
        load_task_dataset(data_cfg),
        data_cfg.get("train_split", "train"),
        data_cfg.get("max_train_samples"),
        seed=int(data_cfg.get("selection_seed", cfg["seeds"]["victim"][0])),
        index_range=data_cfg.get("train_index_range"),
    )
    return prepare_causal_lm_dataset(raw, tokenizer, data_cfg), tokenizer


def discover_grid(cfg: dict[str, Any], structure: str | None = None) -> list[dict[str, Any]]:
    """All trained checkpoints of a grid, from per-checkpoint manifests.

    victim_grid.json only lists the seeds of the last train-victims job, so it
    is incomplete when a grid was trained by several parallel jobs.
    """
    from adaptertrace.io import read_json

    root = Path(get(cfg, "experiment.output_root", "artifact")) / "models" / cfg["experiment"]["name"]
    entries = []
    for manifest in sorted(root.glob("*/seed-*/run_manifest.json")):
        entry = read_json(manifest)
        if structure is None or entry["structure"] == structure:
            entries.append({**entry, "checkpoint": str(manifest.parent)})
    if not entries:
        raise FileNotFoundError(f"No trained checkpoints under {root}")
    return sorted(entries, key=lambda e: int(e["victim_seed"]))


def training_order(dataset: Any, training_cfg: dict[str, Any], seed: int) -> np.ndarray:
    """Exact replay of when each example was consumed.

    Returns position[i] (global example index in the run) for one epoch, or an
    (epochs, n) array of global positions for multi-epoch runs.
    """
    import torch
    from transformers import Trainer, TrainingArguments, default_data_collator

    class _IndexModel(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.w = torch.nn.Parameter(torch.zeros(1))

        def forward(self, input_ids=None, attention_mask=None, labels=None, example_index=None):
            return {"loss": self.w.sum()}

    indexed = dataset.add_column("example_index", list(range(len(dataset))))
    args = TrainingArguments(
        output_dir="/tmp/palimpsest_order",
        seed=seed,
        data_seed=seed,
        per_device_train_batch_size=int(training_cfg.get("train_batch_size", 16)),
        gradient_accumulation_steps=int(training_cfg.get("gradient_accumulation_steps", 1)),
        num_train_epochs=float(training_cfg.get("epochs", 1)),
        report_to=[],
        remove_unused_columns=True,
        dataloader_num_workers=0,
    )
    trainer = Trainer(
        model=_IndexModel(),
        args=args,
        train_dataset=indexed,
        data_collator=default_data_collator,
    )
    n = len(dataset)
    epochs = max(1, int(np.ceil(float(training_cfg.get("epochs", 1)))))
    loader = trainer.get_train_dataloader()
    position = np.empty((epochs, n), dtype=np.int64)
    for epoch in range(epochs):
        # Trainer calls set_epoch on its dataloader at the start of every epoch,
        # which reseeds the shuffle; replay that exactly.
        if hasattr(loader, "set_epoch"):
            loader.set_epoch(epoch)
        seen: list[int] = []
        for batch in loader:
            seen.extend(int(v) for v in batch["example_index"].reshape(-1).tolist())
        if sorted(seen) != list(range(n)):
            raise RuntimeError("Replayed order is not a permutation of the training set")
        position[epoch, np.asarray(seen)] = epoch * n + np.arange(n)  # global position
    return position[0] if epochs == 1 else position


def _answer_loglik(model: Any, dataset: Any, batch_size: int, device: str) -> np.ndarray:
    import torch

    out = np.empty(len(dataset), dtype=np.float64)
    model.eval()
    with torch.no_grad():
        for start in range(0, len(dataset), batch_size):
            rows = dataset[start : start + batch_size]
            ids = torch.tensor(rows["input_ids"])
            mask = torch.tensor(rows["attention_mask"])
            labels = torch.tensor(rows["labels"])
            width = int(mask.sum(dim=1).max())
            ids, mask, labels = ids[:, :width].to(device), mask[:, :width].to(device), labels[:, :width].to(device)
            logits = model(input_ids=ids, attention_mask=mask).logits.float()
            logp = torch.log_softmax(logits[:, :-1], dim=-1)
            target = labels[:, 1:]
            valid = target != -100
            picked = logp.gather(-1, target.clamp(min=0).unsqueeze(-1)).squeeze(-1)
            out[start : start + len(ids)] = (picked * valid).sum(dim=1).cpu().numpy()
    return out


def collect(cfg: dict[str, Any], output: Path, batch_size: int) -> None:
    from adaptertrace.hf import infer_device, load_model

    dataset, _ = _train_dataset(cfg)
    grid = discover_grid(cfg)
    seeds = [int(e["victim_seed"]) for e in grid]
    positions = {s: training_order(dataset, cfg["training"], s) for s in seeds}
    print(f"[palimpsest] replayed {len(seeds)} training orders over {len(dataset)} examples")

    device = infer_device(get(cfg, "inference.device"))
    base = load_model(cfg["model"]).to(device)
    base_ll = _answer_loglik(base, dataset, batch_size, device)
    print("[palimpsest] scored base")
    from peft import PeftModel

    peft_model = PeftModel.from_pretrained(base, grid[0]["checkpoint"], adapter_name=f"s{seeds[0]}")
    for entry in grid[1:]:
        peft_model.load_adapter(entry["checkpoint"], adapter_name=f"s{entry['victim_seed']}")
    adapter_ll = []
    for seed in seeds:
        peft_model.set_adapter(f"s{seed}")
        adapter_ll.append(_answer_loglik(peft_model, dataset, batch_size, device))
        print(f"[palimpsest] scored adapter seed {seed}")
    output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        output,
        seeds=np.asarray(seeds),
        positions=np.stack([positions[s] for s in seeds]),
        base_loglik=base_ll,
        adapter_loglik=np.stack(adapter_ll),
    )
    print(f"[palimpsest] wrote {output}")


def lr_weight(position: np.ndarray, training_cfg: dict[str, Any], n_examples: int) -> np.ndarray:
    """Learning rate at the optimiser step where each example was consumed.

    Linear warmup then linear decay to zero (Trainer default). Examples seen
    while the learning rate is small barely move the adapter, so the expected
    memorisation trace follows the schedule rather than monotone recency.
    Any fixed function of the training position keeps the exact null under
    the shuffling assumption.
    """
    per_step = int(training_cfg.get("train_batch_size", 16)) * int(
        training_cfg.get("gradient_accumulation_steps", 1)
    )
    steps = int(np.ceil(n_examples / per_step)) * max(1, int(np.ceil(float(training_cfg.get("epochs", 1)))))
    warmup = int(np.ceil(steps * float(training_cfg.get("warmup_ratio", 0.06))))
    step = np.asarray(position) // per_step
    weight = np.where(
        step < warmup,
        (step + 1) / max(warmup, 1),
        np.maximum(0.0, (steps - step) / max(steps - warmup, 1)),
    )
    # Multi-epoch runs: an example's trace is the sum over its occurrences.
    return weight.sum(axis=0) if weight.ndim == 2 else weight


def independence_matrix(positions: np.ndarray) -> np.ndarray:
    """indep[i, j] is True when adapters i and j share no epoch shuffle.

    With multi-epoch training the Trainer reseeds each epoch as seed + epoch,
    so grids with nearby seeds reuse identical orders across adapters. Such
    pairs are *not* independent under H0 and must be excluded from null pairs
    and from the reference mean (which must be independent of the owner order).
    """
    pos = positions if positions.ndim == 3 else positions[:, None, :]
    k = pos.shape[0]
    sigs = [{np.argsort(epoch).tobytes() for epoch in pos[i]} for i in range(k)]
    indep = np.array([[i != j and not (sigs[i] & sigs[j]) for j in range(k)] for i in range(k)])
    return indep


def reference_ids(indep: np.ndarray, *members: int) -> list[int]:
    """Adapters independent of every given member (owner, suspect, partner)."""
    k = indep.shape[0]
    return [l for l in range(k) if l not in members and all(indep[l, m] for m in members)]


def _one_sided_p(rho: float, n: int) -> float:
    rho = float(np.clip(rho, -0.999999, 0.999999))
    t = rho * np.sqrt((n - 2) / (1.0 - rho * rho))
    return float(stats.t.sf(t, n - 2))


def analyze(
    npz: Path,
    report: Path,
    subsample_sizes: list[int],
    n_repeats: int,
    seed: int,
    training_cfg: dict[str, Any],
) -> dict[str, Any]:
    data = np.load(npz)
    seeds = data["seeds"].tolist()
    positions = data["positions"]
    base_ll = data["base_loglik"]
    adapter_ll = data["adapter_loglik"]
    k, n = adapter_ll.shape
    result: dict[str, Any] = {"seeds": seeds, "n_examples": int(n), "statistics": {}}
    weights = {
        # multi-epoch: mean global position over occurrences
        "position": positions.mean(axis=1) if positions.ndim == 3 else positions.astype(float),
        "lr": np.stack([lr_weight(p, training_cfg, n) for p in positions]),
    }

    indep = independence_matrix(positions)

    def shadow_ref(i: int, j: int, idx: np.ndarray | slice = slice(None)) -> np.ndarray:
        # Mean of adapters trained on the same data whose shuffles are all
        # independent of the owner's (and suspect's) order: a valid reference.
        others = reference_ids(indep, i, j) if i != j else reference_ids(indep, i)
        return adapter_ll[others][:, idx].mean(axis=0)

    variants = {
        "raw/position": (lambda i, j, idx=slice(None): adapter_ll[j, idx], "position"),
        "base_referenced/position": (
            lambda i, j, idx=slice(None): adapter_ll[j, idx] - base_ll[idx],
            "position",
        ),
        "shadow_referenced/position": (
            lambda i, j, idx=slice(None): adapter_ll[j, idx] - shadow_ref(i, j, idx),
            "position",
        ),
        "shadow_referenced/lr": (
            lambda i, j, idx=slice(None): adapter_ll[j, idx] - shadow_ref(i, j, idx),
            "lr",
        ),
    }
    for name, (score_fn, weight_name) in variants.items():
        w = weights[weight_name]
        rho = np.zeros((k, k))
        pval = np.ones((k, k))
        for i in range(k):  # owner
            for j in range(k):  # suspect
                r = stats.spearmanr(score_fn(i, j), w[i]).statistic
                rho[i, j] = r
                pval[i, j] = _one_sided_p(r, n)
        diag = np.diag(pval)
        off = pval[indep]  # only truly independent (owner, suspect) pairs
        rng = np.random.default_rng(seed)
        power = []
        for m in subsample_sizes:
            if m > n:
                continue
            pos_p, null_p = [], []
            for _ in range(n_repeats):
                idx = rng.choice(n, size=m, replace=False)
                i = int(rng.integers(k))
                j = int(rng.choice(np.flatnonzero(indep[i])))
                pos_p.append(_one_sided_p(stats.spearmanr(score_fn(i, i, idx), w[i, idx]).statistic, m))
                null_p.append(_one_sided_p(stats.spearmanr(score_fn(i, j, idx), w[i, idx]).statistic, m))
            pos_p, null_p = np.asarray(pos_p), np.asarray(null_p)
            power.append(
                {
                    "n_queries": m,
                    "tpr_at_0.05": float(np.mean(pos_p < 0.05)),
                    "tpr_at_0.001": float(np.mean(pos_p < 1e-3)),
                    "fpr_at_0.05": float(np.mean(null_p < 0.05)),
                    "fpr_at_0.001": float(np.mean(null_p < 1e-3)),
                    "median_positive_p": float(np.median(pos_p)),
                }
            )
        result["statistics"][name] = {
            "owner_equals_suspect": {
                "median_rho": float(np.median(np.diag(rho))),
                "median_p": float(np.median(diag)),
                "max_p": float(diag.max()),
                "detected_at_0.001": int(np.sum(diag < 1e-3)),
            },
            "independent_pairs": {
                "n_pairs": int(off.size),
                "fpr_at_0.05": float(np.mean(off < 0.05)),
                "fpr_at_0.001": float(np.mean(off < 1e-3)),
                "ks_uniform_p": float(stats.kstest(off, "uniform").pvalue),
            },
            "rho_matrix": rho.round(5).tolist(),
            "p_matrix": pval.tolist(),
            "power_curve": power,
        }
    report.parent.mkdir(parents=True, exist_ok=True)
    report.write_text(json.dumps(result, indent=2))
    for name, block in result["statistics"].items():
        print(f"== {name}: owner=suspect {block['owner_equals_suspect']}")
        print(f"   independent: {block['independent_pairs']}")
        for row in block["power_curve"]:
            print(f"   {row}")
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, help="victim-grid YAML (e.g. llama31_lineage_victims)")
    parser.add_argument("--stage", choices=["collect", "analyze", "all"], default="all")
    parser.add_argument("--output", default=None, help="npz path for log-likelihoods")
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--subsample", default="250,500,1000,2000,4000,8000")
    parser.add_argument("--repeats", type=int, default=200)
    args = parser.parse_args()
    cfg = load_config(args.config)
    root = Path(get(cfg, "experiment.output_root", "artifact"))
    name = cfg["experiment"]["name"]
    npz = Path(args.output) if args.output else root / "raw" / f"{name}_palimpsest_loglik.npz"
    if args.stage in {"collect", "all"}:
        collect(cfg, npz, args.batch_size)
    if args.stage in {"analyze", "all"}:
        analyze(
            npz,
            root / "reports" / f"{name}_palimpsest.json",
            [int(v) for v in args.subsample.split(",")],
            args.repeats,
            int(get(cfg, "experiment.seed", 7)),
            cfg["training"],
        )


if __name__ == "__main__":
    main()
