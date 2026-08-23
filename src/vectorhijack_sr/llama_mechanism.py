from __future__ import annotations

import copy
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from .collect import _load_checkpoint_model
from .config import get
from .hf import (
    infer_device,
    load_model,
    load_task_dataset,
    load_tokenizer,
    require_hf,
    select_split,
)
from .io import read_json, write_csv, write_json, write_jsonl
from .metrics import kl_divergence
from .probes import LABEL_WORDS, build_prompt_views, dataset_rows, label_token_ids
from .reproducibility import environment_snapshot, git_commit, sha256_file


def _scalar_token_id(value: Any, default: int | None = None) -> int | None:
    """Normalize HF token ids that may be int or list (e.g. Llama-3.1 eos)."""
    if value is None:
        return default
    if isinstance(value, (list, tuple)):
        if not value:
            return default
        return int(value[0])
    return int(value)


def _greedy_scores(
    model: Any, encoded: dict[str, Any], max_new_tokens: int
) -> tuple[list[int], list[np.ndarray]]:
    require_hf()
    pad_token_id = _scalar_token_id(
        getattr(model.config, "pad_token_id", None),
        default=_scalar_token_id(getattr(model.config, "eos_token_id", None)),
    )
    generate_kwargs: dict[str, Any] = {
        "max_new_tokens": max_new_tokens,
        "do_sample": False,
        "return_dict_in_generate": True,
        "output_scores": True,
    }
    if pad_token_id is not None:
        generate_kwargs["pad_token_id"] = pad_token_id
    generated = model.generate(**encoded, **generate_kwargs)
    prompt_len = encoded["input_ids"].shape[1]
    tokens = generated.sequences[0, prompt_len:].detach().cpu().tolist()
    scores = [score[0].detach().float().cpu().numpy() for score in generated.scores]
    return tokens, scores


def _teacher_forced_scores(model: Any, prompt_ids: Any, target_ids: list[int]) -> list[np.ndarray]:
    deps = require_hf()
    torch = deps["torch"]
    device = prompt_ids.device
    target = torch.tensor([target_ids], dtype=prompt_ids.dtype, device=device)
    full = torch.cat([prompt_ids, target], dim=1)
    attention = torch.ones_like(full)
    with torch.inference_mode():
        logits = model(input_ids=full, attention_mask=attention).logits[0]
    start = prompt_ids.shape[1] - 1
    return [logits[start + i].detach().float().cpu().numpy() for i in range(len(target_ids))]


def _topk_overlap(a: np.ndarray, b: np.ndarray, k: int = 10) -> float:
    ia = set(np.argsort(a)[-k:])
    ib = set(np.argsort(b)[-k:])
    return len(ia.intersection(ib)) / max(k, 1)


def run_llama_mechanism(cfg: dict[str, Any]) -> dict[str, Any]:
    deps = require_hf()
    torch = deps["torch"]
    root = Path(get(cfg, "experiment.output_root", "artifact"))
    name = cfg["experiment"]["name"]
    grid = read_json(cfg["inputs"]["victim_grid_json"])
    selected_ids = set(
        cfg["inputs"].get("victim_service_ids", [entry["service_id"] for entry in grid])
    )
    victims = [entry for entry in grid if entry["service_id"] in selected_ids]
    model_cfg = copy.deepcopy(cfg["model"])
    tokenizer = load_tokenizer(model_cfg)
    base = load_model(model_cfg)
    device = infer_device(get(cfg, "inference.device"))
    base.to(device).eval()
    dataset_dict = load_task_dataset(cfg["data"])
    split = select_split(
        dataset_dict,
        cfg["data"].get("probe_split", cfg["data"].get("eval_split", "validation")),
        cfg["data"].get("max_probe_samples"),
        seed=0,
    )
    rows = dataset_rows(split, cfg["data"], None, seed=0)
    task_name = cfg["data"]["task_name"]
    labels = list(cfg.get("llama", {}).get("label_words", LABEL_WORDS[task_name]))
    raw_label_ids = cfg.get("llama", {}).get("label_token_ids")
    if raw_label_ids:
        label_ids = [int(v) for v in raw_label_ids]
    else:
        # Prefer space-prefixed single tokens; for multi-token MNLI words under
        # Llama BPE, fall back to the first answer subtoken used in training.
        label_ids = label_token_ids(
            tokenizer,
            task_name,
            labels=labels,
            allow_first_subtoken=True,
        )
    max_new_tokens = int(get(cfg, "llama.max_new_tokens", 16))
    raw: list[dict[str, Any]] = []
    for victim_entry in victims:
        victim = _load_checkpoint_model(model_cfg, victim_entry["checkpoint"], num_labels=0)
        victim.to(device).eval()
        for probe_seed in cfg["seeds"]["probe"]:
            prompts = build_prompt_views(rows, task_name, int(probe_seed))
            for row in prompts:
                encoded = tokenizer(
                    row["prompt"],
                    return_tensors="pt",
                    truncation=True,
                    max_length=int(cfg["data"].get("max_length", 1024)),
                )
                encoded = {k: v.to(device) for k, v in encoded.items()}
                with torch.inference_mode():
                    v_boundary = victim(**encoded).logits[0, -1].detach().float().cpu().numpy()
                    b_boundary = base(**encoded).logits[0, -1].detach().float().cpu().numpy()
                one_v = v_boundary[label_ids]
                one_b = b_boundary[label_ids]
                target_text = str(labels[int(row["label"])])
                target_ids = tokenizer.encode(" " + target_text, add_special_tokens=False)
                if not target_ids:
                    target_ids = [label_ids[int(row["label"])]]
                target_ids = target_ids[:2]
                v_tf = _teacher_forced_scores(victim, encoded["input_ids"], target_ids)
                b_tf = _teacher_forced_scores(base, encoded["input_ids"], target_ids)
                v_tokens, v_fr = _greedy_scores(victim, encoded, max_new_tokens)
                b_tokens, b_fr = _greedy_scores(base, encoded, max_new_tokens)
                common = min(len(v_fr), len(b_fr))
                fr_kl = [float(kl_divergence(v_fr[i], b_fr[i])) for i in range(common)]
                tf_kl = [
                    float(kl_divergence(v_tf[i], b_tf[i])) for i in range(min(len(v_tf), len(b_tf)))
                ]
                raw.append(
                    {
                        "run_id": name,
                        "service_id": victim_entry["service_id"],
                        "structure": victim_entry["structure"],
                        "victim_seed": int(victim_entry["victim_seed"]),
                        "probe_seed": int(probe_seed),
                        "query_id": int(row["query_id"]),
                        "label": int(row["label"]),
                        "template_id": int(row["template_id"]),
                        "mechanism_only": True,
                        "claim_scope": "trajectory-alignment mechanism; not a general free-running API claim",
                        "victim_logits": one_v.tolist(),
                        "base_logits": one_b.tolist(),
                        "one_token_victim_logits": one_v.tolist(),
                        "one_token_base_logits": one_b.tolist(),
                        "one_token_kl": float(kl_divergence(one_v, one_b)),
                        "one_token_topk_overlap": _topk_overlap(one_v, one_b, k=min(3, len(one_v))),
                        "teacher_forced_token_ids": target_ids,
                        "teacher_forced_kl_by_step": tf_kl,
                        "teacher_forced_kl_mean": float(np.mean(tf_kl)) if tf_kl else float("nan"),
                        "free_running_victim_tokens": v_tokens,
                        "free_running_base_tokens": b_tokens,
                        "free_running_kl_by_step": fr_kl,
                        "free_running_kl_mean": float(np.mean(fr_kl)) if fr_kl else float("nan"),
                        "trajectory_diverged_at": next(
                            (
                                i
                                for i, pair in enumerate(zip(v_tokens, b_tokens, strict=False))
                                if pair[0] != pair[1]
                            ),
                            common,
                        ),
                    }
                )
        del victim
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    frame = pd.DataFrame(raw)
    summary = frame.groupby(["service_id", "structure", "victim_seed"], as_index=False).agg(
        one_token_kl=("one_token_kl", "mean"),
        teacher_forced_kl=("teacher_forced_kl_mean", "mean"),
        free_running_kl=("free_running_kl_mean", "mean"),
        divergence_step=("trajectory_diverged_at", "mean"),
        probes=("query_id", "count"),
    )
    summary["fr_to_tf_kl_ratio"] = summary["free_running_kl"] / summary["teacher_forced_kl"].clip(
        lower=1e-12
    )
    write_jsonl(root / "raw" / f"{name}_llama_mechanism.jsonl", raw)
    write_csv(root / "summaries" / f"{name}_llama_mechanism.csv", summary.to_dict(orient="records"))
    report = {
        "mechanism_only": True,
        "general_free_running_claim": False,
        "victim_services": [entry["service_id"] for entry in victims],
        "summary": summary.to_dict(orient="records"),
    }
    write_json(root / "reports" / f"{name}_llama_mechanism.json", report)
    write_json(
        root / "manifests" / f"{name}_llama_mechanism.json",
        {
            "run_id": name,
            "experiment": cfg["experiment"],
            "config": cfg,
            "victim_grid_sha256": sha256_file(cfg["inputs"]["victim_grid_json"]),
            "environment": environment_snapshot(),
            "git_commit": git_commit(),
            "report": report,
        },
    )
    return report
