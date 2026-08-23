from __future__ import annotations

import copy
from pathlib import Path
from typing import Any

import numpy as np

from .config import get
from .hf import (
    infer_device,
    load_model,
    load_task_dataset,
    load_tokenizer,
    require_hf,
    select_split,
)
from .io import read_json, write_json, write_jsonl
from .probes import (
    build_prompt_views,
    classification_text,
    dataset_rows,
    expand_structural_probe_rows,
    label_token_ids,
    select_margin_sensitive_rows,
)
from .reproducibility import environment_snapshot, set_global_seed


def _load_checkpoint_model(
    base_model_cfg: dict[str, Any],
    checkpoint: str | Path,
    num_labels: int,
) -> Any:
    checkpoint = Path(checkpoint)
    if (checkpoint / "adapter_config.json").exists():
        return load_model(base_model_cfg, num_labels=num_labels, adapter_path=checkpoint)
    cfg = copy.deepcopy(base_model_cfg)
    cfg["model_id"] = str(checkpoint)
    cfg["revision"] = None
    return load_model(cfg, num_labels=num_labels)


def _project_logits(logits: np.ndarray, interface_cfg: dict[str, Any]) -> np.ndarray:
    mode = interface_cfg.get("mode", "full_logits")
    x = np.asarray(logits, dtype=float)
    if interface_cfg.get("round_decimals") is not None:
        x = np.round(x, int(interface_cfg["round_decimals"]))
    if mode == "full_logits":
        return x
    if mode == "topk_logits":
        k = int(interface_cfg.get("top_k", min(5, x.shape[-1])))
        idx = np.argsort(x)[-k:]
        projected = np.full_like(x, fill_value=float(interface_cfg.get("missing_value", -50.0)))
        projected[idx] = x[idx]
        return projected
    if mode == "label_only":
        projected = np.full_like(x, -50.0)
        projected[int(np.argmax(x))] = 0.0
        return projected
    raise ValueError(f"Unsupported interface mode: {mode}")


def _encode_pairs(
    tokenizer: Any,
    rows: list[dict[str, Any]],
    text_fields: list[str],
    max_length: int,
    device: str,
) -> tuple[dict[str, Any], list[int]]:
    first, second = zip(*(classification_text(row, text_fields) for row in rows), strict=True)
    second_arg = None if all(value is None for value in second) else list(second)
    encoded = tokenizer(
        list(first),
        second_arg,
        padding=True,
        truncation=True,
        max_length=max_length,
        return_tensors="pt",
    )
    lengths = encoded["attention_mask"].sum(dim=1).tolist()
    return {key: value.to(device) for key, value in encoded.items()}, [int(v) for v in lengths]


def _predict_logits(model: Any, encoded: dict[str, Any]) -> np.ndarray:
    deps = require_hf()
    torch = deps["torch"]
    with torch.inference_mode():
        return model(**encoded).logits.detach().float().cpu().numpy()


def _select_structural_probe_rows(
    cfg: dict[str, Any],
    source_rows: list[dict[str, Any]],
    base: Any,
    tokenizer: Any,
    device: str,
) -> list[dict[str, Any]]:
    data_cfg = cfg["data"]
    families = list(
        get(
            cfg,
            "probes.families",
            [
                "template_nli",
                "template_contradict",
                "layer_early",
                "layer_late",
                "module_head",
                "lexical_local",
            ],
        )
    )
    variants = expand_structural_probe_rows(source_rows, data_cfg["text_fields"], families=families)
    if not variants:
        return []
    max_length = int(data_cfg.get("max_length", 256))
    batch_size = int(get(cfg, "inference.batch_size", 32))
    scored_rows = source_rows + variants
    logits_by_query: dict[int, np.ndarray] = {}
    for start in range(0, len(scored_rows), batch_size):
        batch = scored_rows[start : start + batch_size]
        encoded, _ = _encode_pairs(
            tokenizer, batch, data_cfg["text_fields"], max_length, device
        )
        logits = _predict_logits(base, encoded)
        for row, values in zip(batch, logits, strict=True):
            logits_by_query[int(row["query_id"])] = values
    return select_margin_sensitive_rows(
        source_rows,
        variants,
        logits_by_query,
        max_variants_per_source=int(get(cfg, "probes.max_variants_per_source", 4)),
        required_families=tuple(
            get(cfg, "probes.required_families", ["template_nli", "layer_early", "module_head"])
        ),
    )


def collect_sequence_classification(
    cfg: dict[str, Any],
    checkpoint_entry: dict[str, Any],
    attacker_model_cfg: dict[str, Any] | None = None,
    tokenizer_override: dict[str, Any] | None = None,
) -> list[dict[str, Any]]:
    deps = require_hf()
    torch = deps["torch"]
    model_cfg = copy.deepcopy(cfg["model"])
    attacker_cfg = copy.deepcopy(attacker_model_cfg or model_cfg)
    data_cfg = cfg["data"]
    num_labels = int(data_cfg["num_labels"])
    victim_tokenizer = load_tokenizer(model_cfg)
    base_tokenizer = load_tokenizer(attacker_cfg, tokenizer_override=tokenizer_override)
    victim = _load_checkpoint_model(model_cfg, checkpoint_entry["checkpoint"], num_labels)
    base = load_model(attacker_cfg, num_labels=num_labels)
    device = infer_device(get(cfg, "inference.device"))
    victim.to(device).eval()
    base.to(device).eval()
    dataset_dict = load_task_dataset(data_cfg)
    split = select_split(
        dataset_dict,
        data_cfg.get("probe_split", data_cfg.get("eval_split", "validation")),
        data_cfg.get("max_probe_samples"),
        seed=int(data_cfg.get("probe_selection_seed", 0)),
    )
    rows = dataset_rows(split, data_cfg, None, seed=0)
    batch_size = int(get(cfg, "inference.batch_size", 32))
    max_length = int(data_cfg.get("max_length", 256))
    interface_cfg = cfg.get("interface", {"mode": "full_logits"})
    output: list[dict[str, Any]] = []
    probe_seeds = [int(value) for value in cfg.get("seeds", {}).get("probe", [7, 42, 123])]
    disjoint_views = bool(get(cfg, "inference.disjoint_probe_views", False))
    structural_active = bool(get(cfg, "probes.structural_active", False))
    canonical_indices = np.arange(len(rows))
    partitions = np.array_split(canonical_indices, len(probe_seeds)) if disjoint_views else None
    for view_index, probe_seed in enumerate(probe_seeds):
        set_global_seed(probe_seed)
        rng = np.random.default_rng(probe_seed)
        source_indices = partitions[view_index] if partitions is not None else canonical_indices
        order = rng.permutation(source_indices)
        view_sources = [rows[int(i)] for i in order]
        if structural_active:
            view_sources = _select_structural_probe_rows(
                cfg, view_sources, base, base_tokenizer, device
            )
            if not view_sources:
                continue
        for start in range(0, len(view_sources), batch_size):
            batch_rows = view_sources[start : start + batch_size]
            victim_inputs, input_lengths = _encode_pairs(
                victim_tokenizer, batch_rows, data_cfg["text_fields"], max_length, device
            )
            base_inputs, base_lengths = _encode_pairs(
                base_tokenizer, batch_rows, data_cfg["text_fields"], max_length, device
            )
            victim_logits = _predict_logits(victim, victim_inputs)
            base_logits = _predict_logits(base, base_inputs)
            for row, v, b, length, base_length in zip(
                batch_rows, victim_logits, base_logits, input_lengths, base_lengths, strict=True
            ):
                record = {
                    "run_id": cfg["experiment"]["name"],
                    "service_id": checkpoint_entry["service_id"],
                    "structure": checkpoint_entry["structure"],
                    "known": bool(checkpoint_entry.get("known", True)),
                    "victim_seed": int(checkpoint_entry["victim_seed"]),
                    "probe_seed": int(probe_seed),
                    "view_id": f"partition-{view_index}"
                    if disjoint_views
                    else f"probe-{probe_seed}",
                    "query_id": int(row["query_id"]),
                    "label": int(row["label"]),
                    "input_length": int(length),
                    "base_input_length": int(base_length),
                    "victim_logits": _project_logits(v, interface_cfg).tolist(),
                    "base_logits": _project_logits(b, interface_cfg).tolist(),
                    "interface": interface_cfg,
                    "attacker_base": {
                        "model_id": attacker_cfg["model_id"],
                        "revision": attacker_cfg.get("revision"),
                        "dtype": attacker_cfg.get("dtype"),
                        "quantization": attacker_cfg.get("quantization"),
                    },
                }
                if row.get("probe_family"):
                    record["probe_family"] = str(row["probe_family"])
                    record["source_query_id"] = int(row.get("source_query_id", row["query_id"]))
                output.append(record)
    del victim, base
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return output


def collect_causal_label_logits(
    cfg: dict[str, Any],
    checkpoint_entry: dict[str, Any],
    attacker_model_cfg: dict[str, Any] | None = None,
    system_prompt_override: str | None = None,
) -> list[dict[str, Any]]:
    deps = require_hf()
    torch = deps["torch"]
    model_cfg = copy.deepcopy(cfg["model"])
    attacker_cfg = copy.deepcopy(attacker_model_cfg or model_cfg)
    data_cfg = cfg["data"]
    victim_tokenizer = load_tokenizer(model_cfg)
    base_tokenizer = load_tokenizer(attacker_cfg)
    victim = _load_checkpoint_model(model_cfg, checkpoint_entry["checkpoint"], num_labels=0)
    base = load_model(attacker_cfg)
    device = infer_device(get(cfg, "inference.device"))
    victim.to(device).eval()
    base.to(device).eval()
    dataset_dict = load_task_dataset(data_cfg)
    split = select_split(
        dataset_dict,
        data_cfg.get("probe_split", data_cfg.get("eval_split", "validation")),
        data_cfg.get("max_probe_samples"),
        seed=int(data_cfg.get("probe_selection_seed", 0)),
    )
    base_rows = dataset_rows(split, data_cfg, None, seed=0)
    task_name = data_cfg["task_name"]
    configured_ids = get(cfg, "llama.label_token_ids")
    victim_ids = (
        [int(v) for v in configured_ids]
        if configured_ids
        else label_token_ids(victim_tokenizer, task_name)
    )
    base_ids = (
        [int(v) for v in configured_ids]
        if configured_ids
        else label_token_ids(base_tokenizer, task_name)
    )
    output: list[dict[str, Any]] = []
    probe_seeds = [int(value) for value in cfg.get("seeds", {}).get("probe", [7, 42, 123])]
    disjoint_views = bool(get(cfg, "inference.disjoint_probe_views", False))
    partitions = (
        np.array_split(np.arange(len(base_rows)), len(probe_seeds)) if disjoint_views else None
    )
    for view_index, probe_seed in enumerate(probe_seeds):
        selected_rows = (
            [base_rows[index] for index in partitions[view_index]]
            if partitions is not None
            else base_rows
        )
        prompt_rows = build_prompt_views(
            selected_rows,
            task_name=task_name,
            probe_seed=probe_seed,
            system_prompt=None,
        )
        for row in prompt_rows:
            victim_prompt = row["prompt"]
            if system_prompt_override:
                victim_prompt = f"{system_prompt_override}\n\n{victim_prompt}"
            base_prompt = row["prompt"]
            victim_encoded = victim_tokenizer(
                victim_prompt,
                return_tensors="pt",
                truncation=True,
                max_length=int(data_cfg.get("max_length", 512)),
            )
            base_encoded = base_tokenizer(
                base_prompt,
                return_tensors="pt",
                truncation=True,
                max_length=int(data_cfg.get("max_length", 512)),
            )
            victim_encoded = {k: v.to(device) for k, v in victim_encoded.items()}
            base_encoded = {k: v.to(device) for k, v in base_encoded.items()}
            with torch.inference_mode():
                v_full = victim(**victim_encoded).logits[0, -1].detach().float().cpu().numpy()
                b_full = base(**base_encoded).logits[0, -1].detach().float().cpu().numpy()
            v = v_full[victim_ids]
            b = b_full[base_ids]
            output.append(
                {
                    "run_id": cfg["experiment"]["name"],
                    "service_id": checkpoint_entry["service_id"],
                    "structure": checkpoint_entry["structure"],
                    "known": bool(checkpoint_entry.get("known", True)),
                    "victim_seed": int(checkpoint_entry["victim_seed"]),
                    "probe_seed": int(probe_seed),
                    "view_id": f"partition-{view_index}"
                    if disjoint_views
                    else f"probe-{probe_seed}",
                    "query_id": int(row["query_id"]),
                    "label": int(row["label"]),
                    "input_length": int(victim_encoded["attention_mask"].sum().item()),
                    "base_input_length": int(base_encoded["attention_mask"].sum().item()),
                    "victim_logits": v.tolist(),
                    "base_logits": b.tolist(),
                    "victim_label_token_ids": victim_ids,
                    "base_label_token_ids": base_ids,
                    "template_id": int(row["template_id"]),
                    "mechanism_only": True,
                    "attacker_base": attacker_cfg,
                }
            )
    del victim, base
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return output


def collect_grid(cfg: dict[str, Any], grid_path: str | Path | None = None) -> list[dict[str, Any]]:
    grid_path = Path(
        grid_path
        or get(cfg, "inputs.victim_grid_json")
        or (
            Path(get(cfg, "experiment.output_root", "artifact"))
            / "models"
            / cfg["experiment"]["name"]
            / "victim_grid.json"
        )
    )
    grid = read_json(grid_path)
    all_rows: list[dict[str, Any]] = []
    for entry in grid:
        if cfg["model"].get("task_type", "sequence_classification") == "causal_lm":
            all_rows.extend(collect_causal_label_logits(cfg, entry))
        else:
            all_rows.extend(collect_sequence_classification(cfg, entry))
    artifact_root = Path(get(cfg, "experiment.output_root", "artifact"))
    output_path = artifact_root / "raw" / f"{cfg['experiment']['name']}_queries.jsonl"
    write_jsonl(output_path, all_rows)
    write_json(
        artifact_root / "manifests" / f"{cfg['experiment']['name']}_collection.json",
        {
            "run_id": f"{cfg['experiment']['name']}::collection",
            "experiment": cfg["experiment"],
            "model": cfg["model"],
            "data": cfg["data"],
            "interface": cfg.get("interface", {}),
            "inference": cfg.get("inference", {}),
            "seeds": cfg.get("seeds", {}),
            "victim_grid_json": str(grid_path),
            "raw_jsonl": str(output_path),
            "counts": {
                "raw_queries": len(all_rows),
                "services": len({row["service_id"] for row in all_rows}),
                "victim_seeds": len({int(row["victim_seed"]) for row in all_rows}),
                "probe_seeds": len({int(row["probe_seed"]) for row in all_rows}),
            },
            "environment": environment_snapshot(),
        },
    )
    return all_rows


def collect_sequence_models(
    cfg: dict[str, Any],
    victim: Any,
    base: Any,
    tokenizer: Any,
    metadata: dict[str, Any],
) -> list[dict[str, Any]]:
    """Collect paired sequence-classification outputs from already loaded models."""
    deps = require_hf()
    torch = deps["torch"]
    data_cfg = cfg["data"]
    device = infer_device(get(cfg, "inference.device"))
    victim.to(device).eval()
    base.to(device).eval()
    dataset_dict = load_task_dataset(data_cfg)
    split = select_split(
        dataset_dict,
        data_cfg.get("probe_split", data_cfg.get("eval_split", "validation")),
        data_cfg.get("max_probe_samples"),
        seed=int(data_cfg.get("probe_selection_seed", 0)),
    )
    rows = dataset_rows(split, data_cfg, None, seed=0)
    batch_size = int(get(cfg, "inference.batch_size", 32))
    max_length = int(data_cfg.get("max_length", 256))
    interface_cfg = cfg.get("interface", {"mode": "full_logits"})
    output: list[dict[str, Any]] = []
    probe_seeds = [int(value) for value in cfg.get("seeds", {}).get("probe", [7, 42, 123])]
    disjoint_views = bool(get(cfg, "inference.disjoint_probe_views", False))
    canonical_indices = np.arange(len(rows))
    partitions = np.array_split(canonical_indices, len(probe_seeds)) if disjoint_views else None
    for view_index, probe_seed in enumerate(probe_seeds):
        rng = np.random.default_rng(probe_seed)
        source_indices = partitions[view_index] if partitions is not None else canonical_indices
        order = rng.permutation(source_indices)
        for start in range(0, len(order), batch_size):
            batch_rows = [rows[i] for i in order[start : start + batch_size]]
            text_pairs = [classification_text(row, data_cfg["text_fields"]) for row in batch_rows]
            first = [pair[0] for pair in text_pairs]
            second_values = [pair[1] for pair in text_pairs]
            second = None if all(value is None for value in second_values) else second_values
            tokenized = tokenizer(
                first,
                second,
                padding=True,
                truncation=True,
                max_length=max_length,
                return_tensors="pt",
            )
            input_lengths = tokenized["attention_mask"].sum(dim=1).tolist()
            tokenized = {k: v.to(device) for k, v in tokenized.items()}
            with torch.inference_mode():
                v_logits = victim(**tokenized).logits.detach().float().cpu().numpy()
                b_logits = base(**tokenized).logits.detach().float().cpu().numpy()
            for row, v, b, length in zip(
                batch_rows, v_logits, b_logits, input_lengths, strict=True
            ):
                output.append(
                    {
                        "run_id": cfg["experiment"]["name"],
                        "service_id": metadata["service_id"],
                        "structure": metadata.get("structure", "base_only"),
                        "known": bool(metadata.get("known", True)),
                        "victim_seed": int(metadata.get("victim_seed", -1)),
                        "probe_seed": int(probe_seed),
                        "view_id": f"partition-{view_index}"
                        if disjoint_views
                        else f"probe-{probe_seed}",
                        "query_id": int(row["query_id"]),
                        "label": int(row["label"]),
                        "input_length": int(length),
                        "victim_logits": _project_logits(v, interface_cfg).tolist(),
                        "base_logits": _project_logits(b, interface_cfg).tolist(),
                        **metadata,
                    }
                )
    return output
