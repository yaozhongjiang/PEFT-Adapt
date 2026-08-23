from __future__ import annotations

import copy
from pathlib import Path
from typing import Any

from .config import get
from .hf import (
    apply_peft,
    dataset_fingerprint,
    load_model,
    load_task_dataset,
    load_tokenizer,
    prepare_causal_lm_dataset,
    prepare_classification_dataset,
    save_adapter_or_model,
    select_split,
    train_with_trainer,
    trainable_parameter_count,
)
from .io import ensure_dir, read_json, write_json
from .reproducibility import environment_snapshot, set_global_seed


def train_victim_grid(cfg: dict[str, Any]) -> list[dict[str, Any]]:
    model_cfg = copy.deepcopy(cfg["model"])
    data_cfg = cfg["data"]
    training_cfg = cfg["training"]
    artifact_root = Path(get(cfg, "experiment.output_root", "artifact"))
    output_root = artifact_root / "models" / cfg["experiment"]["name"]
    ensure_dir(output_root)
    victim_seeds = list(cfg["seeds"]["victim"])
    structures = list(cfg["structures"])
    dataset_dict = load_task_dataset(data_cfg)
    tokenizer = load_tokenizer(model_cfg)
    train_raw = select_split(
        dataset_dict,
        data_cfg.get("train_split", "train"),
        data_cfg.get("max_train_samples"),
        seed=victim_seeds[0],
    )
    eval_raw = select_split(
        dataset_dict,
        data_cfg.get("eval_split", "validation"),
        data_cfg.get("max_eval_samples"),
        seed=victim_seeds[0],
    )
    task_type = model_cfg.get("task_type", "sequence_classification")
    if task_type == "causal_lm":
        train_dataset = prepare_causal_lm_dataset(train_raw, tokenizer, data_cfg)
        eval_dataset = prepare_causal_lm_dataset(eval_raw, tokenizer, data_cfg)
        num_labels = None
        training_cfg = copy.deepcopy(training_cfg)
        training_cfg["data_collator"] = "default"
    elif task_type == "sequence_classification":
        train_dataset = prepare_classification_dataset(train_raw, tokenizer, data_cfg)
        eval_dataset = prepare_classification_dataset(eval_raw, tokenizer, data_cfg)
        num_labels = int(data_cfg["num_labels"])
    else:
        raise ValueError(f"Victim-grid training does not support task_type={task_type!r}")
    results: list[dict[str, Any]] = []
    for spec in structures:
        for victim_seed in victim_seeds:
            set_global_seed(victim_seed)
            model = load_model(model_cfg, num_labels=num_labels)
            model = apply_peft(model, spec, model_cfg.get("task_type", "sequence_classification"))
            structure_name = spec["name"]
            output_dir = output_root / structure_name / f"seed-{victim_seed}"
            manifest_path = output_dir / "run_manifest.json"
            if manifest_path.is_file():
                manifest = read_json(manifest_path)
                results.append({**manifest, "checkpoint": str(output_dir)})
                continue
            result = train_with_trainer(
                model,
                tokenizer,
                train_dataset,
                eval_dataset,
                output_dir,
                training_cfg,
                victim_seed,
            )
            save_adapter_or_model(model, tokenizer, output_dir)
            manifest = {
                "run_id": f"{cfg['experiment']['name']}::{structure_name}::seed-{victim_seed}",
                "experiment": cfg["experiment"]["name"],
                "service_id": f"{structure_name}-seed-{victim_seed}",
                "structure": structure_name,
                "known": bool(spec.get("known", True)),
                "structure_spec": spec,
                "victim_seed": victim_seed,
                "model": model_cfg,
                "data": data_cfg,
                "dataset_fingerprints": {
                    "train": dataset_fingerprint(train_raw),
                    "eval": dataset_fingerprint(eval_raw),
                },
                "training": training_cfg,
                "trainable_parameters": trainable_parameter_count(model),
                "result": result,
                "environment": environment_snapshot(),
            }
            write_json(output_dir / "run_manifest.json", manifest)
            results.append({**manifest, "checkpoint": str(output_dir)})
            del model
    write_json(output_root / "victim_grid.json", results)
    write_json(
        artifact_root / "manifests" / f"{cfg['experiment']['name']}_victim_grid.json",
        {
            "run_id": f"{cfg['experiment']['name']}::victim-grid",
            "experiment": cfg["experiment"],
            "model": model_cfg,
            "data": data_cfg,
            "training": training_cfg,
            "victim_seeds": victim_seeds,
            "structures": structures,
            "counts": {"victim_checkpoints": len(results)},
            "victim_grid_json": str(output_root / "victim_grid.json"),
            "environment": environment_snapshot(),
        },
    )
    return results
