from __future__ import annotations

import copy
import gc
import json
import tempfile
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from .bootstrap import cluster_bootstrap
from .collect import _load_checkpoint_model
from .config import get
from .hf import (
    apply_peft,
    freeze_to_parameter_budget,
    infer_device,
    load_model,
    load_task_dataset,
    load_tokenizer,
    peak_memory_bytes,
    require_hf,
    save_adapter_or_model,
    select_split,
    total_parameter_count,
    trainable_parameter_count,
)
from .io import ensure_dir, file_size_bytes, read_json, read_jsonl, write_csv, write_json
from .probes import classification_text, dataset_rows
from .reproducibility import environment_snapshot, git_commit, set_global_seed, sha256_file


@dataclass
class TrainAccounting:
    wall_seconds: float
    estimated_flops: int
    optimizer_steps: int
    examples_seen: int
    peak_memory_bytes: int


class DistillationRecords:
    def __init__(self, records: list[dict[str, Any]]) -> None:
        self.records = records

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, index: int) -> dict[str, Any]:
        return self.records[index]


def _prepare_distillation_records(
    tokenizer: Any,
    query_rows: list[dict[str, Any]],
    dataset: Any,
    data_cfg: dict[str, Any],
) -> list[dict[str, Any]]:
    by_id = {int(row["query_id"]): row for row in query_rows}
    records: list[dict[str, Any]] = []
    max_length = int(data_cfg.get("max_length", 256))
    for row in dataset_rows(dataset, data_cfg, None, seed=0):
        query_id = int(row["query_id"])
        if query_id not in by_id:
            continue
        first, second = classification_text(row, data_cfg["text_fields"])
        encoded = tokenizer(first, second, truncation=True, max_length=max_length)
        teacher = by_id[query_id]
        records.append(
            {
                **encoded,
                "labels": int(row["label"]),
                "teacher_logits": [float(v) for v in teacher["victim_logits"]],
                "query_id": query_id,
            }
        )
    if not records:
        raise ValueError(
            "No distillation records matched raw query IDs to the configured dataset split"
        )
    return records


def _collator(tokenizer: Any):
    deps = require_hf()
    torch = deps["torch"]

    def collate(features: list[dict[str, Any]]) -> dict[str, Any]:
        model_features = [
            {k: v for k, v in item.items() if k not in {"teacher_logits", "query_id", "labels"}}
            for item in features
        ]
        batch = tokenizer.pad(model_features, return_tensors="pt")
        batch["labels"] = torch.tensor([item["labels"] for item in features], dtype=torch.long)
        batch["teacher_logits"] = torch.tensor(
            [item["teacher_logits"] for item in features], dtype=torch.float32
        )
        batch["query_id"] = torch.tensor([item["query_id"] for item in features], dtype=torch.long)
        return batch

    return collate


def _estimate_batch_flops(model: Any, batch: dict[str, Any]) -> int:
    inputs = {k: v for k, v in batch.items() if k not in {"teacher_logits", "query_id", "labels"}}
    try:
        forward = int(model.floating_point_ops(inputs))
        return max(forward * 3, 1)
    except Exception:
        tokens = int(batch.get("attention_mask", batch["input_ids"]).sum().item())
        # Explicitly labeled estimate: forward/backward over the full deployed
        # model, not merely the trainable subset.
        return int(6 * total_parameter_count(model) * max(tokens, 1))


def train_distilled(
    model: Any,
    tokenizer: Any,
    records: list[dict[str, Any]],
    output_dir: str | Path,
    training_cfg: dict[str, Any],
    seed: int,
    max_flops: int | None = None,
) -> TrainAccounting:
    deps = require_hf()
    torch = deps["torch"]
    DataLoader = torch.utils.data.DataLoader
    set_global_seed(seed)
    device = infer_device(training_cfg.get("device"))
    model.to(device)
    model.train()
    loader = DataLoader(
        DistillationRecords(records),
        batch_size=int(training_cfg.get("train_batch_size", 16)),
        shuffle=True,
        collate_fn=_collator(tokenizer),
        num_workers=int(training_cfg.get("dataloader_num_workers", 0)),
        generator=torch.Generator().manual_seed(seed),
    )
    optimizer = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=float(training_cfg.get("learning_rate", 2e-4)),
        weight_decay=float(training_cfg.get("weight_decay", 0.01)),
    )
    epochs = int(training_cfg.get("epochs", 3))
    max_steps = int(training_cfg.get("max_steps", -1))
    grad_accum = int(training_cfg.get("gradient_accumulation_steps", 1))
    temperature = float(training_cfg.get("temperature", 2.0))
    alpha = float(training_cfg.get("distillation_alpha", 1.0))
    hard_weight = float(training_cfg.get("hard_label_weight", 0.0))
    use_cuda = device.startswith("cuda") and torch.cuda.is_available()
    use_fp16 = use_cuda and bool(training_cfg.get("fp16", False))
    use_bf16 = use_cuda and bool(training_cfg.get("bf16", False))
    if use_fp16 and use_bf16:
        raise ValueError("Choose at most one of fp16 and bf16")
    amp_enabled = use_fp16 or use_bf16
    amp_dtype = torch.float16 if use_fp16 else torch.bfloat16
    try:
        scaler = torch.amp.GradScaler("cuda", enabled=use_fp16)
    except (AttributeError, TypeError):  # PyTorch compatibility path
        scaler = torch.cuda.amp.GradScaler(enabled=use_fp16)
    if use_cuda:
        torch.cuda.reset_peak_memory_stats()
    start = time.perf_counter()
    estimated_flops = 0
    optimizer_steps = 0
    examples_seen = 0
    optimizer.zero_grad(set_to_none=True)
    stop = False
    accumulated = 0

    def optimizer_step() -> None:
        nonlocal optimizer_steps, accumulated
        if accumulated == 0:
            return
        if use_fp16:
            scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(
            [parameter for parameter in model.parameters() if parameter.requires_grad],
            float(training_cfg.get("max_grad_norm", 1.0)),
        )
        if use_fp16:
            scaler.step(optimizer)
            scaler.update()
        else:
            optimizer.step()
        optimizer.zero_grad(set_to_none=True)
        optimizer_steps += 1
        accumulated = 0

    for _epoch in range(epochs):
        for batch_index, batch in enumerate(loader):
            batch = {
                key: value.to(device) if hasattr(value, "to") else value
                for key, value in batch.items()
            }
            teacher = batch.pop("teacher_logits")
            batch.pop("query_id")
            labels = batch["labels"]
            with torch.autocast(
                device_type="cuda",
                dtype=amp_dtype,
                enabled=amp_enabled,
            ):
                outputs = model(**batch)
                student = outputs.logits
                soft_teacher = torch.softmax(teacher / temperature, dim=-1)
                log_student = torch.log_softmax(student / temperature, dim=-1)
                kl = torch.nn.functional.kl_div(
                    log_student, soft_teacher, reduction="batchmean"
                ) * (temperature**2)
                hard = torch.nn.functional.cross_entropy(student, labels)
                loss = alpha * kl + hard_weight * hard
            scaled_loss = loss / grad_accum
            if use_fp16:
                scaler.scale(scaled_loss).backward()
            else:
                scaled_loss.backward()
            accumulated += 1
            batch_flops = _estimate_batch_flops(model, batch)
            estimated_flops += batch_flops
            examples_seen += int(labels.shape[0])
            if accumulated >= grad_accum:
                optimizer_step()
            if max_steps > 0 and optimizer_steps >= max_steps:
                stop = True
            if max_flops is not None and estimated_flops >= max_flops:
                stop = True
            if stop:
                optimizer_step()
                break
        if stop:
            break
        # Do not discard the final partial accumulation of an epoch.
        optimizer_step()
    elapsed = time.perf_counter() - start
    output_dir = ensure_dir(output_dir)
    save_adapter_or_model(model, tokenizer, output_dir)
    accounting = TrainAccounting(
        wall_seconds=elapsed,
        estimated_flops=int(estimated_flops),
        optimizer_steps=int(optimizer_steps),
        examples_seen=int(examples_seen),
        peak_memory_bytes=peak_memory_bytes(device),
    )
    write_json(output_dir / "train_accounting.json", accounting.__dict__)
    return accounting


def evaluate_sequence_model(
    model: Any,
    tokenizer: Any,
    dataset: Any,
    data_cfg: dict[str, Any],
    teacher_model: Any | None = None,
    batch_size: int = 32,
    device: str | None = None,
) -> dict[str, float]:
    deps = require_hf()
    torch = deps["torch"]
    device = infer_device(device)
    model.to(device).eval()
    if teacher_model is not None:
        teacher_model.to(device).eval()
    correct = 0
    agree = 0
    total = 0
    kl_values: list[float] = []
    rows = dataset_rows(dataset, data_cfg, None, seed=0)
    for start in range(0, len(rows), batch_size):
        batch_rows = rows[start : start + batch_size]
        pairs = [classification_text(row, data_cfg["text_fields"]) for row in batch_rows]
        first = [p[0] for p in pairs]
        second_values = [p[1] for p in pairs]
        second = None if all(v is None for v in second_values) else second_values
        encoded = tokenizer(
            first,
            second,
            padding=True,
            truncation=True,
            max_length=int(data_cfg.get("max_length", 256)),
            return_tensors="pt",
        )
        encoded = {k: v.to(device) for k, v in encoded.items()}
        labels = torch.tensor([row["label"] for row in batch_rows], device=device)
        with torch.inference_mode():
            logits = model(**encoded).logits
            pred = logits.argmax(dim=-1)
            if teacher_model is not None:
                teacher_logits = teacher_model(**encoded).logits
                teacher_pred = teacher_logits.argmax(dim=-1)
                agree += int((pred == teacher_pred).sum().item())
                kl = torch.nn.functional.kl_div(
                    torch.log_softmax(logits, dim=-1),
                    torch.softmax(teacher_logits, dim=-1),
                    reduction="none",
                ).sum(dim=-1)
                kl_values.extend(kl.detach().cpu().tolist())
        correct += int((pred == labels).sum().item())
        total += len(batch_rows)
    return {
        "task_accuracy": correct / max(total, 1),
        "agreement": agree / max(total, 1) if teacher_model is not None else float("nan"),
        "kl_to_victim": float(np.mean(kl_values)) if kl_values else float("nan"),
        "examples": total,
    }


def _merge_evaluate(
    checkpoint: str | Path,
    public_model_cfg: dict[str, Any],
    tokenizer: Any,
    eval_dataset: Any,
    data_cfg: dict[str, Any],
    teacher_model: Any,
) -> dict[str, Any]:
    num_labels = int(data_cfg["num_labels"])
    model = _load_checkpoint_model(public_model_cfg, checkpoint, num_labels)
    attach_ok = True
    merged_ok = False
    merged_metrics: dict[str, float] = {}
    if hasattr(model, "merge_and_unload"):
        try:
            merged = model.merge_and_unload(safe_merge=True)
            merged_ok = True
            merged_metrics = evaluate_sequence_model(
                merged, tokenizer, eval_dataset, data_cfg, teacher_model=teacher_model
            )
        except Exception as exc:
            merged_metrics = {"merge_error": repr(exc)}
    return {"attach": attach_ok, "merge": merged_ok, "merged_metrics": merged_metrics}


def _method_model(
    method: dict[str, Any],
    cfg: dict[str, Any],
    target_trainable: int | None,
) -> tuple[Any, Any, dict[str, Any]]:
    public_cfg = copy.deepcopy(cfg["model"])
    num_labels = int(cfg["data"]["num_labels"])
    tokenizer_cfg = method.get("model", public_cfg)
    tokenizer = load_tokenizer(tokenizer_cfg)
    model_cfg = copy.deepcopy(method.get("model", public_cfg))
    model = load_model(model_cfg, num_labels=num_labels)
    kind = method["kind"]
    metadata: dict[str, Any] = {}
    if kind in {"posterior_peft", "oracle_peft", "wrong_peft", "distill_to_peft"}:
        model = apply_peft(
            model, method["structure"], model_cfg.get("task_type", "sequence_classification")
        )
    elif kind == "parameter_matched_student":
        if target_trainable is None:
            raise ValueError("Parameter-matched student requires target trainable parameter count")
        metadata["parameter_match"] = freeze_to_parameter_budget(
            model,
            target_trainable,
            head_patterns=method.get("head_patterns"),
        )
    elif kind in {"plain_student", "compute_matched_student"}:
        for parameter in model.parameters():
            parameter.requires_grad = True
    else:
        raise ValueError(f"Unsupported redeploy method kind: {kind}")
    return model, tokenizer, metadata


def _train_neighbor_adapter(
    cfg: dict[str, Any],
    structure: dict[str, Any],
    output_dir: Path,
    seed: int,
) -> tuple[str, dict[str, Any]]:
    """Train a same-label neighboring-domain adapter without victim queries."""
    task_b = cfg["composition"]["task_b"]
    model_cfg = copy.deepcopy(cfg["model"])
    tokenizer = load_tokenizer(model_cfg)
    dataset_dict = load_task_dataset(task_b)
    train_split = select_split(
        dataset_dict,
        task_b.get("train_split", "train"),
        task_b.get("max_train_samples"),
        seed=seed,
    )
    rows = dataset_rows(train_split, task_b, None, seed=seed)
    # Use hard labels as one-hot teacher logits so the same audited training loop
    # can be reused without querying the victim.
    teacher_rows: list[dict[str, Any]] = []
    num_labels = int(task_b["num_labels"])
    for row in rows:
        logits = np.full(num_labels, -6.0)
        logits[row["label"]] = 6.0
        teacher_rows.append({"query_id": row["query_id"], "victim_logits": logits.tolist()})
    records = _prepare_distillation_records(tokenizer, teacher_rows, train_split, task_b)
    model = load_model(model_cfg, num_labels=num_labels)
    model = apply_peft(model, structure, model_cfg.get("task_type", "sequence_classification"))
    accounting = train_distilled(
        model,
        tokenizer,
        records,
        output_dir,
        cfg["composition"].get("training", cfg["training"]),
        seed=seed,
    )
    return str(output_dir), accounting.__dict__


def _load_adapter_state(checkpoint: str | Path) -> tuple[dict[str, Any], dict[str, Any]]:
    """Load an adapter config and tensor state without executing remote code."""
    deps = require_hf()
    torch = deps["torch"]
    checkpoint = Path(checkpoint)
    config_path = checkpoint / "adapter_config.json"
    if not config_path.exists():
        raise FileNotFoundError(f"Missing PEFT adapter_config.json: {config_path}")
    config = json.loads(config_path.read_text())
    safe_path = checkpoint / "adapter_model.safetensors"
    bin_path = checkpoint / "adapter_model.bin"
    if safe_path.exists():
        from safetensors.torch import load_file

        state = dict(load_file(str(safe_path), device="cpu"))
    elif bin_path.exists():
        try:
            state = dict(torch.load(bin_path, map_location="cpu", weights_only=True))
        except TypeError:  # older PyTorch compatibility
            state = dict(torch.load(bin_path, map_location="cpu"))
    else:
        raise FileNotFoundError(f"Missing adapter weights under {checkpoint}")
    return config, state


def _head_keys(config: dict[str, Any], state: dict[str, Any]) -> list[str]:
    modules = list(dict.fromkeys(config.get("modules_to_save") or []))
    modules.extend(name for name in ["classifier", "score"] if name not in modules)
    output: list[str] = []
    for key in state:
        if any(
            key.endswith(f".{module}.weight")
            or key.endswith(f".{module}.bias")
            or f".{module}.modules_to_save." in key
            for module in modules
        ):
            output.append(key)
    return output


def _make_body_only_adapter(checkpoint: str | Path, output_dir: str | Path) -> dict[str, Any]:
    """Create a temporary adapter containing only composable PEFT body deltas.

    PEFT sequence-classification adapters automatically save their task head via
    ``modules_to_save``. PEFT correctly refuses to add weighted adapters when
    those task heads overlap. For a fair two-task composition experiment we
    therefore combine only the LoRA body deltas and restore each task-specific
    head immediately before evaluating that task. No head parameters are
    averaged, silently dropped, or borrowed from the other task.
    """
    config, state = _load_adapter_state(checkpoint)
    head_keys = _head_keys(config, state)
    head_state = {key: state[key].detach().cpu().clone() for key in head_keys}
    body_state = {key: value for key, value in state.items() if key not in head_state}
    if not body_state:
        raise ValueError(f"Adapter {checkpoint} has no composable body parameters")
    output_dir = ensure_dir(output_dir)
    body_config = copy.deepcopy(config)
    body_config["modules_to_save"] = None
    # TaskType.SEQ_CLS automatically re-injects classifier/score as
    # modules_to_save. Clearing task_type is safe for inference and prevents
    # that implicit wrapper in this body-only copy.
    body_config["task_type"] = None
    (output_dir / "adapter_config.json").write_text(
        json.dumps(body_config, indent=2, sort_keys=True) + "\n"
    )
    from safetensors.torch import save_file

    save_file(body_state, str(output_dir / "adapter_model.safetensors"))
    return {
        "head_state": head_state,
        "head_keys": sorted(head_state),
        "body_keys": sorted(body_state),
    }


def _restore_task_head(model: Any, head_state: dict[str, Any]) -> list[str]:
    """Restore one adapter's task head into a body-composed PEFT model."""
    model_state = model.state_dict()
    compatible: dict[str, Any] = {}
    missing: list[str] = []
    for source_key, value in head_state.items():
        candidates = [source_key]
        # Some PEFT versions save modules_to_save wrappers explicitly. Map
        # those keys back to the ordinary base-model task head.
        if ".modules_to_save." in source_key:
            prefix, suffix = source_key.split(".modules_to_save.", 1)
            suffix = suffix.split(".", 1)[1] if "." in suffix else suffix
            candidates.append(f"{prefix}.{suffix}")
        target = next(
            (
                candidate
                for candidate in candidates
                if candidate in model_state and model_state[candidate].shape == value.shape
            ),
            None,
        )
        if target is None:
            # Last-resort suffix match remains deterministic and shape checked.
            suffix = ".".join(source_key.split(".")[-2:])
            matches = [
                key
                for key, tensor in model_state.items()
                if key.endswith(suffix) and tensor.shape == value.shape
            ]
            target = matches[0] if len(matches) == 1 else None
        if target is None:
            missing.append(source_key)
        else:
            compatible[target] = value.to(dtype=model_state[target].dtype)
    if missing:
        raise KeyError(f"Could not map task-head tensors into composed model: {missing}")
    model.load_state_dict(compatible, strict=False)
    return sorted(compatible)


def _composition_evaluate(
    cfg: dict[str, Any],
    task_a_checkpoint: str | Path,
    task_b_checkpoint: str | Path,
    task_a_eval: Any,
    task_b_eval: Any,
    victim_model: Any,
) -> dict[str, Any]:
    num_labels_a = int(cfg["data"]["num_labels"])
    num_labels_b = int(cfg["composition"]["task_b"]["num_labels"])
    if num_labels_a != num_labels_b:
        return {"supported": False, "reason": "Task A/B label dimensions differ"}
    deps = require_hf()
    PeftModel = deps["PeftModel"]
    tokenizer = load_tokenizer(cfg["model"])

    # First evaluate each complete artifact exactly as deployed, including its
    # own task-specific classifier head.
    base = load_model(cfg["model"], num_labels=num_labels_a)
    individual = PeftModel.from_pretrained(base, str(task_a_checkpoint), adapter_name="task_a")
    try:
        individual.load_adapter(str(task_b_checkpoint), adapter_name="task_b")
    except Exception as exc:
        return {"supported": False, "reason": f"load_adapter failed: {exc!r}"}
    results: dict[str, Any] = {
        "supported": True,
        "composition_protocol": "weighted_body_delta_with_task_specific_heads",
    }
    individual.set_adapter("task_a")
    results["task_a_before"] = evaluate_sequence_model(
        individual, tokenizer, task_a_eval, cfg["data"], teacher_model=victim_model
    )
    individual.set_adapter("task_b")
    results["task_b_before"] = evaluate_sequence_model(
        individual, tokenizer, task_b_eval, cfg["composition"]["task_b"]
    )
    del individual

    # Then combine body deltas only. This avoids the invalid operation of
    # averaging two independently trained classifier heads. Each task is
    # evaluated with its own restored head while sharing the same composed body.
    with tempfile.TemporaryDirectory(prefix="vectorhijack-compose-") as temp_root:
        temp_root_path = Path(temp_root)
        try:
            task_a_meta = _make_body_only_adapter(task_a_checkpoint, temp_root_path / "task_a")
            task_b_meta = _make_body_only_adapter(task_b_checkpoint, temp_root_path / "task_b")
            composed_base = load_model(cfg["model"], num_labels=num_labels_a)
            model = PeftModel.from_pretrained(
                composed_base, str(temp_root_path / "task_a"), adapter_name="task_a"
            )
            model.load_adapter(str(temp_root_path / "task_b"), adapter_name="task_b")
            if not hasattr(model, "add_weighted_adapter"):
                raise AttributeError("PEFT model lacks add_weighted_adapter")
            model.add_weighted_adapter(
                adapters=["task_a", "task_b"],
                weights=list(cfg["composition"].get("weights", [0.5, 0.5])),
                adapter_name="composed",
                combination_type=cfg["composition"].get("combination_type", "linear"),
            )
            model.set_adapter("composed")
            restored_a = _restore_task_head(model, task_a_meta["head_state"])
            results["task_a_composed"] = evaluate_sequence_model(
                model, tokenizer, task_a_eval, cfg["data"], teacher_model=victim_model
            )
            restored_b = _restore_task_head(model, task_b_meta["head_state"])
            results["task_b_composed"] = evaluate_sequence_model(
                model, tokenizer, task_b_eval, cfg["composition"]["task_b"]
            )
            results["restored_task_a_head_keys"] = restored_a
            results["restored_task_b_head_keys"] = restored_b
            results["body_parameter_keys"] = {
                "task_a": task_a_meta["body_keys"],
                "task_b": task_b_meta["body_keys"],
            }
        except Exception as exc:
            results["supported"] = False
            results["composition_error"] = repr(exc)
    return results


def _resolve_posterior_structure(
    method: dict[str, Any],
    cfg: dict[str, Any],
    victim_entry: dict[str, Any],
) -> dict[str, Any]:
    """Resolve a locked Stage-I structure without consulting Stage-II labels.

    The default configuration embeds the selected top posterior candidate. For
    large sweeps, `structure_from` may point to a JSON report with a
    `by_service` map. This function deliberately refuses to infer an oracle
    structure from the victim checkpoint metadata.
    """
    inline = method.get("structure")
    source = method.get("structure_from")
    if source is None:
        if not isinstance(inline, dict):
            raise ValueError("posterior_peft requires an inline `structure` or `structure_from`")
        return copy.deepcopy(inline)
    report = read_json(source)
    service_id = str(victim_entry["service_id"])
    by_service = report.get("by_service", report.get("predictions", {}))
    candidate: Any = None
    if isinstance(by_service, dict):
        candidate = by_service.get(service_id)
    elif isinstance(by_service, list):
        candidate = next(
            (row for row in by_service if str(row.get("service_id")) == service_id),
            None,
        )
    if isinstance(candidate, dict):
        candidate = candidate.get("structure", candidate.get("top1", candidate))
    if isinstance(candidate, str):
        structure_catalog = cfg.get("structure_catalog", {})
        if candidate not in structure_catalog:
            raise KeyError(f"Posterior label {candidate!r} is absent from `structure_catalog`")
        candidate = structure_catalog[candidate]
    if not isinstance(candidate, dict):
        raise KeyError(f"No locked posterior structure for service {service_id!r}")
    return copy.deepcopy(candidate)


def _teacher_rows_from_model(
    model: Any,
    tokenizer: Any,
    dataset: Any,
    data_cfg: dict[str, Any],
    device: str | None = None,
    batch_size: int = 32,
) -> list[dict[str, Any]]:
    """Materialize logits from a local student for a conversion baseline."""
    deps = require_hf()
    torch = deps["torch"]
    resolved_device = infer_device(device)
    model.to(resolved_device).eval()
    rows = dataset_rows(dataset, data_cfg, None, seed=0)
    output: list[dict[str, Any]] = []
    for start in range(0, len(rows), batch_size):
        batch_rows = rows[start : start + batch_size]
        pairs = [classification_text(row, data_cfg["text_fields"]) for row in batch_rows]
        first = [pair[0] for pair in pairs]
        second_values = [pair[1] for pair in pairs]
        second = None if all(value is None for value in second_values) else second_values
        encoded = tokenizer(
            first,
            second,
            padding=True,
            truncation=True,
            max_length=int(data_cfg.get("max_length", 256)),
            return_tensors="pt",
        )
        encoded = {key: value.to(resolved_device) for key, value in encoded.items()}
        with torch.inference_mode():
            logits = model(**encoded).logits.detach().float().cpu().numpy()
        for row, row_logits in zip(batch_rows, logits, strict=True):
            output.append(
                {
                    "query_id": int(row["query_id"]),
                    "victim_logits": row_logits.tolist(),
                }
            )
    return output


def _student_task_b_adaptation(
    cfg: dict[str, Any],
    task_a_checkpoint: str | Path,
    model_cfg: dict[str, Any],
    output_dir: Path,
    task_a_eval: Any,
    task_b_train: Any,
    task_b_eval: Any,
    victim_model: Any,
    seed: int,
) -> tuple[dict[str, Any], TrainAccounting]:
    """Give an unstructured student the same Task-B adaptation opportunity.

    PEFT artifacts are evaluated by non-destructive adapter composition. A
    standalone student cannot load a second adapter, so the fairest operational
    comparator is an additional Task-B fine-tuning stage with its FLOPs and
    artifact cost included. We report both Task-A retention and Task-B utility.
    """
    source_cfg = copy.deepcopy(model_cfg)
    source_cfg["model_id"] = str(task_a_checkpoint)
    source_cfg["revision"] = None
    model = load_model(source_cfg, num_labels=int(cfg["data"]["num_labels"]))
    tokenizer = load_tokenizer(source_cfg)
    task_a_before = evaluate_sequence_model(
        model,
        tokenizer,
        task_a_eval,
        cfg["data"],
        teacher_model=victim_model,
    )
    task_b_before = evaluate_sequence_model(
        model,
        tokenizer,
        task_b_eval,
        cfg["composition"]["task_b"],
    )
    task_b_rows = dataset_rows(
        task_b_train,
        cfg["composition"]["task_b"],
        None,
        seed=seed,
    )
    num_labels = int(cfg["composition"]["task_b"]["num_labels"])
    teacher_rows: list[dict[str, Any]] = []
    for row in task_b_rows:
        logits = np.full(num_labels, -6.0, dtype=float)
        logits[int(row["label"])] = 6.0
        teacher_rows.append({"query_id": int(row["query_id"]), "victim_logits": logits.tolist()})
    records = _prepare_distillation_records(
        tokenizer,
        teacher_rows,
        task_b_train,
        cfg["composition"]["task_b"],
    )
    accounting = train_distilled(
        model,
        tokenizer,
        records,
        output_dir,
        cfg["composition"].get(
            "student_training", cfg["composition"].get("training", cfg["training"])
        ),
        seed=seed,
    )
    task_a_after = evaluate_sequence_model(
        model,
        tokenizer,
        task_a_eval,
        cfg["data"],
        teacher_model=victim_model,
    )
    task_b_after = evaluate_sequence_model(
        model,
        tokenizer,
        task_b_eval,
        cfg["composition"]["task_b"],
    )
    return (
        {
            "supported": True,
            "task_a_before": task_a_before,
            "task_b_before": task_b_before,
            "task_a_composed": task_a_after,
            "task_b_composed": task_b_after,
            "operational_mode": "additional_full_student_finetuning",
        },
        accounting,
    )


def run_redeploy(cfg: dict[str, Any]) -> dict[str, Any]:
    root = Path(get(cfg, "experiment.output_root", "artifact"))
    name = cfg["experiment"]["name"]
    grid = read_json(cfg["inputs"]["victim_grid_json"])
    query_rows_all = list(read_jsonl(cfg["inputs"]["victim_query_jsonl"]))
    target_structure = cfg["inputs"].get("victim_structure")
    victims = [
        entry
        for entry in grid
        if target_structure is None or entry["structure"] == target_structure
    ]
    allowed_seeds = set(int(v) for v in cfg["seeds"]["victim"])
    victims = [entry for entry in victims if int(entry["victim_seed"]) in allowed_seeds]
    if not victims:
        raise ValueError("No victim checkpoint matched redeploy configuration")

    dataset_dict = load_task_dataset(cfg["data"])
    query_dataset = select_split(
        dataset_dict,
        cfg["data"].get("probe_split", cfg["data"].get("eval_split", "validation")),
        cfg["data"].get("max_probe_samples"),
        seed=0,
    )
    eval_dataset = select_split(
        dataset_dict,
        cfg["data"].get("eval_split", "validation"),
        cfg["data"].get("max_eval_samples"),
        seed=1,
    )
    task_b_cfg = cfg["composition"]["task_b"]
    task_b_dict = load_task_dataset(task_b_cfg)
    task_b_train = select_split(
        task_b_dict,
        task_b_cfg.get("train_split", "train"),
        task_b_cfg.get("max_train_samples"),
        seed=2,
    )
    task_b_eval = select_split(
        task_b_dict,
        task_b_cfg.get("eval_split", "validation"),
        task_b_cfg.get("max_eval_samples"),
        seed=3,
    )

    results: list[dict[str, Any]] = []
    paired_rows: list[dict[str, Any]] = []
    for victim_entry in victims:
        seed = int(victim_entry["victim_seed"])
        victim_model = _load_checkpoint_model(
            cfg["model"], victim_entry["checkpoint"], int(cfg["data"]["num_labels"])
        )
        victim_tokenizer = load_tokenizer(cfg["model"])
        victim_queries = [
            row
            for row in query_rows_all
            if row["service_id"] == victim_entry["service_id"]
            and int(row["probe_seed"]) == int(cfg["seeds"].get("query_probe_seed", 7))
        ]
        victim_queries = victim_queries[: int(cfg["budget"]["victim_queries"])]
        if not victim_queries:
            raise ValueError(f"No victim query rows for {victim_entry['service_id']}")

        neighbor_dir, neighbor_accounting_dict = _train_neighbor_adapter(
            cfg,
            cfg["composition"]["task_b_structure"],
            root / "models" / name / f"seed-{seed}" / "task_b_adapter",
            seed + 10_000,
        )
        neighbor_flops = int(neighbor_accounting_dict["estimated_flops"])

        # Original victim upper bound.
        victim_eval = evaluate_sequence_model(
            victim_model,
            victim_tokenizer,
            eval_dataset,
            cfg["data"],
            teacher_model=victim_model,
        )
        victim_merge = _merge_evaluate(
            victim_entry["checkpoint"],
            cfg["model"],
            victim_tokenizer,
            eval_dataset,
            cfg["data"],
            victim_model,
        )
        victim_comp = _composition_evaluate(
            cfg,
            victim_entry["checkpoint"],
            neighbor_dir,
            eval_dataset,
            task_b_eval,
            victim_model,
        )
        results.append(
            {
                "seed": seed,
                "method": "original_victim_adapter",
                "kind": "victim_upper_bound",
                "victim_queries": 0,
                "agreement": victim_eval["agreement"],
                "task_accuracy": victim_eval["task_accuracy"],
                "kl": victim_eval["kl_to_victim"],
                "artifact_bytes": file_size_bytes(victim_entry["checkpoint"]),
                "deployed_parameters": total_parameter_count(victim_model),
                "trainable_parameters": int(victim_entry.get("trainable_parameters", 0)),
                "train_flops": float("nan"),
                "composition_aux_flops": neighbor_flops,
                "total_operational_flops": float("nan"),
                "train_wall_seconds": float("nan"),
                "serve_peak_memory_bytes": peak_memory_bytes(infer_device()),
                "attach": victim_merge["attach"],
                "merge": victim_merge["merge"],
                "merge_accuracy": victim_merge.get("merged_metrics", {}).get(
                    "task_accuracy", float("nan")
                ),
                "composition_task_a": victim_comp.get("task_a_composed", {}).get(
                    "task_accuracy", float("nan")
                ),
                "composition_task_b": victim_comp.get("task_b_composed", {}).get(
                    "task_accuracy", float("nan")
                ),
                "transfer": victim_comp.get("task_b_before", {}).get("task_accuracy", float("nan")),
                "conversion_cost_included": True,
            }
        )

        posterior_method = next(
            method for method in cfg["methods"] if method["kind"] == "posterior_peft"
        )
        posterior_resolved = copy.deepcopy(posterior_method)
        posterior_resolved["structure"] = _resolve_posterior_structure(
            posterior_method, cfg, victim_entry
        )
        posterior_model, posterior_tokenizer, posterior_meta = _method_model(
            posterior_resolved, cfg, None
        )
        posterior_records = _prepare_distillation_records(
            posterior_tokenizer, victim_queries, query_dataset, cfg["data"]
        )
        posterior_dir = root / "models" / name / f"seed-{seed}" / posterior_method["name"]
        posterior_accounting = train_distilled(
            posterior_model,
            posterior_tokenizer,
            posterior_records,
            posterior_dir,
            cfg["training"],
            seed,
        )
        posterior_trainable = trainable_parameter_count(posterior_model)
        posterior_eval = evaluate_sequence_model(
            posterior_model,
            posterior_tokenizer,
            eval_dataset,
            cfg["data"],
            teacher_model=victim_model,
        )
        posterior_merge = _merge_evaluate(
            posterior_dir,
            cfg["model"],
            posterior_tokenizer,
            eval_dataset,
            cfg["data"],
            victim_model,
        )
        posterior_comp = _composition_evaluate(
            cfg,
            posterior_dir,
            neighbor_dir,
            eval_dataset,
            task_b_eval,
            victim_model,
        )
        posterior_row = {
            "seed": seed,
            "method": posterior_method["name"],
            "kind": posterior_method["kind"],
            "victim_queries": len(victim_queries),
            "agreement": posterior_eval["agreement"],
            "task_accuracy": posterior_eval["task_accuracy"],
            "kl": posterior_eval["kl_to_victim"],
            "artifact_bytes": file_size_bytes(posterior_dir),
            "deployed_parameters": total_parameter_count(posterior_model),
            "trainable_parameters": posterior_trainable,
            "train_flops": posterior_accounting.estimated_flops,
            "composition_aux_flops": neighbor_flops,
            "total_operational_flops": posterior_accounting.estimated_flops + neighbor_flops,
            "train_wall_seconds": posterior_accounting.wall_seconds,
            "serve_peak_memory_bytes": posterior_accounting.peak_memory_bytes,
            "attach": posterior_merge["attach"],
            "merge": posterior_merge["merge"],
            "merge_accuracy": posterior_merge.get("merged_metrics", {}).get(
                "task_accuracy", float("nan")
            ),
            "composition_task_a": posterior_comp.get("task_a_composed", {}).get(
                "task_accuracy", float("nan")
            ),
            "composition_task_b": posterior_comp.get("task_b_composed", {}).get(
                "task_accuracy", float("nan")
            ),
            "transfer": posterior_comp.get("task_b_before", {}).get("task_accuracy", float("nan")),
            "conversion_cost_included": True,
            "posterior_structure_source": posterior_method.get(
                "structure_from", "inline_locked_prediction"
            ),
            **posterior_meta,
        }
        results.append(posterior_row)
        target_flops = posterior_accounting.estimated_flops
        trained_checkpoints: dict[str, dict[str, Any]] = {
            "posterior_peft": {
                "checkpoint": str(posterior_dir),
                "flops": posterior_accounting.estimated_flops,
                "model_cfg": cfg["model"],
            }
        }

        for method in [m for m in cfg["methods"] if m["kind"] != "posterior_peft"]:
            method_dir = root / "models" / name / f"seed-{seed}" / method["name"]
            kind = method["kind"]
            conversion_source_flops = 0
            if kind == "distill_to_peft":
                source_name = method.get("source_method", "plain_distilled_student")
                if source_name not in trained_checkpoints:
                    raise ValueError(
                        f"distill_to_peft source '{source_name}' must appear earlier in methods"
                    )
                source = trained_checkpoints[source_name]
                source_cfg = copy.deepcopy(source["model_cfg"])
                source_cfg["model_id"] = source["checkpoint"]
                source_cfg["revision"] = None
                source_model = load_model(source_cfg, num_labels=int(cfg["data"]["num_labels"]))
                source_tokenizer = load_tokenizer(source_cfg)
                student_teacher_rows = _teacher_rows_from_model(
                    source_model,
                    source_tokenizer,
                    query_dataset,
                    cfg["data"],
                    device=get(cfg, "training.device"),
                )
                model, tokenizer, method_meta = _method_model(method, cfg, posterior_trainable)
                method_records = _prepare_distillation_records(
                    tokenizer, student_teacher_rows, query_dataset, cfg["data"]
                )
                conversion_source_flops = int(source["flops"])
                del source_model
            else:
                model, tokenizer, method_meta = _method_model(method, cfg, posterior_trainable)
                method_records = _prepare_distillation_records(
                    tokenizer, victim_queries, query_dataset, cfg["data"]
                )
            max_flops = target_flops if kind == "compute_matched_student" else None
            accounting = train_distilled(
                model,
                tokenizer,
                method_records,
                method_dir,
                cfg["training"],
                seed,
                max_flops=max_flops,
            )
            recovery_flops = accounting.estimated_flops + conversion_source_flops
            evaluation = evaluate_sequence_model(
                model, tokenizer, eval_dataset, cfg["data"], teacher_model=victim_model
            )
            peft_like = kind in {"oracle_peft", "wrong_peft", "distill_to_peft"}
            merge = (
                _merge_evaluate(
                    method_dir,
                    cfg["model"],
                    tokenizer,
                    eval_dataset,
                    cfg["data"],
                    victim_model,
                )
                if peft_like
                else {"attach": False, "merge": False, "merged_metrics": {}}
            )
            if peft_like:
                composition = _composition_evaluate(
                    cfg, method_dir, neighbor_dir, eval_dataset, task_b_eval, victim_model
                )
                composition_flops = neighbor_flops
            else:
                method_model_cfg = copy.deepcopy(method.get("model", cfg["model"]))
                composition, student_b_accounting = _student_task_b_adaptation(
                    cfg,
                    method_dir,
                    method_model_cfg,
                    root / "models" / name / f"seed-{seed}" / f"{method['name']}_task_b",
                    eval_dataset,
                    task_b_train,
                    task_b_eval,
                    victim_model,
                    seed + 20_000,
                )
                composition_flops = student_b_accounting.estimated_flops
            row = {
                "seed": seed,
                "method": method["name"],
                "kind": kind,
                "victim_queries": len(victim_queries),
                "agreement": evaluation["agreement"],
                "task_accuracy": evaluation["task_accuracy"],
                "kl": evaluation["kl_to_victim"],
                "artifact_bytes": file_size_bytes(method_dir),
                "deployed_parameters": total_parameter_count(model),
                "trainable_parameters": trainable_parameter_count(model),
                "train_flops": recovery_flops,
                "composition_aux_flops": composition_flops,
                "total_operational_flops": recovery_flops + composition_flops,
                "train_wall_seconds": accounting.wall_seconds,
                "serve_peak_memory_bytes": accounting.peak_memory_bytes,
                "attach": merge["attach"],
                "merge": merge["merge"],
                "merge_accuracy": merge.get("merged_metrics", {}).get(
                    "task_accuracy", float("nan")
                ),
                "composition_task_a": composition.get("task_a_composed", {}).get(
                    "task_accuracy", float("nan")
                ),
                "composition_task_b": composition.get("task_b_composed", {}).get(
                    "task_accuracy", float("nan")
                ),
                "transfer": composition.get("task_b_before", {}).get("task_accuracy", float("nan")),
                "conversion_cost_included": True,
                "conversion_source_flops": conversion_source_flops,
                **method_meta,
            }
            results.append(row)
            trained_checkpoints[method["name"]] = {
                "checkpoint": str(method_dir),
                "flops": recovery_flops,
                "model_cfg": method.get("model", cfg["model"]),
            }
            del model
            gc.collect()

        seed_rows = [row for row in results if row["seed"] == seed]
        posterior = next(row for row in seed_rows if row["kind"] == "posterior_peft")
        for baseline_kind in [
            "parameter_matched_student",
            "compute_matched_student",
            "distill_to_peft",
        ]:
            candidates = [row for row in seed_rows if row["kind"] == baseline_kind]
            if not candidates:
                continue
            baseline = max(candidates, key=lambda row: row["agreement"])
            paired_rows.append(
                {
                    "seed": seed,
                    "comparison": f"posterior_peft-minus-{baseline_kind}",
                    "delta_agreement": posterior["agreement"] - baseline["agreement"],
                    "delta_task_accuracy": posterior["task_accuracy"] - baseline["task_accuracy"],
                    "delta_merge_accuracy": posterior["merge_accuracy"]
                    - baseline["merge_accuracy"],
                    "delta_composition_task_a": posterior["composition_task_a"]
                    - baseline["composition_task_a"],
                    "delta_composition_task_b": posterior["composition_task_b"]
                    - baseline["composition_task_b"],
                    "delta_train_flops": posterior["train_flops"] - baseline["train_flops"],
                    "delta_total_operational_flops": posterior["total_operational_flops"]
                    - baseline["total_operational_flops"],
                    "operational_flops_ratio": posterior["total_operational_flops"]
                    / max(baseline["total_operational_flops"], 1),
                    "artifact_bytes_ratio": posterior["artifact_bytes"]
                    / max(baseline["artifact_bytes"], 1),
                    "serve_memory_ratio": posterior["serve_peak_memory_bytes"]
                    / max(baseline["serve_peak_memory_bytes"], 1),
                    "attach_advantage": int(bool(posterior["attach"]))
                    - int(bool(baseline["attach"])),
                    "merge_advantage": int(bool(posterior["merge"])) - int(bool(baseline["merge"])),
                }
            )
        del posterior_model, victim_model
        gc.collect()

    result_frame = pd.DataFrame(results)
    delta_frame = pd.DataFrame(paired_rows)
    delta_summary: list[dict[str, Any]] = []
    if not delta_frame.empty:
        for comparison, group in delta_frame.groupby("comparison"):
            row: dict[str, Any] = {
                "comparison": comparison,
                "seeds": int(group["seed"].nunique()),
            }
            for metric in [
                "delta_agreement",
                "delta_task_accuracy",
                "delta_merge_accuracy",
                "delta_composition_task_a",
                "delta_composition_task_b",
                "delta_train_flops",
                "delta_total_operational_flops",
                "operational_flops_ratio",
                "artifact_bytes_ratio",
                "serve_memory_ratio",
                "attach_advantage",
                "merge_advantage",
            ]:
                values = group[metric].dropna().to_numpy(dtype=float)
                if len(values) == 0:
                    row[f"{metric}_mean"] = float("nan")
                    row[f"{metric}_ci_lower"] = float("nan")
                    row[f"{metric}_ci_upper"] = float("nan")
                    continue
                sample = group[["seed", metric]].rename(columns={metric: "value"})
                boot = cluster_bootstrap(
                    sample,
                    cluster_col="seed",
                    statistic=lambda frame: float(frame["value"].mean()),
                    n_bootstrap=int(get(cfg, "bootstrap.n", 2000)),
                    seed=int(get(cfg, "experiment.seed", 7)),
                )
                row[f"{metric}_mean"] = boot["estimate"]
                row[f"{metric}_ci_lower"] = boot["ci_lower"]
                row[f"{metric}_ci_upper"] = boot["ci_upper"]
            delta_summary.append(row)

    write_csv(
        root / "summaries" / f"{name}_redeploy_per_seed.csv",
        result_frame.to_dict(orient="records"),
    )
    write_csv(
        root / "summaries" / f"{name}_redeploy_paired.csv",
        delta_frame.to_dict(orient="records"),
    )
    write_csv(root / "summaries" / f"{name}_redeploy_delta_summary.csv", delta_summary)
    report = {
        "methods": results,
        "paired_deltas": delta_summary,
        "victim_query_budget_equal": True,
        "per_seed_reporting": True,
        "victim_seed_bootstrap": True,
        "training_flops_are_estimated_and_labeled": True,
        "conversion_cost_included": True,
        "task_b_training_or_adapter_acquisition_cost_included": True,
        "plain_students_receive_equal_task_b_training_budget": True,
        "distill_to_peft_is_two_stage": True,
        "primary_superiority_rule": (
            "Posterior PEFT is operationally superior only when the paired 95% "
            "confidence intervals for Task-A and Task-B absolute utility both "
            "exclude zero in its favor, with query, parameter/compute, conversion, "
            "and Task-B acquisition costs reported. Attach/merge capability alone "
            "is not counted as superiority."
        ),
        "flops_measurement": "estimated by model.floating_point_ops or 6*parameters*tokens fallback",
    }
    write_json(root / "reports" / f"{name}_redeploy.json", report)
    write_json(
        root / "manifests" / f"{name}_redeploy.json",
        {
            "run_id": name,
            "experiment": cfg["experiment"],
            "config": cfg,
            "victim_grid_sha256": sha256_file(cfg["inputs"]["victim_grid_json"]),
            "victim_query_sha256": sha256_file(cfg["inputs"]["victim_query_jsonl"]),
            "environment": environment_snapshot(),
            "git_commit": git_commit(),
            "report": report,
        },
    )
    return report
