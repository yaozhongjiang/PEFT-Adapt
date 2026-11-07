from __future__ import annotations

import contextlib
import copy
import hashlib
import importlib.metadata
import inspect
import json
import math
import os
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import numpy as np

from .config import get
from .io import ensure_dir, file_size_bytes, write_json
from .reproducibility import environment_snapshot, set_global_seed


class MissingDependencyError(RuntimeError):
    pass


def require_hf() -> dict[str, Any]:
    try:
        import torch
        from datasets import Dataset, DatasetDict, load_dataset
        from peft import (
            IA3Config,
            LoraConfig,
            PeftModel,
            PrefixTuningConfig,
            TaskType,
            get_peft_model,
        )
        from transformers import (
            AutoModelForCausalLM,
            AutoModelForMaskedLM,
            AutoModelForSequenceClassification,
            AutoTokenizer,
            BitsAndBytesConfig,
            DataCollatorWithPadding,
            Trainer,
            TrainingArguments,
            default_data_collator,
        )
    except ImportError as exc:
        raise MissingDependencyError(
            "Full experiments require torch, transformers, datasets, peft, and accelerate. "
            "Install with `python -m pip install -e .`."
        ) from exc
    return locals()


def model_revision_fingerprint(model_id: str, revision: str | None) -> str:
    return hashlib.sha256(f"{model_id}@{revision or 'main'}".encode()).hexdigest()


def infer_device(device: str | None = None) -> str:
    deps = require_hf()
    torch = deps["torch"]
    if device:
        return device
    return "cuda" if torch.cuda.is_available() else "cpu"


def load_tokenizer(
    model_cfg: dict[str, Any], tokenizer_override: dict[str, Any] | None = None
) -> Any:
    deps = require_hf()
    AutoTokenizer = deps["AutoTokenizer"]
    cfg = copy.deepcopy(model_cfg)
    if tokenizer_override:
        cfg.update(tokenizer_override)
    tokenizer_id = cfg.get("tokenizer_id", cfg["model_id"])
    tokenizer = AutoTokenizer.from_pretrained(
        tokenizer_id,
        revision=cfg.get("tokenizer_revision", cfg.get("revision")),
        use_fast=cfg.get("use_fast", True),
        trust_remote_code=cfg.get("trust_remote_code", False),
        local_files_only=cfg.get("local_files_only", False),
    )
    tokenizer.padding_side = cfg.get("padding_side", tokenizer.padding_side)
    tokenizer.truncation_side = cfg.get("truncation_side", tokenizer.truncation_side)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token or tokenizer.unk_token
    if cfg.get("chat_template"):
        tokenizer.chat_template = cfg["chat_template"]
    return tokenizer


def _dtype(torch: Any, value: str | None) -> Any:
    if value is None or value == "auto":
        return "auto"
    mapping = {
        "float32": torch.float32,
        "fp32": torch.float32,
        "float16": torch.float16,
        "fp16": torch.float16,
        "bfloat16": torch.bfloat16,
        "bf16": torch.bfloat16,
    }
    if value not in mapping:
        raise ValueError(f"Unsupported dtype: {value}")
    return mapping[value]


def load_model(
    model_cfg: dict[str, Any],
    num_labels: int | None = None,
    adapter_path: str | Path | None = None,
    adapter_name: str = "default",
) -> Any:
    deps = require_hf()
    torch = deps["torch"]
    task = model_cfg.get("task_type", "sequence_classification")
    try:
        transformers_major = int(importlib.metadata.version("transformers").split(".", 1)[0])
    except (importlib.metadata.PackageNotFoundError, ValueError):
        transformers_major = 4
    dtype_key = "dtype" if transformers_major >= 5 else "torch_dtype"
    kwargs: dict[str, Any] = {
        "revision": model_cfg.get("revision"),
        "trust_remote_code": model_cfg.get("trust_remote_code", False),
        "local_files_only": model_cfg.get("local_files_only", False),
        dtype_key: _dtype(torch, model_cfg.get("dtype", "auto")),
    }
    kwargs = {k: v for k, v in kwargs.items() if v is not None}
    quant = model_cfg.get("quantization")
    if quant in {"int8", "int4"}:
        BitsAndBytesConfig = deps["BitsAndBytesConfig"]
        kwargs["quantization_config"] = BitsAndBytesConfig(
            load_in_8bit=quant == "int8",
            load_in_4bit=quant == "int4",
            bnb_4bit_compute_dtype=_dtype(torch, model_cfg.get("bnb_compute_dtype", "float16")),
        )
        kwargs["device_map"] = model_cfg.get("device_map", "auto")
    model_id = model_cfg["model_id"]
    if task == "sequence_classification":
        cls = deps["AutoModelForSequenceClassification"]
        model = cls.from_pretrained(model_id, num_labels=num_labels, **kwargs)
    elif task == "causal_lm":
        cls = deps["AutoModelForCausalLM"]
        model = cls.from_pretrained(model_id, **kwargs)
    elif task == "masked_lm":
        cls = deps["AutoModelForMaskedLM"]
        model = cls.from_pretrained(model_id, **kwargs)
    else:
        raise ValueError(f"Unsupported task_type: {task}")
    if adapter_path is not None:
        PeftModel = deps["PeftModel"]
        model = PeftModel.from_pretrained(model, str(adapter_path), adapter_name=adapter_name)
    return model


def resolve_task_type(task: str) -> Any:
    deps = require_hf()
    TaskType = deps["TaskType"]
    if task == "sequence_classification":
        return TaskType.SEQ_CLS
    if task == "causal_lm":
        return TaskType.CAUSAL_LM
    raise ValueError(f"PEFT task type unsupported: {task}")


def apply_peft(model: Any, spec: dict[str, Any], task_type: str) -> Any:
    deps = require_hf()
    family = spec["family"].lower()
    target_modules = spec.get("target_modules")
    modules_to_save = spec.get("modules_to_save")
    common = {"task_type": resolve_task_type(task_type), "inference_mode": False}
    if family in {"lora", "dora", "lora_head", "nonuniform_lora"}:
        LoraConfig = deps["LoraConfig"]
        config = LoraConfig(
            **common,
            r=int(spec.get("rank", 8)),
            lora_alpha=int(spec.get("alpha", max(16, int(spec.get("rank", 8)) * 2))),
            lora_dropout=float(spec.get("dropout", 0.0)),
            target_modules=target_modules,
            modules_to_save=modules_to_save,
            bias=spec.get("bias", "none"),
            use_dora=family == "dora" or bool(spec.get("use_dora", False)),
            rank_pattern=spec.get("rank_pattern", {})
            if family == "nonuniform_lora"
            else spec.get("rank_pattern", {}),
            alpha_pattern=spec.get("alpha_pattern", {}),
        )
        model = deps["get_peft_model"](model, config)
        if family == "lora_head":
            _unfreeze_named(model, spec.get("head_patterns", ["classifier", "score"]))
        return model
    if family == "ia3":
        IA3Config = deps["IA3Config"]
        config = IA3Config(
            **common,
            target_modules=target_modules,
            feedforward_modules=spec.get("feedforward_modules"),
            modules_to_save=modules_to_save,
        )
        return deps["get_peft_model"](model, config)
    if family == "prefix":
        PrefixTuningConfig = deps["PrefixTuningConfig"]
        config = PrefixTuningConfig(
            **common,
            num_virtual_tokens=int(spec.get("num_virtual_tokens", 20)),
            encoder_hidden_size=spec.get("encoder_hidden_size"),
        )
        return deps["get_peft_model"](model, config)
    if family == "head_only":
        _freeze_all(model)
        _unfreeze_named(model, spec.get("head_patterns", ["classifier", "score"]))
        return model
    if family == "bitfit":
        _freeze_all(model)
        for name, parameter in model.named_parameters():
            if name.endswith("bias") or any(
                pattern in name for pattern in spec.get("head_patterns", ["classifier", "score"])
            ):
                parameter.requires_grad = True
        return model
    if family == "top_layer_ft":
        _freeze_all(model)
        patterns = spec.get("target_patterns")
        if not patterns:
            raise ValueError("top_layer_ft requires non-empty target_patterns")
        _unfreeze_named(model, patterns)
        return model
    if family == "full":
        for parameter in model.parameters():
            parameter.requires_grad = True
        return model
    raise ValueError(f"Unsupported PEFT family: {family}")


def _freeze_all(model: Any) -> None:
    for parameter in model.parameters():
        parameter.requires_grad = False


def _unfreeze_named(model: Any, patterns: Iterable[str]) -> None:
    matched = 0
    for name, parameter in model.named_parameters():
        if any(pattern in name for pattern in patterns):
            parameter.requires_grad = True
            matched += parameter.numel()
    if matched == 0:
        raise ValueError(f"No parameters matched unfreeze patterns: {list(patterns)}")


def trainable_parameter_count(model: Any) -> int:
    return int(sum(p.numel() for p in model.parameters() if p.requires_grad))


def total_parameter_count(model: Any) -> int:
    return int(sum(p.numel() for p in model.parameters()))


def freeze_to_parameter_budget(
    model: Any, target_trainable: int, head_patterns: list[str] | None = None
) -> dict[str, Any]:
    """Unfreeze a deterministic suffix of modules closest to a parameter budget."""
    head_patterns = head_patterns or ["classifier", "score"]
    _freeze_all(model)
    named = list(model.named_parameters())
    selected: list[str] = []
    count = 0
    # Always include the task head, then add parameters from the end of the backbone.
    for name, param in named:
        if any(pattern in name for pattern in head_patterns):
            param.requires_grad = True
            selected.append(name)
            count += param.numel()
    for name, param in reversed(named):
        if param.requires_grad:
            continue
        if count >= target_trainable:
            break
        param.requires_grad = True
        selected.append(name)
        count += param.numel()
    return {
        "target_trainable": int(target_trainable),
        "actual_trainable": int(count),
        "selected_parameters": selected,
    }


def dataset_fingerprint(dataset: Any) -> str | None:
    return getattr(dataset, "_fingerprint", None)


def load_task_dataset(data_cfg: dict[str, Any]) -> Any:
    deps = require_hf()
    load_dataset = deps["load_dataset"]
    kwargs: dict[str, Any] = {}
    if data_cfg.get("data_files"):
        kwargs["data_files"] = data_cfg["data_files"]
    dataset = load_dataset(
        data_cfg["dataset_name"],
        data_cfg.get("dataset_config"),
        revision=data_cfg.get("revision"),
        cache_dir=data_cfg.get("cache_dir"),
        **kwargs,
    )
    return dataset


def prepare_classification_dataset(dataset: Any, tokenizer: Any, data_cfg: dict[str, Any]) -> Any:
    text_fields = data_cfg["text_fields"]
    label_field = data_cfg.get("label_field", "label")
    max_length = int(data_cfg.get("max_length", 256))

    def tokenize(batch: dict[str, Any]) -> dict[str, Any]:
        args = [batch[field] for field in text_fields]
        encoded = tokenizer(*args, truncation=True, max_length=max_length)
        encoded["labels"] = batch[label_field]
        return encoded

    remove = [c for c in dataset.column_names if c not in {label_field}]
    return dataset.map(tokenize, batched=True, remove_columns=remove)


def prepare_causal_lm_dataset(dataset: Any, tokenizer: Any, data_cfg: dict[str, Any]) -> Any:
    """Create fixed-length supervised causal-LM examples for label prompting.

    Only answer tokens contribute to the loss. This makes victim training match
    the paper's trajectory-alignment mechanism condition without turning the
    result into a claim about unconstrained free-running generation.
    """
    from .probes import LABEL_WORDS, render_prompt

    if data_cfg.get("format") == "instruction":
        return _prepare_instruction_dataset(dataset, tokenizer, data_cfg)
    label_field = data_cfg.get("label_field", "label")
    task_name = data_cfg["task_name"]
    label_words = list(data_cfg.get("label_words", LABEL_WORDS[task_name]))
    max_length = int(data_cfg.get("max_length", 512))
    use_chat_template = bool(data_cfg.get("use_chat_template", True))
    dataset = dataset.filter(lambda row: int(row[label_field]) >= 0)

    def encode(row: dict[str, Any], index: int) -> dict[str, Any]:
        prompt_row = {"query_id": index, "label": int(row[label_field])}
        for field in data_cfg["text_fields"]:
            prompt_row[field] = str(row[field])
        rendered = render_prompt(prompt_row, task_name, template_index=0)
        if use_chat_template and getattr(tokenizer, "chat_template", None):
            prompt = tokenizer.apply_chat_template(
                [{"role": "user", "content": rendered}],
                tokenize=False,
                add_generation_prompt=True,
            )
        else:
            prompt = rendered.rstrip() + "\n"
        answer = " " + str(label_words[int(row[label_field])])
        if tokenizer.eos_token:
            answer += tokenizer.eos_token
        prompt_ids = tokenizer.encode(prompt, add_special_tokens=True)
        answer_ids = tokenizer.encode(answer, add_special_tokens=False)
        if not answer_ids:
            raise ValueError(f"Empty answer tokenization for task label: {answer!r}")
        keep_prompt = max(max_length - len(answer_ids), 1)
        prompt_ids = prompt_ids[-keep_prompt:]
        input_ids = (prompt_ids + answer_ids)[:max_length]
        prompt_loss_mask = min(len(prompt_ids), len(input_ids))
        labels = [-100] * prompt_loss_mask + input_ids[prompt_loss_mask:]
        attention_mask = [1] * len(input_ids)
        pad = max_length - len(input_ids)
        input_ids += [int(tokenizer.pad_token_id)] * pad
        attention_mask += [0] * pad
        labels += [-100] * pad
        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": labels,
        }

    return dataset.map(
        encode,
        with_indices=True,
        remove_columns=dataset.column_names,
        desc=f"Tokenizing {task_name} as causal label supervision",
    )


def _prepare_instruction_dataset(dataset: Any, tokenizer: Any, data_cfg: dict[str, Any]) -> Any:
    """Supervised instruction tuning: loss on every response token.

    Long responses give each training example many supervised tokens, which is
    what makes per-example memorisation measurable (label-word tasks supervise
    only ~2 tokens per example).
    """
    instruction_field = data_cfg.get("instruction_field", "instruction")
    context_field = data_cfg.get("context_field")
    response_field = data_cfg.get("response_field", "response")
    max_length = int(data_cfg.get("max_length", 512))
    use_chat_template = bool(data_cfg.get("use_chat_template", True))
    dataset = dataset.filter(lambda row: bool(str(row[response_field]).strip()))

    def encode(row: dict[str, Any]) -> dict[str, Any]:
        user = str(row[instruction_field]).strip()
        if context_field and str(row.get(context_field) or "").strip():
            user = f"{user}\n\n{str(row[context_field]).strip()}"
        if use_chat_template and getattr(tokenizer, "chat_template", None):
            prompt = tokenizer.apply_chat_template(
                [{"role": "user", "content": user}], tokenize=False, add_generation_prompt=True
            )
        else:
            prompt = user + "\n"
        answer = str(row[response_field]).strip()
        if tokenizer.eos_token:
            answer += tokenizer.eos_token
        prompt_ids = tokenizer.encode(prompt, add_special_tokens=True)
        answer_ids = tokenizer.encode(answer, add_special_tokens=False)
        prompt_ids = prompt_ids[-max(max_length // 2, max_length - len(answer_ids)) :]
        answer_ids = answer_ids[: max_length - len(prompt_ids)]
        input_ids = prompt_ids + answer_ids
        labels = [-100] * len(prompt_ids) + answer_ids
        attention_mask = [1] * len(input_ids)
        pad = max_length - len(input_ids)
        input_ids += [int(tokenizer.pad_token_id)] * pad
        attention_mask += [0] * pad
        labels += [-100] * pad
        return {"input_ids": input_ids, "attention_mask": attention_mask, "labels": labels}

    return dataset.map(
        encode,
        remove_columns=dataset.column_names,
        desc="Tokenizing instruction data (response-token supervision)",
    )


def select_split(
    dataset_dict: Any,
    split: str,
    max_samples: int | None = None,
    seed: int = 0,
    index_range: list[int] | tuple[int, int] | None = None,
) -> Any:
    dataset = dataset_dict[split]
    if index_range is not None:
        # Restrict to a contiguous slice first (e.g. disjoint halves of a
        # single-split corpus for owner vs. different-data adapters).
        start, stop = int(index_range[0]), min(int(index_range[1]), len(dataset))
        dataset = dataset.select(range(start, stop))
    if max_samples is not None and len(dataset) > max_samples:
        dataset = dataset.shuffle(seed=seed).select(range(max_samples))
    return dataset


def train_with_trainer(
    model: Any,
    tokenizer: Any,
    train_dataset: Any,
    eval_dataset: Any,
    output_dir: str | Path,
    training_cfg: dict[str, Any],
    seed: int,
) -> dict[str, Any]:
    deps = require_hf()
    TrainingArguments = deps["TrainingArguments"]
    Trainer = deps["Trainer"]
    DataCollatorWithPadding = deps["DataCollatorWithPadding"]
    torch = deps["torch"]
    set_global_seed(seed)
    output_dir = ensure_dir(output_dir)
    fp16 = bool(training_cfg.get("fp16", torch.cuda.is_available()))
    bf16 = bool(training_cfg.get("bf16", False))
    batch_size = int(training_cfg.get("train_batch_size", 16))
    grad_accum = int(training_cfg.get("gradient_accumulation_steps", 1))
    epochs = float(training_cfg.get("epochs", 3))
    max_steps = int(training_cfg.get("max_steps", -1))
    estimated_steps = (
        max_steps
        if max_steps > 0
        else max(1, math.ceil(len(train_dataset) / max(batch_size * grad_accum, 1)))
        * max(1, math.ceil(epochs))
    )
    warmup_steps = int(
        training_cfg.get(
            "warmup_steps",
            math.ceil(estimated_steps * float(training_cfg.get("warmup_ratio", 0.06))),
        )
    )
    kwargs = dict(
        output_dir=str(output_dir),
        overwrite_output_dir=True,
        seed=seed,
        data_seed=seed,
        learning_rate=float(training_cfg.get("learning_rate", 2e-4)),
        weight_decay=float(training_cfg.get("weight_decay", 0.01)),
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=int(training_cfg.get("eval_batch_size", 32)),
        gradient_accumulation_steps=grad_accum,
        num_train_epochs=epochs,
        max_steps=max_steps,
        warmup_steps=warmup_steps,
        logging_steps=int(training_cfg.get("logging_steps", 25)),
        save_strategy=training_cfg.get("save_strategy", "no"),
        eval_strategy=training_cfg.get("eval_strategy", "epoch"),
        report_to=[],
        fp16=fp16 and not bf16,
        bf16=bf16,
        remove_unused_columns=True,
        dataloader_num_workers=int(training_cfg.get("dataloader_num_workers", 0)),
        gradient_checkpointing=bool(training_cfg.get("gradient_checkpointing", False)),
    )
    # Transformers has renamed and removed TrainingArguments across major
    # releases. Map the evaluation key, then pass only parameters supported by
    # the installed version. Required scientific settings remain explicit in
    # the run manifest even if a deprecated cosmetic flag disappears.
    signature = inspect.signature(TrainingArguments.__init__)
    parameters = signature.parameters
    if "eval_strategy" not in parameters and "evaluation_strategy" in parameters:
        kwargs["evaluation_strategy"] = kwargs.pop("eval_strategy")
    supported_kwargs = {key: value for key, value in kwargs.items() if key in parameters}
    args = TrainingArguments(**supported_kwargs)
    collator_name = str(training_cfg.get("data_collator", "padding"))
    if collator_name == "default":
        data_collator = deps["default_data_collator"]
    elif collator_name == "padding":
        data_collator = DataCollatorWithPadding(tokenizer)
    else:
        raise ValueError(f"Unsupported data_collator: {collator_name}")
    trainer_kwargs = {
        "model": model,
        "args": args,
        "train_dataset": train_dataset,
        "eval_dataset": eval_dataset,
        "data_collator": data_collator,
    }
    trainer_signature = inspect.signature(Trainer.__init__).parameters
    if "processing_class" in trainer_signature:
        trainer_kwargs["processing_class"] = tokenizer
    elif "tokenizer" in trainer_signature:
        trainer_kwargs["tokenizer"] = tokenizer
    trainer = Trainer(**trainer_kwargs)
    start = time.perf_counter()
    train_output = trainer.train()
    elapsed = time.perf_counter() - start
    eval_metrics = trainer.evaluate()
    trainer.save_model(str(output_dir))
    tokenizer.save_pretrained(str(output_dir))
    result = {
        "train_metrics": train_output.metrics,
        "eval_metrics": eval_metrics,
        "wall_seconds": elapsed,
        "trainable_parameters": trainable_parameter_count(model),
        "total_parameters": total_parameter_count(model),
        "checkpoint_bytes": file_size_bytes(output_dir),
    }
    write_json(output_dir / "training_result.json", result)
    return result


def save_adapter_or_model(model: Any, tokenizer: Any, output_dir: str | Path) -> None:
    output_dir = ensure_dir(output_dir)
    model.save_pretrained(str(output_dir), safe_serialization=True)
    tokenizer.save_pretrained(str(output_dir))


def dynamic_int8_quantize(model: Any) -> Any:
    deps = require_hf()
    torch = deps["torch"]
    model = copy.deepcopy(model).cpu().eval()
    return torch.ao.quantization.quantize_dynamic(model, {torch.nn.Linear}, dtype=torch.qint8)


def peak_memory_bytes(device: str) -> int:
    deps = require_hf()
    torch = deps["torch"]
    if device.startswith("cuda") and torch.cuda.is_available():
        return int(torch.cuda.max_memory_allocated())
    try:
        import psutil

        return int(psutil.Process(os.getpid()).memory_info().rss)
    except ImportError:
        return 0
