from __future__ import annotations

import copy
import inspect
from pathlib import Path
from typing import Any

from .hf import load_model, load_task_dataset, load_tokenizer, require_hf, save_adapter_or_model
from .io import ensure_dir, write_json
from .reproducibility import set_global_seed


def continued_pretrain_backbone(
    model_cfg: dict[str, Any],
    data_cfg: dict[str, Any],
    output_dir: str | Path,
    seed: int,
    max_steps: int = 50,
    learning_rate: float = 5e-5,
    max_samples: int = 2000,
) -> str:
    """Create a real local checkpoint revision via continued language-model training.

    For sequence-classification bases, the backbone is updated with MLM while the
    original public classifier head is preserved exactly. For causal LMs, the
    whole causal checkpoint is continued directly.
    """
    deps = require_hf()
    torch = deps["torch"]
    transformers = __import__("transformers", fromlist=["DataCollatorForLanguageModeling"])
    DataCollatorForLanguageModeling = transformers.DataCollatorForLanguageModeling
    Trainer = deps["Trainer"]
    TrainingArguments = deps["TrainingArguments"]
    output_dir = ensure_dir(output_dir)
    set_global_seed(seed)
    tokenizer = load_tokenizer(model_cfg)
    task_type = model_cfg.get("task_type", "sequence_classification")
    lm_cfg = copy.deepcopy(model_cfg)
    lm_cfg["task_type"] = "causal_lm" if task_type == "causal_lm" else "masked_lm"
    lm_model = load_model(lm_cfg)
    dataset_dict = load_task_dataset(data_cfg)
    split = dataset_dict[data_cfg.get("continued_pretrain_split", data_cfg.get("train_split", "train"))]
    if len(split) > max_samples:
        split = split.shuffle(seed=seed).select(range(max_samples))
    text_fields = data_cfg["text_fields"]
    max_length = int(data_cfg.get("max_length", 256))

    def tokenize(batch: dict[str, Any]) -> dict[str, Any]:
        if len(text_fields) == 1:
            texts = batch[text_fields[0]]
        else:
            texts = [" ".join(map(str, parts)) for parts in zip(*(batch[f] for f in text_fields), strict=True)]
        return tokenizer(
            texts,
            truncation=True,
            max_length=max_length,
            return_special_tokens_mask=True,
        )

    tokenized = split.map(tokenize, batched=True, remove_columns=split.column_names)
    collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer,
        mlm=task_type != "causal_lm",
        mlm_probability=0.15,
    )
    kwargs = {
        "output_dir": str(output_dir / "lm_work"),
        "overwrite_output_dir": True,
        "seed": seed,
        "learning_rate": learning_rate,
        "max_steps": max_steps,
        "per_device_train_batch_size": 8,
        "gradient_accumulation_steps": 1,
        "save_strategy": "no",
        "report_to": [],
        "logging_steps": max(1, max_steps // 5),
        "fp16": bool(torch.cuda.is_available()),
    }
    signature = inspect.signature(TrainingArguments.__init__)
    parameters = signature.parameters
    if "eval_strategy" not in parameters and "evaluation_strategy" in parameters:
        kwargs["evaluation_strategy"] = kwargs.pop("eval_strategy", "no")
    supported_kwargs = {key: value for key, value in kwargs.items() if key in parameters}
    args = TrainingArguments(**supported_kwargs)
    trainer_kwargs = {
        "model": lm_model,
        "args": args,
        "train_dataset": tokenized,
        "data_collator": collator,
    }
    trainer_sig = inspect.signature(Trainer.__init__).parameters
    if "processing_class" in trainer_sig:
        trainer_kwargs["processing_class"] = tokenizer
    elif "tokenizer" in trainer_sig:
        trainer_kwargs["tokenizer"] = tokenizer
    trainer = Trainer(**trainer_kwargs)
    train_result = trainer.train()

    if task_type == "causal_lm":
        final_model = lm_model
    else:
        original_cfg = copy.deepcopy(model_cfg)
        original_cfg["task_type"] = "sequence_classification"
        num_labels = int(data_cfg["num_labels"])
        final_model = load_model(original_cfg, num_labels=num_labels)
        prefix = getattr(final_model, "base_model_prefix", None)
        if not prefix or not hasattr(final_model, prefix) or not hasattr(lm_model, prefix):
            raise RuntimeError(
                "Cannot transfer continued-pretraining backbone: incompatible base_model_prefix"
            )
        target_backbone = getattr(final_model, prefix)
        source_backbone = getattr(lm_model, prefix)
        source_state = source_backbone.state_dict()
        target_state = target_backbone.state_dict()
        # MLM backbones omit pooler weights; keep the original seq-cls pooler intact.
        compatible = {
            key: tensor
            for key, tensor in source_state.items()
            if key in target_state and tensor.shape == target_state[key].shape
        }
        missing, unexpected = target_backbone.load_state_dict(compatible, strict=False)
        allowed_missing = {key for key in missing if key.startswith("pooler.")}
        unexpected_missing = [key for key in missing if key not in allowed_missing]
        if unexpected_missing or unexpected:
            raise RuntimeError(
                "Backbone transfer mismatch; "
                f"missing={unexpected_missing[:5]}, unexpected={unexpected[:5]}"
            )
    final_dir = output_dir / "checkpoint"
    save_adapter_or_model(final_model, tokenizer, final_dir)
    write_json(
        output_dir / "continued_pretraining_manifest.json",
        {
            "seed": seed,
            "max_steps": max_steps,
            "learning_rate": learning_rate,
            "max_samples": max_samples,
            "train_metrics": train_result.metrics,
            "source_model": model_cfg,
            "output_checkpoint": str(final_dir),
            "classifier_head_preserved": task_type != "causal_lm",
        },
    )
    return str(final_dir)
