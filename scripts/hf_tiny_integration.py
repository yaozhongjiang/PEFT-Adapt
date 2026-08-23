#!/usr/bin/env python
"""Offline integration test for Transformers + PEFT execution paths.

This script creates a tiny BERT checkpoint and tokenizer locally, performs one
Trainer step, saves two independent LoRA adapters, reloads/merges an adapter,
and executes the task-head-safe weighted composition protocol. It downloads no
model or dataset and is therefore suitable for CI and artifact review.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
from pathlib import Path

os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")

import torch
from datasets import Dataset

torch.set_num_threads(1)
try:
    torch.set_num_interop_threads(1)
except RuntimeError:
    pass
from peft import LoraConfig, TaskType, get_peft_model  # noqa: E402
from transformers import BertConfig, BertForSequenceClassification, BertTokenizer  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from vectorhijack_sr.collect import _load_checkpoint_model  # noqa: E402
from vectorhijack_sr.hf import (  # noqa: E402
    prepare_classification_dataset,
    save_adapter_or_model,
    train_with_trainer,
)
from vectorhijack_sr.redeploy import (  # noqa: E402
    _composition_evaluate,
    _make_body_only_adapter,
    _merge_evaluate,
)


def _build_local_base(root: Path) -> Path:
    base_dir = root / "tiny-base"
    base_dir.mkdir(parents=True, exist_ok=True)
    vocab = [
        "[PAD]",
        "[UNK]",
        "[CLS]",
        "[SEP]",
        "[MASK]",
        "a",
        "dog",
        "cat",
        "runs",
        "sleeps",
        "animal",
        "is",
        "not",
        "moving",
        "entailment",
        "neutral",
        "contradiction",
        ".",
    ]
    (base_dir / "vocab.txt").write_text("\n".join(vocab) + "\n")
    tokenizer = BertTokenizer(vocab_file=str(base_dir / "vocab.txt"), do_lower_case=True)
    tokenizer.save_pretrained(base_dir)
    torch.manual_seed(1234)
    model = BertForSequenceClassification(
        BertConfig(
            vocab_size=len(vocab),
            hidden_size=32,
            num_hidden_layers=1,
            num_attention_heads=4,
            intermediate_size=64,
            max_position_embeddings=64,
            num_labels=3,
        )
    )
    model.save_pretrained(base_dir, safe_serialization=True)
    return base_dir


def _dataset() -> Dataset:
    return Dataset.from_dict(
        {
            "premise": [
                "a dog runs .",
                "a cat sleeps .",
                "a dog sleeps .",
                "a cat runs .",
                "an animal runs .",
                "an animal sleeps .",
            ],
            "hypothesis": [
                "an animal is moving .",
                "an animal is not moving .",
                "a cat runs .",
                "a dog sleeps .",
                "a dog runs .",
                "a cat sleeps .",
            ],
            "label": [0, 2, 1, 1, 1, 0],
        }
    )


def _make_adapter(base_dir: Path, out: Path, seed: int) -> Path:
    torch.manual_seed(seed)
    base = BertForSequenceClassification.from_pretrained(base_dir)
    model = get_peft_model(
        base,
        LoraConfig(
            task_type=TaskType.SEQ_CLS,
            r=4,
            lora_alpha=8,
            lora_dropout=0.0,
            target_modules=["query", "value"],
            modules_to_save=["classifier"],
        ),
    )
    # Deterministic non-zero independent checkpoints; no external data needed.
    with torch.no_grad():
        for name, parameter in model.named_parameters():
            if parameter.requires_grad:
                parameter.add_(0.005 * seed * torch.randn_like(parameter))
    tokenizer = BertTokenizer.from_pretrained(base_dir)
    save_adapter_or_model(model, tokenizer, out)
    return out


def run(output: Path) -> dict[str, object]:
    shutil.rmtree(output, ignore_errors=True)
    output.mkdir(parents=True)
    base_dir = _build_local_base(output)
    tokenizer = BertTokenizer.from_pretrained(base_dir)
    raw = _dataset()
    data_cfg = {
        "task_name": "mnli",
        "text_fields": ["premise", "hypothesis"],
        "label_field": "label",
        "num_labels": 3,
        "max_length": 32,
    }

    # Actual Trainer compatibility check, not merely model construction.
    tokenized = prepare_classification_dataset(raw, tokenizer, data_cfg)
    train_model = BertForSequenceClassification.from_pretrained(base_dir)
    trainer_result = train_with_trainer(
        train_model,
        tokenizer,
        tokenized,
        tokenized,
        output / "trainer-checkpoint",
        {
            "learning_rate": 1e-4,
            "train_batch_size": 2,
            "eval_batch_size": 2,
            "max_steps": 1,
            "epochs": 1,
            "save_strategy": "no",
            "eval_strategy": "no",
            "logging_steps": 1,
            "fp16": False,
            "bf16": False,
        },
        seed=7,
    )

    adapter_a = _make_adapter(base_dir, output / "adapter-a", seed=11)
    adapter_b = _make_adapter(base_dir, output / "adapter-b", seed=22)
    cfg = {
        "model": {
            "model_id": str(base_dir),
            "tokenizer_id": str(base_dir),
            "revision": None,
            "task_type": "sequence_classification",
            "dtype": "float32",
            "local_files_only": True,
        },
        "data": data_cfg,
        "composition": {
            "task_b": data_cfg,
            "weights": [0.5, 0.5],
            "combination_type": "linear",
        },
    }
    victim = _load_checkpoint_model(cfg["model"], adapter_a, num_labels=3)
    composition = _composition_evaluate(
        cfg,
        adapter_a,
        adapter_b,
        raw,
        raw,
        victim,
    )
    if not composition.get("supported"):
        raise RuntimeError(f"Composition integration failed: {composition}")
    merge = _merge_evaluate(adapter_a, cfg["model"], tokenizer, raw, data_cfg, victim)
    if not merge.get("merge"):
        raise RuntimeError(f"Safe merge integration failed: {merge}")
    body_meta = _make_body_only_adapter(adapter_a, output / "body-only-check")
    if not body_meta["head_keys"] or not body_meta["body_keys"]:
        raise RuntimeError("Body/head split did not retain both parameter groups")

    report = {
        "status": "passed",
        "offline": True,
        "trainer_step": trainer_result,
        "merge": merge,
        "composition": composition,
        "body_head_split": {
            "head_key_count": len(body_meta["head_keys"]),
            "body_key_count": len(body_meta["body_keys"]),
        },
    }
    (output / "integration_report.json").write_text(
        json.dumps(report, indent=2, default=str) + "\n"
    )
    return report


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT / "artifact" / "cache" / "hf_tiny_integration",
    )
    args = parser.parse_args()
    report = run(args.output)
    print(json.dumps({"status": report["status"], "output": str(args.output)}, indent=2))
    sys.stdout.flush()
    sys.stderr.flush()
    # Some old Linux kernels leave native tokenizer/OpenMP teardown threads
    # blocked after all assertions and files have completed. This script is a
    # child-process integration probe, so exit immediately after flushing the
    # verified report rather than letting third-party destructors hang CI.
    os._exit(0)


if __name__ == "__main__":
    main()
