from __future__ import annotations

import os

import pytest

if os.environ.get("RUN_HF_INTEGRATION") != "1":
    pytest.skip("set RUN_HF_INTEGRATION=1 for local HF dataset test", allow_module_level=True)

from datasets import Dataset  # noqa: E402
from transformers import BertTokenizer  # noqa: E402

from vectorhijack_sr.hf import prepare_causal_lm_dataset  # noqa: E402


@pytest.mark.integration
def test_causal_label_dataset_masks_prompt(tmp_path):
    vocab = [
        "[PAD]",
        "[UNK]",
        "[CLS]",
        "[SEP]",
        "[MASK]",
        "premise",
        "hypothesis",
        "answer",
        "with",
        "entailment",
        "neutral",
        "contradiction",
        "a",
        "b",
    ]
    vocab_path = tmp_path / "vocab.txt"
    vocab_path.write_text("\n".join(vocab) + "\n")
    tokenizer = BertTokenizer(vocab_file=str(vocab_path))
    dataset = Dataset.from_dict({"premise": ["a"], "hypothesis": ["b"], "label": [0]})
    prepared = prepare_causal_lm_dataset(
        dataset,
        tokenizer,
        {
            "task_name": "mnli",
            "text_fields": ["premise", "hypothesis"],
            "label_field": "label",
            "max_length": 32,
            "use_chat_template": False,
        },
    )
    row = prepared[0]
    assert len(row["input_ids"]) == 32
    assert len(row["labels"]) == 32
    supervised = [value for value in row["labels"] if value != -100]
    assert supervised
    assert all(mask in {0, 1} for mask in row["attention_mask"])
