from __future__ import annotations

import re
from typing import Any

import numpy as np


DEFAULT_TEMPLATES = {
    "mnli": [
        "Premise: {premise}\nHypothesis: {hypothesis}\nAnswer with entailment, neutral, or contradiction.",
        "Given '{premise}', does '{hypothesis}' follow? Reply entailment, neutral, or contradiction.",
        "Text A: {premise}\nText B: {hypothesis}\nRelation:",
    ],
    "snli": [
        "Premise: {premise}\nHypothesis: {hypothesis}\nAnswer with entailment, neutral, or contradiction.",
        "Given '{premise}', does '{hypothesis}' follow? Reply entailment, neutral, or contradiction.",
        "Text A: {premise}\nText B: {hypothesis}\nRelation:",
    ],
    "sst2": [
        "Review: {sentence}\nSentiment: positive or negative?",
        "Classify the sentiment as positive or negative: {sentence}",
        "Text: {sentence}\nLabel:",
    ],
    "ag_news": [
        "News: {text}\nCategory: World, Sports, Business, or Science?",
        "Classify this headline/article into World, Sports, Business, or Science: {text}",
        "Article: {text}\nSingle category:",
    ],
}

LABEL_WORDS = {
    "mnli": ["entailment", "neutral", "contradiction"],
    "snli": ["entailment", "neutral", "contradiction"],
    "sst2": ["negative", "positive"],
    "ag_news": ["World", "Sports", "Business", "Science"],
}


def dataset_rows(dataset: Any, data_cfg: dict[str, Any], max_samples: int | None, seed: int) -> list[dict[str, Any]]:
    if max_samples is not None and len(dataset) > max_samples:
        dataset = dataset.shuffle(seed=seed).select(range(max_samples))
    rows: list[dict[str, Any]] = []
    for idx, item in enumerate(dataset):
        label = int(item[data_cfg.get("label_field", "label")])
        # Several public datasets (e.g., SNLI) use -1 for examples without a
        # consensus label. They must never enter training, evaluation, or query
        # accounting because cross-entropy treats them as an invalid class.
        if label < 0:
            continue
        row = {"query_id": idx, "label": label}
        for field in data_cfg["text_fields"]:
            row[field] = str(item[field])
        rows.append(row)
    return rows


def classification_text(row: dict[str, Any], text_fields: list[str]) -> tuple[str, str | None]:
    if len(text_fields) == 1:
        return str(row[text_fields[0]]), None
    if len(text_fields) == 2:
        return str(row[text_fields[0]]), str(row[text_fields[1]])
    return " [SEP] ".join(str(row[field]) for field in text_fields), None


def render_prompt(row: dict[str, Any], task_name: str, template_index: int = 0) -> str:
    templates = DEFAULT_TEMPLATES.get(task_name)
    if not templates:
        fields = {k: v for k, v in row.items() if isinstance(v, str)}
        body = "\n".join(f"{k}: {v}" for k, v in fields.items())
        return f"{body}\nAnswer with one label token."
    return templates[template_index % len(templates)].format(**row)


def build_prompt_views(
    rows: list[dict[str, Any]],
    task_name: str,
    probe_seed: int,
    system_prompt: str | None = None,
) -> list[dict[str, Any]]:
    rng = np.random.default_rng(probe_seed)
    templates = DEFAULT_TEMPLATES.get(task_name, ["{text}"])
    output: list[dict[str, Any]] = []
    for row in rows:
        template_index = int(rng.integers(0, len(templates)))
        prompt = render_prompt(row, task_name, template_index)
        if system_prompt:
            prompt = f"{system_prompt}\n\n{prompt}"
        output.append({**row, "prompt": prompt, "template_id": template_index, "probe_seed": probe_seed})
    return output


ENCODER_PROBE_FAMILIES = (
    "template_nli",
    "template_contradict",
    "layer_early",
    "layer_late",
    "module_head",
    "lexical_local",
)

_FAMILY_CODE = {name: index + 1 for index, name in enumerate(ENCODER_PROBE_FAMILIES)}


def _clip_words(text: str, start_frac: float, end_frac: float) -> str:
    words = str(text).split()
    if len(words) < 6:
        return str(text)
    start = int(len(words) * start_frac)
    end = max(start + 1, int(len(words) * end_frac))
    clipped = " ".join(words[start:end]).strip()
    return clipped if clipped else str(text)


def expand_structural_probe_rows(
    rows: list[dict[str, Any]],
    text_fields: list[str],
    families: list[str] | None = None,
) -> list[dict[str, Any]]:
    """Create new encoder probes that change residual geometry, not a subset of ID text.

    Families are black-box stand-ins for the suggested active attacks:
    template prefixes, early/late span drops (layer locality), a hypothesis-only
    cue (classifier/head), and lexical local edits. Canonical rows are not
    duplicated here so they can be merged from the original collection.
    """
    if len(text_fields) < 2:
        raise ValueError("Structural encoder probes require a paired text task")
    premise_field, hypothesis_field = text_fields[0], text_fields[1]
    wanted = [name for name in (families or list(ENCODER_PROBE_FAMILIES)) if name in _FAMILY_CODE]
    output: list[dict[str, Any]] = []
    for row in rows:
        premise = str(row[premise_field])
        hypothesis = str(row[hypothesis_field])
        variants: dict[str, tuple[str, str]] = {
            "template_nli": ("NLI: " + premise, hypothesis),
            "template_contradict": ("Decide if this is a contradiction. " + premise, hypothesis),
            "layer_early": (_clip_words(premise, 0.33, 1.0), hypothesis),
            "layer_late": (_clip_words(premise, 0.0, 0.67), hypothesis),
            "module_head": ("The claim is:", hypothesis),
        }
        lexical = controlled_local_perturbations(premise)
        if len(lexical) > 1:
            variants["lexical_local"] = (lexical[1], hypothesis)
        source_id = int(row["query_id"])
        for family in wanted:
            if family not in variants:
                continue
            new_premise, new_hypothesis = variants[family]
            if new_premise == premise and new_hypothesis == hypothesis:
                continue
            item = dict(row)
            item[premise_field] = new_premise
            item[hypothesis_field] = new_hypothesis
            item["probe_family"] = family
            item["source_query_id"] = source_id
            item["query_id"] = source_id * 100 + _FAMILY_CODE[family]
            output.append(item)
    return output


def logit_margin(logits: np.ndarray) -> float:
    values = np.asarray(logits, dtype=float).ravel()
    if values.size == 0:
        return 0.0
    if values.size == 1:
        return float(values[0])
    ordered = np.sort(values)
    return float(ordered[-1] - ordered[-2])


def select_margin_sensitive_rows(
    source_rows: list[dict[str, Any]],
    variant_rows: list[dict[str, Any]],
    base_logits_by_query: dict[int, np.ndarray],
    *,
    max_variants_per_source: int = 4,
    required_families: tuple[str, ...] = ("template_nli", "layer_early", "module_head"),
) -> list[dict[str, Any]]:
    """Keep a bounded set of new probes using only the local base model.

    Required families are kept when they exist. Remaining slots go to the
    lowest base-margin variants (boundary-near) and the largest margin shift
    versus the canonical example.
    """
    by_source: dict[int, list[dict[str, Any]]] = {}
    for row in variant_rows:
        by_source.setdefault(int(row["source_query_id"]), []).append(row)
    selected: list[dict[str, Any]] = []
    for source in source_rows:
        source_id = int(source["query_id"])
        candidates = list(by_source.get(source_id, []))
        if not candidates:
            continue
        canonical_margin = logit_margin(base_logits_by_query.get(source_id, np.zeros(1)))
        kept: list[dict[str, Any]] = []
        seen: set[str] = set()
        for family in required_families:
            match = next((row for row in candidates if row.get("probe_family") == family), None)
            if match is None or str(match["probe_family"]) in seen:
                continue
            kept.append(match)
            seen.add(str(match["probe_family"]))
        scored: list[tuple[float, dict[str, Any]]] = []
        for row in candidates:
            family = str(row.get("probe_family"))
            if family in seen:
                continue
            margin = logit_margin(base_logits_by_query.get(int(row["query_id"]), np.zeros(1)))
            shift = abs(margin - canonical_margin)
            scored.append((float(margin - 0.25 * shift), row))
        scored.sort(key=lambda item: item[0])
        for _, row in scored:
            if len(kept) >= max_variants_per_source:
                break
            family = str(row.get("probe_family"))
            if family in seen:
                continue
            kept.append(row)
            seen.add(family)
        selected.extend(kept[:max_variants_per_source])
    return selected


def controlled_local_perturbations(text: str) -> list[str]:
    """Deterministic, auditable surface perturbations; no external NLP service."""
    variants = [text]
    replacements = {
        r"\bnot\b": "never",
        r"\bgood\b": "favorable",
        r"\bbad\b": "unfavorable",
        r"\bbecause\b": "since",
        r"\bshows\b": "indicates",
    }
    for pattern, replacement in replacements.items():
        candidate, count = re.subn(pattern, replacement, text, count=1, flags=re.IGNORECASE)
        if count and candidate != text:
            variants.append(candidate)
    if len(text.split()) > 6:
        words = text.split()
        variants.append(" ".join(words[:3] + words[4:]))
    return list(dict.fromkeys(variants))


def label_token_ids(
    tokenizer: Any,
    task_name: str,
    labels: list[str] | None = None,
    *,
    allow_first_subtoken: bool = False,
) -> list[int]:
    label_list = list(labels) if labels is not None else list(LABEL_WORDS[task_name])
    ids: list[int] = []
    for label in label_list:
        encoded = tokenizer.encode(label, add_special_tokens=False)
        if len(encoded) != 1:
            # Match causal-LM training answers, which prefix a leading space.
            encoded = tokenizer.encode(" " + label, add_special_tokens=False)
        if len(encoded) != 1:
            if allow_first_subtoken and encoded:
                encoded = encoded[:1]
            else:
                raise ValueError(
                    f"Label '{label}' is not one token for tokenizer {tokenizer.name_or_path}; "
                    "choose an explicit single-token label map in the configuration."
                )
        ids.append(int(encoded[0]))
    return ids
