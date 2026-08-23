from vectorhijack_sr.features import aggregate_signature
from vectorhijack_sr.probes import (
    expand_structural_probe_rows,
    logit_margin,
    select_margin_sensitive_rows,
)


def test_expand_structural_probes_changes_text():
    rows = [
        {
            "query_id": 3,
            "label": 0,
            "premise": "The good movie was not long because it shows care.",
            "hypothesis": "The film is short.",
        }
    ]
    expanded = expand_structural_probe_rows(rows, ["premise", "hypothesis"])
    families = {row["probe_family"] for row in expanded}
    assert "template_nli" in families
    assert "layer_early" in families
    assert "module_head" in families
    assert all(row["source_query_id"] == 3 for row in expanded)
    assert all(row["premise"] != rows[0]["premise"] or row["hypothesis"] != rows[0]["hypothesis"] for row in expanded)


def test_margin_selection_prefers_required_and_low_margin():
    sources = [{"query_id": 1, "label": 0, "premise": "a", "hypothesis": "b"}]
    variants = [
        {"query_id": 101, "source_query_id": 1, "probe_family": "template_nli"},
        {"query_id": 102, "source_query_id": 1, "probe_family": "layer_early"},
        {"query_id": 103, "source_query_id": 1, "probe_family": "module_head"},
        {"query_id": 104, "source_query_id": 1, "probe_family": "layer_late"},
    ]
    logits = {
        1: [2.0, 0.1, 0.0],
        101: [1.5, 0.2, 0.0],
        102: [1.2, 0.1, 0.0],
        103: [0.4, 0.3, 0.2],
        104: [3.0, 0.0, 0.0],
    }
    selected = select_margin_sensitive_rows(
        sources, variants, logits, max_variants_per_source=3
    )
    families = [row["probe_family"] for row in selected]
    assert "template_nli" in families
    assert "layer_early" in families
    assert "module_head" in families
    assert len(selected) == 3


def test_logit_margin():
    assert logit_margin([3.0, 1.0, 0.0]) == 2.0


def test_aggregate_signature_includes_sensitivity():
    rows = [
        {
            "query_id": 1,
            "probe_family": "canonical",
            "source_query_id": 1,
            "victim_logits": [1.0, 0.0, -1.0],
            "base_logits": [0.5, 0.1, -0.2],
            "input_length": 10,
        },
        {
            "query_id": 101,
            "probe_family": "template_nli",
            "source_query_id": 1,
            "victim_logits": [2.5, 0.0, -1.0],
            "base_logits": [0.5, 0.1, -0.2],
            "input_length": 12,
        },
        {
            "query_id": 103,
            "probe_family": "module_head",
            "source_query_id": 1,
            "victim_logits": [0.6, 0.4, 0.0],
            "base_logits": [0.5, 0.1, -0.2],
            "input_length": 8,
        },
    ]
    # pad to satisfy later callers if needed
    while len(rows) < 8:
        extra = dict(rows[0])
        extra["query_id"] = 10 + len(rows)
        extra["victim_logits"] = [1.0 + 0.01 * len(rows), 0.0, -1.0]
        rows.append(extra)
    sig = aggregate_signature(rows)
    assert sig["sens_template_delta_mean"] > 0
    assert sig["sens_head_delta_mean"] >= 0
    assert "sens_family_std_mean" in sig
