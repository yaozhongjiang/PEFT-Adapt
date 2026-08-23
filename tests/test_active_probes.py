from vectorhijack_sr.active_probes import select_query_ids


def test_active_probe_uses_only_known_fit_seeds():
    rows = []
    for query_id in range(20):
        for structure, scale in [("lora_r8", 1.0), ("lora_r64", 2.0 + query_id / 10)]:
            rows.append(
                {
                    "query_id": query_id,
                    "structure": structure,
                    "known": True,
                    "victim_seed": 11,
                    "victim_logits": [scale, 0.1, 0.0],
                    "base_logits": [0.0, 0.1, 0.0],
                }
            )
        rows.append(
            {
                "query_id": query_id,
                "structure": "dora_r64_unknown",
                "known": False,
                "victim_seed": 44,
                "victim_logits": [9.0, 0.0, 0.0],
                "base_logits": [0.0, 0.0, 0.0],
            }
        )
    result = select_query_ids(rows, fit_seeds={11}, keep_fraction=0.3, min_queries=4, max_queries=8)
    assert result["selection"] == "rank_sensitive_known_only"
    assert 4 <= result["n_selected"] <= 8
    assert set(result["selected_query_ids"]).issubset(set(range(20)))
