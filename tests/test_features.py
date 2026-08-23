import numpy as np

from vectorhijack_sr.features import aggregate_signature, per_query_features


def test_per_query_features_finite():
    result = per_query_features(np.array([1.0, 0.0, -1.0]), np.array([0.5, 0.1, -0.2]))
    assert result["res_l2"] > 0
    assert all(np.isfinite(value) for value in result.values())


def test_aggregate_signature_has_spectrum():
    rows = [
        {
            "victim_logits": [1.0 + i * 0.01, 0.0, -1.0],
            "base_logits": [0.5, 0.1, -0.2],
            "input_length": 10 + i,
        }
        for i in range(20)
    ]
    sig = aggregate_signature(rows)
    assert "spec_effective_rank" in sig
    assert "res_l2_mean" in sig
