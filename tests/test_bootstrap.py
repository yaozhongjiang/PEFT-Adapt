import pandas as pd

from vectorhijack_sr.bootstrap import cluster_bootstrap


def test_cluster_bootstrap_uses_clusters():
    frame = pd.DataFrame(
        {
            "service": ["a", "a", "b", "b", "c", "c"],
            "value": [1, 1, 2, 2, 3, 3],
        }
    )
    result = cluster_bootstrap(
        frame,
        "service",
        lambda sample: float(sample["value"].mean()),
        n_bootstrap=50,
        seed=1,
    )
    assert result["cluster_count"] == 3
    assert result["ci_lower"] <= result["estimate"] <= result["ci_upper"]
