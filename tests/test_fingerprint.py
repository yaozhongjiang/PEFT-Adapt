import pandas as pd

from vectorhijack_sr.fingerprint import FingerprintExperiment


def _signatures():
    rows = []
    for structure, offset in [("r8", 0.0), ("r64", 2.0)]:
        for victim_seed in [11, 22, 33, 44, 55, 66]:
            for probe_seed, jitter in [(7, 0.01), (42, -0.01), (123, 0.02)]:
                rows.append(
                    {
                        "service_id": f"{structure}-{victim_seed}",
                        "structure": structure,
                        "known": True,
                        "victim_seed": victim_seed,
                        "probe_seed": probe_seed,
                        "batch_id": 0,
                        "f1": offset + victim_seed / 100 + jitter,
                        "f2": offset * 0.5 + victim_seed / 200 - jitter,
                    }
                )
    return pd.DataFrame(rows)


def test_lineage_split_is_checkpoint_disjoint():
    frame = _signatures()
    exp = FingerprintExperiment(random_state=2)
    result = exp.run_from_signatures(
        frame,
        feature_cols=["f1", "f2"],
        calibration_victim_seeds=[11, 22],
        validation_victim_seeds=[33, 44],
        test_victim_seeds=[55, 66],
        n_bootstrap=20,
        primary_structure="r64",
    )
    pairs = result["pairs"]
    assert set(pairs["left_seed"]).issubset({55, 66})
    assert set(pairs["right_seed"]).issubset({55, 66})
    assert result["metrics"]["threshold_selected_on_validation_services"] is True
    assert "pair_auc" in result["metrics"]
    assert result["metrics"]["primary_structure"] == "r64"
    assert "primary_pair_auc" in result["metrics"]
