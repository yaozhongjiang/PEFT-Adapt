import numpy as np

from vectorhijack_sr.metrics import wilson_ci
from vectorhijack_sr.unknown import HierarchicalManifoldUnknownDetector, HybridUnknownDetector


def _synthetic_open_set(seed: int = 3):
    rng = np.random.default_rng(seed)
    x_train = np.vstack(
        [
            rng.normal(-2, 0.25, size=(60, 6)),
            rng.normal(0, 0.25, size=(60, 6)),
            rng.normal(2, 0.25, size=(60, 6)),
        ]
    )
    y_train = np.array(["a"] * 60 + ["b"] * 60 + ["c"] * 60)
    groups = np.array([f"svc-{label}-{i // 10}" for label in "abc" for i in range(60)])
    x_known = rng.normal(0, 0.25, size=(30, 6))
    y_known = np.array(["b"] * 30)
    x_unknown = rng.normal(7, 0.3, size=(30, 6))
    return x_train, y_train, groups, x_known, y_known, x_unknown


def test_unknown_detector_rejects_shifted_samples():
    x_train, y_train, groups, x_known, y_known, x_unknown = _synthetic_open_set()
    detector = HybridUnknownDetector(random_state=1, target_known_fpr=0.05).fit(
        x_train, y_train, groups
    )
    metrics = detector.evaluate(x_known, y_known, x_unknown)
    assert metrics["auroc"] > 0.9
    assert metrics["unknown_recall_locked"] > 0.7
    assert metrics["known_reject_rate"] <= 0.15
    assert metrics["known_accept_accuracy"] > 0.8
    assert "unknown_recall_locked_ci" in metrics


def test_hierarchical_manifold_rejects_shifted_samples():
    x_train, y_train, groups, x_known, y_known, x_unknown = _synthetic_open_set(5)
    detector = HierarchicalManifoldUnknownDetector(
        random_state=1, target_known_fpr=0.05
    ).fit(x_train, y_train, groups)
    metrics = detector.evaluate(x_known, y_known, x_unknown)
    assert metrics["rejector"] == "hierarchical_manifold"
    assert metrics["auroc"] > 0.9
    assert metrics["unknown_recall_locked"] > 0.7
    assert metrics["known_reject_rate"] <= 0.2
    assert metrics["mahal_ecdf_auroc"] > 0.8


def test_wilson_ci_bounds():
    p, lo, hi = wilson_ci(8, 10)
    assert abs(p - 0.8) < 1e-9
    assert 0.0 <= lo <= p <= hi <= 1.0
    assert wilson_ci(0, 0)[0] != wilson_ci(0, 0)[0]  # NaN
