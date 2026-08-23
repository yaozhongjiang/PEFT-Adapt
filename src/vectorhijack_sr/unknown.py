from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Sequence

import numpy as np
from sklearn.covariance import LedoitWolf
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GroupKFold

try:
    from sklearn.model_selection import StratifiedGroupKFold
except ImportError:  # pragma: no cover - compatibility fallback
    StratifiedGroupKFold = None
from sklearn.preprocessing import StandardScaler

from .metrics import open_set_classification_rate, unknown_metrics, wilson_ci
from .structural import StructuralClassifier

# Compact residual/spectral core used by the LoRA-manifold Mahalanobis stage.
MANIFOLD_CORE_FEATURES = (
    "res_l2_mean",
    "res_l2_std",
    "res_linf_mean",
    "spec_effective_rank",
    "spec_entropy",
    "spec_energy90",
    "spec_sv1",
    "spec_sv2",
    "logit_cosine_mean",
    "entropy_delta_mean",
    "topk_overlap_mean",
)


@dataclass
class OpenSetPredictions:
    closed_labels: np.ndarray
    final_labels: np.ndarray
    closed_probabilities: np.ndarray
    unknown_scores: np.ndarray
    rejected: np.ndarray
    threshold: float


class HybridUnknownDetector:
    """Cross-fitted conformal/distance/energy unknown detector.

    Pseudo-unknown examples are produced only by leaving an entire known
    structure out. The real unknown test structures are never used for fitting
    or threshold selection.
    """

    def __init__(self, random_state: int = 0, target_known_fpr: float = 0.05) -> None:
        self.random_state = random_state
        self.target_known_fpr = target_known_fpr
        self.closed = StructuralClassifier(random_state=random_state)
        self.meta_scaler = StandardScaler()
        self.meta = LogisticRegression(
            max_iter=2000,
            class_weight="balanced",
            random_state=random_state,
        )
        self.threshold_: float | None = None
        self.feature_names_ = [
            "one_minus_max_prob",
            "posterior_entropy",
            "negative_margin",
            "min_mahalanobis",
            "predicted_mahalanobis",
            "one_minus_conformal_p",
            "energy",
            "distance_ratio",
        ]

    def fit(
        self,
        x_train: np.ndarray,
        y_train: np.ndarray,
        groups: np.ndarray,
        x_validation_known: np.ndarray | None = None,
    ) -> "HybridUnknownDetector":
        x_train = np.asarray(x_train, dtype=float)
        y_train = np.asarray(y_train)
        groups = np.asarray(groups)
        unique_classes = np.unique(y_train)
        if len(unique_classes) < 3:
            raise ValueError("Hybrid unknown detector requires at least three known structures")

        meta_x: list[np.ndarray] = []
        meta_y: list[np.ndarray] = []
        # Leave one complete structure out.  Both pseudo-known and pseudo-unknown
        # meta examples are aggregated at the service-checkpoint level.  This is
        # essential: otherwise hundreds of probe batches from one checkpoint can
        # dominate the meta detector and create an artificially sharp threshold.
        for held_out in unique_classes:
            train_mask = y_train != held_out
            unknown_mask = y_train == held_out
            if len(np.unique(y_train[train_mask])) < 2:
                continue
            model = StructuralClassifier(random_state=self.random_state).fit(
                x_train[train_mask], y_train[train_mask]
            )
            known_features = self._cross_fitted_known_features(
                x_train[train_mask],
                y_train[train_mask],
                groups[train_mask],
            )
            unknown_features = model.summary_features(x_train[unknown_mask])
            known_service = self._aggregate_by_group(known_features, groups[train_mask])
            unknown_service = self._aggregate_by_group(unknown_features, groups[unknown_mask])
            if len(known_service) == 0 or len(unknown_service) == 0:
                continue
            # Preserve all pseudo-unknown services and sample at most three times
            # as many pseudo-known services. The sampling unit is a checkpoint.
            rng = np.random.default_rng(self.random_state + len(meta_x))
            n = min(len(known_service), len(unknown_service) * 3)
            selected = rng.choice(len(known_service), size=n, replace=False)
            meta_x.extend([known_service[selected], unknown_service])
            meta_y.extend(
                [
                    np.zeros(n, dtype=int),
                    np.ones(len(unknown_service), dtype=int),
                ]
            )

        if not meta_x:
            raise ValueError("Could not construct pseudo-unknown meta examples")
        meta_matrix = np.vstack(meta_x)
        meta_labels = np.concatenate(meta_y)
        self.meta.fit(self.meta_scaler.fit_transform(meta_matrix), meta_labels)

        # Calibrate rejection on group-held-out known services, not on the same
        # rows used to fit the closed classifier. In-sample posterior features
        # are systematically overconfident and previously produced a threshold
        # that rejected nearly every independently trained known checkpoint.
        if x_validation_known is None:
            known_summary = self._cross_fitted_known_features(x_train, y_train, groups)
        else:
            provisional = StructuralClassifier(random_state=self.random_state).fit(x_train, y_train)
            known_summary = provisional.summary_features(
                np.asarray(x_validation_known, dtype=float)
            )
        known_scores = self._score_features(known_summary)
        self.threshold_ = float(
            np.quantile(
                known_scores,
                1.0 - self.target_known_fpr,
                method="higher",
            )
        )
        self.closed.fit(x_train, y_train)
        return self

    @staticmethod
    def _aggregate_by_group(features: np.ndarray, groups: np.ndarray) -> np.ndarray:
        """Average detector features per independent service checkpoint."""
        features = np.asarray(features, dtype=float)
        groups = np.asarray(groups)
        if len(features) != len(groups):
            raise ValueError("Feature/group length mismatch")
        rows = [features[groups == group].mean(axis=0) for group in np.unique(groups)]
        return np.vstack(rows) if rows else np.empty((0, features.shape[1]), dtype=float)

    def _cross_fitted_known_features(
        self,
        x: np.ndarray,
        y: np.ndarray,
        groups: np.ndarray,
    ) -> np.ndarray:
        """Return out-of-service detector features aligned with input rows."""
        unique_groups = np.unique(groups)
        if len(unique_groups) < 2:
            raise ValueError(
                "Unknown-detector threshold calibration requires at least two services"
            )
        # A stratified group fold must leave at least one service of every class
        # in training. Fall back to ordinary group folds when a class has too few
        # independent checkpoints for stratification.
        groups_per_class = [len(np.unique(groups[y == label])) for label in np.unique(y)]
        max_splits = min(5, len(unique_groups), min(groups_per_class))
        if StratifiedGroupKFold is not None and max_splits >= 2:
            splitter = StratifiedGroupKFold(
                n_splits=max_splits,
                shuffle=True,
                random_state=self.random_state,
            )
            splits = splitter.split(x, y, groups)
        else:
            n_splits = min(5, len(unique_groups))
            splitter = GroupKFold(n_splits=n_splits)
            splits = splitter.split(x, y, groups)
        aligned: np.ndarray | None = None
        covered = np.zeros(len(x), dtype=bool)
        for train_idx, validation_idx in splits:
            if len(np.unique(y[train_idx])) < 2:
                continue
            model = StructuralClassifier(random_state=self.random_state).fit(
                x[train_idx], y[train_idx]
            )
            fold_features = model.summary_features(x[validation_idx])
            if aligned is None:
                aligned = np.empty((len(x), fold_features.shape[1]), dtype=float)
            aligned[validation_idx] = fold_features
            covered[validation_idx] = True
        if aligned is None or not np.all(covered):
            missing = int(np.sum(~covered))
            raise ValueError(
                f"Could not create group-held-out known calibration features; uncovered={missing}"
            )
        return aligned

    def refit_closed(self, x: np.ndarray, y: np.ndarray) -> "HybridUnknownDetector":
        """Refit only the closed classifier after locking the unknown threshold.

        This permits a service-disjoint calibration checkpoint to set the
        rejection threshold, followed by use of all known calibration services
        for the final closed-set model. The meta detector and threshold remain
        unchanged, so the test services never influence calibration.
        """
        if self.threshold_ is None:
            raise RuntimeError("Lock the unknown threshold with fit() before refitting")
        self.closed.fit(np.asarray(x, dtype=float), np.asarray(y))
        return self

    def _score_features(self, features: np.ndarray) -> np.ndarray:
        return self.meta.predict_proba(self.meta_scaler.transform(features))[:, 1]

    def score_unknown(self, x: np.ndarray) -> np.ndarray:
        return self._score_features(self.closed.summary_features(np.asarray(x, dtype=float)))

    def predict(self, x: np.ndarray, unknown_label: str = "__unknown__") -> OpenSetPredictions:
        if self.threshold_ is None:
            raise RuntimeError("Unknown detector is not fitted")
        details = self.closed.predict_details(x)
        scores = self.score_unknown(x)
        rejected = scores > self.threshold_
        final = details.labels.astype(object)
        final[rejected] = unknown_label
        return OpenSetPredictions(
            closed_labels=details.labels,
            final_labels=final,
            closed_probabilities=details.probabilities,
            unknown_scores=scores,
            rejected=rejected,
            threshold=self.threshold_,
        )

    def evaluate(
        self,
        x_known: np.ndarray,
        y_known: np.ndarray,
        x_unknown: np.ndarray,
        posterior_concentration: float = 0.8,
    ) -> dict[str, Any]:
        known = self.predict(x_known)
        unknown = self.predict(x_unknown)
        known_scores = known.unknown_scores
        unknown_scores = unknown.unknown_scores
        base = unknown_metrics(known_scores, unknown_scores, target_fpr=self.target_known_fpr)
        known_correct = known.closed_labels == np.asarray(y_known)
        base.update(
            {
                "known_closed_accuracy": float(np.mean(known_correct)),
                "known_accept_accuracy": float(
                    np.mean(known_correct[~known.rejected]) if np.any(~known.rejected) else 0.0
                ),
                "known_reject_rate": float(np.mean(known.rejected)),
                "unknown_recall_locked": float(np.mean(unknown.rejected)),
                "false_posterior_concentration_raw": float(
                    np.mean(unknown.closed_probabilities.max(axis=1) >= posterior_concentration)
                ),
                "false_posterior_concentration_accepted": float(
                    np.mean(
                        (unknown.closed_probabilities.max(axis=1) >= posterior_concentration)
                        & (~unknown.rejected)
                    )
                ),
                "false_posterior_concentration_among_accepted": float(
                    np.mean(
                        unknown.closed_probabilities.max(axis=1)[~unknown.rejected]
                        >= posterior_concentration
                    )
                    if np.any(~unknown.rejected)
                    else 0.0
                ),
                "unknown_false_accept_rate_locked": float(np.mean(~unknown.rejected)),
                "oscr": open_set_classification_rate(
                    known_correct,
                    known.closed_probabilities.max(axis=1) - known.unknown_scores,
                    unknown.unknown_scores,
                ),
                "locked_threshold": float(self.threshold_),
            }
        )
        n_known = int(len(known_correct))
        n_unknown = int(len(unknown.rejected))
        _, kacc_lo, kacc_hi = wilson_ci(int(np.sum(known_correct)), n_known)
        _, krej_lo, krej_hi = wilson_ci(int(np.sum(known.rejected)), n_known)
        _, urec_lo, urec_hi = wilson_ci(int(np.sum(unknown.rejected)), n_unknown)
        base.update(
            {
                "known_closed_accuracy_ci": [kacc_lo, kacc_hi],
                "known_reject_rate_ci": [krej_lo, krej_hi],
                "unknown_recall_locked_ci": [urec_lo, urec_hi],
                "wilson_n_known": n_known,
                "wilson_n_unknown": n_unknown,
            }
        )
        return base


class HierarchicalManifoldUnknownDetector:
    """Closed-set hybrid classifier + LoRA-manifold Mahalanobis rejector.

    Protocol (locked primary):

    1. Fit the hybrid closed-set / meta stack on detector-fit known services.
    2. Fit a Ledoit–Wolf Mahalanobis model on the residual/spectral core of the
       same known LoRA manifold.
    3. Lock the unknown threshold on a disjoint known threshold split so that
       known FPR ≈ ``target_known_fpr`` under the Mahalanobis score.
    4. Reject when Mahalanobis exceeds the locked threshold; otherwise emit the
       closed-set structural label.

    Hybrid meta scores remain available as an ablation channel
    (``hybrid_auroc``) but do not enter the locked rejection decision.
    """

    def __init__(
        self,
        random_state: int = 0,
        target_known_fpr: float = 0.05,
        manifold_feature_names: Sequence[str] | None = None,
        rejection_mode: str = "manifold",
    ) -> None:
        self.random_state = random_state
        self.target_known_fpr = target_known_fpr
        self.rejection_mode = str(rejection_mode).lower()
        self.manifold_feature_names = (
            list(manifold_feature_names)
            if manifold_feature_names is not None
            else list(MANIFOLD_CORE_FEATURES)
        )
        self.hybrid = HybridUnknownDetector(
            random_state=random_state,
            target_known_fpr=target_known_fpr,
        )
        self.manifold_scaler = StandardScaler()
        self.manifold_cov = LedoitWolf()
        self.hybrid_threshold_: float | None = None
        self.mahal_threshold_: float | None = None
        self.threshold_: float | None = None
        self.manifold_column_index_: np.ndarray | None = None
        self.feature_names_: list[str] | None = None

    def _select_manifold(self, x: np.ndarray) -> np.ndarray:
        if self.manifold_column_index_ is None:
            raise RuntimeError("Manifold columns are not resolved")
        return np.asarray(x, dtype=float)[:, self.manifold_column_index_]

    def _resolve_manifold_columns(self, n_features: int, feature_names: Sequence[str] | None) -> None:
        if feature_names is None:
            n = min(len(self.manifold_feature_names), n_features)
            self.manifold_column_index_ = np.arange(n, dtype=int)
            self.feature_names_ = [f"f{i}" for i in range(n_features)]
            return
        names = list(feature_names)
        if len(names) != n_features:
            raise ValueError("feature_names length must match x columns")
        present = [c for c in self.manifold_feature_names if c in names]
        if len(present) < 3:
            missing = [c for c in self.manifold_feature_names if c not in names]
            raise ValueError(f"Manifold core features missing from frame: {missing[:5]}...")
        self.manifold_feature_names = present
        self.manifold_column_index_ = np.array(
            [names.index(c) for c in self.manifold_feature_names], dtype=int
        )
        self.feature_names_ = names

    def _mahalanobis(self, x: np.ndarray) -> np.ndarray:
        z = self.manifold_scaler.transform(self._select_manifold(x))
        return self.manifold_cov.mahalanobis(z)

    def score_components(self, x: np.ndarray) -> dict[str, np.ndarray]:
        x = np.asarray(x, dtype=float)
        hybrid = self.hybrid.score_unknown(x)
        mahal = self._mahalanobis(x)
        if self.rejection_mode in {"or", "hierarchical_or"}:
            if self.hybrid_threshold_ is None or self.mahal_threshold_ is None:
                raise RuntimeError("Component thresholds are not locked")
            hybrid_norm = hybrid / max(self.hybrid_threshold_, 1e-12)
            mahal_norm = mahal / max(self.mahal_threshold_, 1e-12)
            combined = np.maximum(hybrid_norm, mahal_norm)
        else:
            # Manifold-primary: raw Mahalanobis is the locked unknown score.
            hybrid_norm = hybrid
            mahal_norm = mahal
            combined = mahal
        return {
            "hybrid": hybrid,
            "mahalanobis": mahal,
            "hybrid_norm": hybrid_norm,
            "mahal_norm": mahal_norm,
            "mahal_ecdf": mahal_norm,
            "combined": combined,
        }

    def fit(
        self,
        x_train: np.ndarray,
        y_train: np.ndarray,
        groups: np.ndarray,
        x_validation_known: np.ndarray | None = None,
        feature_names: Sequence[str] | None = None,
    ) -> "HierarchicalManifoldUnknownDetector":
        x_train = np.asarray(x_train, dtype=float)
        self._resolve_manifold_columns(x_train.shape[1], feature_names)
        self.hybrid.fit(x_train, y_train, groups, x_validation_known=x_validation_known)

        manifold_x = self._select_manifold(x_train)
        self.manifold_scaler.fit(manifold_x)
        z = self.manifold_scaler.transform(manifold_x)
        self.manifold_cov.fit(z)

        if x_validation_known is None:
            val = x_train
        else:
            val = np.asarray(x_validation_known, dtype=float)
        hybrid_val = self.hybrid.score_unknown(val)
        mahal_val = self._mahalanobis(val)

        if self.rejection_mode in {"or", "hierarchical_or"}:
            component_fpr = max(self.target_known_fpr / 2.0, 1e-6)
            self.hybrid_threshold_ = float(
                np.quantile(hybrid_val, 1.0 - component_fpr, method="linear")
            )
            self.mahal_threshold_ = float(
                np.quantile(mahal_val, 1.0 - component_fpr, method="linear")
            )
            # Avoid degenerate zero thresholds on tiny threshold splits.
            self.hybrid_threshold_ = max(self.hybrid_threshold_, 1e-6)
            self.mahal_threshold_ = max(self.mahal_threshold_, 1e-6)
            self.threshold_ = 1.0
        else:
            self.hybrid_threshold_ = float(
                np.quantile(hybrid_val, 1.0 - self.target_known_fpr, method="linear")
            )
            self.mahal_threshold_ = float(
                np.quantile(mahal_val, 1.0 - self.target_known_fpr, method="linear")
            )
            self.threshold_ = float(max(self.mahal_threshold_, 1e-6))
        return self

    def refit_closed(self, x: np.ndarray, y: np.ndarray) -> "HierarchicalManifoldUnknownDetector":
        self.hybrid.refit_closed(x, y)
        return self

    def score_unknown(self, x: np.ndarray) -> np.ndarray:
        return self.score_components(x)["combined"]

    def predict(self, x: np.ndarray, unknown_label: str = "__unknown__") -> OpenSetPredictions:
        if self.threshold_ is None:
            raise RuntimeError("Unknown detector is not fitted")
        details = self.hybrid.closed.predict_details(x)
        scores = self.score_unknown(x)
        rejected = scores > self.threshold_
        final = details.labels.astype(object)
        final[rejected] = unknown_label
        return OpenSetPredictions(
            closed_labels=details.labels,
            final_labels=final,
            closed_probabilities=details.probabilities,
            unknown_scores=scores,
            rejected=rejected,
            threshold=self.threshold_,
        )

    def evaluate(
        self,
        x_known: np.ndarray,
        y_known: np.ndarray,
        x_unknown: np.ndarray,
        posterior_concentration: float = 0.8,
    ) -> dict[str, Any]:
        known = self.predict(x_known)
        unknown = self.predict(x_unknown)
        known_scores = known.unknown_scores
        unknown_scores = unknown.unknown_scores
        base = unknown_metrics(known_scores, unknown_scores, target_fpr=self.target_known_fpr)
        known_correct = known.closed_labels == np.asarray(y_known)

        known_comp = self.score_components(x_known)
        unknown_comp = self.score_components(x_unknown)
        hybrid_only = unknown_metrics(
            known_comp["hybrid"], unknown_comp["hybrid"], target_fpr=self.target_known_fpr
        )
        mahal_only = unknown_metrics(
            known_comp["mahalanobis"],
            unknown_comp["mahalanobis"],
            target_fpr=self.target_known_fpr,
        )

        base.update(
            {
                "rejector": "hierarchical_manifold",
                "rejection_mode": self.rejection_mode,
                "known_closed_accuracy": float(np.mean(known_correct)),
                "known_accept_accuracy": float(
                    np.mean(known_correct[~known.rejected]) if np.any(~known.rejected) else 0.0
                ),
                "known_reject_rate": float(np.mean(known.rejected)),
                "unknown_recall_locked": float(np.mean(unknown.rejected)),
                "false_posterior_concentration_raw": float(
                    np.mean(unknown.closed_probabilities.max(axis=1) >= posterior_concentration)
                ),
                "false_posterior_concentration_accepted": float(
                    np.mean(
                        (unknown.closed_probabilities.max(axis=1) >= posterior_concentration)
                        & (~unknown.rejected)
                    )
                ),
                "false_posterior_concentration_among_accepted": float(
                    np.mean(
                        unknown.closed_probabilities.max(axis=1)[~unknown.rejected]
                        >= posterior_concentration
                    )
                    if np.any(~unknown.rejected)
                    else 0.0
                ),
                "unknown_false_accept_rate_locked": float(np.mean(~unknown.rejected)),
                "oscr": open_set_classification_rate(
                    known_correct,
                    known.closed_probabilities.max(axis=1) - known.unknown_scores,
                    unknown.unknown_scores,
                ),
                "locked_threshold": float(self.threshold_),
                "hybrid_component_threshold": float(self.hybrid_threshold_ or float("nan")),
                "mahal_component_threshold": float(self.mahal_threshold_ or float("nan")),
                "hybrid_auroc": hybrid_only["auroc"],
                "mahal_auroc": mahal_only["auroc"],
                "mahal_ecdf_auroc": mahal_only["auroc"],
                "manifold_features": list(self.manifold_feature_names),
            }
        )
        n_known = int(len(known_correct))
        n_unknown = int(len(unknown.rejected))
        _, kacc_lo, kacc_hi = wilson_ci(int(np.sum(known_correct)), n_known)
        _, krej_lo, krej_hi = wilson_ci(int(np.sum(known.rejected)), n_known)
        _, urec_lo, urec_hi = wilson_ci(int(np.sum(unknown.rejected)), n_unknown)
        base.update(
            {
                "known_closed_accuracy_ci": [kacc_lo, kacc_hi],
                "known_reject_rate_ci": [krej_lo, krej_hi],
                "unknown_recall_locked_ci": [urec_lo, urec_hi],
                "wilson_n_known": n_known,
                "wilson_n_unknown": n_unknown,
            }
        )
        return base
