from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import StandardScaler

from .metrics import equal_error_rate, fpr_at_tpr, safe_roc_auc


@dataclass
class PairModel:
    scaler: StandardScaler
    classifier: LogisticRegression
    feature_cols: list[str]

    def score(self, pair_features: np.ndarray) -> np.ndarray:
        return self.classifier.predict_proba(self.scaler.transform(pair_features))[:, 1]


class FingerprintExperiment:
    """Held-out exact-lineage fingerprinting with service-level resampling."""

    def __init__(self, random_state: int = 0) -> None:
        self.random_state = random_state

    @staticmethod
    def pair_vector(a: np.ndarray, b: np.ndarray) -> np.ndarray:
        denom = np.linalg.norm(a) * np.linalg.norm(b)
        cosine = float(np.dot(a, b) / denom) if denom > 0 else 0.0
        return np.concatenate(
            [
                np.abs(a - b),
                np.square(a - b),
                a * b,
                np.array([cosine, np.linalg.norm(a - b)], dtype=float),
            ]
        )

    def build_pairs(
        self,
        frame: pd.DataFrame,
        feature_cols: list[str],
        victim_seeds: list[int],
    ) -> pd.DataFrame:
        selected = frame[frame["victim_seed"].isin(victim_seeds)].copy()
        # Collapse each checkpoint/view to one signature. Probe seed is a view,
        # while service_id identifies exact lineage.
        grouped = (
            selected.groupby(
                ["service_id", "structure", "victim_seed", "probe_seed"], as_index=False
            )[feature_cols]
            .mean()
            .reset_index(drop=True)
        )
        records: list[dict[str, Any]] = []
        rows = list(grouped.itertuples(index=False))
        for i, left in enumerate(rows):
            left_vec = np.asarray([getattr(left, c) for c in feature_cols], dtype=float)
            for j in range(i + 1, len(rows)):
                right = rows[j]
                # Positives are distinct views of the same exact checkpoint.
                same_lineage = left.service_id == right.service_id
                # Avoid trivially duplicated same-view pairs.
                if same_lineage and left.probe_seed == right.probe_seed:
                    continue
                # Hard negatives are prioritized; cross-family/task negatives
                # remain available for breakdowns but do not dominate training.
                right_vec = np.asarray([getattr(right, c) for c in feature_cols], dtype=float)
                pair = self.pair_vector(left_vec, right_vec)
                if same_lineage:
                    negative_type = "positive_exact_lineage"
                elif left.structure == right.structure:
                    negative_type = "same_family_rank_new_checkpoint"
                else:
                    negative_type = "different_structure"
                records.append(
                    {
                        "left_service": left.service_id,
                        "right_service": right.service_id,
                        "left_seed": int(left.victim_seed),
                        "right_seed": int(right.victim_seed),
                        "left_structure": left.structure,
                        "right_structure": right.structure,
                        "pair_type": negative_type,
                        "label": int(same_lineage),
                        "cluster_id": "|".join(sorted([left.service_id, right.service_id])),
                        "pair_features": pair.tolist(),
                    }
                )
        result = pd.DataFrame(records)
        if result.empty:
            raise ValueError("No service pairs could be constructed")
        return result

    def fit(self, pairs: pd.DataFrame) -> PairModel:
        # Train the main lineage classifier only on exact-lineage positives and
        # the strict same-structure/new-checkpoint hard negative. Easier
        # cross-family negatives are retained for evaluation breakdowns but do
        # not dominate the decision boundary.
        strict = pairs[
            pairs["pair_type"].isin(["positive_exact_lineage", "same_family_rank_new_checkpoint"])
        ].copy()
        x = np.stack(strict["pair_features"].to_numpy())
        y = strict["label"].to_numpy(dtype=int)
        if len(np.unique(y)) < 2:
            raise ValueError(
                "Fingerprint training requires exact-lineage positives and "
                "same-structure/new-checkpoint hard negatives"
            )
        scaler = StandardScaler().fit(x)
        classifier = LogisticRegression(
            max_iter=3000,
            class_weight="balanced",
            random_state=self.random_state,
        ).fit(scaler.transform(x), y)
        return PairModel(scaler=scaler, classifier=classifier, feature_cols=[])

    def _top1_attribution(
        self,
        signatures: pd.DataFrame,
        feature_cols: list[str],
        model: PairModel,
        victim_seeds: list[int],
    ) -> float:
        selected = signatures[signatures["victim_seed"].isin(victim_seeds)].copy()
        services = sorted(selected["service_id"].unique())
        correct: list[bool] = []
        for service in services:
            views = (
                selected[selected["service_id"] == service]
                .groupby("probe_seed", as_index=False)[feature_cols]
                .mean()
            )
            if len(views) < 2:
                continue
            query = views.iloc[0][feature_cols].to_numpy(dtype=float)
            gallery_scores: list[tuple[str, float]] = []
            for candidate in services:
                candidate_views = (
                    selected[selected["service_id"] == candidate]
                    .groupby("probe_seed", as_index=False)[feature_cols]
                    .mean()
                )
                if candidate_views.empty:
                    continue
                gallery = candidate_views.iloc[-1][feature_cols].to_numpy(dtype=float)
                score = float(model.score(self.pair_vector(query, gallery)[None, :])[0])
                gallery_scores.append((candidate, score))
            predicted = max(gallery_scores, key=lambda item: item[1])[0]
            correct.append(predicted == service)
        return float(np.mean(correct)) if correct else float("nan")

    def _service_bootstrap_auc(
        self,
        pairs: pd.DataFrame,
        n_bootstrap: int,
    ) -> dict[str, Any]:
        """Two-way cluster bootstrap over service checkpoints.

        A pair is dependent on both endpoints. Each bootstrap draw samples
        services with replacement and weights a pair by the product of the two
        endpoint multiplicities. This avoids treating the O(n^2) pair table as
        independent observations.
        """
        services = sorted(
            set(pairs["left_service"].astype(str)).union(set(pairs["right_service"].astype(str)))
        )
        if len(services) < 2:
            raise ValueError("Service bootstrap requires at least two checkpoints")
        y = pairs["label"].to_numpy(dtype=int)
        scores = pairs["score"].to_numpy(dtype=float)
        left = pairs["left_service"].astype(str).to_numpy()
        right = pairs["right_service"].astype(str).to_numpy()
        rng = np.random.default_rng(self.random_state)
        estimates: list[float] = []
        for _ in range(n_bootstrap):
            sampled = rng.choice(services, size=len(services), replace=True)
            counts = {service: int(np.sum(sampled == service)) for service in services}
            weights = np.asarray(
                [counts.get(a, 0) * counts.get(b, 0) for a, b in zip(left, right, strict=True)],
                dtype=float,
            )
            mask = weights > 0
            if np.unique(y[mask]).size < 2:
                continue
            estimates.append(
                float(roc_auc_score(y[mask], scores[mask], sample_weight=weights[mask]))
            )
        if not estimates:
            return {
                "ci_lower": float("nan"),
                "ci_upper": float("nan"),
                "service_count": len(services),
            }
        values = np.asarray(estimates, dtype=float)
        return {
            "ci_lower": float(np.quantile(values, 0.025)),
            "ci_upper": float(np.quantile(values, 0.975)),
            "service_count": len(services),
        }

    def run_from_signatures(
        self,
        signatures: pd.DataFrame,
        feature_cols: list[str],
        calibration_victim_seeds: list[int],
        validation_victim_seeds: list[int],
        test_victim_seeds: list[int],
        n_bootstrap: int = 2000,
        target_tmr: float = 0.95,
        primary_structure: str | None = None,
    ) -> dict[str, Any]:
        """Train, lock an operating point on validation services, then test.

        Victim-training seeds/checkpoints in the three partitions must be
        disjoint.  The validation partition is used only to select the FMR
        operating threshold; no test pair participates in threshold selection.
        """
        train_pairs = self.build_pairs(signatures, feature_cols, calibration_victim_seeds)
        validation_pairs = self.build_pairs(signatures, feature_cols, validation_victim_seeds)
        test_pairs = self.build_pairs(signatures, feature_cols, test_victim_seeds)
        model = self.fit(train_pairs)

        validation_pairs = validation_pairs.copy()
        validation_pairs["score"] = model.score(
            np.stack(validation_pairs["pair_features"].to_numpy())
        )
        y_validation = validation_pairs["label"].to_numpy(dtype=int)
        validation_scores = validation_pairs["score"].to_numpy(dtype=float)
        validation_fmr, locked_threshold = fpr_at_tpr(
            y_validation, validation_scores, target_tpr=target_tmr
        )

        test_pairs = test_pairs.copy()
        test_pairs["score"] = model.score(np.stack(test_pairs["pair_features"].to_numpy()))
        y = test_pairs["label"].to_numpy(dtype=int)
        scores = test_pairs["score"].to_numpy(dtype=float)
        positives_mask = y == 1
        negatives_mask = y == 0
        locked_tmr = float(np.mean(scores[positives_mask] >= locked_threshold))
        locked_fmr = float(np.mean(scores[negatives_mask] >= locked_threshold))
        eer, eer_threshold = equal_error_rate(y, scores)
        oracle_fmr95, oracle_threshold = fpr_at_tpr(y, scores, target_tpr=target_tmr)
        metrics: dict[str, Any] = {
            "pair_auc": safe_roc_auc(y, scores),
            "macro_top1": self._top1_attribution(
                signatures, feature_cols, model, test_victim_seeds
            ),
            "eer": eer,
            "eer_threshold": eer_threshold,
            "locked_threshold": float(locked_threshold),
            "validation_fmr_at_target_tmr": float(validation_fmr),
            "test_tmr_at_locked_threshold": locked_tmr,
            "test_fmr_at_locked_threshold": locked_fmr,
            "target_tmr": float(target_tmr),
            "oracle_test_fmr_at_target_tmr_diagnostic": float(oracle_fmr95),
            "oracle_test_threshold_diagnostic": float(oracle_threshold),
            "threshold_selected_on_validation_services": True,
            "calibration_services": int(
                signatures[signatures["victim_seed"].isin(calibration_victim_seeds)][
                    "service_id"
                ].nunique()
            ),
            "validation_services": int(
                signatures[signatures["victim_seed"].isin(validation_victim_seeds)][
                    "service_id"
                ].nunique()
            ),
            "test_services": int(
                signatures[signatures["victim_seed"].isin(test_victim_seeds)][
                    "service_id"
                ].nunique()
            ),
        }

        positives = test_pairs[test_pairs["label"] == 1]
        for pair_type, negatives in test_pairs[test_pairs["label"] == 0].groupby("pair_type"):
            contrast = pd.concat([positives, negatives], ignore_index=True)
            metrics[f"auc_vs_{pair_type}"] = safe_roc_auc(contrast["label"], contrast["score"])
            metrics[f"mean_score_{pair_type}"] = float(negatives["score"].mean())
            metrics[f"locked_fmr_{pair_type}"] = float(
                np.mean(negatives["score"].to_numpy(dtype=float) >= locked_threshold)
            )
        metrics["mean_score_positive_exact_lineage"] = float(positives["score"].mean())

        boot = self._service_bootstrap_auc(test_pairs, n_bootstrap=n_bootstrap)
        metrics["pair_auc_ci_lower"] = boot["ci_lower"]
        metrics["pair_auc_ci_upper"] = boot["ci_upper"]
        metrics["bootstrap_services"] = boot["service_count"]
        metrics["bootstrap_unit"] = "service_checkpoint"
        if primary_structure:
            primary = str(primary_structure)
            subset = test_pairs[
                (test_pairs["left_structure"].astype(str) == primary)
                & (test_pairs["right_structure"].astype(str) == primary)
            ].copy()
            if not subset.empty and subset["label"].nunique() > 1:
                y_p = subset["label"].to_numpy(dtype=int)
                s_p = subset["score"].to_numpy(dtype=float)
                pos = y_p == 1
                neg = y_p == 0
                eer_p, _ = equal_error_rate(y_p, s_p)
                fmr_p, _ = fpr_at_tpr(y_p, s_p, target_tpr=target_tmr)
                boot_p = self._service_bootstrap_auc(subset, n_bootstrap=n_bootstrap)
                metrics["primary_structure"] = primary
                metrics["primary_pair_auc"] = safe_roc_auc(y_p, s_p)
                metrics["primary_eer"] = eer_p
                metrics["primary_oracle_fmr_at_target_tmr"] = float(fmr_p)
                metrics["primary_test_fmr_at_locked_threshold"] = float(
                    np.mean(s_p[neg] >= locked_threshold) if np.any(neg) else float("nan")
                )
                metrics["primary_test_tmr_at_locked_threshold"] = float(
                    np.mean(s_p[pos] >= locked_threshold) if np.any(pos) else float("nan")
                )
                metrics["primary_pair_auc_ci_lower"] = boot_p["ci_lower"]
                metrics["primary_pair_auc_ci_upper"] = boot_p["ci_upper"]
                metrics["primary_bootstrap_services"] = boot_p["service_count"]
        return {
            "metrics": metrics,
            "pairs": test_pairs,
            "validation_pairs": validation_pairs,
            "model": model,
        }
