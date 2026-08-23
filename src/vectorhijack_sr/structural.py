from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
from sklearn.covariance import LedoitWolf
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

from .metrics import entropy


@dataclass
class ClosedSetOutput:
    labels: np.ndarray
    probabilities: np.ndarray
    logits: np.ndarray
    min_mahalanobis: np.ndarray
    predicted_distance: np.ndarray
    conformal_p: np.ndarray


class StructuralClassifier:
    """Closed-set classifier with class-conditional shrinkage distances."""

    def __init__(self, random_state: int = 0, c: float = 1.0) -> None:
        self.random_state = random_state
        self.scaler = StandardScaler()
        self.classifier = LogisticRegression(
            C=c,
            max_iter=3000,
            class_weight="balanced",
            random_state=random_state,
        )
        self.classes_: np.ndarray | None = None
        self.centroids_: dict[Any, np.ndarray] = {}
        self.precisions_: dict[Any, np.ndarray] = {}
        self.calibration_distances_: dict[Any, np.ndarray] = {}

    def fit(self, x: np.ndarray, y: np.ndarray) -> "StructuralClassifier":
        x = np.asarray(x, dtype=float)
        y = np.asarray(y)
        if len(np.unique(y)) < 2:
            raise ValueError("Structural classifier requires at least two classes")
        z = self.scaler.fit_transform(x)
        self.classifier.fit(z, y)
        self.classes_ = self.classifier.classes_
        for label in self.classes_:
            group = z[y == label]
            centroid = group.mean(axis=0)
            if len(group) >= 3:
                precision = LedoitWolf().fit(group).precision_
            else:
                precision = np.eye(group.shape[1])
            distances = _mahalanobis(group, centroid, precision)
            self.centroids_[label] = centroid
            self.precisions_[label] = precision
            self.calibration_distances_[label] = np.sort(distances)
        return self

    def decision_function(self, x: np.ndarray) -> np.ndarray:
        z = self.scaler.transform(np.asarray(x, dtype=float))
        scores = self.classifier.decision_function(z)
        if scores.ndim == 1:
            scores = np.column_stack([-scores, scores])
        return scores

    def predict_proba(self, x: np.ndarray) -> np.ndarray:
        z = self.scaler.transform(np.asarray(x, dtype=float))
        return self.classifier.predict_proba(z)

    def transform(self, x: np.ndarray) -> np.ndarray:
        return self.scaler.transform(np.asarray(x, dtype=float))

    def distances(self, x: np.ndarray) -> np.ndarray:
        if self.classes_ is None:
            raise RuntimeError("Classifier is not fitted")
        z = self.transform(x)
        return np.column_stack(
            [_mahalanobis(z, self.centroids_[label], self.precisions_[label]) for label in self.classes_]
        )

    def predict_details(self, x: np.ndarray) -> ClosedSetOutput:
        if self.classes_ is None:
            raise RuntimeError("Classifier is not fitted")
        probabilities = self.predict_proba(x)
        logits = self.decision_function(x)
        distances = self.distances(x)
        pred_idx = probabilities.argmax(axis=1)
        labels = self.classes_[pred_idx]
        pred_distance = distances[np.arange(len(distances)), pred_idx]
        min_distance = distances.min(axis=1)
        conformal = np.empty(len(labels), dtype=float)
        for i, (label, distance) in enumerate(zip(labels, pred_distance, strict=True)):
            calibration = self.calibration_distances_[label]
            conformal[i] = (1.0 + np.sum(calibration >= distance)) / (len(calibration) + 1.0)
        return ClosedSetOutput(
            labels=labels,
            probabilities=probabilities,
            logits=logits,
            min_mahalanobis=min_distance,
            predicted_distance=pred_distance,
            conformal_p=conformal,
        )

    def summary_features(self, x: np.ndarray) -> np.ndarray:
        details = self.predict_details(x)
        probs = details.probabilities
        sorted_probs = np.sort(probs, axis=1)
        margin = sorted_probs[:, -1] - sorted_probs[:, -2] if probs.shape[1] > 1 else sorted_probs[:, -1]
        energy = -np.log(np.exp(details.logits - details.logits.max(axis=1, keepdims=True)).sum(axis=1)) - details.logits.max(axis=1)
        distances = self.distances(x)
        sorted_dist = np.sort(distances, axis=1)
        distance_ratio = sorted_dist[:, 0] / np.clip(sorted_dist[:, 1], 1e-9, None) if distances.shape[1] > 1 else sorted_dist[:, 0]
        return np.column_stack(
            [
                1.0 - probs.max(axis=1),
                entropy(probs),
                -margin,
                details.min_mahalanobis,
                details.predicted_distance,
                1.0 - details.conformal_p,
                energy,
                distance_ratio,
            ]
        )


def _mahalanobis(x: np.ndarray, centroid: np.ndarray, precision: np.ndarray) -> np.ndarray:
    delta = x - centroid
    squared = np.einsum("ij,jk,ik->i", delta, precision, delta)
    return np.sqrt(np.clip(squared, 0.0, None))
