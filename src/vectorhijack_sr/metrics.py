from __future__ import annotations

import math
from typing import Iterable

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    roc_auc_score,
    roc_curve,
)


def softmax(x: np.ndarray, axis: int = -1) -> np.ndarray:
    z = x - np.max(x, axis=axis, keepdims=True)
    e = np.exp(z)
    return e / np.clip(e.sum(axis=axis, keepdims=True), 1e-12, None)


def entropy(prob: np.ndarray, axis: int = -1) -> np.ndarray:
    p = np.clip(prob, 1e-12, 1.0)
    return -(p * np.log(p)).sum(axis=axis)


def safe_roc_auc(y_true: Iterable[int], scores: Iterable[float]) -> float:
    y = np.asarray(list(y_true), dtype=int)
    s = np.asarray(list(scores), dtype=float)
    if np.unique(y).size < 2:
        return float("nan")
    return float(roc_auc_score(y, s))


def tpr_at_fpr(
    y_true: np.ndarray, scores: np.ndarray, target_fpr: float = 0.05
) -> tuple[float, float]:
    fpr, tpr, thresholds = roc_curve(y_true, scores)
    finite = np.isfinite(thresholds)
    fpr, tpr, thresholds = fpr[finite], tpr[finite], thresholds[finite]
    if thresholds.size == 0:
        return 0.0, float(np.max(scores) + np.finfo(float).eps)
    valid = np.where(fpr <= target_fpr)[0]
    if valid.size == 0:
        idx = int(np.argmin(fpr))
    else:
        idx = int(valid[np.argmax(tpr[valid])])
    return float(tpr[idx]), float(thresholds[idx])


def fpr_at_tpr(
    y_true: np.ndarray, scores: np.ndarray, target_tpr: float = 0.95
) -> tuple[float, float]:
    fpr, tpr, thresholds = roc_curve(y_true, scores)
    finite = np.isfinite(thresholds)
    fpr, tpr, thresholds = fpr[finite], tpr[finite], thresholds[finite]
    if thresholds.size == 0:
        return 1.0, float(np.max(scores) + np.finfo(float).eps)
    valid = np.where(tpr >= target_tpr)[0]
    if valid.size == 0:
        idx = int(np.argmax(tpr))
    else:
        idx = int(valid[np.argmin(fpr[valid])])
    return float(fpr[idx]), float(thresholds[idx])


def equal_error_rate(y_true: np.ndarray, scores: np.ndarray) -> tuple[float, float]:
    fpr, tpr, thresholds = roc_curve(y_true, scores)
    finite = np.isfinite(thresholds)
    fpr, tpr, thresholds = fpr[finite], tpr[finite], thresholds[finite]
    if thresholds.size == 0:
        return 1.0, float(np.max(scores) + np.finfo(float).eps)
    fnr = 1.0 - tpr
    idx = int(np.argmin(np.abs(fpr - fnr)))
    return float((fpr[idx] + fnr[idx]) / 2.0), float(thresholds[idx])


def open_set_classification_rate(
    known_correct: np.ndarray,
    known_confidence: np.ndarray,
    unknown_score: np.ndarray,
) -> float:
    """Numerical OSCR area following the CCR-vs-FPR definition.

    `known_confidence` must increase with acceptance as known. `unknown_score`
    must increase with unknownness. We map both to a shared rejection axis.
    """
    known_correct = np.asarray(known_correct, dtype=bool)
    known_accept_score = np.asarray(known_confidence, dtype=float)
    unknown_accept_score = -np.asarray(unknown_score, dtype=float)
    thresholds = np.unique(np.concatenate([known_accept_score, unknown_accept_score]))
    if thresholds.size < 2:
        return 0.0
    ccr: list[float] = []
    fpr: list[float] = []
    for threshold in thresholds:
        ccr.append(float(np.mean(known_correct & (known_accept_score >= threshold))))
        fpr.append(float(np.mean(unknown_accept_score >= threshold)))
    order = np.argsort(fpr)
    return float(np.trapezoid(np.asarray(ccr)[order], np.asarray(fpr)[order]))


def classification_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    return {"accuracy": float(accuracy_score(y_true, y_pred))}


def unknown_metrics(
    known_scores: np.ndarray,
    unknown_scores: np.ndarray,
    target_fpr: float = 0.05,
) -> dict[str, float]:
    y = np.concatenate(
        [np.zeros(len(known_scores), dtype=int), np.ones(len(unknown_scores), dtype=int)]
    )
    s = np.concatenate([known_scores, unknown_scores])
    tpr, threshold = tpr_at_fpr(y, s, target_fpr=target_fpr)
    return {
        "auroc": safe_roc_auc(y, s),
        "aupr": float(average_precision_score(y, s)),
        "tpr_at_target_fpr": tpr,
        "threshold_at_target_fpr": threshold,
    }


def kl_divergence(p_logits: np.ndarray, q_logits: np.ndarray, axis: int = -1) -> np.ndarray:
    p = softmax(p_logits, axis=axis)
    q = softmax(q_logits, axis=axis)
    return (p * (np.log(np.clip(p, 1e-12, 1.0)) - np.log(np.clip(q, 1e-12, 1.0)))).sum(axis=axis)


def mean_ci(values: Iterable[float], z: float = 1.96) -> tuple[float, float, float]:
    arr = np.asarray(list(values), dtype=float)
    arr = arr[np.isfinite(arr)]
    if arr.size == 0:
        return float("nan"), float("nan"), float("nan")
    mean = float(arr.mean())
    if arr.size == 1:
        return mean, mean, mean
    se = float(arr.std(ddof=1) / math.sqrt(arr.size))
    return mean, mean - z * se, mean + z * se


def wilson_ci(successes: int, n: int, z: float = 1.96) -> tuple[float, float, float]:
    """Wilson score interval for a binomial proportion.

    Returns ``(point, lower, upper)``. When ``n == 0`` all values are NaN.
    """
    if n <= 0:
        return float("nan"), float("nan"), float("nan")
    successes = int(successes)
    n = int(n)
    if successes < 0 or successes > n:
        raise ValueError("successes must lie in [0, n]")
    phat = successes / n
    z2 = z * z
    denom = 1.0 + z2 / n
    center = (phat + z2 / (2.0 * n)) / denom
    half = (z / denom) * math.sqrt((phat * (1.0 - phat) / n) + (z2 / (4.0 * n * n)))
    return float(phat), float(max(0.0, center - half)), float(min(1.0, center + half))
