from __future__ import annotations

from collections.abc import Callable
from typing import Any

import numpy as np
import pandas as pd


def cluster_bootstrap(
    frame: pd.DataFrame,
    cluster_col: str,
    statistic: Callable[[pd.DataFrame], float],
    n_bootstrap: int = 2000,
    seed: int = 0,
    alpha: float = 0.05,
) -> dict[str, Any]:
    """Bootstrap whole services/checkpoints, never individual probes."""
    if cluster_col not in frame:
        raise KeyError(f"Missing cluster column: {cluster_col}")
    clusters = frame[cluster_col].dropna().unique()
    if len(clusters) == 0:
        raise ValueError("No clusters available for bootstrap")
    rng = np.random.default_rng(seed)
    estimates: list[float] = []
    grouped = {cluster: frame[frame[cluster_col] == cluster] for cluster in clusters}
    for _ in range(n_bootstrap):
        sampled = rng.choice(clusters, size=len(clusters), replace=True)
        pieces = []
        for draw_index, cluster in enumerate(sampled):
            piece = grouped[cluster].copy()
            piece["__bootstrap_cluster"] = f"{cluster}:{draw_index}"
            pieces.append(piece)
        sample = pd.concat(pieces, ignore_index=True)
        estimates.append(float(statistic(sample)))
    estimate = float(statistic(frame))
    lower = float(np.nanquantile(estimates, alpha / 2))
    upper = float(np.nanquantile(estimates, 1 - alpha / 2))
    return {
        "estimate": estimate,
        "ci_lower": lower,
        "ci_upper": upper,
        "n_bootstrap": n_bootstrap,
        "cluster_count": int(len(clusters)),
        "samples": estimates,
    }
