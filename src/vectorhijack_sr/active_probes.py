from __future__ import annotations

from collections import defaultdict
from typing import Any, Iterable

import numpy as np

from .metrics import softmax


def _residual_stats(victim_logits: Iterable[float], base_logits: Iterable[float]) -> dict[str, float]:
    v = np.asarray(list(victim_logits), dtype=float)
    b = np.asarray(list(base_logits), dtype=float)
    if v.shape != b.shape or v.size == 0:
        return {"l2": 0.0, "margin_delta": 0.0, "entropy_delta": 0.0}
    residual = v - b
    vp = softmax(v)
    bp = softmax(b)
    v_sorted = np.sort(v)
    b_sorted = np.sort(b)
    v_margin = float(v_sorted[-1] - v_sorted[-2]) if v.size > 1 else float(v_sorted[-1])
    b_margin = float(b_sorted[-1] - b_sorted[-2]) if b.size > 1 else float(b_sorted[-1])
    v_ent = float(-(np.clip(vp, 1e-12, 1.0) * np.log(np.clip(vp, 1e-12, 1.0))).sum())
    b_ent = float(-(np.clip(bp, 1e-12, 1.0) * np.log(np.clip(bp, 1e-12, 1.0))).sum())
    return {
        "l2": float(np.linalg.norm(residual)),
        "margin_delta": float(abs(v_margin - b_margin)),
        "entropy_delta": float(abs(v_ent - b_ent)),
    }


def score_queries_known_only(
    rows: list[dict[str, Any]],
    fit_seeds: set[int],
) -> dict[int, float]:
    """Rank-sensitive excitation using only known calibration checkpoints.

    For each query, take the mean residual L2 / margin / entropy stats per known
    structure on detector-fit seeds, then score the query by the standard
    deviation across those structures. Unknown test families never enter
    selection.
    """
    per_query: dict[int, dict[str, list[dict[str, float]]]] = defaultdict(lambda: defaultdict(list))
    for row in rows:
        if not bool(row.get("known", True)):
            continue
        if int(row.get("victim_seed", -1)) not in fit_seeds:
            continue
        stats = _residual_stats(row.get("victim_logits") or [], row.get("base_logits") or [])
        per_query[int(row["query_id"])][str(row.get("structure", ""))].append(stats)

    scores: dict[int, float] = {}
    for query_id, by_structure in per_query.items():
        if len(by_structure) < 2:
            continue
        l2_means = [float(np.mean([s["l2"] for s in xs])) for xs in by_structure.values()]
        margin_means = [
            float(np.mean([s["margin_delta"] for s in xs])) for xs in by_structure.values()
        ]
        entropy_means = [
            float(np.mean([s["entropy_delta"] for s in xs])) for xs in by_structure.values()
        ]
        # Rank/family-sensitive excitation plus raw margin energy.
        scores[query_id] = float(
            np.std(l2_means) + 0.5 * np.std(margin_means) + 0.25 * np.std(entropy_means)
        )
    return scores


def select_query_ids(
    rows: list[dict[str, Any]],
    fit_seeds: set[int],
    keep_fraction: float = 0.25,
    min_queries: int = 80,
    max_queries: int | None = 400,
) -> dict[str, Any]:
    scores = score_queries_known_only(rows, fit_seeds)
    ranked = sorted(scores, key=scores.get, reverse=True)
    universe = sorted({int(row["query_id"]) for row in rows})
    if not ranked:
        return {
            "selected_query_ids": universe,
            "n_candidates": 0,
            "n_selected": len(universe),
            "n_universe": len(universe),
            "keep_fraction": 1.0,
            "selection": "fallback_all",
        }
    n_keep = int(np.ceil(keep_fraction * len(ranked)))
    n_keep = max(n_keep, min(min_queries, len(ranked)))
    if max_queries is not None:
        n_keep = min(n_keep, max_queries, len(ranked))
    selected = ranked[:n_keep]
    return {
        "selected_query_ids": selected,
        "n_candidates": len(ranked),
        "n_selected": len(selected),
        "n_universe": len(universe),
        "keep_fraction": float(keep_fraction),
        "selection": "rank_sensitive_known_only",
        "mean_score_selected": float(np.mean([scores[i] for i in selected])),
        "mean_score_all": float(np.mean(list(scores.values()))),
    }
