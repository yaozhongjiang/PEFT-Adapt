from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd

from .io import write_csv, write_json


def save_result_bundle(
    output_root: str | Path,
    experiment_name: str,
    raw_rows: list[dict[str, Any]],
    summary_rows: list[dict[str, Any]],
    manifest: dict[str, Any],
    metrics: dict[str, Any],
) -> dict[str, str]:
    from .io import write_jsonl

    root = Path(output_root)
    paths = {
        "raw": str(root / "raw" / f"{experiment_name}.jsonl"),
        "summary": str(root / "summaries" / f"{experiment_name}.csv"),
        "manifest": str(root / "manifests" / f"{experiment_name}.json"),
        "report": str(root / "reports" / f"{experiment_name}.json"),
    }
    write_jsonl(paths["raw"], raw_rows)
    write_csv(paths["summary"], summary_rows)
    write_json(paths["manifest"], manifest)
    write_json(paths["report"], metrics)
    return paths


def latex_table_from_csv(csv_path: str | Path, columns: list[str] | None = None) -> str:
    frame = pd.read_csv(csv_path)
    if columns:
        frame = frame[columns]
    return frame.to_latex(index=False, escape=True, float_format=lambda x: f"{x:.3f}")
