#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from vectorhijack_sr.io import _sanitize_json


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--summaries", default="artifact/summaries")
    parser.add_argument("--output", default="artifact/reports/all_summaries.json")
    args = parser.parse_args()
    summaries = Path(args.summaries)
    result = {}
    for path in sorted(summaries.glob("*.csv")):
        frame = pd.read_csv(path)
        result[path.name] = {
            "rows": len(frame),
            "columns": list(frame.columns),
            "numeric_mean": frame.select_dtypes("number").mean(numeric_only=True).to_dict(),
        }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(_sanitize_json(result), indent=2, allow_nan=False) + "\n", encoding="utf-8"
    )
    print(output)


if __name__ == "__main__":
    main()
