#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json

import pandas as pd

from adaptertrace.bootstrap import cluster_bootstrap
from adaptertrace.io import _sanitize_json


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("csv")
    parser.add_argument("--cluster", required=True)
    parser.add_argument("--metric", required=True)
    parser.add_argument("--n", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=7)
    args = parser.parse_args()
    frame = pd.read_csv(args.csv)
    result = cluster_bootstrap(
        frame,
        args.cluster,
        lambda sample: float(sample[args.metric].mean()),
        n_bootstrap=args.n,
        seed=args.seed,
    )
    result.pop("samples", None)
    print(json.dumps(_sanitize_json(result), indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
