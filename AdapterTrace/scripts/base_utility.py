#!/usr/bin/env python3
"""Held-out utility of the bare public base (no adapter), the zero point for adapter value."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from palimpsest_evasion import task_utility  # noqa: E402
from palimpsest_lineage import _train_dataset  # noqa: E402

from adaptertrace.config import get, load_config  # noqa: E402
from adaptertrace.hf import infer_device, load_model  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    args = parser.parse_args()
    cfg = load_config(args.config)
    _, tokenizer = _train_dataset(cfg)
    device = infer_device(get(cfg, "inference.device"))
    model = load_model(cfg["model"]).to(device)
    value = task_utility(model, cfg, tokenizer, device, 8)
    out = Path(get(cfg, "experiment.output_root", "artifact")) / "reports" / f"{cfg['experiment']['name']}_base_utility.json"
    out.write_text(json.dumps({"base_utility": value}, indent=2))
    print(f"== base utility {value:.4f}")


if __name__ == "__main__":
    main()
