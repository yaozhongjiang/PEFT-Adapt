from __future__ import annotations

import argparse
import json
from typing import Any, Callable

from .base_mismatch import run_base_mismatch
from .collect import collect_grid
from .config import load_config
from .lineage import run_lineage
from .io import _sanitize_json
from .llama_mechanism import run_llama_mechanism
from .open_set import run_open_set
from .redeploy import run_redeploy
from .synthetic import run_synthetic_smoke
from .victims import train_victim_grid


def _config_parser(subparsers: Any, name: str, help_text: str) -> argparse.ArgumentParser:
    parser = subparsers.add_parser(name, help=help_text)
    parser.add_argument("--config", required=True, help="YAML experiment configuration")
    return parser


def _print_result(result: Any) -> None:
    if isinstance(result, dict):
        print(json.dumps(_sanitize_json(result), indent=2, allow_nan=False, default=str))
    else:
        print(result)


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        prog="vectorhijack-sr",
        description="Controlled local experiments for PEFT structural leakage measurement",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)
    _config_parser(subparsers, "smoke", "Run the offline synthetic end-to-end smoke test")
    _config_parser(subparsers, "train-victims", "Train independent local victim checkpoints")
    collect_parser = _config_parser(subparsers, "collect", "Collect paired victim/base outputs")
    collect_parser.add_argument("--grid", help="Optional victim_grid.json path")
    _config_parser(subparsers, "open-set", "Run closed/open-set structural inference")
    _config_parser(
        subparsers, "base-mismatch", "Run fixed-victim real local-base mismatch experiments"
    )
    _config_parser(subparsers, "lineage", "Run held-out exact-lineage fingerprinting")
    _config_parser(subparsers, "redeploy", "Run per-seed cost-matched redeployability experiments")
    _config_parser(subparsers, "llama-mechanism", "Run trajectory-alignment mechanism diagnostics")
    run_all = subparsers.add_parser("run-all", help="Execute a YAML list of commands/configs")
    run_all.add_argument("--plan", required=True, help="YAML file containing a `runs` list")

    args = parser.parse_args(argv)
    if args.command == "run-all":
        plan = load_config(args.plan)
        results = []
        dispatch = _dispatch_table()
        for item in plan.get("runs", []):
            command = item["command"]
            if command not in dispatch:
                raise SystemExit(f"Unknown plan command: {command}")
            cfg = load_config(item["config"])
            results.append(
                {"command": command, "config": item["config"], "result": dispatch[command](cfg)}
            )
        _print_result({"runs": results})
        return

    cfg = load_config(args.config)
    if args.command == "collect":
        result = collect_grid(cfg, grid_path=args.grid)
        _print_result({"rows": len(result)})
        return
    result = _dispatch_table()[args.command](cfg)
    _print_result(result)


def _dispatch_table() -> dict[str, Callable[[dict[str, Any]], Any]]:
    return {
        "smoke": run_synthetic_smoke,
        "train-victims": train_victim_grid,
        "collect": lambda cfg: {"rows": len(collect_grid(cfg))},
        "open-set": run_open_set,
        "base-mismatch": run_base_mismatch,
        "lineage": run_lineage,
        "redeploy": run_redeploy,
        "llama-mechanism": run_llama_mechanism,
    }


if __name__ == "__main__":
    main()
