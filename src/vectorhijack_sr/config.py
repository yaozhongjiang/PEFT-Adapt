from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path
from typing import Any

import yaml


class ConfigError(ValueError):
    """Raised when a configuration is missing a required field."""


def load_config(path: str | Path) -> dict[str, Any]:
    path = Path(path)
    with path.open("r", encoding="utf-8") as handle:
        cfg = yaml.safe_load(handle)
    if not isinstance(cfg, dict):
        raise ConfigError(f"Configuration must be a mapping: {path}")
    cfg = copy.deepcopy(cfg)
    cfg.setdefault("experiment", {})
    cfg["experiment"].setdefault("name", path.stem)
    cfg["experiment"].setdefault("output_root", "artifact")
    cfg["experiment"].setdefault("local_only", False)
    cfg["experiment"]["config_path"] = str(path.resolve())
    cfg["experiment"]["config_sha256"] = hashlib.sha256(path.read_bytes()).hexdigest()
    return cfg


def require(cfg: dict[str, Any], dotted_key: str) -> Any:
    value: Any = cfg
    for part in dotted_key.split("."):
        if not isinstance(value, dict) or part not in value:
            raise ConfigError(f"Missing required configuration key: {dotted_key}")
        value = value[part]
    return value


def get(cfg: dict[str, Any], dotted_key: str, default: Any = None) -> Any:
    value: Any = cfg
    for part in dotted_key.split("."):
        if not isinstance(value, dict) or part not in value:
            return default
        value = value[part]
    return value


def canonical_json(cfg: dict[str, Any]) -> str:
    return json.dumps(cfg, sort_keys=True, separators=(",", ":"), default=str)
