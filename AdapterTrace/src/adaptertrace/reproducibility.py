from __future__ import annotations

import hashlib
import importlib.metadata
import os
import platform
import random
import socket
import subprocess
import sys
from pathlib import Path
from typing import Any

import numpy as np


def set_global_seed(seed: int, deterministic: bool = True) -> None:
    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch

        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
        if deterministic:
            torch.use_deterministic_algorithms(True, warn_only=True)
            os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    except ImportError:
        pass


def sha256_file(path: str | Path) -> str:
    h = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def git_commit(root: str | Path = ".") -> str | None:
    try:
        return subprocess.check_output(
            ["git", "-C", str(root), "rev-parse", "HEAD"],
            text=True,
            stderr=subprocess.DEVNULL,
        ).strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def package_versions(names: list[str] | None = None) -> dict[str, str | None]:
    names = names or [
        "torch",
        "transformers",
        "datasets",
        "peft",
        "accelerate",
        "numpy",
        "scipy",
        "pandas",
        "scikit-learn",
        "bitsandbytes",
    ]
    versions: dict[str, str | None] = {}
    for name in names:
        try:
            versions[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            versions[name] = None
    return versions


def environment_snapshot() -> dict[str, Any]:
    gpu: list[dict[str, Any]] = []
    try:
        import torch

        for idx in range(torch.cuda.device_count()):
            props = torch.cuda.get_device_properties(idx)
            gpu.append(
                {
                    "index": idx,
                    "name": props.name,
                    "total_memory": int(props.total_memory),
                    "compute_capability": [props.major, props.minor],
                }
            )
        torch_info = {
            "version": torch.__version__,
            "cuda_version": torch.version.cuda,
            "cudnn_version": torch.backends.cudnn.version(),
            "cuda_available": torch.cuda.is_available(),
            "gpus": gpu,
        }
    except ImportError:
        torch_info = {"available": False}
    return {
        "python": sys.version,
        "platform": platform.platform(),
        "machine": platform.machine(),
        "processor": platform.processor(),
        "hostname": socket.gethostname(),
        "packages": package_versions(),
        "torch": torch_info,
    }
