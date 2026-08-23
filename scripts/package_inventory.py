#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

EXCLUDE_PARTS = {
    ".git",
    ".pytest_cache",
    "__pycache__",
    "build",
    "dist",
    ".venv",
}
EXCLUDE_NAMES = {"SHA256SUMS", "PACKAGE_INVENTORY.json"}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[1])
    args = parser.parse_args()
    root = args.root.resolve()
    files: list[dict[str, object]] = []
    errors: list[str] = []
    for path in sorted(root.rglob("*")):
        relative = path.relative_to(root)
        if any(part in EXCLUDE_PARTS or part.endswith(".egg-info") for part in relative.parts):
            continue
        if path.name in EXCLUDE_NAMES:
            continue
        if path.is_symlink():
            errors.append(f"symlink not allowed in artifact package: {relative}")
            continue
        if not path.is_file():
            continue
        files.append(
            {
                "path": relative.as_posix(),
                "bytes": path.stat().st_size,
                "sha256": sha256(path),
            }
        )
    inventory = {
        "root": root.name,
        "file_count": len(files),
        "total_bytes": sum(int(row["bytes"]) for row in files),
        "files": files,
        "errors": errors,
        "passed": not errors,
    }
    (root / "PACKAGE_INVENTORY.json").write_text(json.dumps(inventory, indent=2) + "\n")
    (root / "SHA256SUMS").write_text("".join(f"{row['sha256']}  {row['path']}\n" for row in files))
    print(
        json.dumps(
            {key: inventory[key] for key in ["file_count", "total_bytes", "errors", "passed"]},
            indent=2,
        )
    )
    if errors:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
