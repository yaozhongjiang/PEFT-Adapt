#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
from typing import Any

from jsonschema import Draft202012Validator

PROFILES = {
    "smoke": ["synthetic"],
    "paper": ["open_set", "base_mismatch", "lineage", "redeploy"],
    "all": ["open_set", "base_mismatch", "lineage", "redeploy", "llama"],
}


def _reject_constant(value: str) -> None:
    raise ValueError(f"non-finite JSON constant {value}")


def load_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle, parse_constant=_reject_constant)


def _finite_vector(value: Any) -> bool:
    return (
        isinstance(value, list)
        and bool(value)
        and all(isinstance(item, (int, float)) and math.isfinite(float(item)) for item in value)
    )


def validate_jsonl(
    path: Path,
    validator: Draft202012Validator,
    seen_keys: set[tuple[Any, ...]],
) -> tuple[int, list[str], int]:
    count = 0
    duplicates = 0
    errors: list[str] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            count += 1
            try:
                row = json.loads(line, parse_constant=_reject_constant)
            except (json.JSONDecodeError, ValueError) as exc:
                errors.append(f"{path}:{line_number}: invalid JSON: {exc}")
                continue
            for error in validator.iter_errors(row):
                errors.append(f"{path}:{line_number}: {error.message}")
            if not isinstance(row, dict):
                errors.append(f"{path}:{line_number}: row is not an object")
                continue
            for key in ("victim_logits", "base_logits"):
                if key in row and not _finite_vector(row[key]):
                    errors.append(f"{path}:{line_number}: {key} must be a finite non-empty vector")
            if "victim_logits" in row and "base_logits" in row:
                if len(row["victim_logits"]) != len(row["base_logits"]):
                    errors.append(f"{path}:{line_number}: victim/base logit lengths differ")
            uniqueness = (
                row.get("run_id"),
                row.get("service_id"),
                row.get("victim_seed"),
                row.get("probe_seed"),
                row.get("query_id"),
                row.get("mismatch_condition"),
                row.get("observation_mode"),
                row.get("template_id"),
            )
            if uniqueness in seen_keys:
                duplicates += 1
                errors.append(f"{path}:{line_number}: duplicate raw query identity {uniqueness}")
            seen_keys.add(uniqueness)
            if "llama" in path.name:
                if row.get("mechanism_only") is not True:
                    errors.append(
                        f"{path}:{line_number}: Llama result is not marked mechanism_only"
                    )
                if "not a general" not in str(row.get("claim_scope", "")):
                    errors.append(f"{path}:{line_number}: Llama claim_scope is too broad")
    return count, errors, duplicates


def _check_disjoint_manifest(manifest: dict[str, Any], path: Path, errors: list[str]) -> None:
    candidates = [manifest]
    config = manifest.get("config")
    if isinstance(config, dict):
        candidates.append(config)
    for candidate in candidates:
        split = candidate.get("split", candidate.get("splits", {}))
        if not isinstance(split, dict):
            continue
        partitions: dict[str, set[int]] = {}
        for name in (
            "calibration_victim_seeds",
            "detector_fit_victim_seeds",
            "threshold_victim_seeds",
            "validation_victim_seeds",
            "test_victim_seeds",
        ):
            if name in split:
                partitions[name] = {int(value) for value in split[name]}
        test = partitions.get("test_victim_seeds", set())
        for name, values in partitions.items():
            if name != "test_victim_seeds" and values.intersection(test):
                errors.append(f"{path}: {name} overlaps test_victim_seeds")
        train = partitions.get("calibration_victim_seeds")
        validation = partitions.get("validation_victim_seeds")
        if train is not None and validation is not None and train.intersection(validation):
            errors.append(f"{path}: calibration and validation victim seeds overlap")
        fit = partitions.get("detector_fit_victim_seeds")
        threshold = partitions.get("threshold_victim_seeds")
        if fit is not None and threshold is not None and fit.intersection(threshold):
            errors.append(f"{path}: detector-fit and threshold victim seeds overlap")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", default="artifact")
    parser.add_argument("--profile", choices=sorted(PROFILES), default=None)
    parser.add_argument("--expected", action="append", default=[])
    parser.add_argument("--allow-smoke-only", action="store_true")
    args = parser.parse_args()
    root = Path(args.root)
    repo = Path(__file__).resolve().parents[1]
    raw_validator = Draft202012Validator(load_json(repo / "schemas" / "raw_query.schema.json"))
    manifest_validator = Draft202012Validator(
        load_json(repo / "schemas" / "run_manifest.schema.json")
    )
    summary_validator = Draft202012Validator(load_json(repo / "schemas" / "summary.schema.json"))
    errors: list[str] = []
    warnings: list[str] = []
    for directory in ["raw", "summaries", "manifests", "reports"]:
        if not (root / directory).is_dir():
            errors.append(f"missing directory: {root / directory}")
    raw_files = sorted((root / "raw").glob("*.jsonl"))
    summary_files = sorted((root / "summaries").glob("*.csv"))
    manifest_files = sorted((root / "manifests").glob("*.json"))
    report_files = sorted((root / "reports").glob("*.json"))
    for label, files in [
        ("raw JSONL", raw_files),
        ("summary CSV", summary_files),
        ("run manifest", manifest_files),
        ("report JSON", report_files),
    ]:
        if not files:
            errors.append(f"no {label} files")

    raw_rows = 0
    duplicates = 0
    seen_keys: set[tuple[Any, ...]] = set()
    for path in raw_files:
        count, file_errors, file_duplicates = validate_jsonl(path, raw_validator, seen_keys)
        raw_rows += count
        duplicates += file_duplicates
        errors.extend(file_errors)

    summary_rows = 0
    for path in summary_files:
        with path.open("r", encoding="utf-8", newline="") as handle:
            rows = list(csv.DictReader(handle))
        if not rows:
            errors.append(f"empty summary CSV: {path}")
            continue
        summary_rows += len(rows)
        for index, row in enumerate(rows, 2):
            for error in summary_validator.iter_errors(row):
                errors.append(f"{path}:{index}: summary schema: {error.message}")

    for path in manifest_files:
        try:
            manifest = load_json(path)
            for error in manifest_validator.iter_errors(manifest):
                errors.append(f"{path}: manifest schema: {error.message}")
            if isinstance(manifest, dict):
                _check_disjoint_manifest(manifest, path, errors)
        except Exception as exc:
            errors.append(f"invalid JSON {path}: {exc}")
    for path in report_files:
        try:
            report = load_json(path)
            if not isinstance(report, dict):
                errors.append(f"report is not a JSON object: {path}")
        except Exception as exc:
            errors.append(f"invalid JSON {path}: {exc}")

    expected: list[str] = []
    if args.profile:
        expected.extend(PROFILES[args.profile])
    for value in args.expected:
        expected.extend(token.strip() for token in value.split(",") if token.strip())
    if args.allow_smoke_only and not expected:
        expected = PROFILES["smoke"]
    names = " ".join(
        path.name for path in raw_files + summary_files + manifest_files + report_files
    )
    for token in sorted(set(expected)):
        if token not in names:
            errors.append(f"missing expected experiment output containing '{token}'")

    if expected and "llama" not in expected and any("llama" in path.name for path in raw_files):
        warnings.append("Llama mechanism outputs are present but not required by selected profile")

    result = {
        "root": str(root.resolve()),
        "profile": args.profile,
        "expected_tokens": sorted(set(expected)),
        "raw_files": len(raw_files),
        "raw_rows": raw_rows,
        "duplicate_raw_identities": duplicates,
        "summary_files": len(summary_files),
        "summary_rows": summary_rows,
        "manifest_files": len(manifest_files),
        "report_files": len(report_files),
        "warnings": warnings,
        "errors": errors,
        "passed": not errors,
    }
    (root / "reports").mkdir(parents=True, exist_ok=True)
    (root / "reports" / "validation_report.json").write_text(
        json.dumps(result, indent=2, allow_nan=False) + "\n", encoding="utf-8"
    )
    print(json.dumps(result, indent=2, allow_nan=False))
    if errors:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
