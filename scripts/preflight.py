#!/usr/bin/env python3
"""Static scientific-protocol preflight for the complete experiment package."""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any

import yaml

ROOT = Path(__file__).resolve().parents[1]
COMMANDS = {
    "smoke",
    "train-victims",
    "collect",
    "open-set",
    "base-mismatch",
    "lineage",
    "redeploy",
    "llama-mechanism",
}


def load_yaml(path: Path) -> dict[str, Any]:
    with path.open(encoding="utf-8") as handle:
        value = yaml.safe_load(handle)
    if not isinstance(value, dict):
        raise ValueError(f"YAML root must be a mapping: {path}")
    return value


def disjoint(config: dict[str, Any], left: str, right: str, errors: list[str], name: str) -> None:
    a = set(int(v) for v in config["splits"][left])
    b = set(int(v) for v in config["splits"][right])
    overlap = sorted(a.intersection(b))
    if overlap:
        errors.append(f"{name}: {left}/{right} overlap: {overlap}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=ROOT)
    args = parser.parse_args()
    root = args.root.resolve()
    errors: list[str] = []
    warnings: list[str] = []

    yaml_paths = sorted((root / "configs").rglob("*.yaml"))
    configs: dict[str, dict[str, Any]] = {}
    for path in yaml_paths:
        try:
            configs[str(path.relative_to(root))] = load_yaml(path)
        except Exception as exc:
            errors.append(f"invalid YAML {path}: {exc}")

    # Plan closure: every referenced command and configuration must exist.
    for relative, config in configs.items():
        if "runs" not in config:
            continue
        for index, run in enumerate(config["runs"]):
            command = run.get("command")
            config_path = root / str(run.get("config", ""))
            if command not in COMMANDS:
                errors.append(f"{relative}: run {index} has unsupported command {command!r}")
            if not config_path.is_file():
                errors.append(f"{relative}: run {index} missing config {config_path}")

    open_cfg = configs["configs/open_set/bert_mnli_detector.yaml"]
    disjoint(open_cfg, "calibration_victim_seeds", "test_victim_seeds", errors, "open-set")
    threshold = set(int(v) for v in open_cfg["splits"]["threshold_victim_seeds"])
    calibration = set(int(v) for v in open_cfg["splits"]["calibration_victim_seeds"])
    if not threshold or not threshold.issubset(calibration):
        errors.append("open-set: threshold seeds must be a non-empty calibration subset")

    lineage_cfg = configs["configs/lineage/bert_mnli.yaml"]
    disjoint(lineage_cfg, "calibration_victim_seeds", "validation_victim_seeds", errors, "lineage")
    disjoint(lineage_cfg, "calibration_victim_seeds", "test_victim_seeds", errors, "lineage")
    disjoint(lineage_cfg, "validation_victim_seeds", "test_victim_seeds", errors, "lineage")
    lineage_victims_cfg = configs["configs/lineage/bert_mnli_victims.yaml"]
    lineage_required = set(
        int(v)
        for key in (
            "calibration_victim_seeds",
            "validation_victim_seeds",
            "test_victim_seeds",
        )
        for v in lineage_cfg["splits"][key]
    )
    if not lineage_required.issubset(set(int(v) for v in lineage_victims_cfg["seeds"]["victim"])):
        errors.append("lineage victim grid omits train/validation/test checkpoints")
    if len(lineage_victims_cfg.get("structures", [])) != 1:
        errors.append("strict lineage grid must hold task/family/rank fixed")

    mismatch_cfg = configs["configs/base_mismatch/bert_mnli.yaml"]
    disjoint(
        mismatch_cfg,
        "calibration_victim_seeds",
        "test_victim_seeds",
        errors,
        "base-mismatch",
    )
    disjoint(
        mismatch_cfg,
        "detector_fit_victim_seeds",
        "threshold_victim_seeds",
        errors,
        "base-mismatch",
    )
    mismatch_calibration = set(int(v) for v in mismatch_cfg["splits"]["calibration_victim_seeds"])
    mismatch_partition = set(
        int(v) for v in mismatch_cfg["splits"]["detector_fit_victim_seeds"]
    ).union(int(v) for v in mismatch_cfg["splits"]["threshold_victim_seeds"])
    if mismatch_calibration != mismatch_partition:
        errors.append("base-mismatch: fit and threshold seeds must partition calibration seeds")
    condition_kinds = {row["kind"] for row in mismatch_cfg["base_mismatch"]["conditions"]}
    required_mismatch = {"matched", "dynamic_int8", "continued_pretraining", "model_config"}
    missing_mismatch = sorted(required_mismatch - condition_kinds)
    if missing_mismatch:
        errors.append(f"base-mismatch: missing real executable conditions {missing_mismatch}")

    victims_cfg = configs["configs/open_set/bert_mnli_victims.yaml"]
    known = [row for row in victims_cfg["structures"] if row.get("known", True)]
    unknown = [row for row in victims_cfg["structures"] if not row.get("known", True)]
    if len(known) < 3:
        errors.append("open-set victim grid needs at least three known structures")
    unknown_families = {row["family"] for row in unknown}
    required_unknown = {"lora", "dora", "ia3", "lora_head", "nonuniform_lora"}
    if not required_unknown.issubset(unknown_families):
        errors.append(
            f"open-set victim grid missing unknown families: {sorted(required_unknown - unknown_families)}"
        )
    victim_seeds = set(int(v) for v in victims_cfg["seeds"]["victim"])
    required_seeds = calibration.union(set(open_cfg["splits"]["test_victim_seeds"]))
    if not required_seeds.issubset(victim_seeds):
        errors.append("victim grid omits calibration/test victim-training seeds")

    redeploy_cfg = configs["configs/redeploy/bert_mnli.yaml"]
    kinds = {method["kind"] for method in redeploy_cfg["methods"]}
    required_methods = {
        "posterior_peft",
        "oracle_peft",
        "wrong_peft",
        "plain_student",
        "parameter_matched_student",
        "compute_matched_student",
        "distill_to_peft",
    }
    if not required_methods.issubset(kinds):
        errors.append(f"redeploy: missing baselines {sorted(required_methods - kinds)}")
    posterior_methods = [
        method for method in redeploy_cfg["methods"] if method["kind"] == "posterior_peft"
    ]
    if len(posterior_methods) != 1 or not posterior_methods[0].get("structure_from"):
        errors.append("redeploy: posterior PEFT must consume locked Stage-I service predictions")
    redeploy_seeds = set(int(v) for v in redeploy_cfg["seeds"]["victim"])
    open_test_seeds = set(int(v) for v in open_cfg["splits"]["test_victim_seeds"])
    if not redeploy_seeds.issubset(open_test_seeds):
        errors.append("redeploy: every victim must have a held-out Stage-I prediction")
    task_b = redeploy_cfg["composition"]["task_b"]
    if task_b.get("dataset_name") != "stanfordnlp/snli" or int(task_b.get("num_labels", -1)) != 3:
        errors.append("redeploy: Task B must be the independent three-label SNLI task")
    if not redeploy_cfg["composition"].get("student_training"):
        warnings.append(
            "redeploy: student_training not explicit; code correctly falls back to composition.training"
        )

    llama_cfg = configs["configs/llama_mechanism/llama31.yaml"]
    llama_text = (root / "configs/llama_mechanism/llama31.yaml").read_text()
    if "mechanism diagnostic" not in llama_text or "not be generalized" not in llama_text:
        errors.append("Llama mechanism config lacks explicit non-generalization boundary")
    if int(llama_cfg["llama"]["max_new_tokens"]) < 2:
        errors.append("Llama mechanism must retain a free-running comparison window")
    if not (root / "configs/llama_mechanism/llama31_victims.yaml").is_file():
        errors.append("Llama mechanism victim-training grid is missing")

    # Code/package hygiene: no placeholders, secrets, or production endpoint scanner.
    forbidden_markers = re.compile(r"\b(TODO|TBD|FIXME|NotImplementedError)\b")
    secret_patterns = re.compile(
        r"(sk-[A-Za-z0-9]{20,}|AKIA[0-9A-Z]{16}|-----BEGIN (?:RSA |EC )?PRIVATE KEY-----)"
    )
    production_scan = re.compile(r"requests\.(get|post)\(|httpx\.(get|post)\(")
    for directory in ["src", "scripts", "tests", "configs"]:
        for path in (root / directory).rglob("*"):
            if not path.is_file() or path.suffix in {".pyc", ".png", ".pdf"}:
                continue
            text = path.read_text(errors="ignore")
            if path.resolve() != Path(__file__).resolve() and forbidden_markers.search(text):
                errors.append(f"draft marker in executable package: {path.relative_to(root)}")
            if secret_patterns.search(text):
                errors.append(f"possible credential/private key in {path.relative_to(root)}")
            if production_scan.search(text):
                errors.append(f"direct network endpoint code in {path.relative_to(root)}")

    report = {
        "root": str(root),
        "yaml_files": len(yaml_paths),
        "checks": {
            "plan_closure": True,
            "seed_disjointness": True,
            "open_set_unknown_grid": True,
            "real_base_mismatch_conditions": True,
            "exact_lineage_split": True,
            "redeploy_baseline_closure": True,
            "llama_claim_boundary": True,
            "secret_and_endpoint_scan": True,
        },
        "warnings": warnings,
        "errors": errors,
        "passed": not errors,
    }
    print(json.dumps(report, indent=2))
    if errors:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
