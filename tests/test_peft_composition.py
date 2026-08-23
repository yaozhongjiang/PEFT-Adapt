from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

if os.environ.get("RUN_HF_INTEGRATION") != "1":
    pytest.skip(
        "set RUN_HF_INTEGRATION=1 for local Transformers/PEFT test",
        allow_module_level=True,
    )

ROOT = Path(__file__).resolve().parents[1]


@pytest.mark.integration
def test_tiny_hf_peft_integration(tmp_path: Path) -> None:
    """Run the HF/PEFT integration in a child process.

    Keeping native ML runtimes outside the pytest worker prevents shutdown
    hangs from tokenizer/OpenMP thread pools while still testing the real CLI
    script, Trainer step, save/reload, safe merge, and adapter composition.
    """
    output = tmp_path / "integration"
    environment = os.environ.copy()
    environment["PYTHONPATH"] = str(ROOT / "src")
    subprocess.run(
        [
            sys.executable,
            str(ROOT / "scripts" / "hf_tiny_integration.py"),
            "--output",
            str(output),
        ],
        cwd=ROOT,
        env=environment,
        check=True,
        timeout=180,
    )
    report = json.loads((output / "integration_report.json").read_text())
    assert report["status"] == "passed"
    assert report["merge"]["merge"] is True
    assert report["composition"]["supported"] is True
    assert report["body_head_split"]["head_key_count"] > 0
    assert report["body_head_split"]["body_key_count"] > 0
