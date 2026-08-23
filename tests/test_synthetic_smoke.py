from pathlib import Path

from vectorhijack_sr.config import load_config
from vectorhijack_sr.synthetic import run_synthetic_smoke


def test_smoke_writes_complete_bundle(tmp_path):
    cfg_path = Path(__file__).parents[1] / "configs" / "smoke" / "synthetic.yaml"
    cfg = load_config(cfg_path)
    cfg["experiment"]["output_root"] = str(tmp_path / "artifact")
    cfg["synthetic"]["probes_per_service"] = 60
    cfg["fingerprint"]["n_bootstrap"] = 20
    result = run_synthetic_smoke(cfg)
    assert result["counts"]["raw_rows"] > 0
    assert (tmp_path / "artifact" / "raw" / "synthetic_queries.jsonl").exists()
    assert (tmp_path / "artifact" / "summaries" / "synthetic_open_set.json").exists()
    assert (tmp_path / "artifact" / "manifests" / "synthetic_smoke.json").exists()
