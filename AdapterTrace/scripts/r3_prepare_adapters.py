#!/usr/bin/env python3
"""Prepare the adapters served in the self-hosted case study.

Every served adapter gets its own directory with the original weights (or the
thief's modification of them) and a minimal PEFT adapter_config.json that the
inference server can parse. Modifications follow the evasion study:
  svd4      : keep the top 4 singular directions of every update B@A (scale kept)
  mergeL    : lambda * stolen + (1 - lambda) * thief's other-data adapter,
              realised exactly by concatenating the LoRA factors
"""
from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

import torch
from safetensors.torch import load_file, save_file

KEEP = ["r", "lora_alpha", "target_modules", "lora_dropout", "bias", "fan_in_fan_out",
        "peft_type", "task_type", "base_model_name_or_path", "inference_mode",
        "modules_to_save", "use_rslora", "use_dora", "init_lora_weights"]


def load(d: Path):
    cfg = json.loads((d / "adapter_config.json").read_text())
    return cfg, load_file(str(d / "adapter_model.safetensors"))


def write(out: Path, cfg: dict, tensors: dict) -> None:
    out.mkdir(parents=True, exist_ok=True)
    clean = {k: cfg[k] for k in KEEP if k in cfg}
    clean["use_dora"] = False
    (out / "adapter_config.json").write_text(json.dumps(clean, indent=1))
    save_file({k: v.contiguous() for k, v in tensors.items()}, str(out / "adapter_model.safetensors"))


def pairs(t: dict):
    for ka in sorted(k for k in t if "lora_A" in k):
        yield ka, ka.replace("lora_A", "lora_B")


def svd(cfg: dict, t: dict, keep: int):
    out = {}
    for ka, kb in pairs(t):
        A, B = t[ka].float().cuda(), t[kb].float().cuda()
        qb, rb = torch.linalg.qr(B)            # B = qb rb
        qa, ra = torch.linalg.qr(A.T)          # A^T = qa ra
        u, s, vh = torch.linalg.svd(rb @ ra.T)  # B A = qb (rb ra^T) qa^T
        U = qb @ u[:, :keep]
        V = qa @ vh[:keep].T
        sq = s[:keep].sqrt()
        out[kb] = (U * sq).to(t[kb].dtype).cpu()
        out[ka] = (sq[:, None] * V.T).to(t[ka].dtype).cpu()
    c = dict(cfg)
    c["lora_alpha"] = cfg["lora_alpha"] * keep / cfg["r"]  # keep alpha / r
    c["r"] = keep
    return c, out


def merge(cfg: dict, t: dict, cfg2: dict, t2: dict, lam: float):
    assert cfg["r"] == cfg2["r"] and cfg["lora_alpha"] == cfg2["lora_alpha"]
    out = {}
    a, b = lam ** 0.5, (1 - lam) ** 0.5
    for ka, kb in pairs(t):
        out[ka] = torch.cat([a * t[ka].float(), b * t2[ka].float()], 0).to(t[ka].dtype)
        out[kb] = torch.cat([a * t[kb].float(), b * t2[kb].float()], 1).to(t[kb].dtype)
    c = dict(cfg)
    c["r"] = 2 * cfg["r"]
    c["lora_alpha"] = 2 * cfg["lora_alpha"]
    return c, out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--grid", required=True, help=".../models/llama31_dolly_grid/lora_r64")
    ap.add_argument("--other", required=True, help=".../models/llama31_dolly_otherdata/lora_r64")
    ap.add_argument("--plan", required=True, help="JSON with owners, partners, shadows, siblings, others")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    plan = json.loads(Path(args.plan).read_text())
    grid, other, out = Path(args.grid), Path(args.other), Path(args.out)
    served = {}

    def plain(name: str, d: Path) -> None:
        cfg, t = load(d)
        write(out / name, cfg, t)
        served[name] = str(out / name)

    for s in plan["shadows"]:
        plain(f"shadow-{s}", grid / f"seed-{s}")
    for s in plan["siblings"]:
        plain(f"sibling-{s}", grid / f"seed-{s}")
    for s in plan["others"]:
        plain(f"other-{s}", other / f"seed-{s}")
    for o, p in zip(plan["owners"], plan["partners"]):
        cfg, t = load(grid / f"seed-{o}")
        write(out / f"stolen-{o}", cfg, t)
        served[f"stolen-{o}"] = str(out / f"stolen-{o}")
        c, x = svd(cfg, t, 4)
        write(out / f"svd4-{o}", c, x)
        served[f"svd4-{o}"] = str(out / f"svd4-{o}")
        cfg2, t2 = load(other / f"seed-{p}")
        for lam in (0.5, 0.25):
            c, x = merge(cfg, t, cfg2, t2, lam)
            write(out / f"merge{lam}-{o}", c, x)
            served[f"merge{lam}-{o}"] = str(out / f"merge{lam}-{o}")
    (out / "served.json").write_text(json.dumps(served, indent=1))
    print(f"wrote {len(served)} adapters to {out}")


if __name__ == "__main__":
    main()
