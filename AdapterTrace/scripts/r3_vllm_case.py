#!/usr/bin/env python3
"""Self-hosted PEFT service case study: verify run provenance over HTTP.

A vLLM server (OpenAI-compatible API, multi-LoRA) serves stolen, modified and
innocent adapters on the public base. The verifier only talks to the HTTP
endpoint: for each queried training example it sends the token ids of
prompt + response to /v1/completions with max_tokens=1 and
prompt_logprobs=5, and sums the returned log-probabilities of the response
tokens (full-score interface); the top-5 entries also give the weaker
top-5-with-prefill interface. It then runs the committed protocol.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import protocol as P  # noqa: E402


def post(url: str, body: dict, timeout: float = 600) -> dict:
    req = urllib.request.Request(url, data=json.dumps(body).encode(), headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.loads(r.read())


def score_one(url: str, model: str, ids: list[int], n_prompt: int):
    t0 = time.perf_counter()
    out = post(url, {"model": model, "prompt": ids, "max_tokens": 1, "temperature": 0.0, "prompt_logprobs": 5})
    dt = time.perf_counter() - t0
    plp = out["choices"][0]["prompt_logprobs"]
    full = top5 = 0.0
    for pos in range(n_prompt, len(ids)):
        d = plp[pos]
        tok = str(ids[pos])
        lp = d[tok]["logprob"]
        full += lp
        ranked = sorted(v["logprob"] for v in d.values())[::-1]
        kth = ranked[min(4, len(ranked) - 1)] if d[tok].get("rank", 1) > 5 else lp
        top5 += lp if d[tok].get("rank", 1) <= 5 else kth
    usage = out.get("usage", {})
    return full, top5, dt, usage.get("prompt_tokens", len(ids))


def score_model(url: str, model: str, examples: list[tuple[list[int], int]], workers: int):
    t0 = time.perf_counter()
    with ThreadPoolExecutor(workers) as ex:
        res = list(ex.map(lambda e: score_one(url, model, *e), examples))
    wall = time.perf_counter() - t0
    full = np.array([r[0] for r in res])
    top5 = np.array([r[1] for r in res])
    lat = np.array([r[2] for r in res])
    toks = int(sum(r[3] for r in res))
    return full, top5, {"wall_s": wall, "lat_p50_s": float(np.median(lat)), "lat_p95_s": float(np.percentile(lat, 95)),
                        "requests": len(res), "prompt_tokens": toks, "req_per_s": len(res) / wall}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--served", required=True)
    ap.add_argument("--plan", required=True)
    ap.add_argument("--grid", required=True)
    ap.add_argument("--url", default="http://127.0.0.1:8000/v1/completions")
    ap.add_argument("--hf-npz", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--workers", type=int, default=16)
    args = ap.parse_args()

    import importlib.metadata as md
    import mc_null as M
    from palimpsest_lineage import _train_dataset, training_order
    from adaptertrace.config import load_config

    cfg = load_config(args.config)
    plan_all = json.loads(Path(args.plan).read_text())
    served = json.loads(Path(args.served).read_text())
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    timing: dict = {}

    t = time.perf_counter()
    dataset, _ = _train_dataset(cfg)
    timing["load_and_tokenize_data_s"] = time.perf_counter() - t
    n_train = len(dataset)
    versions = {p: md.version(p) for p in ("transformers", "accelerate", "torch", "datasets")}
    shadow_dirs = [Path(served[f"shadow-{s}"]) for s in plan_all["shadows"]]

    # ---- commit phase (owner, training time) ----
    records = {}
    t = time.perf_counter()
    for o in plan_all["owners"]:
        plan = {"n_query": plan_all["n_query"], "query_seed": int(plan_all["query_seed"]),
                "statistic": "spearman(l_S - mean shadows, lr schedule weight)", "alpha": 1e-3,
                "mc_draws": plan_all["mc_draws"], "max_tests": 1,
                "shadow_sha256": [P.sha256_file(d / "adapter_model.safetensors") for d in shadow_dirs]}
        records[o] = P.commit(dataset, cfg["training"], o, Path(served[f"stolen-{o}"]), plan, versions)
    timing["commit_per_owner_s"] = (time.perf_counter() - t) / len(records)
    (out / "commitments.json").write_text(json.dumps(records, indent=1))

    # ---- verify phase (arbiter) ----
    rows = [dataset[i] for i in range(n_train)]
    examples = []
    for r in rows:
        n_valid = int(sum(r["attention_mask"]))
        ids = r["input_ids"][:n_valid]
        n_prompt = next(i for i, l in enumerate(r["labels"]) if l != -100)
        examples.append((ids, n_prompt))

    # warm-up and sequential latency on one adapter
    first = f"stolen-{plan_all['owners'][0]}"
    score_one(args.url, first, *examples[0])
    lat = [score_one(args.url, first, *examples[i])[2] for i in range(1, 101)]
    timing["sequential_latency_p50_s"] = float(np.median(lat))
    timing["sequential_latency_p95_s"] = float(np.percentile(lat, 95))

    nulls = {}
    t = time.perf_counter()
    for n in sorted({plan_all["n_query"], 250}):
        nulls[n] = M.null_rho("dolly", n, plan_all["mc_draws"], 7)
    timing["mc_null_s"] = time.perf_counter() - t

    scores: dict[str, dict] = {}
    serve_stats = {}
    hf = np.load(args.hf_npz)
    hf_seed_idx = {int(s): i for i, s in enumerate(hf["seeds"])}
    agree = {}
    report = {"owners": {}, "timing": timing, "serving": serve_stats, "versions": versions}
    for o in plan_all["owners"]:
        rec = records[o]
        t = time.perf_counter()
        ok = P.check(rec, dataset, Path(served[f"stolen-{o}"]), shadow_dirs)
        t_check = time.perf_counter() - t
        t = time.perf_counter()
        pos = training_order(dataset, cfg["training"], o)
        t_replay = time.perf_counter() - t
        w = M.lr_weight(np.asarray(pos)[None, :], *M.SETTINGS["dolly"])
        q = P.query_subset(rec["plan"], n_train)
        sub = [examples[i] for i in q]
        names = [n for n in served if not n.startswith(("stolen-", "svd4-", "merge")) or n.endswith(f"-{o}")]
        for name in names:
            key = (name, tuple(q))
            if name not in scores or not np.array_equal(scores[name]["q"], q):
                full, top5, st = score_model(args.url, name, sub, args.workers)
                scores[name] = {"q": q, "full": full, "top5": top5}
                serve_stats[f"{name}@owner{o}"] = st
        ref_full = np.mean([scores[f"shadow-{s}"]["full"] for s in plan_all["shadows"]], axis=0)
        ref_top5 = np.mean([scores[f"shadow-{s}"]["top5"] for s in plan_all["shadows"]], axis=0)
        res = {"hash_check": ok, "replay_s": t_replay, "check_s": t_check, "suspects": {}}
        for name in names:
            if name.startswith("shadow-"):
                continue
            entry = {}
            for iface, ref in (("full", ref_full), ("top5", ref_top5)):
                r = scores[name][iface] - ref
                for n in sorted(nulls):
                    rho = P.spearman(r[:n], w[q][:n])
                    entry[f"{iface}/n{n}"] = P.decide(rho, nulls[n], {**rec["plan"], "n_query": n})
            res["suspects"][name] = entry
        if o in hf_seed_idx:  # agreement of HTTP scores with offline HF scoring
            agree[o] = float(np.corrcoef(scores[f"stolen-{o}"]["full"], hf["adapter_loglik"][hf_seed_idx[o]][q])[0, 1])
        report["owners"][str(o)] = res
        print(json.dumps({o: {k: v["full/n%d" % plan_all["n_query"]] for k, v in res["suspects"].items()}}, indent=0), flush=True)
    report["http_vs_hf_corr"] = agree
    (out / "report.json").write_text(json.dumps(report, indent=1, default=float))
    np.savez_compressed(out / "scores.npz", **{f"{k}__{f}": v[f] for k, v in scores.items() for f in ("q", "full", "top5")})
    print("DONE", flush=True)


if __name__ == "__main__":
    main()
