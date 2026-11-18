#!/usr/bin/env python3
"""Render the same job N times in a row on an endpoint and record memory.

Each run gets fresh seeds (identical inputs would be answered from the
worker's Redis result cache without rendering). Every finished job's
``metadata.memory`` (written by handler.py) becomes one CSV row, and the
summary shows whether container memory keeps climbing on one worker.

    RUNPOD_API_KEY=... RUNPOD_ENDPOINT_ID=... \\
        python3 scripts/memory_soak.py --input job.json --runs 12 --csv soak.csv

``job.json`` is a request body: {"input": {"workflow": {...}, "images": [...]}}.
Stdlib only.
"""

from __future__ import annotations

import argparse
import copy
import csv
import json
import os
import random
import sys
import time
import urllib.error
import urllib.request

API = "https://api.runpod.ai/v2"
SEED_KEYS = ("noise_seed", "seed")
FIELDS = (
    "run", "job_id", "status", "worker_id", "job_index", "exec_s",
    "before_current_mib", "after_current_mib", "max_mib", "after_peak_mib",
    "before_anon_mib", "after_anon_mib", "after_file_mib",
    "before_comfy_rss_mib", "after_comfy_rss_mib", "after_gpu_used_mib", "error",
)


def request(method: str, url: str, key: str, body: dict | None = None) -> dict:
    data = None if body is None else json.dumps(body).encode()
    req = urllib.request.Request(url, data=data, method=method, headers={
        "Authorization": f"Bearer {key}",
        "Content-Type": "application/json",
        # Cloudflare in front of the API rejects urllib's default agent
        "User-Agent": "ltx25-memory-soak/1.0",
    })
    with urllib.request.urlopen(req, timeout=60) as resp:
        return json.load(resp)


def reseed(job: dict) -> dict:
    job = copy.deepcopy(job)
    for node in (job.get("input", {}).get("workflow") or {}).values():
        inputs = node.get("inputs") if isinstance(node, dict) else None
        if isinstance(inputs, dict):
            for k in SEED_KEYS:
                if isinstance(inputs.get(k), int):
                    inputs[k] = random.randrange(2**32)
    return job


def row_from(run: int, job_id: str, status: dict) -> dict:
    out = status.get("output") or {}
    mem = (out.get("metadata") or {}).get("memory") or out.get("memory") or {}
    b, a = mem.get("before") or {}, mem.get("after") or {}
    return {
        "run": run, "job_id": job_id, "status": status.get("status"),
        "worker_id": a.get("worker_id") or b.get("worker_id"),
        "job_index": a.get("job_index"),
        "exec_s": round((status.get("executionTime") or 0) / 1000, 1),
        "before_current_mib": b.get("current_mib"), "after_current_mib": a.get("current_mib"),
        "max_mib": a.get("max_mib"), "after_peak_mib": a.get("peak_mib"),
        "before_anon_mib": b.get("anon_mib"), "after_anon_mib": a.get("anon_mib"),
        "after_file_mib": a.get("file_mib"),
        "before_comfy_rss_mib": b.get("comfy_rss_mib"), "after_comfy_rss_mib": a.get("comfy_rss_mib"),
        "after_gpu_used_mib": a.get("gpu_used_mib"),
        "error": (out.get("error") or status.get("error") or "")[:200],
    }


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--input", required=True, help="request body JSON file")
    ap.add_argument("--runs", type=int, default=12)
    ap.add_argument("--csv", default="memory_soak.csv")
    ap.add_argument("--poll", type=float, default=5.0)
    args = ap.parse_args()
    key, endpoint = os.environ.get("RUNPOD_API_KEY"), os.environ.get("RUNPOD_ENDPOINT_ID")
    if not key or not endpoint:
        print("set RUNPOD_API_KEY and RUNPOD_ENDPOINT_ID", file=sys.stderr)
        return 2
    base = f"{API}/{endpoint}"
    job = json.load(open(args.input))
    rows = []
    with open(args.csv, "w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=FIELDS)
        writer.writeheader()
        for run in range(1, args.runs + 1):
            job_id = request("POST", f"{base}/run", key, reseed(job))["id"]
            while True:
                status = request("GET", f"{base}/status/{job_id}", key)
                if status.get("status") in ("COMPLETED", "FAILED", "CANCELLED", "TIMED_OUT"):
                    break
                time.sleep(args.poll)
            row = row_from(run, job_id, status)
            rows.append(row)
            writer.writerow(row)
            fh.flush()
            print(
                f"run {run:2d} {row['status']:9s} worker={row['worker_id']} idx={row['job_index']} "
                f"exec={row['exec_s']}s mem {row['before_current_mib']}→{row['after_current_mib']} "
                f"(anon {row['before_anon_mib']}→{row['after_anon_mib']}, max {row['max_mib']}) "
                f"comfy_rss {row['after_comfy_rss_mib']} gpu {row['after_gpu_used_mib']} {row['error']}",
                flush=True,
            )
    anon = [(r["run"], r["after_anon_mib"]) for r in rows if r["after_anon_mib"] is not None]
    if len(anon) >= 2:
        first, last = anon[0][1], anon[-1][1]
        print(f"\nanon memory after run {anon[0][0]}: {first} MiB, after run {anon[-1][0]}: {last} MiB "
              f"(Δ {round(last - first, 1)} MiB over {len(anon) - 1} runs)")
    workers = {r["worker_id"] for r in rows if r["worker_id"]}
    if len(workers) > 1:
        print(f"note: jobs ran on {len(workers)} workers {sorted(workers)} — compare per worker")
    return 0 if all(r["status"] == "COMPLETED" for r in rows) else 1


if __name__ == "__main__":
    sys.exit(main())
