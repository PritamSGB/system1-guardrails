#!/usr/bin/env python3
"""Experiment 1: accuracy, consistency, cost and latency of three guards across four models.

Every check is its own API request. The three checks for a message go out at the same time, one
connection each, so the recorded latency is the time until all three answers are back.

21 queries x 3 checks x 3 repeats x 4 models = 756 calls, about Rs 3.6 in total.
Models and repeats are shuffled per query so no model always goes first.

    python3 experiment1_accuracy.py
    python3 experiment1_accuracy.py --ids 108,109 --repeats 1

Results are appended to results/experiment1_<timestamp>.jsonl as each query finishes, so a
Ctrl-C keeps what has run. Pass that file to report.py to score it.
"""
import json
import random
import statistics
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime
from pathlib import Path

import guards
import models

REPEATS = 3
RESULTS = Path(__file__).parent / "results"


def run_all_checks(clients, model, query, claude) -> dict:
    """The three checks as three requests sent at once. client_ms is the wall time for all three."""
    start = time.perf_counter()
    with ThreadPoolExecutor(max_workers=len(guards.CHECKS)) as pool:
        calls = [f.result() for f in [pool.submit(models.run_check, c, model, query, check, claude)
                                      for c, check in zip(clients, guards.CHECKS)]]
    wall = (time.perf_counter() - start) * 1000
    return {
        "verdicts": {c["check"]: c["verdict"] for c in calls},
        "calls": calls,
        "client_ms": wall,
        "sequential_ms": sum(c["client_ms"] for c in calls),  # what one-after-another would have cost
        "tokens": {k: sum(c["tokens"][k] for c in calls) for k in ("input", "cached", "output", "reasoning")},
        "cost_inr": sum(c["cost_inr"] for c in calls),
    }


def main() -> None:
    args = sys.argv
    ids = [int(x) for x in args[args.index("--ids") + 1].split(",")] if "--ids" in args else None
    repeats = int(args[args.index("--repeats") + 1]) if "--repeats" in args else REPEATS
    rows = guards.load_queries(ids)

    clients, claude, available = models.connect(n=len(guards.CHECKS))
    model_list = [m for m in guards.MODELS if m in available]
    try:
        models.warm_up(clients, model_list, claude)
    except models.APIError as e:
        sys.exit(str(e))

    RESULTS.mkdir(exist_ok=True)
    out = RESULTS / f"experiment1_{datetime.now():%Y%m%d_%H%M%S}.jsonl"
    print(f"{len(rows)} queries x {len(guards.CHECKS)} checks x {repeats} repeats x {len(model_list)} models "
          f"= {len(rows) * len(guards.CHECKS) * repeats * len(model_list)} calls")
    print(f"Writing to {out}\n")

    try:
        for row in rows:
            order = model_list * repeats
            random.shuffle(order)
            runs = {m: [] for m in model_list}
            for m in order:
                try:
                    runs[m].append(run_all_checks(clients, m, row["query"], claude))
                except models.APIError as e:
                    print(f"  id {row['id']} {m}: {e}")
            print(f"[id {row['id']}] {row['query'][:60]}")
            for m, rs in runs.items():
                if rs:
                    cells = " ".join(f"{c}={'/'.join(fmt(r['verdicts'][c]) for r in rs)}" for c in guards.CHECKS)
                    print(f"  {guards.DISPLAY_NAME[m]:<18} {cells}  "
                          f"{statistics.median(r['client_ms'] for r in rs):.0f}ms")
            with out.open("a") as f:
                f.write(json.dumps({**row, "runs": runs}, ensure_ascii=False) + "\n")
    except KeyboardInterrupt:
        print("\nInterrupted; the completed queries are saved.")
    for c in clients:
        c.close()
    print(f"\nDone. python3 report.py {out.name}")


def fmt(verdict: dict) -> str:
    return str(verdict["level"]) if "level" in verdict else f"{verdict['p_yes']:.2f}"


if __name__ == "__main__":
    main()
