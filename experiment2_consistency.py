#!/usr/bin/env python3
"""Experiment 2: ask the same question ten times and see whether the answer holds.

Only the sex-determination check, on three messages:
  108  a question about the law on sex determination   -> no
  109  a scan report that says "Sex: not disclosed"    -> no
  101  control, a clear request to learn the sex       -> yes

Five models, including gpt-5.6-luna at medium reasoning, to see whether a stronger model is
steadier. 3 queries x 10 runs x 5 models = 150 calls, about Rs 0.9 in total. Calls go one at a
time in shuffled order, so nothing overlaps.

    python3 experiment2_consistency.py
    python3 experiment2_consistency.py --ids 108 --runs 5
"""
import json
import random
import statistics
import sys
from datetime import datetime
from pathlib import Path

import guards
import models

QUERY_IDS = [108, 109, 101]
RUNS = 10
RESULTS = Path(__file__).parent / "results"


def main() -> None:
    args = sys.argv
    ids = [int(x) for x in args[args.index("--ids") + 1].split(",")] if "--ids" in args else QUERY_IDS
    runs_per = int(args[args.index("--runs") + 1]) if "--runs" in args else RUNS
    rows = guards.load_queries(ids)

    clients, claude, model_list = models.connect(n=1)
    try:
        models.warm_up(clients, model_list, claude)
    except models.APIError as e:
        sys.exit(str(e))
    client = clients[0]

    RESULTS.mkdir(exist_ok=True)
    out = RESULTS / f"experiment2_{datetime.now():%Y%m%d_%H%M%S}.jsonl"
    print(f"{len(rows)} queries x {runs_per} runs x {len(model_list)} models "
          f"= {len(rows) * runs_per * len(model_list)} calls, sex determination only")
    print(f"Writing to {out}\n")

    total = 0.0
    for row in rows:
        order = model_list * runs_per
        random.shuffle(order)
        runs = {m: [] for m in model_list}
        for m in order:
            try:
                r = models.run_check(client, m, row["query"], guards.SEX, claude)
                runs[m].append({**r["verdict"], "client_ms": r["client_ms"], "tokens": r["tokens"],
                                "cost_inr": r["cost_inr"], "raw": r["raw"]})
            except models.APIError as e:
                print(f"  id {row['id']} {m}: {e}")
        print(f"[id {row['id']}] truth={row['truth'][guards.SEX]}  {row['query'][:60]}")
        for m, rs in runs.items():
            if not rs:
                continue
            ps = [r["p_yes"] for r in rs]
            confs = [r["confidence"] for r in rs if r["confidence"] is not None]
            print(f"  {guards.DISPLAY_NAME[m]:<18} "
                  f"{''.join('Y' if r['said_yes'] else 'N' for r in rs)}  "
                  f"p {min(ps):.2f}/{statistics.median(ps):.2f}/{max(ps):.2f}"
                  + (f"  conf {min(confs)}-{max(confs)}" if confs else ""))
        total += sum(r["cost_inr"] for rs in runs.values() for r in rs)
        with out.open("a") as f:
            f.write(json.dumps({**row, "runs": runs}, ensure_ascii=False) + "\n")
    client.close()
    print(f"\nTotal cost Rs {total:.2f}. python3 report.py --experiment2 {out.name}")


if __name__ == "__main__":
    main()
