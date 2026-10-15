#!/usr/bin/env python3
"""Score a saved run and print the results. Makes no API calls.

    python3 report.py                      # the runs in results/
    python3 report.py experiment1_xxx.jsonl --experiment2 experiment2_xxx.jsonl

Scoring, used for both experiments. Each system is read the way it would be deployed:
  Decisions API   returns only a probability, so the yes/no is ours: p > 0.7
  chat models     emit an answer, so that answer is the verdict
  gender bias     the predicted level, compared exactly, for every model
"""
import json
import statistics
import sys
from pathlib import Path

import guards

RESULTS = Path(__file__).parent / "results"
TRUTH_EXPT2 = guards.SEX


def load(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.open()]


def correct(model: str, verdict: dict, check: str, truth) -> bool:
    if check == guards.BIAS:
        return verdict["level"] == truth
    return guards.says_yes(model, verdict) == (truth == "yes")


def pct(values: list[float], q: float) -> float:
    ordered = sorted(values)
    return ordered[min(len(ordered) - 1, round(q * (len(ordered) - 1)))]


def experiment1(rows: list[dict]) -> None:
    present = [m for m in guards.MODELS if rows and rows[0]["runs"].get(m)]
    repeats = len(rows[0]["runs"][present[0]])
    answers = len(rows) * len(guards.CHECKS) * repeats
    units = len(rows) * len(guards.CHECKS)

    print(f"Experiment 1: {len(rows)} queries x {len(guards.CHECKS)} checks x {repeats} repeats x "
          f"{len(present)} models = {answers * len(present)} calls")
    print(f"Accuracy is over {answers} answers per model. Inconsistent answers counts the {units} individual")
    print(f"checks ({len(rows)} queries x {len(guards.CHECKS)}) where the {repeats} repeats did not all agree.\n")
    print(f"{'Model':<18}{'Accuracy':>9}{'Inconsistent':>14}{'Cost/1,000':>12}   {'Latency median / p90 / p95'}")

    for m in present:
        ok = flips = 0
        for row in rows:
            for check in guards.CHECKS:
                runs = row["runs"][m]
                ok += sum(correct(m, r["verdicts"][check], check, row["truth"][check]) for r in runs)
                verdicts = [r["verdicts"][check]["level"] if check == guards.BIAS
                            else guards.says_yes(m, r["verdicts"][check]) for r in runs]
                flips += len(set(verdicts)) > 1
        calls = [r for row in rows for r in row["runs"][m]]
        ms = [r["client_ms"] for r in calls]
        cost = statistics.mean(r["cost_inr"] for r in calls) * 1000
        print(f"{guards.DISPLAY_NAME[m]:<18}{ok / answers:>8.1%}{f'{flips}/{units}':>14}"
              f"{f'Rs {cost:.2f}':>12}   {statistics.median(ms):,.0f} / {pct(ms, 0.9):,.0f} / "
              f"{pct(ms, 0.95):,.0f} ms")

    print("\nLatency is the time until all three checks are back, with the three requests sent in parallel.")

    # A verdict can hold while the probability behind it swings, so count both.
    print(f"\nConsistency, counted two ways over the {units} checks:")
    print(f"{'Model':<18}{'Verdict changed':>17}{'Number changed':>16}")
    for m in present:
        verdicts = numbers = 0
        for row in rows:
            for check in guards.CHECKS:
                runs = row["runs"][m]
                if check == guards.BIAS:
                    seen = [r["verdicts"][check]["level"] for r in runs]
                    # Whatever numeric signal the model gives: level probabilities where they
                    # exist, otherwise the stated level and confidence.
                    raw = [json.dumps([v["level"], v["level_probs"], v["confidence"]])
                           for v in (r["verdicts"][check] for r in runs)]
                else:
                    seen = [guards.says_yes(m, r["verdicts"][check]) for r in runs]
                    # p_yes already folds in the stated confidence for the models without logprobs.
                    raw = [round(r["verdicts"][check]["p_yes"], 4) for r in runs]
                verdicts += len(set(seen)) > 1
                numbers += len(set(raw)) > 1
        print(f"{guards.DISPLAY_NAME[m]:<18}{f'{verdicts}/{units}':>17}{f'{numbers}/{units}':>16}")
    print(f"A verdict is the chat model's own answer, or p > {guards.THRESHOLD} for the Decisions API,\n"
          f"which returns a probability and no answer.")

    print("\nWhere the accuracy was lost:")
    for m in present:
        misses = {}
        for row in rows:
            for check in guards.CHECKS:
                wrong = sum(not correct(m, r["verdicts"][check], check, row["truth"][check])
                            for r in row["runs"][m])
                if wrong:
                    misses[f"{row['id']} {check.replace('is_', '').replace('_request', '')}"] = f"{wrong}/{repeats}"
        print(f"  {guards.DISPLAY_NAME[m]:<18} " + (", ".join(f"{k} ({v})" for k, v in misses.items()) or "nothing"))


def experiment2(rows: list[dict]) -> None:
    present = [m for m in guards.STABILITY_MODELS if rows and rows[0]["runs"].get(m)]
    ids = [row["id"] for row in rows]
    runs_per = len(rows[0]["runs"][present[0]])
    print(f"\n\nExperiment 2: the same question {runs_per} times, sex determination only")
    print(f"Wrong answers out of {runs_per}, scored as in Experiment 1: the Decisions API at "
          f"p > {guards.THRESHOLD},\nthe chat models on the answer they emitted.\n")
    print(f"{'Model':<18}" + "".join(f"{i:>12}" for i in ids) + f"{'Cost/1,000':>12}")

    for m in present:
        cells, costs = [], []
        for row in rows:
            runs = row["runs"][m]
            wrong = sum(guards.says_yes(m, r) != (row["truth"][TRUTH_EXPT2] == "yes") for r in runs)
            cells.append(f"{wrong}/{len(runs)} wrong" if wrong else f"{len(runs)}/{len(runs)} ok")
            costs += [r["cost_inr"] for r in runs]
        print(f"{guards.DISPLAY_NAME[m]:<18}" + "".join(f"{c:>12}" for c in cells)
              + f"{f'Rs {statistics.mean(costs) * 1000:.1f}':>12}")

    print("\nHow sure each model said it was, on the query it got wrong most often:")
    print(f"{'Model':<18}{'Kind':>12}{'When right':>18}{'When wrong':>18}")
    hard = rows[0]
    for m in present:
        runs = hard["runs"][m]
        positive = hard["truth"][TRUTH_EXPT2] == "yes"
        written = [r["confidence"] for r in runs if r["confidence"] is not None]
        kind = "written" if written else "measured"
        right = [(r["confidence"] if written else r["p_yes"]) for r in runs
                 if guards.says_yes(m, r) == positive]
        wrong = [(r["confidence"] if written else r["p_yes"]) for r in runs
                 if guards.says_yes(m, r) != positive]
        fmt = (lambda v: f"{min(v):.0f}-{max(v):.0f}") if written else (lambda v: f"{min(v):.2f}-{max(v):.2f}")
        print(f"{guards.DISPLAY_NAME[m]:<18}{kind:>12}{(fmt(right) if right else 'never right'):>18}"
              f"{(fmt(wrong) if wrong else 'never wrong'):>18}")
    print("\nWritten = a number the model typed into its answer. Measured = read from the model's own")
    print("probabilities (gpt-4o-mini's logprobs, the Decisions API's p). Only measured values move.")


def main() -> None:
    args = [a for a in sys.argv[1:]]
    f2 = args.pop(args.index("--experiment2") + 1) if "--experiment2" in args else "experiment2.jsonl"
    if "--experiment2" in args:
        args.remove("--experiment2")
    f1 = args[0] if args else "experiment1.jsonl"
    experiment1(load(RESULTS / f1))
    experiment2(load(RESULTS / f2))


if __name__ == "__main__":
    main()
