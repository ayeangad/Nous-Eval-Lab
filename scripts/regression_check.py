"""Regression check: baseline run vs candidate run -> verdict + report.

Demo: baseline = all-gold run, candidate = seeded partial-buggy run.
  uv run python scripts/regression_check.py --demo
Explicit:
  uv run python scripts/regression_check.py --baseline results/<b> --candidate results/<c>
Exit code 1 when a regression is flagged (CI-friendly).
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from nous_eval_lab.analysis.regression import compare, load_results, render_markdown  # noqa: E402
from nous_eval_lab.runners.local import load_tasks, run_benchmark  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--baseline", default="")
    ap.add_argument("--candidate", default="")
    ap.add_argument("--demo", action="store_true")
    ap.add_argument("--config", default="configs/base.yaml")
    args = ap.parse_args()
    cfg = yaml.safe_load(open(ROOT / args.config))

    if args.demo:
        tasks = load_tasks(ROOT / cfg["benchmark"]["tasks_path"])
        gold = {t.task_id: t.gold_code for t in tasks}
        # seeded regression: 2 of 8 tasks revert to buggy
        mixed = dict(gold)
        for t in sorted(tasks, key=lambda t: t.task_id)[:2]:
            mixed[t.task_id] = t.buggy_code
        bdir = run_benchmark(tasks, gold, seed=7, run_dir=ROOT / "results/regression_demo")
        cdir = run_benchmark(tasks, mixed, seed=7, run_dir=ROOT / "results/regression_demo")
        base, cand = load_results(bdir), load_results(cdir)
    else:
        base, cand = load_results(args.baseline), load_results(args.candidate)
    cmp = compare(base, cand, cfg)
    print(render_markdown(cmp))
    raise SystemExit(1 if cmp["regression"] else 0)


if __name__ == "__main__":
    main()
