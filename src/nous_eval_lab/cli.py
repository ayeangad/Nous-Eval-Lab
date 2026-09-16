"""CLI entry: run SWE-bench-informed slice with gold candidate."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import yaml

from .runners.local import load_tasks, run_benchmark


def main() -> None:
    ap = argparse.ArgumentParser(description="Nous Eval Lab — run benchmark slice")
    ap.add_argument("--config", default="configs/base.yaml")
    ap.add_argument("--candidate", default="gold", choices=["gold", "buggy"])
    args = ap.parse_args()

    cfg = yaml.safe_load(open(args.config))
    tasks = load_tasks(cfg["benchmark"]["tasks_path"])
    cmap = {t.task_id: (t.gold_code if args.candidate == "gold" else t.buggy_code) for t in tasks}
    out = run_benchmark(tasks, cmap, seed=int(cfg.get("seed", 42)), run_dir=cfg.get("run_dir", "results"))
    meta = json.load(open(out / "meta.json"))
    print(f"wrote {out} pass_rate={meta['pass_rate']:.2f} ({meta['n_passed']}/{meta['n_tasks']})")


if __name__ == "__main__":
    main()
