"""Seeded local runner: tasks → grader → JSONL artifacts.

Deterministic: same seed + same tasks + same candidates → same results.
Writes results/<run-id>/results.jsonl + meta.json.
"""
from __future__ import annotations

import datetime as _dt
import hashlib
import json
import random
from pathlib import Path

from ..benchmarks.swebench_informed.grader import grade
from ..benchmarks.swebench_informed.task_schema import Task


def make_run_id(seed: int, task_ids: list[str]) -> str:
    h = hashlib.sha256(("|".join(sorted(task_ids)) + f"|{seed}").encode()).hexdigest()[:8]
    ts = _dt.datetime.now(_dt.timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    return f"run_{ts}_s{seed}_{h}"


def load_tasks(tasks_path: str | Path) -> list[Task]:
    tasks: list[Task] = []
    with open(tasks_path) as f:
        for line in f:
            line = line.strip()
            if line:
                tasks.append(Task.from_dict(json.loads(line)))
    return tasks


def run_benchmark(
    tasks: list[Task],
    candidate_map: dict[str, str],
    seed: int,
    run_dir: str | Path,
    run_id: str | None = None,
) -> Path:
    """candidate_map: task_id → candidate code string. Missing → buggy_code baseline."""
    rng = random.Random(seed)
    order = list(tasks)
    rng.shuffle(order)
    task_ids = [t.task_id for t in tasks]
    run_id = run_id or make_run_id(seed, task_ids)
    out = Path(run_dir) / run_id
    out.mkdir(parents=True, exist_ok=False)
    rows = []
    for task in order:
        code = candidate_map.get(task.task_id, task.buggy_code)
        res = grade(code, task)
        rows.append(
            {
                "run_id": run_id,
                "seed": seed,
                "task_id": res.task_id,
                "passed": res.passed,
                "f2p_passed": res.f2p_passed,
                "f2p_total": res.f2p_total,
                "p2p_passed": res.p2p_passed,
                "p2p_total": res.p2p_total,
                "reason": res.reason,
                "outcome": res.outcome,
                "failed_stage": res.failed_stage,
                "capability": task.capability,
                "difficulty": task.difficulty,
                "variants": task.variants,
            }
        )
    with open(out / "results.jsonl", "w") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")
    passed = sum(1 for r in rows if r["passed"])
    meta = {
        "run_id": run_id,
        "seed": seed,
        "n_tasks": len(rows),
        "n_passed": passed,
        "pass_rate": passed / len(rows) if rows else 0.0,
    }
    with open(out / "meta.json", "w") as f:
        json.dump(meta, f, indent=2)
    return out
