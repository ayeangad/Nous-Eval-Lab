"""Heuristic judges: deterministic, zero-cost, no API keys.

- heuristic_v1 (p2p_only): runs ONLY pass_to_pass tests. Labels pass iff
  all P2P pass. By construction it cannot see F2P failures -- the point is
  to demonstrate what a regression-only judge misses.
- heuristic_v2 (checklist): full F2P+P2P grading plus style bans
  (print/warnings/bare-except). Strictness ablation vs the rubric judge.
"""
from __future__ import annotations

from nous_eval_lab.benchmarks.swebench_informed.grader import _run_snippets
from nous_eval_lab.benchmarks.swebench_informed.task_schema import Task

from .base import Verdict

BANNED_PATTERNS = ("print(", "warnings.warn", "except Exception:", "except:")


def _load_tasks() -> dict[str, Task]:
    import json
    from pathlib import Path

    path = Path(__file__).resolve().parents[1] / "benchmarks/swebench_informed/tasks.jsonl"
    out = {}
    for line in path.read_text().splitlines():
        if line.strip():
            t = Task.from_dict(json.loads(line))
            out[t.task_id] = t
    return out


class P2POnlyJudge:
    version = "heuristic_v1"

    def __init__(self) -> None:
        self.tasks = _load_tasks()

    def judge(self, sample: dict) -> Verdict:
        task = self.tasks[sample["task_id"]]
        passed, err = _run_snippets(sample["model_answer"], task.pass_to_pass)[:2]
        ok = passed == len(task.pass_to_pass)
        return Verdict(
            label="pass" if ok else "fail",
            confidence="high" if ok else "medium",
            rationale=f"P2P-only: {passed}/{len(task.pass_to_pass)} pass. {err or 'all P2P passed'}",
            judge_version=self.version,
        )


class ChecklistJudge:
    version = "heuristic_v2"

    def __init__(self) -> None:
        self.tasks = _load_tasks()

    def judge(self, sample: dict) -> Verdict:
        task = self.tasks[sample["task_id"]]
        code = sample["model_answer"]
        f2p, f2p_err = _run_snippets(code, task.fail_to_pass)[:2]
        p2p, p2p_err = _run_snippets(code, task.pass_to_pass)[:2]
        hits = [p for p in BANNED_PATTERNS if p in code]
        behavior_ok = f2p == len(task.fail_to_pass) and p2p == len(task.pass_to_pass)
        if not behavior_ok:
            first = f2p_err or p2p_err
            return Verdict("fail", "high", f"checklist behavior fail: {first}", self.version)
        if hits:
            return Verdict("fail", "low", f"behavior passes but banned patterns {hits}", self.version)
        return Verdict("pass", "high", "checklist: F2P+P2P pass, no banned patterns", self.version)
