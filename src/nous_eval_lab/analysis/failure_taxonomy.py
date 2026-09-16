"""Failure taxonomy: separate the SOURCE of a failure.

Vocabulary (model vs prompt vs grader vs spec, per the role):
- model_wrong: answer is wrong per spec and tests catch it.
- grader_gap: human and suite disagree (grader_sensitive) -- tests are
  incomplete, not the model/judge.
- spec_ambiguous: sample flagged ambiguous; reasonable judges may differ.
- judge_leniency: judge passed what humans failed (missed failure).
- judge_strictness: judge failed what humans passed (false alarm).
- prompt_phrasing: disagreement isolated to adversarial prompt_kind on an
  otherwise-passing answer family (suggestive, not conclusive).
"""
from __future__ import annotations


def attribute(sample: dict, label_row: dict, judge_label: str) -> tuple[str, str]:
    human = label_row["human_label"]
    if label_row.get("grader_sensitive") and human != _suite_label(sample):
        return "grader_gap", "human spec judgment diverges from test suite"
    if label_row.get("ambiguous"):
        return "spec_ambiguous", "annotator flagged genuine specification ambiguity"
    if judge_label == human:
        return "model_wrong" if human == "fail" else "none", "judge and human agree"
    if judge_label == "pass":
        return "judge_leniency", "judge missed a human-marked failure"
    return "judge_strictness", "judge flagged a human-marked pass"


def _suite_label(sample: dict) -> str:
    """Suite verdict for a sample (grader as oracle). Imported lazily."""
    import json
    from pathlib import Path

    from ..benchmarks.swebench_informed.grader import grade
    from ..benchmarks.swebench_informed.task_schema import Task

    if not hasattr(_suite_label, "_tasks"):
        tasks = {}
        p = Path(__file__).resolve().parents[1] / "benchmarks/swebench_informed/tasks.jsonl"
        for line in p.read_text().splitlines():
            if line.strip():
                t = Task.from_dict(json.loads(line))
                tasks[t.task_id] = t
        _suite_label._tasks = tasks  # type: ignore[attr-defined]
    task = _suite_label._tasks[sample["task_id"]]  # type: ignore[attr-defined]
    return "pass" if grade(sample["model_answer"], task).passed else "fail"


def prevalence(rows: list[dict]) -> dict[str, dict[str, int | float]]:
    """rows: {source, ...} -> counts + share."""
    from collections import Counter

    c = Counter(r["source"] for r in rows)
    n = len(rows)
    return {k: {"n": v, "share": v / n} for k, v in sorted(c.items())}
