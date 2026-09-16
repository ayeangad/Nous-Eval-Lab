"""QA checks for benchmark tasks."""
from __future__ import annotations

from .grader import grade
from .task_schema import VALID_VARIANTS, Task


def qa_task(task: Task) -> list[str]:
    """Return list of QA problems (empty = valid)."""
    problems: list[str] = []
    if not task.task_id:
        problems.append("empty task_id")
    if len(task.issue_text.strip()) < 20:
        problems.append("issue_text too short / underspecified unintentionally")
    if not task.capability:
        problems.append("missing capability tags")
    if not task.variants:
        problems.append("missing variants (need happy_path + at least one of edge/underspecified/adversarial)")
    if "happy_path" not in task.variants:
        problems.append("variants must include happy_path")
    if "adversarial" in task.variants and len(task.adversarial_issue_text.strip()) < 10:
        problems.append("adversarial variant declared but adversarial_issue_text missing")
    for v in task.fail_to_pass_variants + task.pass_to_pass_variants:
        if v not in VALID_VARIANTS:
            problems.append(f"bad test variant tag: {v!r}")
    # Behavioral contract: gold must pass, buggy must fail F2P (invalid-task detector).
    try:
        g = grade(task.gold_code, task)
        if not g.passed:
            problems.append(f"gold_code fails own tests: {g.reason[:160]}")
    except Exception as e:  # noqa: BLE001
        problems.append(f"gold_code grader crash: {e}")
    try:
        b = grade(task.buggy_code, task)
        if b.outcome == "candidate_exec_error":
            problems.append(f"buggy_code does not even exec: {b.reason[:160]}")
        elif b.f2p_passed == b.f2p_total:
            problems.append("buggy_code passes F2P: task does not discriminate the capability gap")
    except Exception as e:  # noqa: BLE001
        problems.append(f"buggy_code grader crash: {e}")
    return problems
