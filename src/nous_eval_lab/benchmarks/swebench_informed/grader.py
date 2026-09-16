"""Deterministic grader: candidate patch → F2P/P2P tests → PASS/FAIL.

Ground truth comes from this grader, which is kept strictly separate
from judges and human gold labels (see locked architecture).
"""
from __future__ import annotations

from dataclasses import dataclass

from .task_schema import Task


@dataclass
class GradeResult:
    task_id: str
    passed: bool
    f2p_passed: int
    f2p_total: int
    p2p_passed: int
    p2p_total: int
    reason: str
    outcome: str = "pass"
    failed_stage: str = "none"


def _run_snippets(candidate_code: str, snippets: list[str]) -> tuple[int, str, str]:
    """Exec candidate code once, then each test snippet. Returns (n_passed, first_error, error_kind)."""
    namespace: dict = {}
    try:
        exec(compile(candidate_code, "<candidate>", "exec"), namespace)
    except Exception as e:  # noqa: BLE001 — grader must not crash
        return 0, f"candidate_exec_error: {type(e).__name__}: {e}", "candidate_exec_error"
    passed = 0
    first_error = ""
    first_kind = ""
    for i, snippet in enumerate(snippets):
        try:
            exec(compile(snippet, f"<test_{i}>", "exec"), namespace)
            passed += 1
        except AssertionError as e:
            if not first_error:
                first_error = f"assert_failed test_{i}: {e or snippet[:120]}"
                first_kind = "fail_assert"
        except Exception as e:  # noqa: BLE001
            if not first_error:
                first_error = f"error test_{i}: {type(e).__name__}: {e}"
                first_kind = "test_error"
    return passed, first_error, first_kind


def grade(candidate_code: str, task: Task) -> GradeResult:
    f2p_passed, f2p_err, f2p_kind = _run_snippets(candidate_code, task.fail_to_pass)
    # If candidate itself is broken, P2P run will repeat the exec error; still report stage precisely.
    p2p_passed, p2p_err, p2p_kind = _run_snippets(candidate_code, task.pass_to_pass)
    f2p_ok = f2p_passed == len(task.fail_to_pass)
    p2p_ok = p2p_passed == len(task.pass_to_pass)
    passed = f2p_ok and p2p_ok
    if passed:
        return GradeResult(
            task_id=task.task_id,
            passed=True,
            f2p_passed=f2p_passed,
            f2p_total=len(task.fail_to_pass),
            p2p_passed=p2p_passed,
            p2p_total=len(task.pass_to_pass),
            reason="all tests passed",
            outcome="pass",
            failed_stage="none",
        )
    if f2p_err.startswith("candidate_exec_error"):
        outcome, stage = "candidate_exec_error", "candidate_exec"
    elif not f2p_ok:
        outcome, stage = (f2p_kind or "fail_assert"), "fail_to_pass"
    elif p2p_err.startswith("candidate_exec_error"):
        outcome, stage = "candidate_exec_error", "candidate_exec"
    else:
        outcome, stage = (p2p_kind or "fail_assert"), "pass_to_pass"
    reason = "; ".join(r for r in [f2p_err, p2p_err] if r) or "failed"
    return GradeResult(
        task_id=task.task_id,
        passed=passed,
        f2p_passed=f2p_passed,
        f2p_total=len(task.fail_to_pass),
        p2p_passed=p2p_passed,
        p2p_total=len(task.pass_to_pass),
        reason=reason,
        outcome=outcome,
        failed_stage=stage,
    )
