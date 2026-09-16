import json

from nous_eval_lab.benchmarks.swebench_informed.grader import grade
from nous_eval_lab.benchmarks.swebench_informed.qa import qa_task
from nous_eval_lab.benchmarks.swebench_informed.task_schema import Task


def _load_tasks():
    with open("src/nous_eval_lab/benchmarks/swebench_informed/tasks.jsonl") as f:
        return [Task.from_dict(json.loads(line)) for line in f if line.strip()]


def test_gold_passes():
    for t in _load_tasks():
        r = grade(t.gold_code, t)
        assert r.passed, f"{t.task_id}: {r.reason}"
        assert r.outcome == "pass" and r.failed_stage == "none"


def test_buggy_fails_f2p():
    for t in _load_tasks():
        r = grade(t.buggy_code, t)
        assert not r.passed, t.task_id
        assert r.f2p_passed < r.f2p_total, t.task_id
        # buggy must still pass happy-path P2P (demonstrates overfitting gap)
        assert r.p2p_passed == r.p2p_total, f"{t.task_id}: {r.reason}"
        assert r.failed_stage == "fail_to_pass", t.task_id
        assert r.outcome in ("fail_assert", "test_error"), t.task_id


def test_qa_clean():
    for t in _load_tasks():
        assert qa_task(t) == [], t.task_id


def test_invalid_patch_distinguished():
    t = _load_tasks()[0]
    r = grade("def broken(:\n  pass", t)
    assert not r.passed
    assert r.outcome == "candidate_exec_error" and r.failed_stage == "candidate_exec"


def test_qa_catches_undiscriminating_task():
    import copy

    t = _load_tasks()[0]
    d = copy.deepcopy(t)
    d.buggy_code = d.gold_code  # buggy == gold: no capability gap
    assert any("discriminate" in p for p in qa_task(d))


def test_schema_rejects_bad():
    import pytest

    with pytest.raises(ValueError):
        Task.from_dict({"task_id": "x"})
