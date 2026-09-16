"""Deterministic stand-ins for LLM judges (documented as such).

Real LLM judges need API keys, cost, and latency -- wrong defaults for a
reproducible calibration harness. These stubs isolate the *methodology*
(same gold set, kappa/P-R/confidence/drift machinery) with two honestly
labeled surface-feature strategies:

- stub_llm_freeform_v1: no rubric, no tests. Scores surface features
  (entrypoint defined, no raise, substantive length, explanatory comment).
  Expected to be fooled by plausible-but-wrong code.
- stub_llm_rubric_v2: runs the full suite (like heuristic_v2) but WITHOUT
  style bans; instead, uncertainty markers lower confidence. Expected to
  match the grader except on grader-blind (grader_sensitive) samples.

Use scripts/run_judges.py --include-api with OPENAI_API_KEY set for the
live judge (see llm_api.py); CI never requires it.
"""
from __future__ import annotations

from .base import Verdict
from .heuristic import _load_tasks
from ..benchmarks.swebench_informed.grader import _run_snippets

UNCERTAINTY_MARKERS = (
    "print(", "warn", "except", "TODO", "root", "USD",
    "rstrip", "tuple(", "iter(", "is_integer", "copy.copy",
)


class StubFreeformJudge:
    version = "stub_llm_freeform_v1"

    def __init__(self) -> None:
        self.tasks = _load_tasks()

    def judge(self, sample: dict) -> Verdict:
        from ..benchmarks.swebench_informed.task_schema import Task  # noqa: F401
        code = sample["model_answer"]
        task = self.tasks[sample["task_id"]]
        feats = {
            "entrypoint": f"def {task.entrypoint}" in code,
            "no_raise": "raise" not in code,
            "substantive": len(code.strip()) > 120,
            "explanatory": "#" in code or '"""' in code,
        }
        score = sum(feats.values())
        label = "pass" if score >= 3 else "fail"
        hits = [k for k, v in feats.items() if v]
        if (label == "pass" and feats["explanatory"]) or (label == "fail" and not feats["entrypoint"]):
            conf = "high"
        else:
            conf = "medium"
        return Verdict(label, conf, f"freeform surface score {score}/4: {hits}", self.version)


class StubRubricJudge:
    version = "stub_llm_rubric_v2"

    def __init__(self) -> None:
        self.tasks = _load_tasks()

    def judge(self, sample: dict) -> Verdict:
        task = self.tasks[sample["task_id"]]
        code = sample["model_answer"]
        f2p, f2p_err, _ = _run_snippets(code, task.fail_to_pass)
        p2p, p2p_err, _ = _run_snippets(code, task.pass_to_pass)
        ok = f2p == len(task.fail_to_pass) and p2p == len(task.pass_to_pass)
        markers = [m for m in UNCERTAINTY_MARKERS if m in code]
        if not ok:
            first = f2p_err or p2p_err
            conf = "low" if markers else "high"
            return Verdict("fail", conf, f"rubric behavior fail: {first}", self.version)
        if markers:
            return Verdict("pass", "low", f"rubric pass with uncertainty markers {markers}", self.version)
        return Verdict("pass", "high", "rubric: suite passes, no uncertainty markers", self.version)


JUDGES = {
    "stub_llm_freeform_v1": StubFreeformJudge,
    "stub_llm_rubric_v2": StubRubricJudge,
}
