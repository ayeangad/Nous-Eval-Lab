"""Task schema for SWE-bench-informed capability benchmark.

This is NOT a SWE-bench reproduction. It preserves the
issue → patch → tests → grader pattern with lightweight,
deterministic tasks targeting capability gaps.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


VALID_VARIANTS = {"happy_path", "underspecified", "edge", "adversarial"}
VALID_DIFFICULTY = {"easy", "medium", "hard"}

REQUIRED_FIELDS = {
    "task_id",
    "issue_text",
    "buggy_code",
    "gold_code",
    "fail_to_pass",
    "pass_to_pass",
}


@dataclass
class Task:
    task_id: str
    issue_text: str
    buggy_code: str
    gold_code: str
    fail_to_pass: list[str]
    pass_to_pass: list[str]
    capability: list[str] = field(default_factory=list)
    difficulty: str = "medium"
    failure_modes: list[str] = field(default_factory=list)
    variants: list[str] = field(default_factory=list)
    adversarial_issue_text: str = ""
    entrypoint: str = ""
    fail_to_pass_variants: list[str] = field(default_factory=list)
    pass_to_pass_variants: list[str] = field(default_factory=list)

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "Task":
        missing = REQUIRED_FIELDS - set(d.keys())
        if missing:
            raise ValueError(f"task {d.get('task_id', '?')} missing fields: {sorted(missing)}")
        if not d["fail_to_pass"]:
            raise ValueError(f"task {d['task_id']}: fail_to_pass must be non-empty")
        if not d["pass_to_pass"]:
            raise ValueError(f"task {d['task_id']}: pass_to_pass must be non-empty")
        difficulty = d.get("difficulty", "medium")
        if difficulty not in VALID_DIFFICULTY:
            raise ValueError(f"task {d['task_id']}: bad difficulty {difficulty!r}")
        variants = list(d.get("variants", []))
        for v in variants:
            if v not in VALID_VARIANTS:
                raise ValueError(f"task {d['task_id']}: bad variant {v!r}")
        f2p_v = list(d.get("fail_to_pass_variants") or ["underspecified"] * len(d["fail_to_pass"]))
        p2p_v = list(d.get("pass_to_pass_variants") or ["happy_path"] * len(d["pass_to_pass"]))
        if len(f2p_v) != len(d["fail_to_pass"]):
            raise ValueError(f"task {d['task_id']}: fail_to_pass_variants length mismatch")
        if len(p2p_v) != len(d["pass_to_pass"]):
            raise ValueError(f"task {d['task_id']}: pass_to_pass_variants length mismatch")
        for v in f2p_v + p2p_v:
            if v not in VALID_VARIANTS:
                raise ValueError(f"task {d['task_id']}: bad test variant {v!r}")
        return cls(
            task_id=d["task_id"],
            issue_text=d["issue_text"],
            buggy_code=d["buggy_code"],
            gold_code=d["gold_code"],
            fail_to_pass=list(d["fail_to_pass"]),
            pass_to_pass=list(d["pass_to_pass"]),
            capability=list(d.get("capability", [])),
            difficulty=difficulty,
            failure_modes=list(d.get("failure_modes", [])),
            variants=variants,
            adversarial_issue_text=d.get("adversarial_issue_text", ""),
            entrypoint=d.get("entrypoint", ""),
            fail_to_pass_variants=f2p_v,
            pass_to_pass_variants=p2p_v,
        )
