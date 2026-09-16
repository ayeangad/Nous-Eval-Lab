"""Judge protocol. A judge maps (prompt, model_answer) -> verdict.

Kept strictly separate from the deterministic grader (ground truth
signal) and from human gold labels. See locked architecture.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol


@dataclass
class Verdict:
    label: str  # "pass" | "fail"
    confidence: str  # "low" | "medium" | "high"
    rationale: str
    judge_version: str


class Judge(Protocol):
    version: str

    def judge(self, sample: dict) -> Verdict:
        ...
