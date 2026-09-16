"""Optional live LLM judge (OpenAI-compatible API, stdlib only).

Gated behind OPENAI_API_KEY; never required for tests or CI. When the key
is absent, scripts skip it with a clear message. Responses are cached to
disk (judge_cache.jsonl) for reproducibility and cost control.
"""
from __future__ import annotations

import hashlib
import json
import os
import urllib.request
from pathlib import Path

from .base import Verdict

CACHE = Path("data/predictions/_cache/judge_cache.jsonl")

PROMPT = """You are a strict code-review judge. Given the task spec and a candidate
implementation, decide PASS (implements the spec, including edge cases) or
FAIL. Reply with exactly one JSON object: {{"label": "pass"|"fail",
"confidence": "low"|"medium"|"high", "rationale": "<one sentence>"}}.

SPEC:
{spec}

CANDIDATE:
{code}
"""


def _cache_key(model: str, spec: str, code: str) -> str:
    return hashlib.sha256(f"{model}|{spec}|{code}".encode()).hexdigest()


def _cache_get(key: str) -> dict | None:
    if not CACHE.exists():
        return None
    with open(CACHE) as f:
        for line in f:
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                continue
            if row.get("key") == key:
                return row.get("value")
    return None


def _cache_put(key: str, value: dict) -> None:
    CACHE.parent.mkdir(parents=True, exist_ok=True)
    with open(CACHE, "a") as f:
        f.write(json.dumps({"key": key, "value": value}) + "\n")


class ApiLlmJudge:
    """Live judge. version encodes model + rubric revision."""

    def __init__(self, model: str = "gpt-4o-mini", rubric: str = "rubric_v1") -> None:
        self.model = model
        self.version = f"api_llm_{rubric}_{model}"
        self._key = os.environ.get("OPENAI_API_KEY", "")
        self._base = os.environ.get("OPENAI_BASE_URL", "https://api.openai.com/v1")

    @property
    def available(self) -> bool:
        return bool(self._key)

    def judge(self, sample: dict) -> Verdict:
        if not self._key:
            raise RuntimeError("OPENAI_API_KEY not set; skipping live judge")
        spec, code = sample["prompt"], sample["model_answer"]
        ck = _cache_key(self.model, spec, code)
        hit = _cache_get(ck)
        if hit is None:
            body = json.dumps({
                "model": self.model,
                "messages": [{"role": "user", "content": PROMPT.format(spec=spec, code=code)}],
                "temperature": 0,
                "response_format": {"type": "json_object"},
            }).encode()
            req = urllib.request.Request(
                f"{self._base}/chat/completions", data=body,
                headers={"Content-Type": "application/json",
                         "Authorization": f"Bearer {self._key}"},
            )
            with urllib.request.urlopen(req, timeout=60) as r:
                payload = json.load(r)
            text = payload["choices"][0]["message"]["content"]
            hit = json.loads(text)
            _cache_put(ck, hit)
        label = "pass" if str(hit.get("label", "")).lower().startswith("pass") else "fail"
        conf = str(hit.get("confidence", "medium")).lower()
        if conf not in ("low", "medium", "high"):
            conf = "medium"
        return Verdict(label, conf, str(hit.get("rationale", ""))[:300], self.version)
