"""Run judge(s) over the gold samples -> data/predictions/<version>/preds.jsonl.

Usage:
  uv run python scripts/run_judges.py
  uv run python scripts/run_judges.py --judges heuristic_v1,stub_llm_rubric_v2
  uv run python scripts/run_judges.py --judges api --include-api  # needs OPENAI_API_KEY
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from nous_eval_lab.judges.heuristic import ChecklistJudge, P2POnlyJudge  # noqa: E402
from nous_eval_lab.judges.stub import StubFreeformJudge, StubRubricJudge  # noqa: E402

OFFLINE = {
    "heuristic_v1": P2POnlyJudge,
    "heuristic_v2": ChecklistJudge,
    "stub_llm_freeform_v1": StubFreeformJudge,
    "stub_llm_rubric_v2": StubRubricJudge,
}


def load_samples() -> list[dict]:
    with open(ROOT / "data/gold/samples.jsonl") as f:
        return [json.loads(l) for l in f if l.strip()]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--judges", default=",".join(OFFLINE),
                    help="comma-separated versions, or 'api'")
    ap.add_argument("--include-api", action="store_true")
    args = ap.parse_args()

    names = [n.strip() for n in args.judges.split(",") if n.strip()]
    samples = load_samples()
    for name in names:
        if name == "api":
            from nous_eval_lab.judges.llm_api import ApiLlmJudge

            judge = ApiLlmJudge()
            if not judge.available and not args.include_api:
                print("api judge skipped: OPENAI_API_KEY not set (pass --include-api with key)")
                continue
        else:
            judge = OFFLINE[name]()
        out = ROOT / "data/predictions" / judge.version
        out.mkdir(parents=True, exist_ok=True)
        t0 = time.time()
        with open(out / "preds.jsonl", "w") as f:
            for s in samples:
                v = judge.judge(s)
                f.write(json.dumps({
                    "sample_id": s["sample_id"], "judge_version": v.judge_version,
                    "label": v.label, "confidence": v.confidence,
                    "rationale": v.rationale, "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                }) + "\n")
        print(f"{judge.version}: {len(samples)} preds in {time.time()-t0:.1f}s -> {out/'preds.jsonl'}")


if __name__ == "__main__":
    main()
