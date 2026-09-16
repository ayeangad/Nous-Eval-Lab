# Nous Eval Lab

SWE-bench-**informed** agent evaluation lab (not a SWE-bench reproduction).

Investigated SWE-bench; chose a lightweight deterministic slice preserving
`issue → patch → tests → grader` while targeting underspecified issues,
edge cases, and regression risk. Built to answer four questions:

1. Can we reproduce an agent benchmark reliably?
2. Can we design tasks targeting specific capability gaps?
3. How reliable are judges against human labels?
4. Can we detect evaluation regressions automatically?

## Results (all numbers from real runs in this repo)

Benchmark: **8 tasks** across instruction following, edge-case reasoning,
regression preservation, API behavior, error handling.
`gold` candidate: **8/8**. `buggy` candidate: **0/8** — every buggy patch
passes all pass-to-pass tests while failing fail-to-pass. P2P-only
measurement would score the buggy agent 100%; capability-targeted F2P
tests score it 0%. That gap is the benchmark-design observation.

Judge calibration (200 gold samples, 50/50/50/50 happy/underspecified/
edge/adversarial; single annotator — inter-annotator agreement explicitly
not estimated):

| judge | agreement | Cohen's κ (95% CI) | fail recall |
|---|---|---|---|
| heuristic_v1 (P2P-only) | 0.620 | 0.317 (0.24–0.41) | 0.367 |
| heuristic_v2 (checklist+style) | 0.900 | 0.795 (0.72–0.87) | 0.883 |
| stub_llm_freeform_v1 | 0.630 | 0.206 (0.07–0.34) | 0.750 |
| stub_llm_rubric_v2 | 0.930 | 0.858 (0.78–0.92) | 0.883 |

(`stub_*` are deterministic stand-ins, not real LLMs; live judge is
opt-in via `OPENAI_API_KEY`. See `docs/judge-calibration.md`.)

Reading guide: all judge numbers are **agreement with human gold labels**
from a single annotator — not objective truth. κ measures agreement
beyond chance, not percent correct; agreement, precision, and recall are
reported separately and mean different things.

Notable findings (see `results/calibration/report.md`):

- The P2P-only judge's **high-confidence** verdicts are 51% accurate —
  confidence without grounding is overconfidence.
- The rubric judge's residual errors are **100% grader-gap**: 14 samples
  where the test suite passes but the spec judgment fails. Judge quality
  cannot exceed grader validity.
- Style bans (print/warnings/bare-except) caused all 6 of v2's false
  alarms — strictness misattributed to behavior.
- Judge–judge drift: v1↔v2 κ=0.36 (same backbone, different rubric);
  v2↔rubric κ=0.94 (style-ban ablation only).

Failure analysis: agent failures split by outcome (`fail_assert` ×6,
`test_error` ×2 across the 8 buggy patches); judge disagreements split by
source (`docs/failure-analysis.md`).

Regression: seeded 2-of-8 task revert detected — overall **−25.0pp**,
localized to `edge_case_reasoning −50.0pp` (`results/regression_demo/`).

## Reproduce

```bash
uv sync
uv run pytest -q                                   # 24 tests
uv run nous-eval --candidate gold                  # 8/8
uv run nous-eval --candidate buggy                 # 0/8
uv run python scripts/build_gold.py --check        # 200-sample gold vs grader
uv run python scripts/run_judges.py                # 4 judges, no API keys
uv run python scripts/calibrate.py                 # report + summary.json
uv run python scripts/regression_check.py --demo   # exits 1: regression flagged
docker build -t nous-eval-lab . && docker run --rm nous-eval-lab
```

Docs: `docs/benchmark-design.md`, `docs/judge-calibration.md`,
`docs/failure-analysis.md`. Day-1 frozen baseline: `results/baselines/day1/`
(tag `v0.1.0-day1`).
