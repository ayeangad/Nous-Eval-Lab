# Judge calibration

Centerpiece experiment: same 200-sample gold set judged by 4
configurations, then reliability / error-profile / confidence / drift.

## Annotation protocol

- Single annotator (`annotator_01`): the repo author. Each of the 200
  samples judged against the issue SPEC (not against the test suite).
- Inter-annotator agreement is therefore NOT estimated. The schema
  reserves `annotator_id` for future double-labeling (target: 50–75
  overlap samples, human↔human κ reported separately from judge↔human κ).

## Gold-set construction (`scripts/build_gold.py`, seed 42)

- 8 tasks × 25 samples = 200; strata exactly 50 happy_path /
  50 underspecified / 50 edge / 50 adversarial (rotation in generator).
- 12 answers/task (gold, gold-style, buggy, buggy-style, 5 mutants,
  2 ambiguous, 1 grader-gap) × 2 prompt phrasings + 1 empty response.
- Class balance: 80 pass / 120 fail. Flags: 34 ambiguous, 22
  grader-sensitive.
- `samples.jsonl` (prompt + answer) is strictly separate from
  `labels.jsonl` (human judgment). Predictions live under
  `data/predictions/<judge_version>/`.
- `--check` verifies every non-grader-sensitive label matches the
  grader: 0 unexpected mismatches; 14 intentional, flagged divergences
  where the suite passes but the spec judgment fails
  (e.g. `clamp/g1_negatives`, `parse/g1_int_float`, `dedup/g1_scale`).

## Judge versions

- `heuristic_v1` (P2P-only): regression-test mimic; blind to F2P by design.
- `heuristic_v2` (checklist): full suite + style bans
  (`print(`, `warnings.warn`, bare `except`).
- `stub_llm_freeform_v1`: surface features only (entrypoint, no raise,
  length, comments). A documented stand-in, not an LLM.
- `stub_llm_rubric_v2`: full suite, no style bans; uncertainty markers
  lower confidence instead of flipping the label.
- `api_llm_*` (`judges/llm_api.py`): opt-in live judge, disk-cached,
  skipped without `OPENAI_API_KEY`.

## Methods

- κ: Cohen's kappa (own implementation, `calibration.py`); 95% CIs via
  1000-seed bootstrap (`drift.py`); agreement CIs via Wilson
  (`analysis/statistics.py`).
- P/R: per-class precision/recall + confusion; error profile sliced by
  stratum / failure_category / prompt_kind / task.
- Confidence: accuracy bucketed by stated low/medium/high.
- Drift: judge-vs-gold κ by prompt_kind and stratum; pairwise
  judge–judge κ matrix; bootstrap CI stability.

## Results (current, `results/calibration/report.md`)

All figures below are agreement with single-annotator human gold labels.
κ is agreement beyond chance, not percent correct.

- v1 κ=0.317: P2P-only judging misses F2P failures (fail recall 0.367);
  74% of its disagreements are leniency; high-confidence accuracy 0.51.
- v2 κ=0.795: all 6 false alarms are style bans, all flagged low
  confidence (low bucket accuracy 0.0 — uncertainty pointed the right
  way, toward strictness).
- freeform κ=0.206: surface features fail both ways (46% strictness).
- rubric κ=0.858: residual errors 100% grader-gap — the ceiling is the
  suite, not the judge. Low-confidence bucket 0.85 vs high 0.94.
- Drift: prompt-kind κ differs (v1: 0.35 adv vs 0.27 std); v1↔v2
  judge-judge κ=0.36 — changing the rubric changes the measurement more
  than changing the answers.

## Limitations

Single annotator; synthetic mutant answers (real model trajectories
would be stronger); stubs instead of live LLMs in the default path;
200 samples give wide CIs on slices (reported, not hidden).
