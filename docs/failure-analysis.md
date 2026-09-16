# Failure analysis

## Agent failures (8 buggy patches vs 8 tasks)

All 8 fail at `fail_to_pass` with P2P intact — the suite's happy-path
coverage cannot see the capability gap. Outcomes: `fail_assert` ×6
(clamp, dedup, merge, truncate, top_k, access), `test_error` ×2
(parse raises ValueError on commas; safe_div raises ZeroDivisionError).
No `candidate_exec_error`: all patches are runnable, all are wrong.
Source attribution: `model_wrong` (behavior contradicts spec and the
suite catches it).

## Judge disagreements (vs 200-sample gold)

Per-judge disagreement sources (`results/calibration/summary.json`):

- heuristic_v1 (76 disagreements): leniency 74%, grader_gap 18%,
  spec_ambiguous 8%. The P2P-only design explains the leniency mass.
- heuristic_v2 (20): grader_gap 70%, spec_ambiguous 20%, strictness 10%
  (6 style-ban false alarms, all low confidence).
- freeform (74): strictness 46%, leniency 19%, spec_ambiguous 22%,
  grader_gap 13%. Surface features misfire symmetrically.
- rubric (14): grader_gap 100%. Every residual is a suite-validity
  issue, e.g. `parse_price/g1_int_float` (int 5 == 5.0 passes, spec
  requires float), `clamp/g1_negatives` (wrong for negative ranges),
  `dedup/g1_scale` (order breaks past n=1000).

## Recommendations

1. Training/data: add negative-range, tab-whitespace, scale, and
   type-strictness cases to the suites (the 14 grader-gap samples are
   the exact test backlog).
2. Judge prompts: forbid style-based fails unless behavior fails (v2 →
   rubric ablation gained κ 0.795 → 0.858 with zero new misses).
3. Benchmark: P2P-only gates must never certify a patch; require F2P
   coverage per capability (the Day-1 observation, now ×8).
