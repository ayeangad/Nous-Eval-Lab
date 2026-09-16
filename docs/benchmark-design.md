# Benchmark design (SWE-bench-informed slice)

NOT a SWE-bench reproduction. Preserves `issue → patch → tests → grader`
with lightweight deterministic tasks for rapid judge + failure-analysis
iteration. Roadmap: real SWE-bench Lite instances.

- Why this benchmark: happy-path patches pass while underspecified/edge
  behavior fails. Day-1 proof: a deliberately buggy patch passed 5/5
  pass-to-pass while failing fail-to-pass. Now reproduced across 8 tasks:
  every buggy patch passes all P2P and fails F2P.
- Capability gap: instruction following, edge-case reasoning, regression
  preservation (no input mutation), API behavior, error handling.
- Why existing benchmarks insufficient here: full SWE-bench is
  heavy/non-deterministic per-task Docker; this slice runs 8 tasks in
  <1s with zero network, so the calibration loop stays tight.
- Generation: hand-designed from the gap; each task has happy path +
  underspecified/edge/adversarial variants, `failure_modes`, per-test
  variant tags (`fail_to_pass_variants`), and an adversarial paraphrase
  of the issue text.
- QA (`qa.py`): capability tags, variants incl. happy_path, adversarial
  text when declared, non-empty F2P/P2P, plus the behavioral contract —
  gold must pass its own tests, buggy must fail F2P (else the task does
  not discriminate and is rejected; tested in `test_grader.py`).
- Grader validity: `GradeResult.outcome` distinguishes `pass`,
  `fail_assert`, `test_error`, `candidate_exec_error` with `failed_stage`
  (candidate_exec / fail_to_pass / pass_to_pass) — an invalid patch is
  never conflated with a wrong-but-runnable one.
- Invalid task: gold fails, buggy passes F2P, missing variant coverage,
  or adversarial variant without adversarial text.
- Known limitations: 8 hand-written pure-function tasks (no real repos,
  no tools/network); per-test variant tags default when omitted;
  mutant-based gold answers (see judge-calibration doc) are synthetic.
  Variant deltas in regression reports are task-pass granularity.
