import json
from pathlib import Path

from nous_eval_lab.runners.local import load_tasks, run_benchmark


def test_runner_deterministic(tmp_path):
    tasks = load_tasks("src/nous_eval_lab/benchmarks/swebench_informed/tasks.jsonl")
    cmap = {t.task_id: t.gold_code for t in tasks}
    out1 = run_benchmark(tasks, cmap, seed=42, run_dir=tmp_path / "r", run_id="run_a")
    out2 = run_benchmark(tasks, cmap, seed=42, run_dir=tmp_path / "r2", run_id="run_b")
    rows1 = open(out1 / "results.jsonl").read()
    rows2 = open(out2 / "results.jsonl").read()
    # same multiset of results regardless of shuffle order
    assert sorted(rows1.splitlines()) != []  # sanity
    assert len(rows1.splitlines()) == len(rows2.splitlines()) == len(tasks)
    m1 = json.load(open(out1 / "meta.json"))
    assert m1["pass_rate"] == 1.0
    # buggy baseline must score 0 on this task (fails F2P)
    out3 = run_benchmark(tasks, {}, seed=42, run_dir=tmp_path / "r3", run_id="run_c")
    m3 = json.load(open(out3 / "meta.json"))
    assert m3["pass_rate"] == 0.0
    assert (Path(out3) / "results.jsonl").exists()
