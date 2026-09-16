"""Regression detection: baseline vs candidate runs.

Compares overall + per-capability + per-variant pass rates. Flags when a
drop meets the configured engineering alert thresholds (see configs) AND
the sample size meets min_sample_size. Thresholds are NOT significance
tests; use analysis.statistics.wilson_ci for uncertainty.
"""
from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path


def load_results(run_dir: str | Path) -> list[dict]:
    rows = []
    with open(Path(run_dir) / "results.jsonl") as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def rate(rows: list[dict]) -> float:
    return sum(1 for r in rows if r["passed"]) / len(rows) if rows else 0.0


def breakdown(rows: list[dict], key: str) -> dict[str, float]:
    groups: dict[str, list[dict]] = defaultdict(list)
    for r in rows:
        vals = r.get(key) or ["none"]
        if isinstance(vals, str):
            vals = [vals]
        for v in vals:
            groups[v].append(r)
    return {k: rate(v) for k, v in sorted(groups.items())}


def compare(baseline: list[dict], candidate: list[dict], cfg: dict) -> dict:
    reg = cfg.get("regression", {})
    overall_pp = float(reg.get("overall_drop_pp", 3.0))
    cap_pp = float(reg.get("capability_drop_pp", 5.0))
    min_n = int(reg.get("min_sample_size", 20))
    base_rate, cand_rate = rate(baseline), rate(candidate)
    delta_pp = (cand_rate - base_rate) * 100
    flags = []
    if len(candidate) >= min_n and delta_pp <= -overall_pp:
        flags.append(f"overall {delta_pp:+.1f}pp <= -{overall_pp}pp")
    cap_deltas = {}
    for cap, br in breakdown(baseline, "capability").items():
        cr = breakdown(candidate, "capability").get(cap)
        if cr is None:
            continue
        d = (cr - br) * 100
        cap_deltas[cap] = round(d, 2)
        if len(candidate) >= min_n and d <= -cap_pp:
            flags.append(f"capability '{cap}' {d:+.1f}pp <= -{cap_pp}pp")
    var_deltas = {
        k: round((breakdown(candidate, "variants").get(k, 0.0) - v) * 100, 2)
        for k, v in breakdown(baseline, "variants").items()
    }
    return {
        "n_baseline": len(baseline), "n_candidate": len(candidate),
        "baseline_rate": round(base_rate, 4), "candidate_rate": round(cand_rate, 4),
        "delta_pp": round(delta_pp, 2),
        "capability_deltas_pp": cap_deltas, "variant_deltas_pp": var_deltas,
        "regression": bool(flags), "flags": flags,
        "note": "thresholds are engineering alerts, not significance tests",
    }


def render_markdown(cmp: dict) -> str:
    lines = ["# Regression report", "",
             f"baseline {cmp['baseline_rate']:.2%} (n={cmp['n_baseline']}) -> "
             f"candidate {cmp['candidate_rate']:.2%} (n={cmp['n_candidate']}) = "
             f"{cmp['delta_pp']:+.1f}pp",
             f"verdict: {'REGRESSION' if cmp['regression'] else 'no regression'}"]
    for f in cmp["flags"]:
        lines.append(f"- FLAG: {f}")
    lines.append("")
    lines.append("capability deltas (pp): " + ", ".join(
        f"{k} {v:+.1f}" for k, v in sorted(cmp["capability_deltas_pp"].items())))
    lines.append("variant deltas (pp): " + ", ".join(
        f"{k} {v:+.1f}" for k, v in sorted(cmp["variant_deltas_pp"].items())))
    lines.append("")
    lines.append(f"_{cmp['note']}_")
    return "\n".join(lines) + "\n"
