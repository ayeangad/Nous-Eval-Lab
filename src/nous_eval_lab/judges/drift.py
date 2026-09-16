"""Drift analysis: does judge behavior shift across prompts, strata, versions?

- by_group_kappa: judge-vs-gold kappa sliced by prompt_kind / stratum.
- version_drift: pairwise judge-vs-judge kappa (do versions disagree?).
- bootstrap_kappa_ci: nonparametric CI for kappa (B resamples, seeded).
"""
from __future__ import annotations

import random

from .calibration import cohen_kappa


def by_group_kappa(rows: list[dict], group: str) -> dict[str, dict[str, float]]:
    groups: dict[str, list[dict]] = {}
    for r in rows:
        groups.setdefault(str(r.get(group)), []).append(r)
    out = {}
    for g, v in sorted(groups.items()):
        yt = [x["human_label"] for x in v]
        yp = [x["judge_label"] for x in v]
        out[g] = {"n": len(v), "agreement": sum(a == b for a, b in zip(yt, yp)) / len(v),
                  "kappa": cohen_kappa(yt, yp)}
    return out


def version_drift(preds_by_version: dict[str, dict[str, str]]) -> dict[str, dict[str, float]]:
    """preds_by_version: version -> {sample_id: label}. Pairwise judge agreement."""
    versions = sorted(preds_by_version)
    out: dict[str, dict[str, float]] = {}
    for a in versions:
        out[a] = {}
        for b in versions:
            ids = sorted(set(preds_by_version[a]) & set(preds_by_version[b]))
            ya = [preds_by_version[a][i] for i in ids]
            yb = [preds_by_version[b][i] for i in ids]
            out[a][b] = round(cohen_kappa(ya, yb), 4) if ids else 0.0
    return out


def bootstrap_kappa_ci(
    y_true: list[str], y_pred: list[str], b: int = 1000, seed: int = 42, level: float = 0.95
) -> tuple[float, float]:
    rng = random.Random(seed)
    n = len(y_true)
    stats = []
    for _ in range(b):
        idx = [rng.randrange(n) for _ in range(n)]
        try:
            stats.append(cohen_kappa([y_true[i] for i in idx], [y_pred[i] for i in idx]))
        except AssertionError:
            continue
    stats.sort()
    alpha = 1 - level
    lo = stats[int(alpha / 2 * len(stats))]
    hi = stats[int((1 - alpha / 2) * len(stats)) - 1]
    return round(lo, 4), round(hi, 4)
