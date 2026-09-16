"""Judge calibration: agreement (kappa), per-class P/R, confusion,
error profiles, confidence-vs-correctness. All computed against human gold.
"""
from __future__ import annotations

from collections import Counter


def cohen_kappa(y_true: list[str], y_pred: list[str]) -> float:
    """Cohen's kappa for two raters on nominal labels."""
    assert len(y_true) == len(y_pred) and len(y_true) > 0
    labels = sorted(set(y_true) | set(y_pred))
    n = len(y_true)
    po = sum(t == p for t, p in zip(y_true, y_pred)) / n
    pe = sum(
        (sum(1 for t in y_true if t == lab) / n) * (sum(1 for p in y_pred if p == lab) / n)
        for lab in labels
    )
    return 1.0 if pe == 1.0 else (po - pe) / (1 - pe)


def confusion(y_true: list[str], y_pred: list[str]) -> dict[str, int]:
    c = Counter(zip(y_true, y_pred))
    return {"TP": c[("pass", "pass")], "TN": c[("fail", "fail")],
            "FP": c[("fail", "pass")], "FN": c[("pass", "fail")]}


def per_class_pr(y_true: list[str], y_pred: list[str]) -> dict[str, dict[str, float]]:
    out = {}
    for lab in ("pass", "fail"):
        tp = sum(1 for t, p in zip(y_true, y_pred) if p == lab and t == lab)
        fp = sum(1 for t, p in zip(y_true, y_pred) if p == lab and t != lab)
        fn = sum(1 for t, p in zip(y_true, y_pred) if p != lab and t == lab)
        out[lab] = {
            "precision": tp / (tp + fp) if tp + fp else 0.0,
            "recall": tp / (tp + fn) if tp + fn else 0.0,
            "support": sum(1 for t in y_true if t == lab),
        }
    return out


def confidence_table(rows: list[dict]) -> dict[str, dict[str, float]]:
    """Accuracy of judge verdicts bucketed by stated confidence."""
    out = {}
    for conf in ("low", "medium", "high"):
        sub = [r for r in rows if r["confidence"] == conf]
        if not sub:
            continue
        acc = sum(1 for r in sub if r["judge_label"] == r["human_label"]) / len(sub)
        out[conf] = {"n": len(sub), "accuracy": acc}
    return out


def error_profile(rows: list[dict]) -> dict[str, dict[str, int | float]]:
    """Disagreement rate sliced by stratum, failure category, prompt kind."""
    out: dict[str, dict] = {}
    for key in ("stratum", "failure_category", "prompt_kind", "task_id"):
        groups: dict[str, list[dict]] = {}
        for r in rows:
            groups.setdefault(str(r.get(key) or "none"), []).append(r)
        out[key] = {
            g: {"n": len(v),
                "disagree": sum(1 for x in v if x["judge_label"] != x["human_label"]),
                "rate": sum(1 for x in v if x["judge_label"] != x["human_label"]) / len(v)}
            for g, v in sorted(groups.items())
        }
    return out


def summarize(rows: list[dict]) -> dict:
    y_true = [r["human_label"] for r in rows]
    y_pred = [r["judge_label"] for r in rows]
    n = len(rows)
    agree = sum(t == p for t, p in zip(y_true, y_pred))
    return {
        "n": n,
        "agreement": agree / n,
        "kappa": cohen_kappa(y_true, y_pred),
        "confusion": confusion(y_true, y_pred),
        "per_class": per_class_pr(y_true, y_pred),
        "confidence": confidence_table(rows),
        "errors": error_profile(rows),
    }
