from collections import Counter

from nous_eval_lab.judges.calibration import (
    cohen_kappa,
    confidence_table,
    confusion,
    per_class_pr,
)


def test_kappa_perfect():
    assert cohen_kappa(["pass", "fail", "pass"], ["pass", "fail", "pass"]) == 1.0


def test_kappa_chance():
    # po == pe -> 0.0
    assert cohen_kappa(["pass", "pass", "fail", "fail"],
                       ["pass", "fail", "pass", "fail"]) == 0.0


def test_kappa_partial():
    k = cohen_kappa(["pass", "pass", "fail", "fail"], ["pass", "pass", "pass", "fail"])
    assert 0.0 < k < 1.0


def test_confusion_counts():
    c = confusion(["pass", "fail", "fail", "pass"], ["pass", "pass", "fail", "fail"])
    assert c == {"TP": 1, "TN": 1, "FP": 1, "FN": 1}


def test_per_class_pr_support():
    pr = per_class_pr(["pass", "pass", "fail"], ["pass", "fail", "fail"])
    assert pr["pass"]["support"] == 2 and pr["fail"]["support"] == 1
    assert pr["pass"]["precision"] == 1.0 and pr["pass"]["recall"] == 0.5


def test_confidence_table_accuracy():
    rows = [
        {"confidence": "high", "judge_label": "pass", "human_label": "pass"},
        {"confidence": "high", "judge_label": "pass", "human_label": "fail"},
        {"confidence": "low", "judge_label": "fail", "human_label": "fail"},
    ]
    t = confidence_table(rows)
    assert t["high"] == {"n": 2, "accuracy": 0.5}
    assert t["low"] == {"n": 1, "accuracy": 1.0}
