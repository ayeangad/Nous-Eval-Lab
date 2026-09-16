import json

from nous_eval_lab.analysis.regression import compare
from nous_eval_lab.analysis.statistics import wilson_ci
from nous_eval_lab.judges.drift import bootstrap_kappa_ci


def _rows(n_pass, n_fail):
    return [{"passed": True}] * n_pass + [{"passed": False}] * n_fail


def test_wilson_bounds():
    assert wilson_ci(0, 0) == (0.0, 1.0)
    lo, hi = wilson_ci(8, 8)
    assert lo < 1.0 <= hi
    lo, hi = wilson_ci(0, 8)
    assert lo == 0.0 and hi > 0.0


def test_regression_flags_seeded_drop():
    cfg = {"regression": {"overall_drop_pp": 3.0, "capability_drop_pp": 5.0, "min_sample_size": 5}}
    base = [{"passed": True, "capability": ["c"], "variants": ["happy_path"]}] * 8
    cand = ([{"passed": True, "capability": ["c"], "variants": ["happy_path"]}] * 6
            + [{"passed": False, "capability": ["c"], "variants": ["happy_path"]}] * 2)
    cmp = compare(base, cand, cfg)
    assert cmp["regression"] and cmp["delta_pp"] == -25.0


def test_regression_quiet_on_identical():
    cfg = {"regression": {"overall_drop_pp": 3.0, "capability_drop_pp": 5.0, "min_sample_size": 5}}
    base = _rows(6, 2)
    assert compare(base, list(base), cfg)["regression"] is False


def test_regression_respects_min_n():
    cfg = {"regression": {"overall_drop_pp": 3.0, "capability_drop_pp": 5.0, "min_sample_size": 50}}
    assert compare(_rows(8, 0), _rows(0, 8), cfg)["regression"] is False


def test_bootstrap_ci_ordered():
    lo, hi = bootstrap_kappa_ci(["pass"] * 80 + ["fail"] * 120,
                                ["pass"] * 70 + ["fail"] * 10 + ["pass"] * 10 + ["fail"] * 110,
                                b=50, seed=1)
    assert lo <= hi
