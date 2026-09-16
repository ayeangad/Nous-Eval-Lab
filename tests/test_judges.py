"""Judge experiment tests: determinism, schema, and the core empirical claims."""
import json

from nous_eval_lab.judges.calibration import cohen_kappa
from nous_eval_lab.judges.heuristic import ChecklistJudge, P2POnlyJudge
from nous_eval_lab.judges.stub import StubFreeformJudge, StubRubricJudge

JUDGES = [P2POnlyJudge(), ChecklistJudge(), StubFreeformJudge(), StubRubricJudge()]


def _samples():
    with open("data/gold/samples.jsonl") as f:
        return [json.loads(l) for l in f if l.strip()]


def _labels():
    with open("data/gold/labels.jsonl") as f:
        return {json.loads(l)["sample_id"]: json.loads(l)["human_label"] for l in f if l.strip()}


def test_verdict_schema_and_determinism():
    s = _samples()[0]
    for j in JUDGES:
        v1, v2 = j.judge(s), j.judge(s)
        assert (v1.label, v1.confidence) == (v2.label, v2.confidence)
        assert v1.label in ("pass", "fail") and v1.confidence in ("low", "medium", "high")


def test_checklist_beats_p2p_only():
    samples, labels = _samples(), _labels()
    k = {}
    for j in JUDGES:
        yp = [j.judge(s).label for s in samples]
        k[j.version] = cohen_kappa([labels[s["sample_id"]] for s in samples], yp)
    assert k["heuristic_v2"] > k["heuristic_v1"]
    assert k["stub_llm_rubric_v2"] > k["stub_llm_freeform_v1"]


def test_p2p_only_never_rejects_human_pass():
    # By construction v1 runs P2P only, and every human-pass answer passes P2P.
    samples, labels = _samples(), _labels()
    j = P2POnlyJudge()
    for s in samples:
        if labels[s["sample_id"]] == "pass":
            assert j.judge(s).label == "pass", s["sample_id"]
