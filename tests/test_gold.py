"""Gold-set construction tests: verify the generator (method), not just the artifact."""
from collections import Counter

from scripts.build_gold import build, check


def test_stratified_200():
    samples, labels = build()
    assert len(samples) == 200 == len(labels)
    assert Counter(s["stratum"] for s in samples) == {
        "happy_path": 50, "underspecified": 50, "edge": 50, "adversarial": 50}
    ids = [s["sample_id"] for s in samples]
    assert len(set(ids)) == 200
    assert {l["sample_id"] for l in labels} == set(ids)


def test_both_classes_and_flags():
    _, labels = build()
    c = Counter(l["human_label"] for l in labels)
    assert c["pass"] > 50 and c["fail"] > 50
    assert sum(1 for l in labels if l["ambiguous"]) > 0
    assert sum(1 for l in labels if l["grader_sensitive"]) > 0
    assert {l["annotator_id"] for l in labels} == {"annotator_01"}


def test_labels_match_grader_except_flagged():
    samples, labels = build()
    assert check(samples, labels) == 0
