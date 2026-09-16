"""Calibrate judges against human gold -> results/calibration/report.md.

Joins data/gold/{samples,labels}.jsonl with data/predictions/*/preds.jsonl
and computes kappa, per-class P/R, confusion, error profiles, confidence
analysis, drift slices, and failure-source prevalence. Prints a summary and
writes the full markdown report + machine-readable summary.json.
"""
from __future__ import annotations

import json
import sys
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from nous_eval_lab.analysis.failure_taxonomy import attribute, prevalence  # noqa: E402
from nous_eval_lab.analysis.statistics import wilson_ci  # noqa: E402
from nous_eval_lab.judges.calibration import summarize  # noqa: E402
from nous_eval_lab.judges.drift import bootstrap_kappa_ci, by_group_kappa, version_drift  # noqa: E402


def load() -> tuple[list[dict], dict[str, dict], dict[str, dict], dict[str, dict[str, dict]]]:
    samples = [json.loads(l) for l in open(ROOT / "data/gold/samples.jsonl")]
    labels = {json.loads(l)["sample_id"]: json.loads(l)
              for l in open(ROOT / "data/gold/labels.jsonl")}
    preds: dict[str, dict[str, dict]] = {}
    for d in sorted((ROOT / "data/predictions").glob("*/preds.jsonl")):
        ver = d.parent.name
        if ver.startswith("_"):
            continue
        preds[ver] = {json.loads(l)["sample_id"]: json.loads(l)
                      for l in open(d) if l.strip()}
    smap = {s["sample_id"]: s for s in samples}
    return samples, smap, labels, preds


def main() -> None:
    samples, smap, labels, preds = load()
    out = ROOT / "results/calibration"
    out.mkdir(parents=True, exist_ok=True)
    summary = {}
    by_ver: dict[str, dict[str, str]] = {}
    md = ["# Judge calibration report", "",
          f"gold: {len(samples)} samples, single annotator (annotator_01); "
          "inter-annotator agreement not estimated.", ""]
    for ver in sorted(preds):
        rows = []
        for sid, p in preds[ver].items():
            if sid not in labels:
                continue
            lab = labels[sid]
            rows.append({
                "sample_id": sid, "human_label": lab["human_label"],
                "judge_label": p["label"], "confidence": p["confidence"],
                "stratum": smap[sid]["stratum"], "prompt_kind": smap[sid]["prompt_kind"],
                "task_id": smap[sid]["task_id"],
                "failure_category": lab["failure_category"] or "none",
            })
        rep = summarize(rows)
        yt = [r["human_label"] for r in rows]
        yp = [r["judge_label"] for r in rows]
        agree_n = sum(a == b for a, b in zip(yt, yp))
        lo, hi = bootstrap_kappa_ci(yt, yp)
        wlo, whi = wilson_ci(agree_n, len(rows))
        # failure sources on disagreements
        att = []
        for r in rows:
            src, ev = attribute(smap[r["sample_id"]], labels[r["sample_id"]], r["judge_label"])
            att.append({"sample_id": r["sample_id"], "source": src, "evidence": ev,
                        "agree": r["human_label"] == r["judge_label"]})
        disag_sources = prevalence([a for a in att if not a["agree"]])
        pk = by_group_kappa(rows, "prompt_kind")
        st = by_group_kappa(rows, "stratum")
        conf_acc = {k: round(v["accuracy"], 3) for k, v in rep["confidence"].items()}
        pk_k = {k: round(v["kappa"], 3) for k, v in pk.items()}
        st_k = {k: round(v["kappa"], 3) for k, v in st.items()}
        disag_share = {k: round(v["share"], 3) for k, v in disag_sources.items()}
        summary[ver] = {**rep, "kappa_ci95": [lo, hi], "agree_ci95": [round(wlo, 4), round(whi, 4)],
                        "disagreement_sources": disag_sources}
        by_ver[ver] = {r["sample_id"]: r["judge_label"] for r in rows}
        md += [f"## {ver} (n={rep['n']})",
               f"agreement {rep['agreement']:.3f} (95% CI {wlo:.3f}-{whi:.3f}), "
               f"kappa {rep['kappa']:.3f} (95% CI {lo}-{hi})",
               f"confusion {rep['confusion']}",
               f"per-class P/R: pass P={rep['per_class']['pass']['precision']:.3f} "
               f"R={rep['per_class']['pass']['recall']:.3f}; fail "
               f"P={rep['per_class']['fail']['precision']:.3f} "
               f"R={rep['per_class']['fail']['recall']:.3f}",
               f"confidence: {conf_acc}",
               f"prompt-kind kappa: {pk_k}",
               f"stratum kappa: {st_k}",
               f"disagreement sources: {disag_share}", ""]
    md += ["## Version drift (judge-judge kappa)", f"{version_drift(by_ver)}", ""]
    (out / "summary.json").write_text(json.dumps(summary, indent=2))
    (out / "report.md").write_text("\n".join(md))
    print("\n".join(md))
    print(f"\nwrote {out/'report.md'} and summary.json")


if __name__ == "__main__":
    main()
