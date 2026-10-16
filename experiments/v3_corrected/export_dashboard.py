"""
Collect every corrected run (and the earlier run of the same model, where one exists) into one file
for the Streamlit dashboard (dashboard_v3.py). No database is needed.

Each answer becomes a row in the format the old dashboard's metrics use
(src/analysis/metrics.py): pred, correct_label, correct, prob_correct.
Method results (prior correction, PriDe, majority vote, PAV, averaging) come from
v3_results.json and hf_results.json, so run update_results.sh first.

    .venv/bin/python3 experiments/v3_corrected/export_dashboard.py
Output: dashboard_data/v3_dashboard.json
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(HERE))
from build_report import EARLIER, OLD, earlier_records  # noqa: E402

SLOT = ["A", "B", "C", "D"]
NUM = ["1", "2", "3", "4"]
OUT = ROOT / "dashboard_data" / "v3_dashboard.json"


def read_rows(path):
    rows = {}
    for line in path.read_text().splitlines():
        try:
            r = json.loads(line)
        except json.JSONDecodeError:
            continue
        if r.get("error") is None:
            rows[(r["item_id"], r["rotation"])] = r          # one answer per question and rotation
    return list(rows.values())


def to_metric_rows(rows, fmt):
    labels = SLOT if fmt == "letter" else NUM
    out = []
    for r in rows:
        pick = r.get("pick_slot")
        row = {"item_id": r["item_id"], "rotation": r["rotation"], "pred": labels[pick] if pick is not None else None,
               "correct_label": labels[r["gold_slot"]], "correct": int(r["correct"])}
        if "probs" in r:
            row["prob_correct"] = round(r["probs"][r["gold_slot"]], 5)
        out.append(row)
    return out


def main():
    v3 = json.loads((HERE / "v3_results.json").read_text())
    hf = json.loads((HERE / "hf_results.json").read_text())
    runs, earlier = {}, {}
    for kind, folder in (("text", "raw"), ("probabilities", "raw_hf")):
        for p in sorted((HERE / folder).glob("*__*__*.jsonl")):
            rows = read_rows(p)
            if len(rows) < 760:                                    # unfinished run
                continue
            first = rows[0]
            fmt, ds = first["format"], first["dataset"]
            model = first["model"].replace("-Instruct", "") if kind == "probabilities" else first["model"]
            key = f"{model} | {ds} | {fmt}"
            res = (hf if kind == "probabilities" else v3).get(key, {})
            methods = res.get("methods") or {}
            if kind == "text" and res.get("mitigation"):
                m = res["mitigation"]
                methods = {"single": {"accuracy": m["single_order"]}, "majority": {"accuracy": m["majority"]},
                           "pav": {"accuracy": m["pav"]},
                           "pav_adaptive": {"accuracy": m["pav_adaptive"], "calls": m["pav_adaptive_calls"]}}
            runs[key] = {
                "model": model, "dataset": ds, "format": fmt,
                "source": first.get("provider", "cloud") if kind == "text" else "probabilities",
                "letters": SLOT if fmt == "letter" else NUM,
                "rows": to_metric_rows(rows, fmt),
                "methods": {k: {x: v[x] for x in ("accuracy", "calls", "vs_single") if x in v} for k, v in methods.items()},
                "prior_corrected_picks": (res.get("spread_prior_heldout") or {}).get("pred_share_all"),
                "label_mass": res.get("label_mass_mean"),                                   # probability runs
                "unreadable": (res.get("bias") or {}).get("invalid_final"),                  # text runs
            }
            # earlier run of the same model (letters only), as in the report
            src = None
            if fmt == "letter":
                if model in EARLIER:
                    src = EARLIER[model][0]
                elif model in OLD:
                    src = OLD[model][0][0]
                else:
                    src = res.get("hf_id") or model
            if src:
                recs = earlier_records(src, ds)
                if recs:
                    earlier[key] = {"source": src, "rows": [
                        {"item_id": i, "pred": SLOT[p] if p is not None else None, "correct_label": SLOT[g], "correct": int(p == g)}
                        for i, p, g in recs]}
    OUT.parent.mkdir(exist_ok=True)
    OUT.write_text(json.dumps({"runs": runs, "earlier": earlier}, separators=(",", ":")))
    print(f"{len(runs)} runs, {len(earlier)} with an earlier run -> {OUT} ({OUT.stat().st_size / 1e6:.1f} MB)")


if __name__ == "__main__":
    main()
