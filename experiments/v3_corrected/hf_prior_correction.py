"""
Label-free prior correction for the HF runs (answers chosen from letter probabilities,
so they are NOT affected by the text-parsing problem).

For each (model, dataset): p(letter | prompt) renormalised over A-D; the model's letter
prior is the mean of these vectors over all prompts (no gold labels used, as in Batch
Calibration, Zhou et al., ICLR 2024). Corrected score = log p(letter | prompt) - log prior(letter).
Every prompt is corrected on its own, so the result does not depend on the shuffle design.

    .venv/bin/python3 experiments/v3_corrected/hf_prior_correction.py
"""
from __future__ import annotations

import csv
import json
import math
import sys
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
L = ["A", "B", "C", "D"]
RUNS = ROOT / "results_and_runs/runs_20260316235711.csv"
RESULTS = ROOT / "results_and_runs/results_20260316235711.csv"
MODELS = ["Qwen/Qwen2.5-0.5B-Instruct", "meta-llama/Llama-3.2-1B-Instruct"]


def summarize(rows, pick):
    acc_pos = {}
    for lab in L:
        sub = [r for r in rows if r["gold"] == lab]
        acc_pos[lab] = sum(pick(r) == lab for r in sub) / len(sub)
    preds = Counter(pick(r) for r in rows)
    return {
        "accuracy": sum(pick(r) == r["gold"] for r in rows) / len(rows),
        "acc_by_position": acc_pos,
        "sensitivity_gap": max(acc_pos.values()) - min(acc_pos.values()),
        "pred_share": {lab: preds[lab] / len(rows) for lab in L},
    }


def main():
    csv.field_size_limit(sys.maxsize)
    runs = {r["id"]: (r["model"], r["dataset"]) for r in csv.DictReader(open(RUNS))}
    data = defaultdict(list)
    for r in csv.DictReader(open(RESULTS)):
        key = runs.get(r["run_id"])
        if key and key[0] in MODELS:
            lp = json.loads(r["logprobs"])
            p = [math.exp(lp[lab]) for lab in L]
            z = sum(p)
            data[key].append({"gold": r["correct_label"], "p": [x / z for x in p]})
    out = {}
    for (model, ds), rows in sorted(data.items()):
        prior = [sum(r["p"][i] for r in rows) / len(rows) for i in range(4)]
        before = summarize(rows, lambda r: L[max(range(4), key=lambda i: r["p"][i])])
        after = summarize(rows, lambda r: L[max(range(4), key=lambda i: math.log(r["p"][i] + 1e-12) - math.log(prior[i]))])
        out[f"{model} | {ds}"] = {"n": len(rows), "prior": dict(zip(L, prior)), "before": before, "after": after}
        print(f"\n### {model} | {ds} (n={len(rows)})   label-free prior: " + " ".join(f"{l}={v:.0%}" for l, v in zip(L, prior)))
        for name, s in (("before", before), ("after prior correction", after)):
            print(f"  {name:<24} acc={s['accuracy']:.1%}  gap={s['sensitivity_gap']:.1%}  "
                  f"acc A/B/C/D=" + "/".join(f"{s['acc_by_position'][l]:.0%}" for l in L) +
                  "  predicted A/B/C/D=" + "/".join(f"{s['pred_share'][l]:.0%}" for l in L))
    (Path(__file__).resolve().parent / "hf_prior_correction.json").write_text(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
