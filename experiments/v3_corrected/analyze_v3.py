"""
Analysis of the corrected v3 cloud runs (experiments/v3_corrected/raw/*.jsonl).

Per model x dataset x format:
  bias      : accuracy, accuracy by position of the correct answer, sensitivity gap,
              RStd (std of per-position recall, Zheng et al. ICLR 2024), share of picks per slot,
              invalid rate, CircularEval (all 4 rotations correct), fluctuation (answer content
              changes across rotations)
  mitigation: majority vote over the 4 cyclic rotations, PAV (label-free EM),
              PAV-adaptive (EIG, tau=0.95, randomised start, 20 seeds), majority stop-at-2
  letters vs numbers: paired exact McNemar test over (item, rotation)

    .venv/bin/python3 experiments/v3_corrected/analyze_v3.py
"""
from __future__ import annotations

import json
import math
import random
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(ROOT / "experiments/v3_pav"))
from offline_eval import (K, SEEDS, boot_ci, fit_C, frac_correct, log_post, majority,  # noqa: E402
                          sequential)

SLOT = ["A", "B", "C", "D"]


def load():
    rows = defaultdict(list)
    for p in sorted((HERE / "raw").glob("*.jsonl")):
        for line in p.read_text().splitlines():
            try:
                r = json.loads(line)
            except json.JSONDecodeError:                      # a line still being written by a running job
                continue
            if r.get("error") is None:
                rows[(r["model"], r["dataset"], r["format"])].append(r)
    return rows


def mcnemar_exact(b, c):
    n = b + c
    if n == 0:
        return 1.0
    k = min(b, c)
    return min(1.0, 2 * sum(math.comb(n, i) for i in range(k + 1)) / 2 ** n)


def bias_stats(rows):
    acc_pos = []
    for s in range(K):
        sub = [r for r in rows if r["gold_slot"] == s]
        acc_pos.append(sum(r["correct"] for r in sub) / len(sub) if sub else float("nan"))
    picks = [r["pick_slot"] for r in rows if r["pick_slot"] is not None]
    by_item = defaultdict(list)
    for r in rows:
        by_item[r["item_id"]].append(r)
    full = [v for v in by_item.values() if len(v) == K]
    errs = [r for r in rows if not r["correct"]]
    share = lambda sub, key: {SLOT[s]: sum(r[key] == s for r in sub) / len(sub) for s in range(K)} if sub else None
    return {
        "gold_share": share(rows, "gold_slot"),
        "pred_share_all": share(rows, "pick_slot"),      # over all prompts; the remainder had no readable answer
        "n_errors": len(errs),
        "errors_pred_at": share(errs, "pick_slot"),       # wrong answers by the position picked (no-answer rows in no slot)
        "errors_correct_at": share(errs, "gold_slot"),    # wrong answers by the position of the correct answer
        "n_rows": len(rows),
        "accuracy": sum(r["correct"] for r in rows) / len(rows),
        "acc_by_position": dict(zip(SLOT, acc_pos)),
        "sensitivity_gap": max(acc_pos) - min(acc_pos),
        "rstd": float(np.std(acc_pos)) * 100,
        "pick_share": {SLOT[s]: picks.count(s) / len(picks) if picks else 0 for s in range(K)},
        "invalid_first_try": sum(r["parse"] in ("invalid", "format_retry") for r in rows) / len(rows),
        "invalid_final": sum(r["pick_slot"] is None for r in rows) / len(rows),
        "circular_eval": sum(all(r["correct"] for r in v) for v in full) / len(full) if full else None,
        "fluctuation": sum(len({r["pick_idx"] for r in v}) > 1 or any(r["pick_idx"] is None for r in v)
                           for v in full) / len(full) if full else None,
    }


def to_items(rows):
    by_item = defaultdict(list)
    for r in rows:
        by_item[r["item_id"]].append(r)
    items = []
    for iid, v in by_item.items():
        if len(v) != K:
            continue
        v.sort(key=lambda r: r["rotation"])
        obs = [([r["order"].index(c) for c in range(K)], r["pick_slot"]) for r in v]
        items.append({"id": iid, "gold": v[0]["gold_idx"], "obs": obs})
    return items


def mitigation_stats(items):
    n = len(items)
    single = float(np.mean([np.mean([p is not None and so[it["gold"]] == p for so, p in it["obs"]]) for it in items]))
    maj_i = [frac_correct(majority(it), it["gold"]) for it in items]
    C = fit_C(items)
    pav_i = [frac_correct(log_post(it, C), it["gold"]) for it in items]
    # PAV-adaptive with 2-fold cross-fitted C, randomised query order
    idx = list(range(n)); random.Random(0).shuffle(idx)
    fold = {i: (0 if j < n // 2 else 1) for j, i in enumerate(idx)}
    Cf = [fit_C([items[i] for i in range(n) if fold[i] != f]) for f in (0, 1)]
    ad_i = np.zeros(n); calls = 0.0
    for seed in range(SEEDS):
        rs = random.Random(seed)
        for i, it in enumerate(items):
            a, used = sequential(it, Cf[fold[i]], 0.95, "eig", rs)
            ad_i[i] += a / SEEDS; calls += used / (SEEDS * n)
    m2 = np.zeros(n); calls2 = 0.0
    for seed in range(SEEDS):
        rs = random.Random(seed)
        for i, it in enumerate(items):
            obs = rs.sample(it["obs"], K)
            f2 = majority({"obs": obs}, upto=2)
            if max(f2) == 2:
                m2[i] += frac_correct(f2, it["gold"]) / SEEDS; calls2 += 2 / (SEEDS * n)
            else:
                m2[i] += frac_correct(majority(it), it["gold"]) / SEEDS; calls2 += K / (SEEDS * n)
    return {
        "n_items": n,
        "single_order": single,
        "majority": float(np.mean(maj_i)),
        "pav": float(np.mean(pav_i)),
        "pav_minus_majority_ci95": boot_ci(maj_i, pav_i),
        "pav_adaptive": float(ad_i.mean()), "pav_adaptive_calls": calls,
        "pav_adaptive_minus_majority_ci95": boot_ci(maj_i, ad_i),
        "majority_stop2": float(m2.mean()), "majority_stop2_calls": calls2,
        "label_free_acc_by_position": dict(zip(SLOT, np.diag(C).tolist())),
    }


def main():
    rows = load()
    out = {}
    models = sorted({m for m, _, _ in rows})
    for model in models:
        for ds in ("allenai_ai2_arc", "allenai_openbookqa"):
            for fmt in ("letter", "number"):
                rs = rows.get((model, ds, fmt))
                if not rs:
                    continue
                rec = {"bias": bias_stats(rs)}
                items = to_items(rs)
                if len(items) >= 20:
                    rec["mitigation"] = mitigation_stats(items)
                out[f"{model} | {ds} | {fmt}"] = rec
            L, N = rows.get((model, ds, "letter")), rows.get((model, ds, "number"))
            if L and N:
                key = lambda r: (r["item_id"], r["rotation"])
                nmap = {key(r): r["correct"] for r in N}
                pairs = [(r["correct"], nmap[key(r)]) for r in L if key(r) in nmap]
                b = sum(1 for a, c in pairs if a == 1 and c == 0)
                c = sum(1 for a, c2 in pairs if a == 0 and c2 == 1)
                out[f"{model} | {ds} | letters_vs_numbers"] = {
                    "pairs": len(pairs), "letters_only_correct": b, "numbers_only_correct": c,
                    "mcnemar_exact_p": mcnemar_exact(b, c)}

    (HERE / "v3_results.json").write_text(json.dumps(out, indent=2))
    for k, v in out.items():
        if "bias" in v:
            b, m = v["bias"], v.get("mitigation", {})
            print(f"\n### {k}  (rows={b['n_rows']})")
            print(f"  accuracy {b['accuracy']:.1%} | by position " + " ".join(f"{s}={b['acc_by_position'][s]:.0%}" for s in SLOT)
                  + f" | gap {b['sensitivity_gap']:.1%} | RStd {b['rstd']:.1f} | picks " + " ".join(f"{s}={b['pick_share'][s]:.0%}" for s in SLOT))
            print(f"  invalid first try {b['invalid_first_try']:.1%}, final {b['invalid_final']:.1%} | CircularEval {b['circular_eval']:.1%} | fluctuation {b['fluctuation']:.1%}")
            if m:
                lo, hi = m["pav_minus_majority_ci95"]; lo2, hi2 = m["pav_adaptive_minus_majority_ci95"]
                print(f"  single {m['single_order']:.1%} | majority {m['majority']:.1%} | PAV {m['pav']:.1%} [{lo:+.1f},{hi:+.1f}] | "
                      f"PAV-adaptive {m['pav_adaptive']:.1%} @ {m['pav_adaptive_calls']:.2f} calls [{lo2:+.1f},{hi2:+.1f}] | "
                      f"majority-stop2 {m['majority_stop2']:.1%} @ {m['majority_stop2_calls']:.2f} calls")
        else:
            print(f"\n### {k}: letters-only-correct={v['letters_only_correct']} numbers-only-correct={v['numbers_only_correct']} "
                  f"McNemar exact p={v['mcnemar_exact_p']:.3g} (pairs={v['pairs']})")


if __name__ == "__main__":
    main()
