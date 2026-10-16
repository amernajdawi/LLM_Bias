"""
Every bias-reduction method, tested on each open-weight model run in raw_hf/ (run_hf.py).

Small models such as Qwen2.5-0.5B have real position bias, so a method that works shows it here. No method uses
the gold answers to fit anything; the gold answers are only used to score.

One call per question (each of the 4 rotations is scored as its own one-call trial):
  single            ask once, take the most likely label
  prior (batch)     divide by the model's average label distribution over the questions being
                    answered (Batch Calibration, Zhou et al., ICLR 2024)
  prior (held-out)  the same average, but taken from OTHER questions asked once each
                    (2 folds, 20 random splits)
  prior (transfer)  the average taken from the other dataset
  PriDe             prior from 5% of the questions asked in all 4 rotations, applied to the rest
                    (Zheng et al., ICLR 2024; 20 random draws)
Four calls per question (all 4 rotations):
  prob. average     average each option's probability over the rotations
  log-prob average  the same in log space; cancels any fixed per-slot preference exactly
  majority vote     one vote per rotation, ties split
  PAV               label-free position-aware voting (experiments/v3_pav), also cross-fitted
Adaptive: PAV-adaptive (EIG, tau 0.95, cross-fitted, 20 seeds) and majority stop-at-2,
exactly as in analyze_v3.py.

Tests are paired over questions: 95% bootstrap CI and a two-sided sign-flip test on the
per-question difference, Holm-adjusted within each model and family of comparisons.

    OMP_NUM_THREADS=1 .venv/bin/python3 experiments/v3_corrected/analyze_hf.py
"""
from __future__ import annotations

import json
import math
import random
import sys
from collections import defaultdict
from itertools import combinations
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(ROOT / "experiments/v3_pav"))
from offline_eval import K, SEEDS, fit_C, frac_correct, log_post, majority, sequential  # noqa: E402

SLOT = ["A", "B", "C", "D"]
DSETS = ["allenai_ai2_arc", "allenai_openbookqa"]
FMTS = ["letter", "number"]
EPS = 1e-12
SPLITS = 20

METHODS = {  # key: (name, calls per question)
    "single": ("ask once", "1"),
    "prior_batch": ("prior correction (batch)", "1"),
    "prior_heldout": ("prior correction (held-out)", "1"),
    "prior_transfer": ("prior correction (other dataset)", "1"),
    "pride": ("PriDe", "1 + 40 once"),
    "avg_prob": ("probability average", "4"),
    "avg_logprob": ("log-probability average", "4"),
    "majority": ("majority vote", "4"),
    "pav": ("PAV", "4"),
    "pav_crossfit": ("PAV (cross-fitted)", "4"),
    "pav_adaptive": ("PAV-adaptive", "adaptive"),
    "majority_stop2": ("majority stop-at-2", "adaptive"),
}
ONE_CALL = ("single", "prior_batch", "prior_heldout", "prior_transfer", "pride")


# ---------------------------------------------------------------- data
HF_IDS = {"Qwen2.5-0.5B-Instruct": "Qwen/Qwen2.5-0.5B-Instruct"}   # runs saved before records carried hf_id


def load():
    """Every complete block in raw_hf/, keyed (model name, dataset, format)."""
    blocks = {}
    for path in sorted((HERE / "raw_hf").glob("*__*__*.jsonl")):
        _, ds, fmt = path.stem.split("__")
        if ds not in DSETS or fmt not in FMTS:
            continue
        by, bad = defaultdict(dict), 0
        for line in path.read_text().splitlines():
            try:
                r = json.loads(line)
            except json.JSONDecodeError:
                bad += 1
                continue
            by[r["item_id"]][r["rotation"]] = r
        if bad:
            print(f"WARNING {path.name}: {bad} unreadable line(s) skipped")
        ids = sorted(i for i, v in by.items() if len(v) == K)
        if len(ids) < 20:
            continue
        rot = lambda f: np.array([[by[i][k][f] for k in range(K)] for i in ids])
        O = rot("order")                                          # [question, rotation, slot] -> option
        gold = np.array([by[i][0]["gold_idx"] for i in ids])
        first = by[ids[0]][0]
        blocks[(first["model"].replace("-Instruct", ""), ds, fmt)] = {
            "hf_id": first.get("hf_id") or HF_IDS.get(first["model"], first["model"]),
            "ids": ids, "P": rot("probs"), "O": O, "S": np.argsort(O, axis=2),       # S: option -> slot
            "GS": (O == gold[:, None, None]).argmax(axis=2),                           # slot of the gold option
            "gold": gold, "mass": rot("label_mass")}
    return blocks


# ---------------------------------------------------------------- one call per question
def one_call(b, prior=None):
    """Picked slot [question, rotation] when each prompt is answered on its own."""
    s = np.log(b["P"] + EPS)
    if prior is not None:
        s = s - np.log(prior)
    return s.argmax(axis=2)


def per_question(b, pick):
    correct = (pick == b["GS"]).astype(float)
    content = np.take_along_axis(b["O"], pick[..., None], axis=2)[..., 0]
    return {
        "acc": correct.mean(axis=1),
        "by_slot": np.take_along_axis(correct, np.argsort(b["GS"], axis=1), axis=1),   # [question, gold slot]
        "circular": correct.min(axis=1),
        "fluct": (content.min(axis=1) != content.max(axis=1)).astype(float),
        "picks": (pick[..., None] == np.arange(K)).mean(axis=1),
    }


def average_runs(runs):
    """runs: (per_question result, mask of questions it scored). Average each question over its runs."""
    tot = {k: np.zeros(v.shape) for k, v in runs[0][0].items()}
    cnt = np.zeros(len(runs[0][1]))
    for pq, mask in runs:
        for k in tot:
            tot[k][mask] += pq[k][mask]
        cnt += mask
    assert cnt.min() > 0
    return {k: v / cnt.reshape(-1, *([1] * (v.ndim - 1))) for k, v in tot.items()}


def m_prior_heldout(b):
    n = len(b["gold"]); runs = []
    for seed in range(SPLITS):
        rng = np.random.default_rng(seed)
        perm = rng.permutation(n)
        for test in (perm[: n // 2], perm[n // 2:]):
            train = np.setdiff1d(perm, test)
            asked = rng.integers(0, K, len(train))               # each training question asked once
            mask = np.zeros(n, bool); mask[test] = True
            runs.append((per_question(b, one_call(b, b["P"][train, asked].mean(axis=0))), mask))
    return average_runs(runs)


def m_pride(b, frac=0.05):
    n = len(b["gold"]); m = max(1, round(frac * n)); runs = []
    logP = np.log(b["P"] + EPS)
    for seed in range(SPLITS):
        est = np.random.default_rng(100 + seed).choice(n, m, replace=False)
        z = logP[est].mean(axis=1)                               # per question: mean over the 4 rotations
        q = np.exp(z - z.max(axis=1, keepdims=True)); q /= q.sum(axis=1, keepdims=True)
        mask = np.ones(n, bool); mask[est] = False
        runs.append((per_question(b, one_call(b, q.mean(axis=0))), mask))
    return average_runs(runs)


def spread(b, pick):
    """Where the answers go when asked once: shares over all prompts and over the wrong answers only."""
    err = pick != b["GS"]
    share = lambda m, x: {SLOT[s]: float(((x == s) & m).sum() / m.sum()) for s in range(K)} if m.any() else None
    every = np.ones_like(err)
    return {"n_rows": int(err.size), "accuracy": float(1 - err.mean()), "gold_share": share(every, b["GS"]),
            "pred_share_all": share(every, pick), "n_errors": int(err.sum()),
            "errors_pred_at": share(err, pick), "errors_correct_at": share(err, b["GS"])}


def spread_heldout(b):
    """spread() after the held-out prior correction, averaged over the same 20 splits as m_prior_heldout()."""
    n = len(b["gold"]); outs = []
    for seed in range(SPLITS):
        rng = np.random.default_rng(seed)
        perm = rng.permutation(n)
        pick = np.empty_like(b["GS"])
        for test in (perm[: n // 2], perm[n // 2:]):
            train = np.setdiff1d(perm, test)
            asked = rng.integers(0, K, len(train))
            pick[test] = one_call(b, b["P"][train, asked].mean(axis=0))[test]
        outs.append(spread(b, pick))
    avg = lambda key: {s: float(np.mean([o[key][s] for o in outs])) for s in SLOT}
    return {"n_rows": outs[0]["n_rows"], "accuracy": float(np.mean([o["accuracy"] for o in outs])),
            "gold_share": outs[0]["gold_share"], "pred_share_all": avg("pred_share_all"),
            "n_errors": int(round(np.mean([o["n_errors"] for o in outs]))),
            "errors_pred_at": avg("errors_pred_at"), "errors_correct_at": avg("errors_correct_at")}


def summary(pq):
    acc_slot = pq["by_slot"].mean(axis=0)
    return {"acc_by_slot": dict(zip(SLOT, acc_slot.tolist())), "gap": float(acc_slot.max() - acc_slot.min()),
            "rstd": float(acc_slot.std() * 100), "circular_eval": float(pq["circular"].mean()),
            "fluctuation": float(pq["fluct"].mean()), "pick_share": dict(zip(SLOT, pq["picks"].mean(axis=0).tolist()))}


# ---------------------------------------------------------------- four calls per question
def option_scores(b, log):
    X = np.log(b["P"] + EPS) if log else b["P"]
    return np.take_along_axis(X, b["S"], axis=2).mean(axis=1)   # [question, option]


def rotation_curve(b):
    """Per-question accuracy when only r of the 4 rotations are asked (mean over every subset of size r),
    log-probability average with and without batch prior correction. Descriptive, not tested."""
    logP = np.log(b["P"] + EPS)
    out = {}
    for name, X in (("plain", logP), ("prior_corrected", logP - np.log(b["P"].reshape(-1, K).mean(axis=0)))):
        X = np.take_along_axis(X, b["S"], axis=2)                    # [question, rotation, option]
        out[name] = {r: np.mean([[frac_correct(list(s), g) for s, g in zip(X[:, list(sub)].mean(axis=1), b["gold"])]
                                 for sub in combinations(range(K), r)], axis=0) for r in range(1, K + 1)}
    return out


def voting(b):
    """Majority, PAV, PAV-adaptive and majority stop-at-2 on the hard picks, as in analyze_v3.py."""
    pick = one_call(b)
    items = [{"id": i, "gold": int(b["gold"][q]), "obs": [(b["S"][q, k].tolist(), int(pick[q, k])) for k in range(K)]}
             for q, i in enumerate(b["ids"])]
    n = len(items)
    maj = np.array([frac_correct(majority(it), it["gold"]) for it in items])
    C = fit_C(items)
    pav = np.array([frac_correct(log_post(it, C), it["gold"]) for it in items])
    idx = list(range(n)); random.Random(0).shuffle(idx)
    fold = {i: (0 if j < n // 2 else 1) for j, i in enumerate(idx)}
    Cf = [fit_C([items[i] for i in range(n) if fold[i] != f]) for f in (0, 1)]
    pav_cf = np.array([frac_correct(log_post(it, Cf[fold[i]]), it["gold"]) for i, it in enumerate(items)])
    ad = np.zeros(n); ad_calls = 0.0
    for seed in range(SEEDS):
        rs = random.Random(seed)
        for i, it in enumerate(items):
            a, used = sequential(it, Cf[fold[i]], 0.95, "eig", rs)
            ad[i] += a / SEEDS; ad_calls += used / (SEEDS * n)
    m2 = np.zeros(n); m2_calls = 0.0
    for seed in range(SEEDS):
        rs = random.Random(seed)
        for i, it in enumerate(items):
            obs = rs.sample(it["obs"], K)
            f2 = majority({"obs": obs}, upto=2)
            if max(f2) == 2:
                m2[i] += frac_correct(f2, it["gold"]) / SEEDS; m2_calls += 2 / (SEEDS * n)
            else:
                m2[i] += frac_correct(majority(it), it["gold"]) / SEEDS; m2_calls += K / (SEEDS * n)
    acc = {"majority": maj, "pav": pav, "pav_crossfit": pav_cf, "pav_adaptive": ad, "majority_stop2": m2}
    return acc, {"pav_adaptive": ad_calls, "majority_stop2": m2_calls}, C


# ---------------------------------------------------------------- statistics
def paired(a, b, reps=5000, flips=20000, seed=0):
    """b - a over questions (points): mean, 95% bootstrap CI, two-sided sign-flip p."""
    d = (np.asarray(b, float) - np.asarray(a, float)) * 100
    rng = np.random.default_rng(seed)
    boot = d[rng.integers(0, len(d), (reps, len(d)))].mean(axis=1)
    null = np.abs((rng.choice([-1.0, 1.0], (flips, len(d))) * d).mean(axis=1))
    lo, hi = np.percentile(boot, [2.5, 97.5])
    return {"diff": float(d.mean()), "ci95": [float(lo), float(hi)],
            "p": (1 + int((null >= abs(d.mean()) - 1e-9).sum())) / (flips + 1)}


def gap_ci(by_a, by_b, reps=5000, seed=0):
    """95% bootstrap CI (points) for gap(a) - gap(b), i.e. how much b shrinks the position gap."""
    idx = np.random.default_rng(seed).integers(0, len(by_a), (reps, len(by_a)))
    g = lambda by: np.ptp(by[idx].mean(axis=1), axis=1)
    lo, hi = np.percentile((g(by_a) - g(by_b)) * 100, [2.5, 97.5])
    return [float(lo), float(hi)]


def holm(tests):
    """Adds p_holm to every test dict in the list (one family)."""
    order = sorted(range(len(tests)), key=lambda i: tests[i]["p"])
    run = 0.0
    for r, i in enumerate(order):
        run = max(run, (len(tests) - r) * tests[i]["p"])
        tests[i]["p_holm"] = min(1.0, run)


def fmt_p(p):
    return "<0.0001" if p < 1e-4 else f"{p:.4f}" if p < 0.01 else f"{p:.3f}"


# ---------------------------------------------------------------- main
def main():
    blocks = load()
    out, per_q_acc = {}, {}
    fams = defaultdict(lambda: ([], [], []))           # per model: (vs ask-once, method pairs, letters vs numbers)
    for (name, ds, fmt), b in blocks.items():
        fam_single, fam_pairs, _ = fams[name]
        assert (np.sort(b["GS"], axis=1) == np.arange(K)).all(), "each question needs its answer once in every slot"
        pq = {"single": per_question(b, one_call(b)),
              "prior_batch": per_question(b, one_call(b, b["P"].reshape(-1, K).mean(axis=0))),
              "prior_heldout": m_prior_heldout(b), "pride": m_pride(b)}
        other = blocks.get((name, DSETS[1 - DSETS.index(ds)], fmt))
        if other is not None:
            pq["prior_transfer"] = per_question(b, one_call(b, other["P"].reshape(-1, K).mean(axis=0)))
        acc = {k: v["acc"] for k, v in pq.items()}
        for k, log in (("avg_prob", False), ("avg_logprob", True)):
            acc[k] = np.array([frac_correct(list(s), g) for s, g in zip(option_scores(b, log), b["gold"])])
        vote_acc, calls, C = voting(b)
        acc.update(vote_acc)
        per_q_acc[(name, ds, fmt)] = acc

        picked = np.take_along_axis(b["O"], one_call(b)[..., None], axis=2)[..., 0]     # option chosen per rotation
        votes = np.sort(np.stack([(picked == c).sum(axis=1) for c in range(K)], axis=1), axis=1)[:, ::-1]
        rec = {"hf_id": b["hf_id"], "n_questions": len(b["ids"]), "label_mass_mean": float(b["mass"].mean()),
               "label_mass_p05": float(np.percentile(b["mass"], 5)),
               "votes_no_single_winner": float((votes[:, 0] == votes[:, 1]).mean()),
               "votes_all_different": float((votes[:, 0] == 1).mean()),
               "spread_single": spread(b, one_call(b)),
               "spread_prior_heldout": spread_heldout(b),
               "pav_label_free_acc_by_slot": dict(zip(SLOT, np.diag(C).tolist())), "methods": {}}
        for k in METHODS:
            if k not in acc:
                continue
            m = {"name": METHODS[k][0], "calls": calls.get(k, METHODS[k][1]), "accuracy": float(acc[k].mean())}
            if k in pq:
                m.update(summary(pq[k]))
            if k != "single":
                m["vs_single"] = paired(acc["single"], acc[k]); fam_single.append(m["vs_single"])
                if k in pq:
                    m["gap_reduction_ci95"] = gap_ci(pq["single"]["by_slot"], pq[k]["by_slot"])
            rec["methods"][k] = m
        for a, c in (("majority", "pav"), ("majority", "pav_adaptive"), ("majority", "prior_heldout"),
                     ("majority", "avg_logprob"), ("prior_heldout", "pride"), ("prior_heldout", "avg_logprob")):
            t = paired(acc[a], acc[c]); fam_pairs.append(t)
            rec.setdefault("pairs", {})[f"{c} vs {a}"] = t
        curve = rotation_curve(b)
        rec["rotations_curve"] = {name: {str(r): float(v.mean()) for r, v in c.items()} for name, c in curve.items()}
        rec["rotations_curve_ci95_vs_single"] = {
            str(r): paired(acc["single"], v)["ci95"] for r, v in curve["prior_corrected"].items()}
        out[f"{name} | {ds} | {fmt}"] = rec

    for name, ds in sorted({(n, d) for n, d, _ in blocks}):
        if (name, ds, "letter") in per_q_acc and (name, ds, "number") in per_q_acc:
            bl, bn = blocks[(name, ds, "letter")], blocks[(name, ds, "number")]
            if not (bl["ids"] == bn["ids"] and (bl["O"] == bn["O"]).all()):
                continue                                      # one format still running: compare later
            rec = {}
            for k in ("single", "prior_heldout", "majority", "avg_logprob"):
                t = paired(per_q_acc[(name, ds, "letter")][k], per_q_acc[(name, ds, "number")][k]); fams[name][2].append(t)
                rec[f"{k}: numbers minus letters"] = t
            # prompt-level exact McNemar on ask-once, same layout as analyze_v3.py
            cl, cn = one_call(bl) == bl["GS"], one_call(bn) == bn["GS"]
            b, c = int((cl & ~cn).sum()), int((~cl & cn).sum())
            rec.update({"pairs": int(cl.size), "letters_only_correct": b, "numbers_only_correct": c,
                        "mcnemar_exact_p": min(1.0, 2 * sum(math.comb(b + c, i) for i in range(min(b, c) + 1)) / 2 ** (b + c))
                        if b + c else 1.0})
            out[f"{name} | {ds} | letters_vs_numbers"] = rec
    for model_fams in fams.values():                    # Holm within each model, so adding a model changes nothing else
        for fam in model_fams:
            if fam:
                holm(fam)
    (HERE / "hf_results.json").write_text(json.dumps(out, indent=2))

    for key, rec in out.items():
        if "methods" not in rec:
            print(f"\n### {key}")
            for name, t in rec.items():
                if isinstance(t, dict):
                    print(f"  {name:<40} {t['diff']:+5.1f} pts  [{t['ci95'][0]:+.1f}, {t['ci95'][1]:+.1f}]  "
                          f"p={fmt_p(t['p'])}  Holm p={fmt_p(t['p_holm'])}")
            print(f"  ask once, per prompt: letters-only-correct={rec['letters_only_correct']} "
                  f"numbers-only-correct={rec['numbers_only_correct']} McNemar exact p={rec['mcnemar_exact_p']:.3g}")
            continue
        print(f"\n### {key}  ({rec['n_questions']} questions x 4 rotations; probability on the 4 labels: "
              f"mean {rec['label_mass_mean']:.2f}, 5th pct {rec['label_mass_p05']:.2f})")
        print(f"  {'method':<34}{'calls':>11}{'acc':>8}   {'vs ask-once [95% CI]':<24}{'Holm p':>9}"
              f"{'gap':>7}{'RStd':>6}{'Circ':>7}{'Fluct':>7}   picks A/B/C/D")
        for k, m in rec["methods"].items():
            calls = f"{m['calls']:.2f}" if isinstance(m["calls"], float) else m["calls"]
            vs = (f"{m['vs_single']['diff']:+5.1f} [{m['vs_single']['ci95'][0]:+.1f}, {m['vs_single']['ci95'][1]:+.1f}]"
                  if "vs_single" in m else "")
            hp = fmt_p(m["vs_single"]["p_holm"]) if "vs_single" in m else ""
            bias = (f"{m['gap']:7.1%}{m['rstd']:6.1f}{m['circular_eval']:7.1%}{m['fluctuation']:7.1%}   "
                    + "/".join(f"{m['pick_share'][s]:.0%}" for s in SLOT)) if "gap" in m else "   (order-free: sees all 4 rotations)"
            print(f"  {m['name']:<34}{calls:>11}{m['accuracy']:8.1%}   {vs:<24}{hp:>9}{bias}")
        s = rec["methods"]["single"]["acc_by_slot"]
        print("  ask-once accuracy by answer slot: " + " ".join(f"{x}={s[x]:.0%}" for x in SLOT)
              + " | PAV label-free estimate: " + " ".join(f"{x}={rec['pav_label_free_acc_by_slot'][x]:.0%}" for x in SLOT))
        for name, t in rec["pairs"].items():
            print(f"  {name:<36} {t['diff']:+5.1f} pts  [{t['ci95'][0]:+.1f}, {t['ci95'][1]:+.1f}]  Holm p={fmt_p(t['p_holm'])}")
        for name, c in rec["rotations_curve"].items():
            print(f"  rotations asked 1/2/3/4 ({name:<15}): " + " / ".join(f"{c[r]:.1%}" for r in sorted(c)))
    print(f"\nSaved {HERE / 'hf_results.json'}")


if __name__ == "__main__":
    main()
