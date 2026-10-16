"""
Is the prior-correction gain for the HF models real?

Three tests on the saved probability runs (results_and_runs/*.csv):
  1. chance  : exact McNemar test (same prompts before vs after) + 95% bootstrap CI over questions
  2. held-out: prior learned on half of the questions, accuracy measured on the other half
               (20 random splits)
  3. transfer: prior learned on one dataset, applied to the other

    .venv/bin/python3 experiments/v3_corrected/hf_significance.py
"""
from __future__ import annotations

import csv
import json
import math
import random
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
L = ["A", "B", "C", "D"]
MODELS = ["Qwen/Qwen2.5-0.5B-Instruct", "meta-llama/Llama-3.2-1B-Instruct"]
DSETS = ["allenai_ai2_arc", "allenai_openbookqa"]


def load():
    csv.field_size_limit(sys.maxsize)
    runs = {r["id"]: (r["model"], r["dataset"]) for r in csv.DictReader(open(ROOT / "results_and_runs/runs_20260316235711.csv"))}
    data = defaultdict(list)
    for r in csv.DictReader(open(ROOT / "results_and_runs/results_20260316235711.csv")):
        key = runs.get(r["run_id"])
        if key and key[0] in MODELS:
            lp = json.loads(r["logprobs"])
            p = np.array([math.exp(lp[x]) for x in L]); p /= p.sum()
            data[key].append({"item": r["item_id"], "gold": L.index(r["correct_label"]), "p": p})
    return data


def prior_of(rows):
    return np.mean([r["p"] for r in rows], axis=0)


def correct(rows, prior=None):
    out = []
    for r in rows:
        s = np.log(r["p"] + 1e-12) - (np.log(prior) if prior is not None else 0)
        out.append(int(np.argmax(s) == r["gold"]))
    return np.array(out)


def gap(rows, c):
    acc = [c[[i for i, r in enumerate(rows) if r["gold"] == g]].mean() for g in range(4)]
    return max(acc) - min(acc)


def mcnemar(a, b):
    n01 = int(((a == 0) & (b == 1)).sum()); n10 = int(((a == 1) & (b == 0)).sum())
    n, k = n01 + n10, min(n01, n10)
    p = 1.0 if n == 0 else min(1.0, 2 * sum(math.comb(n, i) for i in range(k + 1)) / 2 ** n)
    return n01, n10, p


def boot(rows, a, b, reps=5000, seed=0):
    """95% CI (points) for mean(b-a) and for gap(a)-gap(b), resampling whole questions."""
    by = defaultdict(list)
    for i, r in enumerate(rows):
        by[r["item"]].append(i)
    keys = list(by)
    rng = np.random.default_rng(seed)
    d_acc, d_gap = [], []
    for _ in range(reps):
        idx = [i for k in rng.choice(keys, len(keys)) for i in by[k]]
        sub = [rows[i] for i in idx]
        aa, bb = a[idx], b[idx]
        d_acc.append((bb.mean() - aa.mean()) * 100)
        d_gap.append((gap(sub, aa) - gap(sub, bb)) * 100)
    return np.percentile(d_acc, [2.5, 97.5]), np.percentile(d_gap, [2.5, 97.5])


def main():
    data = load()
    for model in MODELS:
        print(f"\n######## {model}")
        for ds in DSETS:
            rows = data[(model, ds)]
            before = correct(rows)
            after = correct(rows, prior_of(rows))
            n01, n10, p = mcnemar(before, after)
            (alo, ahi), (glo, ghi) = boot(rows, before, after)
            print(f"\n== {ds}  ({len(rows)} prompts, {len({r['item'] for r in rows})} questions)")
            print(f"  1. chance:   accuracy {before.mean():.1%} -> {after.mean():.1%}  "
                  f"(gain {100 * (after.mean() - before.mean()):+.1f} pts, 95% CI {alo:+.1f} to {ahi:+.1f}); "
                  f"fixed {n01}, broken {n10}, exact McNemar p = {p:.2g}")
            print(f"              position gap {gap(rows, before):.1%} -> {gap(rows, after):.1%}  "
                  f"(reduction 95% CI {glo:+.1f} to {ghi:+.1f} pts)")
            # 2. held-out: learn prior on half the questions, test on the other half
            items = sorted({r["item"] for r in rows})
            gains, gaps_b, gaps_a = [], [], []
            for seed in range(20):
                rnd = random.Random(seed); sh = items[:]; rnd.shuffle(sh)
                train = set(sh[: len(sh) // 2])
                tr = [r for r in rows if r["item"] in train]; te = [r for r in rows if r["item"] not in train]
                b, a = correct(te), correct(te, prior_of(tr))
                gains.append((a.mean() - b.mean()) * 100); gaps_b.append(gap(te, b)); gaps_a.append(gap(te, a))
            print(f"  2. held-out: gain on unseen questions {np.mean(gains):+.1f} pts (min {min(gains):+.1f}, max {max(gains):+.1f} over 20 splits); "
                  f"gap {np.mean(gaps_b):.1%} -> {np.mean(gaps_a):.1%}")
        # 3. transfer between datasets
        for src, dst in ((DSETS[0], DSETS[1]), (DSETS[1], DSETS[0])):
            rs, rd = data[(model, src)], data[(model, dst)]
            b, a = correct(rd), correct(rd, prior_of(rs))
            n01, n10, p = mcnemar(b, a)
            print(f"  3. transfer: prior from {src[8:]} applied to {dst[8:]}: {b.mean():.1%} -> {a.mean():.1%} "
                  f"({100 * (a.mean() - b.mean()):+.1f} pts, p = {p:.2g}); gap {gap(rd, b):.1%} -> {gap(rd, a):.1%}")


if __name__ == "__main__":
    main()
