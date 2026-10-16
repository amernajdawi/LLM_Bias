"""
Offline evaluation of PAV (Position-Aware Voting) on answers that were already saved.

Each item is asked under R permutations. The model returns a slot (or nothing).
Majority vote counts every pick equally. PAV instead models the model's
position behaviour with a K x K matrix

    C[t, s] = P(model picks slot s | the option it "believes" is at slot t)

learned WITHOUT gold labels by EM (Dawid-Skene style; the latent class is the
believed option content). Posterior for item i:

    P(y_i = c | picks) ∝ prod_r C[slot_r(c), pick_r]

A pick of content c while c sat in a slot the model is drawn to anyway is weak
evidence; a pick from a slot the model usually avoids is strong evidence.
Majority vote is the special case C = p*I + (1-p)/(K-1)*(1-I).

PAV-adaptive asks permutations one at a time and stops once the posterior
max >= tau (C is estimated on the other half of the items: 2-fold cross-fitting).

    .venv/bin/python3 experiments/v3_pav/offline_eval.py
"""
from __future__ import annotations

import json
import random
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from src.data.loaders import load_dataset_by_name
from src.data.shuffle import build_permutations

L = ["A", "B", "C", "D"]
K = 4
DS = {
    "allenai_ai2_arc": ("allenai/ai2_arc", "ARC-Challenge"),
    "allenai_openbookqa": ("allenai/openbookqa", "main"),
}


def items_for(ds):
    path, cfg = DS[ds]
    return load_dataset_by_name(path, cfg, "test", L, 200)


# ---------------------------------------------------------------- data loading
def load_letter_baseline(model, ds):
    """Main-results letter runs: 4 permutations per item (correct at A,B,C,D)."""
    rows = json.load(open(ROOT / f"outputs/results/{model}_{ds}.json"))["results"]
    items = {it.id: it for it in items_for(ds)}
    perms = {(p.item_id, p.correct_label): p for p in build_permutations(list(items.values()), L)}
    obs_by_item = {}
    for r in rows:
        p = perms.get((r["item_id"], r["correct_label"]))
        if p is None:
            continue
        it = items[r["item_id"]]
        slot_of = [None] * K                         # slot_of[c] = slot where original content c was shown
        for s, lab in enumerate(L):
            slot_of[it.choices.index(p.options[lab])] = s
        pick = L.index(r["pred"]) if r.get("pred") in L else None
        obs_by_item.setdefault(it.id, []).append((slot_of, pick))
    return [{"id": iid, "gold": items[iid].correct_idx, "obs": obs} for iid, obs in obs_by_item.items()]


def load_ncv_numeric(model, ds):
    """experiments/v2 NCV runs: 4 cyclic content orders, numeric labels."""
    rows = json.load(open(ROOT / f"experiments/v2/outputs/ncv_{model}_{ds}.json"))["results"]
    data = []
    for r in rows:
        obs = []
        for run in r["numeric_runs"]:
            order = run["order"]                     # display slot k shows choices[order[k]]
            slot_of = [order.index(c) for c in range(K)]
            pick = order.index(run["pred_idx"]) if run["pred_idx"] is not None else None
            obs.append((slot_of, pick))
        data.append({"id": r["item_id"], "gold": r["correct_idx"], "obs": obs})
    return data


# ---------------------------------------------------------------- methods
def frac_correct(scores, gold):
    """Accuracy credit with exact ties split evenly (no lucky tie-breaking)."""
    best = max(scores)
    winners = [c for c in range(K) if abs(scores[c] - best) < 1e-12]
    return (gold in winners) / len(winners)


def majority(item, upto=None):
    votes = [0.0] * K
    for slot_of, pick in item["obs"][:upto]:
        if pick is not None:
            votes[slot_of.index(pick)] += 1
    return votes


def log_post(item, C, upto=None):
    lp = np.zeros(K)
    logC = np.log(C)
    for slot_of, pick in item["obs"][:upto]:
        if pick is None:
            continue
        for c in range(K):
            lp[c] += logC[slot_of[c], pick]
    return lp


SEEDS = 20


def _posterior(obs, C):
    lp = log_post({"obs": obs}, C)
    q = np.exp(lp - lp.max())
    return lp, q / q.sum()


def sequential(item, C, tau, policy, rs):
    """Ask stored orders one at a time; stop when posterior max >= tau.
    policy='random': random sequence. policy='eig': first order random, then the
    remaining order with the largest expected information gain under C."""
    remaining = list(item["obs"])
    asked = [remaining.pop(rs.randrange(len(remaining)))]
    while True:
        lp, q = _posterior(asked, C)
        if q.max() >= tau or not remaining:
            return frac_correct(lp, item["gold"]), len(asked)
        if policy == "random":
            asked.append(remaining.pop(rs.randrange(len(remaining))))
            continue
        h_now = -(q * np.log(q + 1e-12)).sum()
        best, best_gain = 0, -1.0
        for j, (slot_of, _) in enumerate(remaining):
            # predictive distribution of the pick s, and expected posterior entropy
            exp_h = 0.0
            for s in range(K):
                like = np.array([C[slot_of[c], s] for c in range(K)])
                ps = float((q * like).sum())
                post = q * like / ps
                exp_h += ps * -(post * np.log(post + 1e-12)).sum()
            if h_now - exp_h > best_gain:
                best, best_gain = j, h_now - exp_h
        asked.append(remaining.pop(best))


def fit_C(data, iters=200, alpha=1.0):
    p = 0.7
    C = np.full((K, K), (1 - p) / (K - 1)); np.fill_diagonal(C, p)
    for _ in range(iters):
        counts = np.full((K, K), alpha)
        for item in data:
            lp = log_post(item, C)
            q = np.exp(lp - lp.max()); q /= q.sum()
            for slot_of, pick in item["obs"]:
                if pick is None:
                    continue
                for c in range(K):
                    counts[slot_of[c], pick] += q[c]
        C_new = counts / counts.sum(axis=1, keepdims=True)
        if np.abs(C_new - C).max() < 1e-7:
            C = C_new
            break
        C = C_new
    return C


def inverse_prior(data):
    """Simpler competitor: weight each vote by 1 / how often the model picks that slot overall."""
    freq = np.full(K, 1e-9)
    for it in data:
        for _, pick in it["obs"]:
            if pick is not None:
                freq[pick] += 1
    w = (freq.sum() / K) / freq
    out = []
    for it in data:
        s = [0.0] * K
        for slot_of, pick in it["obs"]:
            if pick is not None:
                s[slot_of.index(pick)] += w[pick]
        out.append(s)
    return out


def boot_ci(a, b, reps=5000, seed=0):
    """95% paired bootstrap CI (over items) for mean(b - a), in points."""
    a, b = np.asarray(a), np.asarray(b)
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(a), size=(reps, len(a)))
    d = (b[idx] - a[idx]).mean(axis=1) * 100
    return np.percentile(d, 2.5), np.percentile(d, 97.5)


def evaluate(data, name):
    n = len(data)
    single = np.mean([np.mean([(pick is not None and slot_of[it["gold"]] == pick) for slot_of, pick in it["obs"]]) for it in data])
    maj_i = [frac_correct(majority(it), it["gold"]) for it in data]
    maj = np.mean(maj_i)
    inv_i = [frac_correct(s, it["gold"]) for s, it in zip(inverse_prior(data), data)]
    C = fit_C(data)
    pav_i = [frac_correct(log_post(it, C), it["gold"]) for it in data]
    pav = np.mean(pav_i)
    tied = [sorted(majority(it))[-1] == sorted(majority(it))[-2] for it in data]
    ties = sum(tied)
    lo, hi = boot_ci(maj_i, pav_i)
    lo2, hi2 = boot_ci(inv_i, pav_i)
    gain_tied = sum(p - m for p, m, t in zip(pav_i, maj_i, tied) if t) * 100 / n
    gain_untied = sum(p - m for p, m, t in zip(pav_i, maj_i, tied) if not t) * 100 / n

    # gold-based accuracy by position vs PAV's label-free estimate (diagonal of C)
    acc_pos = np.zeros(K); cnt = np.zeros(K)
    for it in data:
        for slot_of, pick in it["obs"]:
            t = slot_of[it["gold"]]
            cnt[t] += 1; acc_pos[t] += (pick == t)
    acc_pos /= np.maximum(cnt, 1)

    # adaptive: 2-fold cross-fitting for C, stop when posterior max >= tau
    rng = random.Random(0)
    idx = list(range(n)); rng.shuffle(idx)
    folds = [set(idx[: n // 2]), set(idx[n // 2:])]
    # Sequential versions. The stored orders always start with gold at slot A, so the
    # query sequence is randomised per item and averaged over SEEDS to avoid that leak.
    C_folds = [fit_C([data[i] for i in range(n) if i not in folds[f]]) for f in (0, 1)]
    fold_of = {i: f for f in (0, 1) for i in folds[f]}
    adaptive, adaptive_ci = {}, {}
    for policy in ("random", "eig"):
        for tau in (0.90, 0.95):
            per_item = np.zeros(n); calls = 0.0
            for seed in range(SEEDS):
                rs = random.Random(seed)
                for i, it in enumerate(data):
                    Cf = C_folds[fold_of[i]]
                    acc_i, used = sequential(it, Cf, tau, policy, rs)
                    per_item[i] += acc_i / SEEDS; calls += used / (SEEDS * n)
            adaptive[(policy, tau)] = (float(per_item.mean()), calls)
            adaptive_ci[(policy, tau)] = boot_ci(maj_i, per_item)
    # cheap baseline: plain majority vote that stops after 2 random orders if both agree
    maj2 = np.zeros(n); calls2 = 0.0
    for seed in range(SEEDS):
        rs = random.Random(seed)
        for i, it in enumerate(data):
            obs = rs.sample(it["obs"], len(it["obs"]))
            first2 = majority({"obs": obs}, upto=2)
            if max(first2) == 2:
                calls2 += 2 / (SEEDS * n); maj2[i] += frac_correct(first2, it["gold"]) / SEEDS
            else:
                calls2 += len(obs) / (SEEDS * n); maj2[i] += frac_correct(majority(it), it["gold"]) / SEEDS
    maj2_acc, maj2_calls = float(maj2.mean()), calls2

    # can PAV rank positions (best -> worst) without gold labels?
    def ranks(x):
        order = np.argsort(x, kind="stable")
        r = np.empty(len(x)); r[order] = np.arange(len(x))
        for v in np.unique(x):                        # average ranks for ties
            r[x == v] = r[x == v].mean()
        return r
    rho = float(np.corrcoef(ranks(np.diag(C)), ranks(acc_pos))[0, 1])

    print(f"\n### {name}  (n={n} items, {len(data[0]['obs'])} permutations each)")
    print(f"  single order (avg over orders) : {single:.1%}")
    print(f"  majority vote (ties split)     : {maj:.1%}   [items with tied top vote: {ties}]")
    print(f"  inverse-prior weighted vote    : {np.mean(inv_i):.1%}")
    print(f"  PAV (EM, label-free)           : {pav:.1%}   ({(pav - maj) * 100:+.1f} pts vs majority, 95% CI [{lo:+.1f}, {hi:+.1f}]; vs inverse-prior CI [{lo2:+.1f}, {hi2:+.1f}])")
    print(f"  PAV gain from tied items       : {gain_tied:+.1f} pts ; from untied items: {gain_untied:+.1f} pts")
    print(f"  majority, stop if 2 random orders agree : {maj2_acc:.1%} using {maj2_calls:.2f} calls/item")
    for (policy, tau), (a, c) in adaptive.items():
        lo3, hi3 = adaptive_ci[(policy, tau)]
        print(f"  PAV-adaptive {policy:<6} tau={tau:.2f}     : {a:.1%} using {c:.2f} calls/item (vs {len(data[0]['obs'])}); vs full majority CI [{lo3:+.1f}, {hi3:+.1f}]")
    print(f"  Spearman(label-free C[t,t], gold acc by slot) = {rho:.2f}")
    print("  gold accuracy by slot          :", " ".join(f"{L[t]}={acc_pos[t]:.0%}" for t in range(K)))
    print("  PAV label-free estimate C[t,t] :", " ".join(f"{L[t]}={C[t, t]:.0%}" for t in range(K)))
    print("  PAV pull toward each slot      :", " ".join(f"{L[s]}={(C[:, s].sum() - C[s, s]) / (K - 1):.0%}" for s in range(K)))
    return {"single": single, "majority": maj, "inverse_prior": float(np.mean(inv_i)), "pav": pav, "ties": ties,
            "pav_minus_majority_ci95": [lo, hi], "pav_minus_inverse_prior_ci95": [lo2, hi2],
            "adaptive": {f"{p}@{t}": v for (p, t), v in adaptive.items()},
            "adaptive_minus_majority_ci95": {f"{p}@{t}": v for (p, t), v in adaptive_ci.items()},
            "majority_stop2": [maj2_acc, maj2_calls], "spearman_diag_vs_gold": rho, "C": C.tolist(),
            "gold_acc_by_slot": acc_pos.tolist()}


def main():
    results = {}
    runs = [
        ("qwen:7b-chat letters (main baseline)", lambda ds: load_letter_baseline("qwen:7b-chat", ds)),
        ("kimi-k2-thinking:cloud letters (main baseline)", lambda ds: load_letter_baseline("kimi-k2-thinking:cloud", ds)),
        ("qwen:7b-chat numbers (v2 NCV runs)", lambda ds: load_ncv_numeric("qwen:7b-chat", ds)),
    ]
    for name, loader in runs:
        for ds in DS:
            try:
                data = loader(ds)
            except FileNotFoundError:
                continue
            results[f"{name} | {ds}"] = evaluate(data, f"{name} | {ds}")
    out = Path(__file__).resolve().parent / "offline_results.json"
    out.write_text(json.dumps(results, indent=2))
    print(f"\nSaved {out}")


if __name__ == "__main__":
    main()
