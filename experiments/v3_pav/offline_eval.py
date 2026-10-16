"""
PAV (Position-Aware Voting): the label-free voting method tested in experiments/v3_corrected.

Each question is asked under several option orders; the model returns a slot (or nothing).
PAV models the model's position behaviour with a K x K matrix

    C[t, s] = P(model picks slot s | the option it "believes" is at slot t)

learned WITHOUT gold labels by EM (Dawid-Skene style; the latent class is the believed option).
Posterior for item i:  P(y_i = c | picks) ∝ prod_r C[slot_r(c), pick_r].
Majority vote is the special case C = p*I + (1-p)/(K-1)*(1-I).
PAV-adaptive (sequential) asks orders one at a time and stops once the posterior max >= tau.

Item format: {"id", "gold": option index, "obs": [(slot_of list, picked slot or None), ...]}.
"""
from __future__ import annotations

import numpy as np

K = 4
SEEDS = 20


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


def boot_ci(a, b, reps=5000, seed=0):
    """95% paired bootstrap CI (over items) for mean(b - a), in points."""
    a, b = np.asarray(a), np.asarray(b)
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(a), size=(reps, len(a)))
    d = (b[idx] - a[idx]).mean(axis=1) * 100
    return np.percentile(d, 2.5), np.percentile(d, 97.5)
