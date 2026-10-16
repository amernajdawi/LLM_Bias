"""
Build the results report (single HTML page) from the result files. Every number in the
page is computed here from the data; nothing is typed in by hand.

Inputs : experiments/v3_corrected/v3_results.json      (analyze_v3.py)
         experiments/v3_corrected/hf_results.json       (analyze_hf.py)
         outputs/results/<model>_<dataset>.json         (old runs, for the before/after table)
Output : path given as argv[1] (default experiments/v3_corrected/report.html)
"""
from __future__ import annotations

import csv
import functools
import html
import json
import math
import sys
from collections import defaultdict
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
SLOT = ["A", "B", "C", "D"]
DS_NAME = {"allenai_ai2_arc": "ARC-Challenge", "allenai_openbookqa": "OpenBookQA"}
NEW_MODELS = ["gemma4:31b-cloud", "gpt-oss:120b", "gpt-oss:20b", "nemotron-3-nano:30b"]
SHOW = {"gemma4:31b-cloud": "gemma4:31b", "gpt-oss:120b": "gpt-oss:120b", "gpt-oss:20b": "gpt-oss:20b",
        "nemotron-3-nano:30b": "nemotron-3-nano:30b"}
# old runs to compare against: (old results file stem, label, note)
OLD = {
    "gemma4:31b-cloud": [("gemma4:31b-cloud", "gemma4:31b-cloud, cloud run", "same model"),
                         ("gemma4:31b", "gemma4:31b, local run", "same model")],
    "gpt-oss:120b": [("gpt-oss:120b", "gpt-oss:120b, cloud run", "same model")],
    "nemotron-3-nano:30b": [("nemotron3:33b", "nemotron3:33b, local run", "same family")],
    "ministral-3:14b": [("ministral-3:14b-cloud", "ministral-3:14b-cloud, cloud run", "same model, now run locally in 4-bit")],
}


def pct(x, d=1):
    return "–" if x is None else f"{x * 100:.{d}f}%"


def pts(x, d=1):
    return "–" if x is None else f"{x * 100:.{d}f}"


def esc(s):
    return html.escape(str(s))


@functools.lru_cache(maxsize=None)
def _csv_rows():
    """Earlier runs kept in results_and_runs/ (Hugging Face, OpenAI and some cloud runs), keyed (model, dataset)."""
    csv.field_size_limit(sys.maxsize)
    runs = {r["id"]: (r["model"], r["dataset"]) for r in csv.DictReader(open(ROOT / "results_and_runs/runs_20260316235711.csv"))}
    out = defaultdict(list)
    for r in csv.DictReader(open(ROOT / "results_and_runs/results_20260316235711.csv")):
        if r["run_id"] in runs:
            out[runs[r["run_id"]]].append((r["item_id"], SLOT.index(r["pred"]) if r["pred"] in SLOT else None,
                                           SLOT.index(r["correct_label"])))
    return out


@functools.lru_cache(maxsize=None)
def earlier_records(src, ds):
    """Earlier letters run of a model: (item id, picked position or None, position of the correct answer).
    Looks in outputs/results/<src>_<ds>.json first, then in the results_and_runs/ CSV export."""
    p = ROOT / f"outputs/results/{src}_{ds}.json"
    if p.exists():
        return tuple((r["item_id"], SLOT.index(r["pred"]) if r.get("pred") in SLOT else None, SLOT.index(r["correct_label"]))
                     for r in json.loads(p.read_text())["results"])
    return tuple(_csv_rows().get((src, ds), []))


def old_stats(stem, ds):
    R = earlier_records(stem, ds)
    if not R:
        return None
    by = defaultdict(list)
    for iid, pick, _ in R:
        by[iid].append(pick)
    acc_pos = {s: sum(p == g for _, p, g in R if g == si) / max(1, sum(1 for _, _, g in R if g == si)) for si, s in enumerate(SLOT)}
    return {"accuracy": sum(p == g for _, p, g in R) / len(R), "acc_by_position": acc_pos,
            "same_letter": sum(1 for v in by.values() if len(set(v)) == 1), "items": len(by)}


# ------------------------------------------------------------------ SVG charts
def grouped_bars(series, height=230, width=600, label="chart", names=SLOT, tip="{name}, correct answer at {s}: {v}"):
    """series: list of (name, {slot: value}, css_class). Values in [0, 1]. names: x-axis labels for A-D."""
    left, right, top, bottom = 44, 12, 12, 34
    ph, pw = height - top - bottom, width - left - right
    n = len(series)
    group_w = pw / len(SLOT)
    bar_w = min(38, (group_w * 0.7 - 2 * (n - 1)) / n)
    out = [f'<svg class="chart" viewBox="0 0 {width} {height}" role="img" aria-label="{esc(label)}">']
    for t in (0, 0.25, 0.5, 0.75, 1.0):
        y = top + ph * (1 - t)
        cls = "base" if t == 0 else "grid"
        out.append(f'<line class="{cls}" x1="{left}" x2="{width - right}" y1="{y:.1f}" y2="{y:.1f}"/>')
        out.append(f'<text class="tick" x="{left - 8}" y="{y + 4:.1f}" text-anchor="end">{int(t * 100)}%</text>')
    for gi, s in enumerate(SLOT):
        gx = left + gi * group_w + (group_w - (n * bar_w + (n - 1) * 2)) / 2
        for si, (name, vals, cls) in enumerate(series):
            v = vals.get(s) or 0.0
            h = max(ph * v, 0.0)
            x = gx + si * (bar_w + 2)
            y = top + ph - h
            r = min(4, h / 2, bar_w / 2)
            path = (f"M{x:.1f},{top + ph:.1f} V{y + r:.1f} Q{x:.1f},{y:.1f} {x + r:.1f},{y:.1f} "
                    f"H{x + bar_w - r:.1f} Q{x + bar_w:.1f},{y:.1f} {x + bar_w:.1f},{y + r:.1f} V{top + ph:.1f} Z")
            out.append(f'<path class="bar {cls}" d="{path}"><title>{tip.format(name=esc(name), s=names[gi], v=f"{v * 100:.1f}%")}</title></path>')
        out.append(f'<text class="tick" x="{left + gi * group_w + group_w / 2:.1f}" y="{height - 12}" text-anchor="middle">{names[gi]}</text>')
    out.append("</svg>")
    return "".join(out)


def slot_profile(vals, cls="s-new"):
    """Tiny four-bar profile (A-D) for table cells."""
    w, h, bw, gap = 64, 26, 13, 3
    out = [f'<svg class="spark" viewBox="0 0 {w} {h}" role="img" aria-label="accuracy by position">']
    out.append(f'<line class="base" x1="0" x2="{w}" y1="{h - 0.5}" y2="{h - 0.5}"/>')
    for i, s in enumerate(SLOT):
        v = vals.get(s) or 0.0
        bh = max((h - 2) * v, 1.0)
        out.append(f'<rect class="bar {cls}" x="{i * (bw + gap) + 1}" y="{h - 1 - bh:.1f}" width="{bw}" height="{bh:.1f}" rx="2">'
                   f'<title>correct at {s}: {v * 100:.1f}%</title></rect>')
    out.append("</svg>")
    return "".join(out)


def legend(items):
    return '<div class="legend">' + "".join(
        f'<span><i class="key {cls}"></i>{esc(name)}</span>' for name, cls in items) + "</div>"


FW, FPAD = 132, 12


def _fx(v, ax):
    return FPAD + (v - ax[0]) / (ax[1] - ax[0]) * (FW - 2 * FPAD)


def forest(d, lo, hi, sig, ax):
    """Gain (dot) and 95% CI (whisker) on a shared points scale; filled dot = significant after Holm."""
    h = 22
    label = f"gain {d:+.1f} points, 95% CI {lo:+.1f} to {hi:+.1f}, {'significant' if sig else 'not significant'} after Holm"
    return (f'<svg class="forest" viewBox="0 0 {FW} {h}" role="img" aria-label="{label}">'
            f'<line class="zero" x1="{_fx(0, ax):.1f}" x2="{_fx(0, ax):.1f}" y1="1" y2="{h - 1}"/>'
            f'<line class="ci" x1="{_fx(lo, ax):.1f}" x2="{_fx(hi, ax):.1f}" y1="{h / 2}" y2="{h / 2}"/>'
            f'<circle class="pt{"" if sig else " open"}" cx="{_fx(d, ax):.1f}" cy="{h / 2}" r="4.5"><title>{label}</title></circle></svg>')


def forest_axis(ax, step=10):
    ticks = range(int(math.ceil(ax[0] / step) * step), int(ax[1]) + 1, step)
    return (f'<svg class="forest" viewBox="0 0 {FW} 14" aria-hidden="true">' + "".join(
        f'<text class="tick" x="{_fx(t, ax):.1f}" y="11" text-anchor="middle">{"0" if t == 0 else f"{t:+d}".replace("-", "−")}</text>'
        for t in ticks) + "</svg>")


QWEN_ROWS = [  # (method key in hf_results.json, name, short description)
    ("single", "Ask once", "one rotation, most likely label"),
    ("prior_heldout", "Prior correction", "label preference learned from other questions, divided out"),
    ("pride", "PriDe", "preference learned from 5% of questions asked in all rotations"),
    ("majority", "Majority vote", "one vote per rotation"),
    ("pav", "PAV", "position-aware vote, no answer key"),
    ("pav_adaptive", "PAV-adaptive", "stops when 95% sure"),
    ("avg_prob", "Probability average", "each option's probability averaged over the rotations"),
    ("avg_logprob", "Log-probability average", "the same, in log space"),
]


def qwen_parts(qw):
    """Method table rows, slot chart and the numbers quoted in the text, for Qwen2.5-0.5B with letters."""
    blk = {ds: qw.get(f"Qwen2.5-0.5B | {ds} | letter") for ds in DS_NAME}
    if not all(blk.values()):
        return None
    M = {ds: b["methods"] for ds, b in blk.items()}
    cis = [M[ds][k]["vs_single"]["ci95"] for ds in DS_NAME for k, _, _ in QWEN_ROWS if k != "single"]
    ax = (math.floor(min(0, min(c[0] for c in cis)) / 5) * 5, math.ceil(max(c[1] for c in cis) / 5) * 5)
    rows = []
    for k, name, note in QWEN_ROWS:
        ms = [M[ds][k] for ds in DS_NAME]
        calls = " / ".join(f"{m['calls']:.2f}" for m in ms) if isinstance(ms[0]["calls"], float) else ms[0]["calls"].replace(" + 40 once", " (+40 once)")
        cells = []
        for m in ms:
            if k == "single":
                cells.append(f"<td class='num'>{pct(m['accuracy'])}</td><td class='sub-inline'>baseline</td>")
                continue
            t = m["vs_single"]; lo, hi = t["ci95"]
            cells.append(f"<td class='num'>{pct(m['accuracy'])}</td>"
                         f"<td><div class='gain'>{forest(t['diff'], lo, hi, t['p_holm'] < 0.05, ax)}"
                         f"<span class='num'>{t['diff']:+.1f}<span class='sub'>{lo:+.1f} to {hi:+.1f}</span></span></div></td>")
        rows.append(f"<tr><td>{esc(name)}<span class='sub'>{esc(note)}</span></td><td class='num'>{esc(calls)}</td>{''.join(cells)}</tr>")

    single = {ds: M[ds]["single"] for ds in DS_NAME}
    fav = {ds: max(SLOT, key=lambda s: single[ds]["pick_share"][s]) for ds in DS_NAME}
    ob = "allenai_openbookqa"
    fav_ob = fav[ob]
    chart = (legend([("Asked once", "s-old"), ("With prior correction", "s-new")]) +
             grouped_bars([("Asked once", M[ob]["single"]["acc_by_slot"], "s-old"),
                           ("With prior correction", M[ob]["prior_heldout"]["acc_by_slot"], "s-new")],
                          label="Qwen2.5-0.5B on OpenBookQA: accuracy by position of the correct answer, asked once and with prior correction"))

    def rng(vals, f=pct):
        lo, hi = min(vals), max(vals)
        return f(lo, 0) if f(lo, 0) == f(hi, 0) else f"{f(lo, 0)}–{f(hi, 0)}"

    def gain(k):
        return [M[ds][k]["vs_single"]["diff"] for ds in DS_NAME]

    def sig_all(k):
        return all(M[ds][k]["vs_single"]["p_holm"] < 0.05 for ds in DS_NAME)

    pav_vs_maj = [blk[ds]["pairs"]["pav vs majority"] for ds in DS_NAME]
    one = gain("prior_heldout") + gain("pride")
    la = {ds: M[ds]["avg_logprob"]["vs_single"]["diff"] for ds in DS_NAME}
    votes = [blk[ds]["votes_no_single_winner"] for ds in DS_NAME]
    calls_ad = [M[ds]["pav_adaptive"]["calls"] for ds in DS_NAME]
    pav_d = " and ".join(f"{t['diff']:+.1f}{' points' if i == 0 else ''} on {DS_NAME[ds]}" for i, (ds, t) in enumerate(zip(DS_NAME, pav_vs_maj)))
    maj = gain("majority")
    maj_txt = f"{maj[0]:+.1f} points" if f"{min(maj):+.1f}" == f"{max(maj):+.1f}" else f"{min(maj):+.1f} to {max(maj):+.1f} points"
    t_one = (f"Prior correction and PriDe cost one call per question and use no answer key. They raised accuracy by "
             f"{min(one):+.1f} to {max(one):+.1f} points"
             + (", each gain significant after correction for multiple tests" if sig_all("prior_heldout") and sig_all("pride") else "")
             + f". The gap between the best and worst position fell from {pts(single[ob]['gap'])} to "
               f"{pts(M[ob]['prior_heldout']['gap'])} points on OpenBookQA.")
    t_avg = (f"Averaging the log-probabilities over the four rotations gave the largest gain: {la['allenai_ai2_arc']:+.1f} points on "
             f"ARC-Challenge and {la[ob]:+.1f} on OpenBookQA. Every option sits once in every position, so the pull toward "
             f"{fav_ob} affects all options equally and cancels out, while the model's knowledge of the content adds up.")
    t_vote = (f"Voting on the four final answers helped less ({maj_txt} for majority voting). "
              f"With so many answers on {fav_ob}, the four votes had no single winner for "
              f"{rng(votes)} of the questions, and the probabilities that could break the tie are thrown away. PAV was "
              + ("not significantly different from majority voting on either dataset"
                 if all(t["ci95"][0] <= 0 <= t["ci95"][1] for t in pav_vs_maj) else "significantly different from majority voting on at least one dataset")
              + f" ({pav_d}), and PAV-adaptive needed "
              f"{rng(calls_ad, lambda x, d: f'{x:.1f}')} calls per question because this model is rarely sure.")
    num = {ds: qw.get(f"Qwen2.5-0.5B | {ds} | number") for ds in DS_NAME}
    t_num = ""
    if all(num.values()) and min(b["n_questions"] for b in num.values()) >= 190:
        N = {ds: b["methods"] for ds, b in num.items()}
        two = lambda k: " and ".join(f"{N[ds][k]['vs_single']['diff']:+.1f}" for ds in DS_NAME)
        top2 = [sum(sorted(N[ds]["single"]["pick_share"].values())[-2:]) for ds in DS_NAME]
        t_num = (f"<li>With the labels (1)–(4) instead of A–D the model is less accurate (asked once: "
                 f"{' and '.join(pct(N[ds]['single']['accuracy']) for ds in DS_NAME)}) and puts {rng(top2)} of its answers "
                 f"on just two labels, but the same methods help. Averaging log-probabilities over the four rotations adds "
                 f"{two('avg_logprob')} points, the prior correction {two('prior_heldout')}, and majority voting {two('majority')} "
                 f"(ARC-Challenge and OpenBookQA).</li>")
    fill = {
        "q_t_num": t_num,
        "q_slot": fav_ob if len(set(fav.values())) == 1 else "one position",
        "q_pull": rng([single[ds]["pick_share"][fav[ds]] for ds in DS_NAME]),
        "q_fav_acc": rng([single[ds]["acc_by_slot"][fav[ds]] for ds in DS_NAME]),
        "q_other_acc": rng([sum(v for s, v in single[ds]["acc_by_slot"].items() if s != fav[ds]) / 3 for ds in DS_NAME]),
        "q_acc_b": pct(single[ob]["accuracy"]), "q_acc_a": pct(M[ob]["prior_heldout"]["accuracy"]),
        "q_la": pct(M[ob]["avg_logprob"]["accuracy"]), "q_maj": pct(M[ob]["majority"]["accuracy"]),
        "q_mass": rng([blk[ds]["label_mass_mean"] for ds in DS_NAME]),
        "q_n": str(blk[ob]["n_questions"]),
        "qwen_rows": "".join(rows), "qwen_chart": chart,
        "q_axis": forest_axis(ax), "q_t_one": t_one, "q_t_avg": t_avg, "q_t_vote": t_vote,
    }
    return fill


# ------------------------------------------------------------------ where the answers go
EARLIER = {  # corrected model -> earlier run of the same model (results file stem or CSV model name, label)
    "gemma4:31b-cloud": ("gemma4:31b-cloud", "Earlier cloud run"),
    "gpt-oss:120b": ("gpt-oss:120b", "Earlier cloud run"),
    "ministral-3:14b": ("ministral-3:14b-cloud", "Earlier cloud run"),
}
NUM_NAMES = ["1", "2", "3", "4"]
WORDS = {1: "one", 2: "two", 3: "three", 4: "four", 5: "five", 6: "six", 7: "seven", 8: "eight", 9: "nine", 10: "ten"}


def earlier_for(key, src=None, label="Earlier run"):
    """(source, label) of the earlier letters run of the same model, or None if there is none."""
    if key in EARLIER:
        return EARLIER[key]
    src = src or key
    return (src, label) if any(earlier_records(src, ds) for ds in DS_NAME) else None


def spread_of(rows):
    """rows: (picked position or None, position of the correct answer). Shares over all prompts and over wrong answers."""
    errs = [r for r in rows if r[0] != r[1]]
    sh = lambda sub, i: {SLOT[s]: sum(r[i] == s for r in sub) / len(sub) for s in range(4)} if sub else None
    return {"n_rows": len(rows), "accuracy": 1 - len(errs) / len(rows), "gold_share": sh(rows, 1),
            "pred_share_all": sh(rows, 0), "n_errors": len(errs), "errors_pred_at": sh(errs, 0),
            "errors_correct_at": sh(errs, 1)}


def earlier_spread(src, ds):
    rows = [(pick, gold) for _, pick, gold in earlier_records(src, ds)]
    return spread_of(rows) if rows else None


def spread_panel(idx, title, cols, names, hidden, note=""):
    """cols: (label, spread, css class) for each run shown, earliest first."""
    series = [("Ground truth", cols[-1][1]["gold_share"], "s-gt")] + [(f"{l} picks", x["pred_share_all"], c) for l, x, c in cols]
    chart = (legend([("Ground truth: where the correct answer is", "s-gt")] + [(f"{l}: where the model's answer is", c) for l, _, c in cols]) +
             grouped_bars(series, label=f"{title}: share of prompts at each position, correct answer and model answer",
                          names=names, tip="{name}: {v} of prompts at position {s}"))
    head = ("<tr><th rowspan='2'>Position</th>"
            f"<th colspan='{len(cols)}'>% errors predicted at</th><th colspan='{len(cols)}'>% errors where correct at</th></tr>"
            "<tr>" + "".join(f"<th class='num'>{esc(l)}</th>" for l, _, _ in cols) * 2 + "</tr>")
    rows = []
    for si, s in enumerate(SLOT):
        rows.append(f"<tr><td>{names[si]}</td>"
                    + "".join(f"<td class='num'>{pct(x['errors_pred_at'][s]) if x['n_errors'] else '–'}</td>" for _, x, _ in cols)
                    + "".join(f"<td class='num'>{pct(x['errors_correct_at'][s]) if x['n_errors'] else '–'}</td>" for _, x, _ in cols) + "</tr>")
    none = [1 - sum(x["errors_pred_at"].values()) if x["n_errors"] else 0.0 for _, x, _ in cols]
    if max(none) >= 0.0005:
        rows.append("<tr><td>No readable answer</td>" + "".join(f"<td class='num'>{pct(v)}</td>" for v in none)
                    + "".join("<td class='num'>–</td>" for _ in cols) + "</tr>")

    def says(label, x):
        if not x["n_errors"]:
            return f"{label}: no wrong answers out of {x['n_rows']} prompts."
        e = x["errors_pred_at"]
        top = max(SLOT, key=lambda s: e[s])
        where = (f"they are spread over all four positions ({pct(min(e.values()), 0)}–{pct(max(e.values()), 0)} each)"
                 if max(e.values()) - min(e.values()) <= 0.15 else f"{pct(e[top])} of them were {names[SLOT.index(top)]}")
        return f"{label}: {x['n_errors']} wrong answers out of {x['n_rows']} prompts ({pct(x['accuracy'])} accuracy); {where}."
    cap = " ".join(says(l, x) for l, x, _ in cols) + (f" {note}" if note else "")
    return (f"<div class='spread' data-spread='{idx}'{' hidden' if hidden else ''}>"
            f"<figure>{chart}<figcaption>{esc(title)}. {esc(cap)}</figcaption></figure>"
            f"<div class='table-wrap'><table><thead>{head}</thead><tbody>{''.join(rows)}</tbody></table></div></div>")


def spread_section(runs):
    """runs: (model key, model name, dataset, format, [(label, spread, css)] for the new run, caption note,
    earlier run as (source, label) or None). Returns the picker options and the panels."""
    default = next((i for i, r in enumerate(runs) if r[0] == "gpt-oss:120b" and r[2] == "allenai_ai2_arc" and r[3] == "letter"), 0)
    groups, panels = defaultdict(list), []
    for i, (key, name, ds, fmt, new_cols, note, earlier) in enumerate(runs):
        src = earlier if fmt == "letter" else None
        old = earlier_spread(src[0], ds) if src else None
        cols = ([(src[1], old, "s-old")] if old else []) + new_cols
        label = f"{name} · {DS_NAME[ds]} · {'letters A–D' if fmt == 'letter' else 'numbers 1–4'}"
        groups[fmt].append(f"<option value='{i}'{' selected' if i == default else ''}>{esc(label)}{' · with earlier run' if old else ''}</option>")
        panels.append(spread_panel(i, label, cols, SLOT if fmt == "letter" else NUM_NAMES, i != default, note))
    options = "".join(f"<optgroup label='{'Letters A–D' if fmt == 'letter' else 'Numbers 1–4'}'>{''.join(o)}</optgroup>"
                      for fmt, o in groups.items())
    return options, "".join(panels)


# ------------------------------------------------------------------ page
def build(out_path: Path):
    v3 = json.loads((HERE / "v3_results.json").read_text())
    qw_path = HERE / "hf_results.json"
    qw = json.loads(qw_path.read_text()) if qw_path.exists() else {}

    def get(model, ds, fmt):
        """A block counts only once at least 95% of its 800 prompts are answered (runs still in progress are left out)."""
        r = v3.get(f"{model} | {ds} | {fmt}")
        return r if r and r["bias"]["n_rows"] >= 760 else None

    others = sorted({k.split(" | ")[0] for k in v3} - set(NEW_MODELS))       # models added later, in name order
    present = [m for m in NEW_MODELS + others if any(get(m, ds, "letter") for ds in DS_NAME)]

    # ---- before/after table rows
    ba_rows = []
    for m in present:
        for stem, lab, note in OLD.get(m) or ([(m, f"{SHOW.get(m, m)}, earlier run", "same model")] if earlier_for(m) else []):
            for ds in DS_NAME:
                o, n = old_stats(stem, ds), get(m, ds, "letter")
                if not o or not n:
                    continue
                b = n["bias"]
                ba_rows.append(
                    f"<tr><td>{esc(lab)}<span class='sub'>{esc(note)} as corrected run: {esc(SHOW.get(m, m))}</span></td>"
                    f"<td>{DS_NAME[ds]}</td>"
                    f"<td class='num'><s class='old'>{pct(o['accuracy'])}</s></td>"
                    f"<td class='num'><strong>{pct(b['accuracy'])}</strong></td>"
                    f"<td class='num'>{o['same_letter']}/{o['items']}</td>"
                    f"<td class='num'>{pct(b['invalid_final'])}</td></tr>")

    # ---- headline example: gemma4 ARC
    g_old = old_stats("gemma4:31b", "allenai_ai2_arc")
    g_new = get("gemma4:31b-cloud", "allenai_ai2_arc", "letter")
    hero_chart = ""
    if g_old and g_new:
        hero_chart = (legend([("Old run (flawed answer reading)", "s-old"), ("Corrected run", "s-new")]) +
                      grouped_bars([("Old run", g_old["acc_by_position"], "s-old"),
                                    ("Corrected run", g_new["bias"]["acc_by_position"], "s-new")],
                                   label="gemma4:31b on ARC-Challenge: accuracy by position of the correct answer, old vs corrected"))

    # ---- corrected bias table
    bias_rows = []
    gaps = []
    for m in present:
        for ds in DS_NAME:
            r = get(m, ds, "letter")
            if not r:
                continue
            b = r["bias"]
            gaps.append(b["sensitivity_gap"])
            ap = b["acc_by_position"]
            bias_rows.append(
                f"<tr><td>{esc(SHOW.get(m, m))}</td><td>{DS_NAME[ds]}</td>"
                f"<td class='num'>{pct(b['accuracy'])}</td>"
                f"<td>{slot_profile(ap)}</td>"
                f"<td class='num slots'>" + " ".join(f"<span>{s} {ap[s] * 100:.0f}</span>" for s in SLOT) + "</td>"
                f"<td class='num'>{pts(b['sensitivity_gap'])}</td><td class='num'>{b['rstd']:.1f}</td>"
                f"<td class='num'>{pct(b['circular_eval'])}</td><td class='num'>{pct(b['fluctuation'])}</td>"
                f"<td class='num'>{pct(b['invalid_final'])}</td></tr>")

    # ---- letters vs numbers
    ln_rows = []
    for m in present:
        for ds in DS_NAME:
            L, N, T = get(m, ds, "letter"), get(m, ds, "number"), v3.get(f"{m} | {ds} | letters_vs_numbers")
            if not (L and N and T):
                continue
            ln_rows.append(
                f"<tr><td>{esc(SHOW.get(m, m))}</td><td>{DS_NAME[ds]}</td>"
                f"<td class='num'>{pct(L['bias']['accuracy'])}</td><td class='num'>{pct(N['bias']['accuracy'])}</td>"
                f"<td class='num'>{pts(L['bias']['sensitivity_gap'])}</td><td class='num'>{pts(N['bias']['sensitivity_gap'])}</td>"
                f"<td class='num'>{T['letters_only_correct']} / {T['numbers_only_correct']}</td>"
                f"<td class='num'>{T['mcnemar_exact_p']:.2g}</td></tr>")

    # ---- mitigation
    mit_rows = []
    for m in present:
        for ds in DS_NAME:
            for fmt in ("letter", "number"):
                r = get(m, ds, fmt)
                if not r or "mitigation" not in r:
                    continue
                x = r["mitigation"]
                lo, hi = x["pav_minus_majority_ci95"]
                mit_rows.append(
                    f"<tr><td>{esc(SHOW.get(m, m))}</td><td>{DS_NAME[ds]}</td><td>{'letters' if fmt == 'letter' else 'numbers'}</td>"
                    f"<td class='num'>{pct(x['single_order'])}</td><td class='num'>{pct(x['majority'])}</td>"
                    f"<td class='num'>{pct(x['pav'])}</td><td class='num'>{lo:+.1f} to {hi:+.1f}</td>"
                    f"<td class='num'>{pct(x['pav_adaptive'])} <span class='sub-inline'>{x['pav_adaptive_calls']:.2f} calls</span></td>"
                    f"<td class='num'>{pct(x['majority_stop2'])} <span class='sub-inline'>{x['majority_stop2_calls']:.2f} calls</span></td></tr>")

    # ---- open-weight models (probabilities): Qwen2.5-0.5B is the worked example, others are listed too
    qfill = qwen_parts(qw)
    assert qfill, "hf_results.json needs both Qwen2.5-0.5B letter blocks"
    hf_names = sorted({k.split(" | ")[0] for k, r in qw.items() if "methods" in r}, key=lambda n: (n != "Qwen2.5-0.5B", n))
    hf_done = lambda n, ds, fmt: qw.get(f"{n} | {ds} | {fmt}") if (qw.get(f"{n} | {ds} | {fmt}") or {}).get("n_questions", 0) >= 190 else None
    q_ln = defaultdict(list)
    for name in hf_names:
        for ds in DS_NAME:
            Lq, Nq, Tq = hf_done(name, ds, "letter"), hf_done(name, ds, "number"), qw.get(f"{name} | {ds} | letters_vs_numbers")
            if not (Lq and Nq and Tq):
                continue
            l, n = Lq["methods"]["single"], Nq["methods"]["single"]
            q_ln[name].append((ds, n["accuracy"] - l["accuracy"], Tq["mcnemar_exact_p"], l["gap"], n["gap"]))
            ln_rows.append(
                f"<tr><td>{esc(name)}<span class='sub'>probabilities, asked once</span></td><td>{DS_NAME[ds]}</td>"
                f"<td class='num'>{pct(l['accuracy'])}</td><td class='num'>{pct(n['accuracy'])}</td>"
                f"<td class='num'>{pts(l['gap'])}</td><td class='num'>{pts(n['gap'])}</td>"
                f"<td class='num'>{Tq['letters_only_correct']} / {Tq['numbers_only_correct']}</td>"
                f"<td class='num'>{Tq['mcnemar_exact_p']:.2g}</td></tr>")

    # ---- every open-weight model, letters: each method's accuracy and gain over asking once
    OPEN_COLS = ["single", "prior_heldout", "pride", "majority", "pav", "avg_logprob"]
    open_rows, others_short = [], []
    for name in hf_names:
        for ds in DS_NAME:
            r = hf_done(name, ds, "letter")
            if not r:
                continue
            cells = []
            for k in OPEN_COLS:
                m = r["methods"][k]
                if k == "single":
                    cells.append(f"<td class='num'>{pct(m['accuracy'])}</td>")
                    continue
                t = m["vs_single"]
                gain = f"{t['diff']:+.1f}"
                cells.append(f"<td class='num'>{pct(m['accuracy'])}<span class='sub'>"
                             f"{f'<strong>{gain}</strong>' if t['p_holm'] < 0.05 else gain}</span></td>")
            open_rows.append(f"<tr><td>{esc(name)}</td><td>{DS_NAME[ds]}</td><td class='num'>{pct(r['label_mass_mean'], 0)}</td>{''.join(cells)}</tr>")
        ob = hf_done(name, "allenai_openbookqa", "letter")
        if name != "Qwen2.5-0.5B" and ob:
            t = ob["methods"]["avg_logprob"]["vs_single"]
            others_short.append((name, ob["methods"]["single"]["accuracy"], ob["methods"]["avg_logprob"]["accuracy"],
                                 t["p_holm"] < 0.05 and t["diff"] > 0))
    other_open_short = ""
    if others_short:
        same = [o for o in others_short if o[3]]
        other_open_short = " ".join(
            (f"The same holds for {n}: averaging raises it from {pct(a)} to {pct(b)} on OpenBookQA." if ok else
             f"For {n}, averaging changed OpenBookQA accuracy from {pct(a)} to {pct(b)}, which is not a significant gain.")
            for n, a, b, ok in others_short)

    # ---- where the answers go (one panel per run, picked from a list)
    spread_runs = []
    for fmt in ("letter", "number"):
        for m in present:
            for ds in DS_NAME:
                r = get(m, ds, fmt)
                if r and "gold_share" in r["bias"]:
                    x = r.get("mitigation")
                    note = (f"With the fixes (4 calls per question, so no favourite position): majority vote {pct(x['majority'])}, "
                            f"PAV {pct(x['pav'])}, compared with {pct(x['single_order'])} asked once. Prior correction needs the "
                            f"model's probabilities, which a text-only model does not give." if x else "")
                    spread_runs.append((m, SHOW.get(m, m), ds, fmt, [("Corrected run", r["bias"], "s-new")], note, earlier_for(m)))
        for name in hf_names:
            for ds in DS_NAME:
                q = hf_done(name, ds, fmt)
                if q and "spread_prior_heldout" in q:
                    spread_runs.append((
                        name, name, ds, fmt,
                        [("New run, asked once", q["spread_single"], "s-new"),
                         ("New run with prior correction", q["spread_prior_heldout"], "s-fix")],
                        f"Averaging over the four rotations (4 calls per question) gives the same answer whatever the order, "
                        f"so it has no favourite position, and reaches {pct(q['methods']['avg_logprob']['accuracy'])} accuracy.",
                        earlier_for(name, q["hf_id"], "Earlier run, older prompt")))
    spread_options, spread_panels = spread_section(spread_runs)
    with_old = list(dict.fromkeys(r[1] for r in spread_runs if r[3] == "letter" and r[6]))
    spread_earlier = (", ".join(with_old[:-1]) + " and " + with_old[-1]) if len(with_old) > 1 else "".join(with_old)

    # ---- key numbers for the summary
    def acc(m, ds, fmt="letter"):
        r = get(m, ds, fmt)
        return r["bias"]["accuracy"] if r else None

    g_old_arc = g_old["accuracy"] if g_old else None
    gap_lo, gap_hi = (min(gaps), max(gaps)) if gaps else (None, None)
    accs = [get(m, ds, "letter")["bias"]["accuracy"] for m in present for ds in DS_NAME if get(m, ds, "letter")]

    # letters vs numbers, in words
    ln = []
    for m in present:
        for ds in DS_NAME:
            L, N, T = get(m, ds, "letter"), get(m, ds, "number"), v3.get(f"{m} | {ds} | letters_vs_numbers")
            if L and N and T:
                ln.append((N["bias"]["accuracy"] - L["bias"]["accuracy"], T["mcnemar_exact_p"]))
    if ln:
        n_up = sum(d > 0 for d, _ in ln); n_down = sum(d < 0 for d, _ in ln)
        sig = sum(p < 0.05 for _, p in ln)
        biggest = max(abs(d) for d, _ in ln)
        ln_text = (f"Across {len(ln)} model and dataset pairs, numbers were more accurate in {n_up} and letters in {n_down}; "
                   f"the largest difference was {biggest * 100:.1f} points, and {sig} of the {len(ln)} differences "
                   f"{'is' if sig == 1 else 'are'} significant (exact McNemar test, p &lt; 0.05).")
        if biggest < 0.05:
            ln_text += (" With answers read the same way for both formats, numbers are not the cure the earlier "
                        "stack_v1 results suggested. The earlier jump from about 25% to above 90% came from the flawed "
                        "letter reading, not from the switch to numbers.")
    else:
        ln_text = "The numeric runs of the cloud models are not complete yet."
    for name, rows in q_ln.items():
        ln_text += (f" For {name} asked once, numbers changed accuracy by "
                    + " and ".join(f"{d * 100:+.1f} points on {DS_NAME[ds]} (p = {p:.2g})" for ds, d, p, _, _ in rows)
                    + "; its position gap went from " + " and ".join(f"{pts(gl)} to {pts(gn)}" for _, _, _, gl, gn in rows)
                    + " points.")

    # mitigation, in words
    mits = [get(m, ds, f)["mitigation"] for m in present for ds in DS_NAME for f in ("letter", "number")
            if get(m, ds, f) and "mitigation" in get(m, ds, f)]
    if mits:
        maj_d = [x["majority"] - x["single_order"] for x in mits]
        better = sum(x["pav_minus_majority_ci95"][0] > 0 for x in mits)
        worse = sum(x["pav_minus_majority_ci95"][1] < 0 for x in mits)
        calls = sum(x["pav_adaptive_calls"] for x in mits) / len(mits)
        ad_d = [x["pav_adaptive"] - x["majority"] for x in mits]
        mit_text = (f"Majority voting over the four rotations changed accuracy by {min(maj_d) * 100:+.1f} to "
                    f"{max(maj_d) * 100:+.1f} points compared with a single order. PAV was significantly better than "
                    f"majority voting in {better} of {len(mits)} settings and significantly worse in {worse}. "
                    f"PAV-adaptive needed {calls:.2f} calls per question on average instead of 4, with accuracy within "
                    f"{min(ad_d) * 100:+.1f} to {max(ad_d) * 100:+.1f} points of full majority voting.")
    else:
        mit_text = "The mitigation analysis needs the complete runs."

    # PAV against plain majority voting, every setting with a CI (cloud runs and Qwen)
    pav_cis = [(f"{m} | {ds} | {f}", x["pav"] - x["majority"], x["pav_minus_majority_ci95"])
               for m in present for ds in DS_NAME for f in ("letter", "number")
               if get(m, ds, f) and "mitigation" in get(m, ds, f) for x in [get(m, ds, f)["mitigation"]]]
    pav_cis += [(k, r["pairs"]["pav vs majority"]["diff"] / 100, r["pairs"]["pav vs majority"]["ci95"])
                for k, r in qw.items() if "methods" in r and r["n_questions"] >= 190]

    def setting(key):
        m, ds, fmt = [s.strip() for s in key.split("|")]
        return f"{SHOW.get(m, m)} on {DS_NAME[ds]}{', numbers' if fmt == 'number' else ''}"

    better = [f"{setting(k)}, {d * 100:+.1f} points" for k, d, (lo, _) in pav_cis if lo > 0]
    p_worse = sum(hi < 0 for _, _, (_, hi) in pav_cis)
    vote_short = (f"The proposed position-aware vote (PAV) was significantly better than plain majority voting in {len(better)} "
                  f"of {len(pav_cis)} settings" + (f" ({'; '.join(better)})" if better else "")
                  + f" and significantly worse in {p_worse}"
                  + (", so on this evidence it does not improve on majority voting." if not better else "."))
    if mits:
        vote_short = (f"On the cloud models, voting over four rotations changed accuracy by {min(maj_d) * 100:+.1f} to "
                      f"{max(maj_d) * 100:+.1f} points, because little bias is left to remove. " + vote_short)

    fill = {
        "g_old": pct(g_old_arc), "g_new": pct(acc("gemma4:31b-cloud", "allenai_ai2_arc")),
        "gap_lo": pts(gap_lo), "gap_hi": pts(gap_hi),
        "acc_lo": pct(min(accs)) if accs else "–", "acc_hi": pct(max(accs)) if accs else "–",
        "hero_chart": hero_chart, "ba_rows": "".join(ba_rows), "bias_rows": "".join(bias_rows),
        "ln_rows": "".join(ln_rows), "mit_rows": "".join(mit_rows), "models": ", ".join(SHOW.get(m, m) for m in present),
        "ln_text": ln_text, "mit_text": mit_text, "vote_short": vote_short, **qfill,
        "spread_options": spread_options, "spread_panels": spread_panels, "spread_earlier": spread_earlier,
        "n_models": WORDS.get(len(present), str(len(present))), "hf_models": ", ".join(hf_names),
        "open_rows": "".join(open_rows), "other_open_short": other_open_short,
    }
    page = (HERE / "report_template.html").read_text(encoding="utf-8")
    for k, v in fill.items():
        page = page.replace("{{" + k + "}}", v)
    missing = [k for k in fill if "{{" + k + "}}" in page]
    assert not missing and "{{" not in page, f"unfilled placeholders: {missing}"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(page, encoding="utf-8")
    print(f"wrote {out_path}")

if __name__ == "__main__":
    build(Path(sys.argv[1]) if len(sys.argv) > 1 else HERE / "report.html")
