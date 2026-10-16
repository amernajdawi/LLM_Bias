"""
Position Bias Explorer: Streamlit dashboard for the corrected (v3) results.
Reads dashboard_data/v3_dashboard.json (no database needed).

    .venv/bin/python3 -m streamlit run dashboard_v3.py
Refresh the data: bash experiments/v3_corrected/update_results.sh && .venv/bin/python3 experiments/v3_corrected/export_dashboard.py
"""
import html
import json
import math
import re
import statistics
import sys
from pathlib import Path

import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from plotly.subplots import make_subplots

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
from src.analysis.metrics import (  # noqa: E402  same metrics as the earlier dashboard
    accuracy_by_position,
    accuracy_confidence_interval,
    accuracy_position_significance_test,
    chi_square_option_proportion,
    error_correct_position_proportion,
    error_prediction_proportion,
    option_proportion,
    overall_accuracy,
    sensitivity_gap,
)

DS_NAME = {"allenai_ai2_arc": "ARC-Challenge", "allenai_openbookqa": "OpenBookQA"}
DS_SHORT = {"allenai_ai2_arc": "ARC", "allenai_openbookqa": "OBQA"}
FMT_NAME = {"letter": "Letters A–D", "number": "Numbers 1–4"}
COMPANY = [("gemma", "Google"), ("gpt-oss", "OpenAI"), ("nemotron", "NVIDIA"), ("ministral", "Mistral"),
           ("llama", "Meta"), ("qwen", "Alibaba")]
WHERE = {"cloud": "Ollama Cloud", "local": "Local GPU (Ollama)", "openai": "OpenAI API"}
METHODS = [  # key in the data, name, model calls per question
    ("prior_heldout", "Prior correction", "1 call"),
    ("pride", "PriDe", "1 call"),
    ("majority", "Majority vote", "4 calls"),
    ("pav", "PAV", "4 calls"),
    ("pav_adaptive", "PAV-adaptive", "up to 4"),
    ("avg_logprob", "Averaging", "4 calls"),
]
SIMPLE_FIXES = [  # the three fixes shown on the summary page, in plain words
    ("avg_logprob", "Ask 4 times, combine the confidence"),
    ("majority", "Ask 4 times, take the most common answer"),
    ("prior_heldout", "Ask once, remove the favourite-letter habit"),
]
VERDICTS = [  # (upper limit of position bias in points, label, badge colour, icon, status colour)
    (5, "Fair", "green", ":material/check_circle:", "#0ca30c"),
    (15, "Some bias", "orange", ":material/error:", "#fab219"),
    (float("inf"), "Strong bias", "red", ":material/warning:", "#d03b3b"),
]
GROUPS = ["All models", "Large (7B+)", "Small (≤2B)", "Read from probabilities", "Choose…"]
FONT = "Public Sans, -apple-system, BlinkMacSystemFont, Segoe UI, Roboto, sans-serif"
# Validated palette (blue / orange / aqua pass the colour-blindness and contrast checks in both modes).
PALETTE = {
    "light": dict(ink="#14161b", ink2="#52545a", grid="#e6e5e0", surface="#fbfbfa", gray="#a9abb0",
                  new="#2a78d6", old="#eb6834", fix="#1baf7a", neg="#e34948", mid="#f0efec", pos="#256abf",
                  seq=["#cde2fb", "#9ec5f4", "#6da7ec", "#3987e5", "#256abf", "#184f95", "#0d366b"]),
    "dark": dict(ink="#ecedef", ink2="#b9bcc4", grid="#2c2e33", surface="#161718", gray="#6b6e75",
                 new="#3987e5", old="#d95926", fix="#199e70", neg="#e66767", mid="#383835", pos="#5598e7",
                 seq=["#0d366b", "#104281", "#184f95", "#256abf", "#3987e5", "#6da7ec", "#b7d3f6"]),
}
PLOT_CONFIG = {"displaylogo": False, "toImageButtonOptions": {"format": "png", "scale": 2},
               "modeBarButtonsToRemove": ["zoom2d", "pan2d", "select2d", "lasso2d", "zoomIn2d", "zoomOut2d",
                                          "autoScale2d", "resetScale2d"]}
DEFAULTS = {"group": "All models", "custom": [], "ds": list(DS_NAME), "fmt": ["letter"], "sort": "Accuracy",
            "chk_ds": "allenai_ai2_arc", "dd_ds": "allenai_ai2_arc", "dd_fmt": "letter"}


# ------------------------------------------------------------------ data
def company(name):
    low = name.lower()
    return next((c for prefix, c in COMPANY if low.startswith(prefix)), "—")


def size_of(name):
    m = re.search(r"(?<![\d.])(\d+(?:\.\d+)?)[bB](?![a-zA-Z])", name)
    return (m.group(1) + "B", float(m.group(1))) if m else ("—", None)


def mcnemar(b, c):
    n = b + c
    return 1.0 if n == 0 else min(1.0, 2 * sum(math.comb(n, i) for i in range(min(b, c) + 1)) / 2 ** n)


@st.cache_data(show_spinner="Loading results…")
def load():
    data = json.loads((ROOT / "dashboard_data" / "v3_dashboard.json").read_text())
    runs, earlier = data["runs"], data["earlier"]
    text_bases = {r["model"].replace("-cloud", "") for r in runs.values() if r["source"] != "probabilities"}
    records = []
    for key, r in runs.items():
        rows, L = r["rows"], r["letters"]
        base = r["model"].replace("-cloud", "").replace(" (Ollama)", "")
        probs = r["source"] == "probabilities"
        label = base + (" (probabilities)" if probs and base in text_bases else "")
        acc_pos = accuracy_by_position(rows, L)
        picks = option_proportion(rows, L)
        top = max(L, key=lambda x: picks[x])
        size, size_num = size_of(base)
        e = earlier.get(key)
        records.append({
            "key": key, "model": r["model"], "base": base, "label": label, "company": company(base), "size": size,
            "size_num": size_num, "read": "Probabilities" if probs else "Text",
            "where": ("Local GPU (Ollama)" if "(Ollama)" in r["model"] else "CPU (Hugging Face)") if probs else WHERE.get(r["source"], r["source"]),
            "dataset": r["dataset"], "fmt": r["format"], "letters": L, "n": len(rows),
            "acc": overall_accuracy(rows), "acc_pos": [acc_pos[x] for x in L], "gap": sensitivity_gap(acc_pos),
            "picks": [picks[x] for x in L], "top": top, "top_share": picks[top],
            "n_err": sum(1 for x in rows if x["correct"] != 1),
            "err_pred": error_prediction_proportion(rows, L), "err_gold": error_correct_position_proportion(rows, L),
            "ci": accuracy_confidence_interval(rows, L), "chi": chi_square_option_proportion(rows, L),
            "z_first_last": accuracy_position_significance_test(rows, L, L[0], L[-1]),
            "methods": r["methods"], "fix_picks": r.get("prior_corrected_picks"),
            "label_mass": r.get("label_mass"), "unreadable": r.get("unreadable"),
            "earlier": None if not e else {
                "source": e["source"], "acc": overall_accuracy(e["rows"]),
                "acc_pos": accuracy_by_position(e["rows"], list("ABCD")),
                "gap": sensitivity_gap(accuracy_by_position(e["rows"], list("ABCD"))),
                "picks": option_proportion(e["rows"], list("ABCD")),
                "ci": accuracy_confidence_interval(e["rows"], list("ABCD")),
                "err_pred": error_prediction_proportion(e["rows"], list("ABCD")),
                "err_gold": error_correct_position_proportion(e["rows"], list("ABCD")),
                "n": len(e["rows"]), "n_err": sum(1 for x in e["rows"] if x["correct"] != 1)},
            "_rows": {(x["item_id"], x["rotation"]): x["correct"] for x in rows},
        })
    df = pd.DataFrame(records)
    # letters vs numbers: the same questions and rotations, paired answer by answer
    pairs = []
    for (label, ds), g in df.groupby(["label", "dataset"]):
        if set(g.fmt) != {"letter", "number"}:
            continue
        lr, nr = g[g.fmt == "letter"].iloc[0], g[g.fmt == "number"].iloc[0]
        common = lr["_rows"].keys() & nr["_rows"].keys()
        b = sum(1 for k in common if lr["_rows"][k] == 1 and nr["_rows"][k] == 0)
        c = sum(1 for k in common if lr["_rows"][k] == 0 and nr["_rows"][k] == 1)
        pairs.append({"label": label, "dataset": ds, "letters": lr.acc, "numbers": nr.acc, "diff": nr.acc - lr.acc,
                      "letters_only": b, "numbers_only": c, "p": mcnemar(b, c), "pairs": len(common)})
    return df.drop(columns="_rows"), pd.DataFrame(pairs)


def is_large(r):
    return r.size_num is None or (isinstance(r.size_num, float) and math.isnan(r.size_num)) or r.size_num >= 7


def is_small(r):
    return not is_large(r) and r.size_num <= 2


def simple_models(df):
    """One row per model for the simple pages: letters A–D, averaged over both datasets, text answers preferred."""
    out = []
    for base, g in df[df.fmt == "letter"].groupby("base"):
        g = g[g.read == ("Text" if (g.read == "Text").any() else "Probabilities")]
        first = g.iloc[0]
        e = [x for x in g.earlier if x]
        same = bool(e) and len(e) == len(g) and all(x["source"].replace("-cloud", "") == first.model.replace("-cloud", "") for x in e)
        out.append({"base": base, "label": first.label, "company": first.company, "size": first["size"],
                    "size_num": first.size_num, "read": first.read, "acc": g.acc.mean(), "bias": g.gap.mean() * 100,
                    "old_acc": statistics.mean(x["acc"] for x in e) if same and first.read == "Text" else None})
    return pd.DataFrame(out)


def verdict(points):
    return next(v for v in VERDICTS if points < v[0])


# ------------------------------------------------------------------ chart and text helpers
def palette():
    t = getattr(getattr(st, "context", None), "theme", None)
    return PALETTE["dark" if getattr(t, "type", None) == "dark" else "light"]


def layout(P, height, **extra):
    axis = dict(gridcolor=P["grid"], linecolor=P["grid"], zeroline=False, automargin=True,
                tickfont=dict(color=P["ink2"], size=13), title_font=dict(color=P["ink2"], size=12))
    base = dict(height=height, margin=dict(l=4, r=4, t=30, b=4), font=dict(family=FONT, size=13, color=P["ink2"]),
                paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)", xaxis=axis, yaxis=dict(axis),
                legend=dict(orientation="h", yanchor="bottom", y=1.0, xanchor="left", x=0, title=None,
                            font=dict(color=P["ink2"], size=13), bgcolor="rgba(0,0,0,0)"),
                hoverlabel=dict(bgcolor=P["surface"], bordercolor=P["grid"], font=dict(family=FONT, color=P["ink"], size=13)),
                barcornerradius=4, bargap=0.38, bargroupgap=0.12)
    for k, v in extra.items():
        base[k] = {**base[k], **v} if isinstance(v, dict) and isinstance(base.get(k), dict) else v
    for ax in ("xaxis", "yaxis"):
        font = base[ax].pop("title_font", None)
        if isinstance(base[ax].get("title"), str):
            base[ax]["title"] = dict(text=base[ax]["title"], font=font)
    return base


def show(fig, key):
    st.plotly_chart(fig, width="stretch", theme=None, config=PLOT_CONFIG, key=key)


def scale(colors):
    return [[i / (len(colors) - 1), c] for i, c in enumerate(colors)]


def pct(x, d=1):
    return "—" if x is None or (isinstance(x, float) and math.isnan(x)) else f"{x * 100:.{d}f}%"


def position_labels(fmts):
    if fmts == {"letter"}:
        return list("ABCD")
    if fmts == {"number"}:
        return list("1234")
    return ["A / 1", "B / 2", "C / 3", "D / 4"]


def row_label(r, both_ds, both_fmt):
    return r.label + (f" · {DS_SHORT[r.dataset]}" if both_ds else "") + (f" · {'letters' if r.fmt == 'letter' else 'numbers'}" if both_fmt else "")


def header(eyebrow, title, lede=None):
    st.markdown(f"<div class='eyebrow'>{html.escape(eyebrow)}</div>", unsafe_allow_html=True)
    st.title(title, anchor=False)
    if lede:
        st.markdown(f"<p class='lede'>{lede}</p>", unsafe_allow_html=True)


def section(title, sub=None):
    st.subheader(title, anchor=False)
    if sub:
        st.markdown(f"<p class='sub'>{sub}</p>", unsafe_allow_html=True)


def takeaway(text):
    with st.container(border=True):
        st.markdown(f":material/lightbulb: **Takeaway** · {text}")


def question(number, q, answer, body):
    st.markdown(f"<div class='qnum'>Question {number}</div>", unsafe_allow_html=True)
    st.subheader(q, anchor=False)
    st.markdown(f"<p class='answer'>{answer}</p><p class='body'>{body}</p>", unsafe_allow_html=True)


# ------------------------------------------------------------------ app shell
st.set_page_config(page_title="Position Bias Explorer", page_icon=":material/bar_chart:", layout="wide",
                   initial_sidebar_state="collapsed")
st.markdown("""<style>
.block-container {max-width: 1180px; padding-top: 2.4rem; padding-bottom: 4rem;}
.eyebrow {text-transform: uppercase; letter-spacing: .08em; font-size: .72rem; font-weight: 600; opacity: .62;}
h1 {letter-spacing: -.01em;}
.lede {font-size: 1.12rem; line-height: 1.65; opacity: .85; max-width: 52rem; margin-top: -.4rem;}
.sub {opacity: .74; font-size: .95rem; line-height: 1.5; max-width: 64rem; margin-top: -.6rem;}
.qnum {font-size: .8rem; font-weight: 600; letter-spacing: .06em; text-transform: uppercase; opacity: .6; margin-bottom: -.8rem;}
.answer {font-size: 1.35rem; font-weight: 650; line-height: 1.35; margin: .2rem 0 .5rem;}
.body {font-size: 1.02rem; line-height: 1.65; opacity: .85;}
.legend-note {font-size: .9rem; opacity: .75;}
[data-testid="stMetricValue"] {font-weight: 600;}
</style>""", unsafe_allow_html=True)

df, lvn = load()
P = palette()
for k, v in DEFAULTS.items():
    st.session_state.setdefault(k, v)
all_labels = sorted(df.label.unique(), key=lambda s: (company(s), s.lower()))
SM = simple_models(df)


# ------------------------------------------------------------------ summary (the simple story)
def page_summary():
    header("Results · multiple-choice questions", "Do AI models pick answers by their position?",
           "We asked AI models 400 multiple-choice questions. Every question was asked <b>4 times</b>, and each time the "
           "right answer was in a different place: A, B, C or D. A fair model is just as good wherever the right answer is.")
    with st.container(horizontal=True):
        st.badge(f"{len(SM)} AI models", icon=":material/neurology:", color="gray")
        st.badge(f"{SM.company.nunique()} companies", icon=":material/apartment:", color="gray")
        st.badge("400 questions × 4 places", icon=":material/quiz:", color="gray")
        st.badge(f"{int(df.n.sum()):,} answers checked", icon=":material/fact_check:", color="gray")

    # 1 · were the old results right?
    st.divider()
    old = SM[SM.old_acc.notna()].sort_values("acc")
    a, b = st.columns([1, 1.35], gap="large", vertical_alignment="center")
    with a:
        question(1, "Were the old results right?", "No. The old code read many answers wrongly.",
                 f"The old code often misread the replies: for example, it wrote down \"A\" when it could not read one. "
                 f"So the models looked biased and weak. "
                 f"For the same {len(old)} models, the old results were <b>{old.old_acc.min():.0%}–{old.old_acc.max():.0%}</b> "
                 f"correct. Read correctly, they are <b>{old.acc.min():.0%}–{old.acc.max():.0%}</b> correct.")
    with b:
        fig = go.Figure()
        xs, ys = [], []
        for r in old.itertuples():
            xs += [r.old_acc, r.acc, None]; ys += [r.label, r.label, None]
        fig.add_scatter(x=xs, y=ys, mode="lines", line=dict(color=P["grid"], width=4), hoverinfo="skip", showlegend=False)
        fig.add_scatter(x=old.old_acc.tolist(), y=old.label.tolist(), mode="markers+text", name="Old result",
                        marker=dict(color=P["old"], size=14, line=dict(color=P["surface"], width=2)),
                        text=[f"{x:.0%}" for x in old.old_acc], textposition="middle left", textfont=dict(color=P["ink2"], size=12),
                        hovertemplate="<b>%{x:.0%}</b> correct in the old result<br>%{y}<extra></extra>")
        fig.add_scatter(x=old.acc.tolist(), y=old.label.tolist(), mode="markers+text", name="Read correctly",
                        marker=dict(color=P["new"], size=14, line=dict(color=P["surface"], width=2)),
                        text=[f"{x:.0%}" for x in old.acc], textposition="middle right", textfont=dict(color=P["ink"], size=12),
                        hovertemplate="<b>%{x:.0%}</b> correct when read correctly<br>%{y}<extra></extra>")
        fig.update_layout(**layout(P, 80 + 46 * len(old), xaxis=dict(range=[0, 1.12], tickformat=".0%", title="Right answers",
                                                                      tickvals=[0, .25, .5, .75, 1]),
                                   yaxis=dict(showgrid=False, tickfont=dict(color=P["ink"], size=13))))
        show(fig, "sum_old")

    # 2 · do models prefer a position?
    st.divider()
    big, small = SM[[is_large(r) for r in SM.itertuples()]], SM[[is_small(r) for r in SM.itertuples()]]
    big_ok = sum(verdict(x)[1] != "Strong bias" for x in big.bias)
    a, b = st.columns([1, 1.35], gap="large", vertical_alignment="center")
    with a:
        question(2, "Do the models prefer a position?",
                 ("Most big models: no, or only a little. Small models: yes, strongly." if len(small) else
                  "Most models: no, or only a little."),
                 f"<b>Position bias</b> = how much better a model is when the right answer is in its favourite place than in its "
                 f"least favourite place. 0 means fair. {big_ok} of {len(big)} big models stay under 15 points. "
                 + (f"The {len(small)} small models reach <b>{small.bias.min():.0f}–{small.bias.max():.0f} points</b>: "
                    "they are right mostly when the answer happens to be in their favourite place." if len(small) else ""))
        st.markdown("<p class='legend-note'>" + " · ".join(
            f"<b>{lab}</b>: {'under 5' if hi == 5 else ('5 to 15' if hi == 15 else '15 or more')} points"
            for hi, lab, *_ in VERDICTS) + "</p>", unsafe_allow_html=True)
    with b:
        s = SM.sort_values("bias")
        fig = go.Figure()
        for hi, lab, _, _, colour in VERDICTS:
            part = s[[verdict(x)[1] == lab for x in s.bias]]
            if len(part):
                fig.add_bar(x=part.bias.tolist(), y=part.label.tolist(), orientation="h", name=lab, marker_color=colour,
                            text=[f"{x:.0f}" for x in part.bias], textposition="outside", cliponaxis=False,
                            textfont=dict(color=P["ink"], size=12),
                            hovertemplate="<b>%{x:.1f} points</b> position bias<br>%{y}<extra>" + lab + "</extra>")
        fig.update_layout(**layout(P, 90 + 34 * len(s), bargap=0.35, barmode="overlay",
                                   xaxis=dict(title="Position bias (points)", range=[0, max(20, s.bias.max() * 1.12)]),
                                   yaxis=dict(showgrid=False, categoryorder="array", categoryarray=s.label.tolist(),
                                              tickfont=dict(color=P["ink"], size=13))))
        show(fig, "sum_bias")

    # 3 · can we fix it?
    st.divider()
    prob = df[(df.fmt == "letter") & (df.read == "Probabilities")]
    fixed = []
    for base, g in prob.groupby("base"):
        fixed.append({"base": base, "once": g.acc.mean(), **{k: statistics.mean(m[k]["accuracy"] for m in g.methods)
                                                             for k, _ in SIMPLE_FIXES}})
    fixed = pd.DataFrame(fixed).sort_values("once")
    a, b = st.columns([1, 1.35], gap="large", vertical_alignment="center")
    with a:
        sm = fixed[fixed.base.isin(set(small.base))]
        question(3, "Can we fix it?", "Yes. Best fix: ask 4 times and combine the answers.",
                 "Ask the same question 4 times, with the answers moved each time, and add up how sure the model is about "
                 "each answer. The favourite place then helps every answer equally, so it stops mattering. "
                 + (" ".join(f"{r.base} goes from <b>{r.once:.0%} to {r.avg_logprob:.0%}</b>." for r in sm.itertuples()) + " "
                    if len(sm) else "")
                 + "Big models gain less, because they had little bias to fix. This fix needs the model's confidence, "
                   "which we can see only for models we run ourselves.")
    with b:
        fig = go.Figure()
        fig.add_bar(x=fixed.once.tolist(), y=fixed.base.tolist(), orientation="h", name="Asked once", marker_color=P["gray"],
                    text=[f"{x:.0%}" for x in fixed.once], textposition="outside", cliponaxis=False,
                    textfont=dict(color=P["ink2"], size=12), hovertemplate="<b>%{x:.0%}</b> right, asked once<br>%{y}<extra></extra>")
        fig.add_bar(x=fixed.avg_logprob.tolist(), y=fixed.base.tolist(), orientation="h", name="With the best fix",
                    marker_color=P["new"], text=[f"{x:.0%}" for x in fixed.avg_logprob], textposition="outside",
                    cliponaxis=False, textfont=dict(color=P["ink"], size=12),
                    hovertemplate="<b>%{x:.0%}</b> right with the best fix<br>%{y}<extra></extra>")
        fig.update_layout(**layout(P, 100 + 58 * len(fixed), barmode="group", bargap=0.3, bargroupgap=0.08,
                                   xaxis=dict(range=[0, 1.12], tickformat=".0%", title="Right answers", tickvals=[0, .25, .5, .75, 1]),
                                   yaxis=dict(showgrid=False, tickfont=dict(color=P["ink"], size=13), autorange="reversed")))
        show(fig, "sum_fix")

    st.markdown("**Which fix works best?** Average gain over the "
                f"{len(fixed)} models where every fix could be tried:")
    gains = []
    for k, name in SIMPLE_FIXES:
        gains.append((name, statistics.mean((m[k]["accuracy"] - m["single"]["accuracy"]) * 100 for m in prob.methods)))
    cols = st.columns(len(gains))
    for col, (name, g) in zip(cols, gains):
        col.metric(name, f"{g:+.1f} points", border=True,
                   help="Average gain in right answers, in percentage points, over the models where the fix could be tried.")
    st.caption("Numbers on this page use letters A–D and the average of both question sets (ARC-Challenge and OpenBookQA). "
               "Every number for each set separately is on **All details**.")


# ------------------------------------------------------------------ check one model (simple)
def page_check():
    header("Check a model", "How does one model behave?", "Pick a model to see if it prefers a position, and what changed since the old result.")
    names = sorted(SM.label.tolist(), key=lambda s: (company(s), s.lower()))
    if st.session_state.get("chk_model") not in names:
        st.session_state["chk_model"] = "gpt-oss:120b" if "gpt-oss:120b" in names else names[0]
    c1, c2 = st.columns([2, 1.4], vertical_alignment="bottom")
    model = c1.selectbox("Model", names, key="chk_model")
    ds = c2.segmented_control("Question set", list(DS_NAME), format_func=DS_NAME.get, key="chk_ds") or "allenai_ai2_arc"
    hit = df[(df.label == model) & (df.dataset == ds) & (df.fmt == "letter")]
    if hit.empty:
        st.info("No result for this model and question set.", icon=":material/info:")
        return
    r = hit.iloc[0]
    L, e = r.letters, r.earlier
    hi, lab, colour, icon, _ = verdict(r.gap * 100)
    with st.container(horizontal=True, vertical_alignment="center"):
        st.badge(lab, icon=icon, color=colour)
        st.badge(r.company, icon=":material/apartment:", color="gray")
        if r["size"] != "—":
            st.badge(r["size"], icon=":material/straighten:", color="gray")

    m1, m2, m3 = st.columns(3)
    m1.metric("Right answers", f"{r.acc:.0%}", border=True,
              delta=f"{(r.acc - e['acc']) * 100:+.0f} points since the old result" if e else None)
    m2.metric("Position bias", f"{r.gap * 100:.0f} point{'' if round(r.gap * 100) == 1 else 's'}", border=True, delta_color="inverse",
              help="How much better the model is when the right answer is in its favourite place than in its least favourite place. 0 means fair.")
    m3.metric("Favourite letter", f"{r.top} · {r.top_share:.0%} of answers", border=True,
              help="A fair model picks each letter about 25% of the time.")

    acc = dict(zip(L, r.acc_pos))
    if r.gap < 0.05:
        say = f"This model is fair: it picks each letter about equally often, and it is right about as often wherever the right answer is."
    else:
        rest = statistics.mean(acc[x] for x in L if x != r.top)
        say = (f"This model likes **{r.top}**: it answers {r.top} {r.top_share:.0%} of the time (fair is 25%). "
               f"When the right answer is {r.top}, it is right {acc[r.top]:.0%} of the time. Otherwise only {rest:.0%}.")
    if e and r.read == "Text":
        et = max("ABCD", key=lambda x: e["picks"][x])
        say += f" In the old result it seemed to answer **{et}** {e['picks'][et]:.0%} of the time, because the old code read the replies wrongly."
    with st.container(border=True):
        st.markdown(f":material/insights: {say}")

    section("Which letter does it choose?", "The line shows a fair share: 25% for each letter.")
    fig = go.Figure()
    if e and r.read == "Text":
        fig.add_bar(name="Old result", x=L, y=[e["picks"][x] for x in "ABCD"], marker_color=P["old"],
                    text=[f"{e['picks'][x]:.0%}" for x in "ABCD"], textposition="outside", textfont=dict(color=P["ink2"]),
                    hovertemplate="<b>%{y:.0%}</b> of answers<br>old result, letter %{x}<extra></extra>")
    fig.add_bar(name="Now", x=L, y=r.picks, marker_color=P["new"], text=[f"{x:.0%}" for x in r.picks],
                textposition="outside", textfont=dict(color=P["ink"]),
                hovertemplate="<b>%{y:.0%}</b> of answers<br>now, letter %{x}<extra></extra>")
    fig.add_hline(y=0.25, line=dict(color=P["ink2"], width=1.2), layer="below")
    fig.update_traces(cliponaxis=False)
    fig.update_layout(**layout(P, 360, barmode="group", showlegend=bool(e and r.read == "Text"),
                               yaxis=dict(range=[0, 1.08], tickformat=".0%", title="Share of answers", tickvals=[0, .25, .5, .75, 1]),
                               xaxis=dict(title="Letter", showgrid=False, tickfont=dict(size=15, color=P["ink"]))))
    show(fig, "chk_picks")

    with st.expander("See more: right answers for each position, and the wrong answers", icon=":material/expand_more:"):
        fig = go.Figure()
        fig.add_bar(name="Now", x=L, y=r.acc_pos, marker_color=P["new"], text=[f"{x:.0%}" for x in r.acc_pos],
                    textposition="outside", textfont=dict(color=P["ink"]),
                    hovertemplate="<b>%{y:.0%}</b> right<br>right answer at %{x}<extra></extra>")
        fig.update_layout(**layout(P, 320, showlegend=False, yaxis=dict(range=[0, 1.1], tickformat=".0%", title="Right answers"),
                                   xaxis=dict(title="Where the right answer was", showgrid=False)))
        show(fig, "chk_acc")
        fc = pd.DataFrame({"Letter": L, "Wrong answers that chose it": [r.err_pred.get(x, 0) * 100 for x in L],
                           "Wrong answers where the right answer was here": [r.err_gold.get(x, 0) * 100 for x in L]})
        st.dataframe(fc, hide_index=True, width="stretch", column_config={
            c: st.column_config.ProgressColumn(c, format="%.0f%%", min_value=0, max_value=100) for c in fc.columns[1:]})


# ------------------------------------------------------------------ all details (for experts)
def reset_filters():
    for k in ("group", "custom", "ds", "fmt"):
        st.session_state[k] = DEFAULTS[k]


def page_details():
    header("All details", "Every result, for each question set",
           "The full results behind the summary: every model, both question sets, letters and numbers, every fix and every test.")
    with st.container(border=True):
        f1, f2, f3, f4 = st.columns([2.6, 1.6, 1.6, 0.9], vertical_alignment="bottom")
        group = f1.pills("Models", GROUPS, key="group") or "All models"
        pick_ds = f2.segmented_control("Question set", list(DS_NAME), format_func=DS_SHORT.get, selection_mode="multi", key="ds") or list(DS_NAME)
        pick_fmt = f3.segmented_control("Labels", list(FMT_NAME), format_func=FMT_NAME.get, selection_mode="multi", key="fmt") or ["letter"]
        f4.button("Reset", icon=":material/restart_alt:", on_click=reset_filters, width="stretch")
        if group == "Choose…":
            st.multiselect("Choose models", all_labels, key="custom", placeholder="Pick one or more")
    keep = {"All models": lambda r: True, "Large (7B+)": is_large, "Small (≤2B)": is_small,
            "Read from probabilities": lambda r: r.read == "Probabilities",
            "Choose…": lambda r: r.label in (st.session_state.custom or all_labels)}[group]
    sel = df[pd.Series([keep(r) for r in df.itertuples()], index=df.index) & df.dataset.isin(pick_ds) & df.fmt.isin(pick_fmt)]
    if sel.empty:
        st.info("No results match these filters. Press **Reset**.", icon=":material/filter_alt_off:")
        return
    ctx = dict(sel=sel, both_ds=sel.dataset.nunique() > 1, both_fmt=sel.fmt.nunique() > 1, pick_ds=pick_ds)
    tabs = st.tabs(["Position heatmap", "Every fix", "A–D or 1–4", "Old vs new", "One model", "Tests", "Table"])
    with tabs[0]:
        details_heatmap(**ctx)
    with tabs[1]:
        details_fixes(**ctx)
    with tabs[2]:
        details_lvn(**ctx)
    with tabs[3]:
        details_old_new(**ctx)
    with tabs[4]:
        details_model(**ctx)
    with tabs[5]:
        details_stats(**ctx)
    with tabs[6]:
        details_table(**ctx)


def details_heatmap(sel, both_ds, both_fmt, pick_ds):
    left, right = st.columns([3, 1], vertical_alignment="bottom")
    with left:
        section("Right answers for each position of the right answer",
                "An even row means no position bias; one dark cell means the model is right mostly when the answer sits there.")
    right.segmented_control("Sort rows by", ["Accuracy", "Position gap", "Name"], key="sort")
    s = sel.copy()
    s["row"] = [r.label + (f" · {'letters' if r.fmt == 'letter' else 'numbers'}" if both_fmt else "") for r in s.itertuples()]
    by = st.session_state.sort or "Accuracy"
    order = (s.groupby("row").acc.mean().sort_values().index.tolist() if by == "Accuracy" else
             s.groupby("row").gap.max().sort_values().index.tolist() if by == "Position gap" else
             sorted(s.row.unique(), key=str.lower, reverse=True))
    dss = [d for d in DS_NAME if d in set(s.dataset)]
    fig = make_subplots(rows=1, cols=len(dss), shared_yaxes=True, horizontal_spacing=0.025, subplot_titles=[DS_NAME[d] for d in dss])
    xl = position_labels(set(s.fmt))
    for i, d in enumerate(dss):
        m = s[s.dataset == d].set_index("row")
        z = [[m.loc[r, "acc_pos"][j] if r in m.index else None for j in range(4)] for r in order]
        fig.add_trace(go.Heatmap(
            z=z, x=xl, y=order, zmin=0, zmax=1, colorscale=scale(P["seq"]), xgap=2, ygap=2, showscale=i == len(dss) - 1,
            texttemplate="%{z:.0%}", textfont=dict(size=12),
            hovertemplate="<b>%{z:.1%}</b> right<br>%{y}<br>right answer at %{x}<extra>" + DS_NAME[d] + "</extra>",
            colorbar=dict(title=dict(text="Right", side="top"), tickformat=".0%", thickness=10, lenmode="pixels",
                          len=max(90, min(220, 26 * len(order))), y=1, yanchor="top", outlinewidth=0,
                          tickfont=dict(color=P["ink2"]))), row=1, col=i + 1)
    fig.update_layout(**layout(P, 90 + 30 * len(order), margin=dict(t=40)))
    fig.update_xaxes(showgrid=False, side="top", tickfont=dict(color=P["ink2"]))
    fig.update_yaxes(showgrid=False, tickfont=dict(color=P["ink"]))
    fig.update_annotations(font=dict(color=P["ink"], size=14), yshift=18)
    show(fig, "det_heatmap")


def details_fixes(sel, both_ds, both_fmt, pick_ds):
    with st.expander("What each fix does", icon=":material/menu_book:"):
        st.markdown("""
| Fix | Calls per question | Needs probabilities | What it does |
|---|---|---|---|
| Prior correction | 1 | yes | Learns, without answer keys, how much the model favours each label and divides that out |
| PriDe | 1 | yes | The same, with the preference learned from 5% of the questions asked in all rotations |
| Majority vote | 4 | no | Asks all four rotations and keeps the option chosen most often |
| PAV | 4 | no | Majority vote that weighs each vote by the model's learned position preference |
| PAV-adaptive | up to 4 | no | PAV that stops asking once it is 95% sure |
| Averaging | 4 | yes | Adds up each option's log-probability over the four rotations, so a fixed pull toward one position cancels out |
""")
    section("Gain from each fix (points over asking once)", "Blue improved accuracy, red lowered it. ✓ = significant after Holm "
            "correction (tested where answers come from probabilities). Empty: the fix needs probabilities the model does not give.")
    s = sel.copy()
    s["row"] = [row_label(r, both_ds, both_fmt) for r in s.itertuples()]
    s = s.sort_values(["read", "acc"], ascending=[True, False])
    rows, z, txt, hov = [], [], [], []
    for r in s.itertuples():
        single = r.methods.get("single", {}).get("accuracy", r.acc)
        zr, tr, hr = [], [], []
        for key, name, calls in METHODS:
            m = r.methods.get(key)
            if not m:
                zr.append(None); tr.append(""); hr.append("not available for this run"); continue
            gv = (m["accuracy"] - single) * 100
            sig = (m.get("vs_single") or {}).get("p_holm", 1) < 0.05
            zr.append(gv); tr.append(f"{gv:+.1f}{' ✓' if sig else ''}")
            hr.append(f"{pct(m['accuracy'])} with {name.lower()} ({calls})<br>{pct(single)} asked once")
        rows.append(r.row); z.append(zr); txt.append(tr); hov.append(hr)
    zmax = max(5.0, max((abs(v) for zr in z for v in zr if v is not None), default=5))
    fig = go.Figure(go.Heatmap(
        z=z, x=[f"{n}<br><span style='font-size:11px'>{c}</span>" for _, n, c in METHODS], y=rows, text=txt,
        customdata=hov, texttemplate="%{text}", textfont=dict(size=12), zmid=0, zmin=-zmax, zmax=zmax, xgap=2, ygap=2,
        colorscale=[[0, P["neg"]], [0.5, P["mid"]], [1, P["pos"]]],
        hovertemplate="<b>%{z:+.1f} points</b><br>%{customdata}<br>%{y}<extra></extra>",
        colorbar=dict(title=dict(text="Points", side="top"), thickness=10, lenmode="pixels",
                      len=max(90, min(220, 26 * len(rows))), y=1, yanchor="top", outlinewidth=0, tickfont=dict(color=P["ink2"]))))
    fig.update_layout(**layout(P, 110 + 30 * len(rows), margin=dict(t=56), yaxis=dict(autorange="reversed")))
    fig.update_xaxes(side="top", showgrid=False, tickfont=dict(color=P["ink"]))
    fig.update_yaxes(showgrid=False, tickfont=dict(color=P["ink"]))
    show(fig, "det_fixes")
    tt = pd.DataFrame([{"Run": r.row, "Asked once": r.methods.get("single", {}).get("accuracy", r.acc) * 100,
                        **{f"{n} ({c})": (r.methods[k]["accuracy"] * 100 if k in r.methods else None) for k, n, c in METHODS}}
                       for r in s.itertuples()])
    st.dataframe(tt, hide_index=True, width="stretch",
                 column_config={c: st.column_config.NumberColumn(c, format="%.1f%%") for c in tt.columns[1:]})


def details_lvn(sel, both_ds, both_fmt, pick_ds):
    s = lvn[lvn.label.isin(set(sel.label)) & lvn.dataset.isin(pick_ds)].copy()
    if s.empty:
        st.info("No model in the selection was run with both label formats.", icon=":material/info:")
        return
    sig = s[s.p < 0.05]
    takeaway(f"Numbers did better in {int((s['diff'] > 0).sum())} of {len(s)} runs and letters in {int((s['diff'] < 0).sum())}. "
             f"{len(sig)} differences are significant" + (f", {int((sig['diff'] < 0).sum())} of them in favour of letters." if len(sig) else ".")
             + f" The largest gap is {s['diff'].abs().max() * 100:.1f} points.")
    section("Numbers minus letters (points)", "Bars to the right: numbers did better. Labelled bars are significant (exact McNemar test, p < 0.05).")
    s["row"] = [r.label + (f" · {DS_SHORT[r.dataset]}" if both_ds else "") for r in s.itertuples()]
    s = s.sort_values("diff")
    fig = go.Figure(go.Bar(
        x=(s["diff"] * 100).tolist(), y=s.row.tolist(), orientation="h",
        marker_color=[P["pos"] if d > 0 else P["neg"] for d in s["diff"]],
        text=[f"{d * 100:+.1f}" if p < 0.05 else "" for d, p in zip(s["diff"], s.p)], textposition="outside",
        textfont=dict(color=P["ink"], size=12), cliponaxis=False, customdata=[list(t) for t in zip(s.letters, s.numbers, s.p)],
        hovertemplate="<b>%{x:+.1f} points</b><br>letters %{customdata[0]:.1%} · numbers %{customdata[1]:.1%}"
                      "<br>McNemar p = %{customdata[2]:.3g}<br>%{y}<extra></extra>"))
    span = max(5, s["diff"].abs().max() * 100 + 3)
    fig.update_layout(**layout(P, 100 + 30 * len(s), bargap=0.45, showlegend=False,
                               xaxis=dict(range=[-span, span], title="Numbers minus letters (points)", zeroline=True,
                                          zerolinecolor=P["ink2"], zerolinewidth=1),
                               yaxis=dict(showgrid=False, tickfont=dict(color=P["ink"]))))
    show(fig, "det_lvn")
    st.dataframe(pd.DataFrame([{"Model": r.label, "Dataset": DS_NAME[r.dataset], "Letters": r.letters * 100, "Numbers": r.numbers * 100,
                                "Difference (pts)": r.diff * 100, "Only letters right": r.letters_only,
                                "Only numbers right": r.numbers_only, "McNemar p": r.p} for r in s.sort_values("diff").itertuples()]),
                 hide_index=True, width="stretch", column_config={
                     "Letters": st.column_config.NumberColumn(format="%.1f%%"), "Numbers": st.column_config.NumberColumn(format="%.1f%%"),
                     "Difference (pts)": st.column_config.NumberColumn(format="%+.1f"), "McNemar p": st.column_config.NumberColumn(format="%.3g")})


def details_old_new(sel, both_ds, both_fmt, pick_ds):
    s = sel[(sel.fmt == "letter") & sel.earlier.notna()].copy()
    if s.empty:
        st.info("No model in the selection has an earlier run. Include letters A–D.", icon=":material/info:")
        return
    st.caption("Earlier runs read answers with the old code (first A–D character, or \"A\" when unreadable); the open Hugging Face "
               "models used an older prompt. nemotron-3-nano is compared with the earlier nemotron3:33b (same family).")
    s["row"] = [r.label + (f" · {DS_SHORT[r.dataset]}" if both_ds else "") for r in s.itertuples()]
    s["old"] = [e["acc"] for e in s.earlier]
    s = s.sort_values("acc")
    fig = go.Figure()
    xs, ys = [], []
    for r in s.itertuples():
        xs += [r.old, r.acc, None]; ys += [r.row, r.row, None]
    fig.add_scatter(x=xs, y=ys, mode="lines", line=dict(color=P["grid"], width=3), hoverinfo="skip", showlegend=False)
    fig.add_scatter(x=s.old.tolist(), y=s.row.tolist(), mode="markers", name="Earlier run",
                    marker=dict(color=P["old"], size=12, line=dict(color=P["surface"], width=2)),
                    hovertemplate="<b>%{x:.1%}</b> earlier run<br>%{y}<extra></extra>")
    fig.add_scatter(x=s.acc.tolist(), y=s.row.tolist(), mode="markers", name="New run",
                    marker=dict(color=P["new"], size=12, line=dict(color=P["surface"], width=2)),
                    hovertemplate="<b>%{x:.1%}</b> new run<br>%{y}<extra></extra>")
    fig.update_layout(**layout(P, 110 + 34 * len(s), xaxis=dict(range=[0, 1.02], tickformat=".0%", title="Accuracy"),
                               yaxis=dict(showgrid=False, tickfont=dict(color=P["ink"]))))
    show(fig, "det_old_new")
    tt = pd.DataFrame([{"Model": r.label, "Dataset": DS_NAME[r.dataset], "Earlier run": r.earlier["source"],
                        "Earlier accuracy": r.old * 100, "New accuracy": r.acc * 100, "Change (pts)": (r.acc - r.old) * 100,
                        "Earlier gap (pts)": r.earlier["gap"] * 100, "New gap (pts)": r.gap * 100,
                        "Earlier most-picked": f"{max('ABCD', key=lambda x: r.earlier['picks'][x])} · {max(r.earlier['picks'].values()):.0%}"}
                       for r in s.sort_values("acc", ascending=False).itertuples()])
    st.dataframe(tt, hide_index=True, width="stretch", column_config={
        "Earlier accuracy": st.column_config.NumberColumn(format="%.1f%%"), "New accuracy": st.column_config.NumberColumn(format="%.1f%%"),
        "Change (pts)": st.column_config.NumberColumn(format="%+.1f"), "Earlier gap (pts)": st.column_config.NumberColumn(format="%.1f"),
        "New gap (pts)": st.column_config.NumberColumn(format="%.1f")})


def details_model(sel, both_ds, both_fmt, pick_ds):
    labels_here = [x for x in all_labels if x in set(sel.label)]
    if st.session_state.get("dd_model") not in labels_here:
        st.session_state["dd_model"] = "Qwen2.5-0.5B" if "Qwen2.5-0.5B" in labels_here else labels_here[0]
    c1, c2, c3 = st.columns([2.4, 1.6, 1.6], vertical_alignment="bottom")
    model = c1.selectbox("Model", labels_here, key="dd_model")
    ds = c2.segmented_control("Question set", list(DS_NAME), format_func=DS_NAME.get, key="dd_ds") or "allenai_ai2_arc"
    fmt = c3.segmented_control("Labels", list(FMT_NAME), format_func=FMT_NAME.get, key="dd_fmt") or "letter"
    hit = df[(df.label == model) & (df.dataset == ds) & (df.fmt == fmt)]
    if hit.empty:
        st.info("This model has no run for that question set and label format.", icon=":material/info:")
        return
    r = hit.iloc[0]
    L, e = r.letters, r.earlier
    g1, g2 = st.columns(2, gap="large")
    with g1:
        section("Where the answers go", "Line: fair share, 25% for each position.")
        fig = go.Figure()
        if e:
            fig.add_bar(name="Earlier run", x=L, y=[e["picks"][x] for x in "ABCD"], marker_color=P["old"],
                        hovertemplate="<b>%{y:.1%}</b> of answers<br>earlier run, position %{x}<extra></extra>")
        fig.add_bar(name="New run, asked once", x=L, y=r.picks, marker_color=P["new"],
                    hovertemplate="<b>%{y:.1%}</b> of answers<br>new run, position %{x}<extra></extra>")
        if r.fix_picks:
            fig.add_bar(name="New run + prior correction", x=L, y=[r.fix_picks[x] for x in "ABCD"], marker_color=P["fix"],
                        hovertemplate="<b>%{y:.1%}</b> of answers<br>after prior correction, position %{x}<extra></extra>")
        fig.add_hline(y=0.25, line=dict(color=P["ink2"], width=1.2), layer="below")
        fig.update_layout(**layout(P, 360, yaxis=dict(range=[0, 1], tickformat=".0%", title="Share of answers"),
                                   xaxis=dict(title="Position", showgrid=False), barmode="group"))
        show(fig, "det_picks")
    with g2:
        section("Accuracy in each position", "Thin lines: 95% confidence intervals (Wilson).")
        fig = go.Figure()
        if e:
            fig.add_bar(name="Earlier run", x=L, y=[e["acc_pos"][x] for x in "ABCD"], marker_color=P["old"],
                        error_y=dict(type="data", symmetric=False, color=P["ink2"], thickness=1.2, width=4,
                                     array=[e["ci"][x]["upper"] - e["acc_pos"][x] for x in "ABCD"],
                                     arrayminus=[e["acc_pos"][x] - e["ci"][x]["lower"] for x in "ABCD"]),
                        hovertemplate="<b>%{y:.1%}</b> correct<br>earlier run, answer at %{x}<extra></extra>")
        fig.add_bar(name="New run, asked once", x=L, y=r.acc_pos, marker_color=P["new"],
                    error_y=dict(type="data", symmetric=False, color=P["ink2"], thickness=1.2, width=4,
                                 array=[r.ci[x]["upper"] - a for x, a in zip(L, r.acc_pos)],
                                 arrayminus=[a - r.ci[x]["lower"] for x, a in zip(L, r.acc_pos)]),
                    hovertemplate="<b>%{y:.1%}</b> correct<br>new run, answer at %{x}<extra></extra>")
        fig.update_layout(**layout(P, 360, yaxis=dict(range=[0, 1.05], tickformat=".0%", title="Accuracy"),
                                   xaxis=dict(title="Position of the correct answer", showgrid=False), barmode="group",
                                   showlegend=bool(e)))
        show(fig, "det_acc")
    fc = pd.DataFrame({"Position": L, "Picked by the wrong answers": [r.err_pred.get(x, 0) * 100 for x in L],
                       "Where the correct answer was": [r.err_gold.get(x, 0) * 100 for x in L]})
    cfg = {c: st.column_config.ProgressColumn(c, format="%.1f%%", min_value=0, max_value=100) for c in fc.columns[1:]}
    if e:
        fc["Picked (earlier run)"] = [e["err_pred"].get(x, 0) * 100 for x in "ABCD"]
        fc["Correct answer was (earlier run)"] = [e["err_gold"].get(x, 0) * 100 for x in "ABCD"]
        cfg.update({c: st.column_config.ProgressColumn(c, format="%.1f%%", min_value=0, max_value=100) for c in fc.columns[3:]})
    st.dataframe(fc, hide_index=True, width="stretch", column_config=cfg)
    if e:
        st.caption(f"Earlier run: {e['source']}.")


def details_stats(sel, both_ds, both_fmt, pick_ds):
    s = sel.copy()
    s["row"] = [row_label(r, both_ds, both_fmt) for r in s.itertuples()]
    section("Is the choice of position different from chance?", "Chi-square test of the share of answers per position against 25% each.")
    st.dataframe(pd.DataFrame([{"Run": r.row, "χ²": r.chi["chi2"], "p-value (approx.)": r.chi["p_value"],
                                "Significant": "Yes" if r.chi["is_significant"] else "No"} for r in s.itertuples()]),
                 hide_index=True, width="stretch", column_config={"χ²": st.column_config.NumberColumn(format="%.1f")})
    section("First position against last position", "Two-proportion z-test: accuracy when the correct answer is first (A / 1) against last (D / 4).")
    st.dataframe(pd.DataFrame([{"Run": r.row, "Accuracy first": r.acc_pos[0] * 100, "Accuracy last": r.acc_pos[-1] * 100,
                                "z": r.z_first_last.get("z_score", 0.0), "p-value": r.z_first_last.get("p_value", 1.0),
                                "Significant": "Yes" if r.z_first_last.get("is_significant") else "No"} for r in s.itertuples()]),
                 hide_index=True, width="stretch", column_config={
                     "Accuracy first": st.column_config.NumberColumn(format="%.1f%%"), "Accuracy last": st.column_config.NumberColumn(format="%.1f%%"),
                     "z": st.column_config.NumberColumn(format="%.2f"), "p-value": st.column_config.NumberColumn(format="%.3g")})
    section("Accuracy by position with 95% confidence intervals", "Wilson score intervals, 200 answers per position.")
    st.dataframe(pd.DataFrame([{"Run": r.row, "Position": x, "Accuracy": r.ci[x]["mean"] * 100,
                                "Lower": r.ci[x]["lower"] * 100, "Upper": r.ci[x]["upper"] * 100}
                               for r in s.itertuples() for x in r.letters]),
                 hide_index=True, width="stretch", height=420,
                 column_config={c: st.column_config.NumberColumn(format="%.1f%%") for c in ("Accuracy", "Lower", "Upper")})


def details_table(sel, both_ds, both_fmt, pick_ds):
    tbl = []
    for (label, fmt), g in sel.groupby(["label", "fmt"]):
        r0 = g.iloc[0]
        row = {"Model": label, "Company": r0.company, "Size": r0["size"], "Read from": r0.read, "Ran on": r0["where"]}
        if both_fmt:
            row["Labels"] = FMT_NAME[fmt]
        for d in DS_NAME:
            x = g[g.dataset == d]
            row[f"Accuracy {DS_SHORT[d]}"] = None if x.empty else x.acc.iloc[0] * 100
            row[f"By position {DS_SHORT[d]}"] = None if x.empty else x.acc_pos.iloc[0]
            row[f"Gap {DS_SHORT[d]}"] = None if x.empty else x.gap.iloc[0] * 100
        tbl.append(row)
    t = pd.DataFrame(tbl)
    t = t.sort_values(next(c for c in t.columns if c.startswith("Accuracy")), ascending=False)
    t = t[[c for c in t.columns if not (c.endswith(("ARC", "OBQA")) and t[c].isna().all())]]
    st.dataframe(t, hide_index=True, width="stretch", column_config={
        **{f"Accuracy {v}": st.column_config.ProgressColumn(f"Accuracy {v}", format="%.1f%%", min_value=0, max_value=100) for v in DS_SHORT.values()},
        **{f"By position {v}": st.column_config.BarChartColumn(f"By position {v}", y_min=0, y_max=1) for v in DS_SHORT.values()},
        **{f"Gap {v}": st.column_config.NumberColumn(f"Gap {v} (pts)", format="%.1f") for v in DS_SHORT.values()},
    })


# ------------------------------------------------------------------ how it was done
def page_how():
    header("How it was done", "How the results were made", "The method in five steps, and the words used on this site.")
    a, b = st.columns([1.1, 1], gap="large")
    with a:
        st.markdown("""
1. **Questions.** 200 science questions from ARC-Challenge and 200 from OpenBookQA. Each has four answers.
2. **Ask four times.** Every question is asked four times. Each time the answers move one place, so the right
   answer is once at A, once at B, once at C and once at D.
3. **Read the answer carefully.** A reply counts only if it clearly names one answer. If not, the question is asked
   once more. An unreadable reply is never counted as "A" (the mistake in the old code).
4. **Compare the positions.** If a model is right much more often when the answer is at one letter, it has position bias.
5. **Try fixes.** The same answers are used to test each fix, so the comparison is fair.
""")
        st.markdown("**Statistics used:** paired tests over questions with Holm correction, exact McNemar tests (A–D vs 1–4), "
                    "and Wilson intervals. **References:** Zheng et al., ICLR 2024 (PriDe); Zhou et al., ICLR 2024 (Batch "
                    "Calibration); Pezeshkpour & Hruschka, NAACL Findings 2024; Wei et al., 2024.")
    with b:
        st.markdown("""
**Words used here**

| Word | Meaning |
|---|---|
| Position bias | How much better a model is when the right answer is in its favourite place than in its least favourite place. 0 = fair. |
| Points | The difference between two percentages: 40% → 46% is +6 points. |
| Asked once | One question, one answer, no fix. |
| Confidence (probabilities) | How sure the model is about each answer. We can see it only for models we run ourselves. |
| Big / small model | Big: 7 billion parameters or more. Small: 2 billion or fewer. |
""")
        st.markdown("**Models**")
        mt = (df.drop_duplicates("label")[["label", "company", "size", "read", "where"]]
              .rename(columns={"label": "Model", "company": "Company", "size": "Size", "read": "Read from", "where": "Ran on"})
              .sort_values(["Company", "Model"]))
        st.dataframe(mt, hide_index=True, width="stretch")


pg = st.navigation([
    st.Page(page_summary, title="Summary", icon=":material/home:", default=True),
    st.Page(page_check, title="Check a model", icon=":material/search:", url_path="check"),
    st.Page(page_details, title="All details", icon=":material/table_chart:", url_path="details"),
    st.Page(page_how, title="How it was done", icon=":material/info:", url_path="how"),
], position="top")
pg.run()
