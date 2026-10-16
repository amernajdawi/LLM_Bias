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
GROUPS = ["All models", "Large (7B+)", "Small (≤2B)", "Read from probabilities", "Choose…"]
FONT = "Public Sans, -apple-system, BlinkMacSystemFont, Segoe UI, Roboto, sans-serif"
# Validated palette (blue / orange / aqua pass the colour-blindness and contrast checks in both modes).
PALETTE = {
    "light": dict(ink="#14161b", ink2="#52545a", grid="#e6e5e0", surface="#fbfbfa",
                  new="#2a78d6", old="#eb6834", fix="#1baf7a", neg="#e34948", mid="#f0efec", pos="#256abf",
                  seq=["#cde2fb", "#9ec5f4", "#6da7ec", "#3987e5", "#256abf", "#184f95", "#0d366b"]),
    "dark": dict(ink="#ecedef", ink2="#b9bcc4", grid="#2c2e33", surface="#161718",
                 new="#3987e5", old="#d95926", fix="#199e70", neg="#e66767", mid="#383835", pos="#5598e7",
                 seq=["#0d366b", "#104281", "#184f95", "#256abf", "#3987e5", "#6da7ec", "#b7d3f6"]),
}
PLOT_CONFIG = {"displaylogo": False, "toImageButtonOptions": {"format": "png", "scale": 2},
               "modeBarButtonsToRemove": ["zoom2d", "pan2d", "select2d", "lasso2d", "zoomIn2d", "zoomOut2d",
                                          "autoScale2d", "resetScale2d"]}
DEFAULTS = {"group": "All models", "custom": [], "ds": list(DS_NAME), "fmt": ["letter"], "sort": "Accuracy",
            "dd_ds": "allenai_ai2_arc", "dd_fmt": "letter"}


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
            "key": key, "model": r["model"], "label": label, "company": company(base), "size": size, "size_num": size_num,
            "read": "Probabilities" if probs else "Text",
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


# ------------------------------------------------------------------ chart and text helpers
def palette():
    t = getattr(getattr(st, "context", None), "theme", None)
    return PALETTE["dark" if getattr(t, "type", None) == "dark" else "light"]


def layout(P, height, **extra):
    axis = dict(gridcolor=P["grid"], linecolor=P["grid"], zeroline=False, automargin=True,
                tickfont=dict(color=P["ink2"], size=12), title_font=dict(color=P["ink2"], size=12))
    base = dict(height=height, margin=dict(l=4, r=4, t=30, b=4), font=dict(family=FONT, size=13, color=P["ink2"]),
                paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)", xaxis=axis, yaxis=dict(axis),
                legend=dict(orientation="h", yanchor="bottom", y=1.0, xanchor="left", x=0, title=None,
                            font=dict(color=P["ink2"], size=12), bgcolor="rgba(0,0,0,0)"),
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


def insight(col, eyebrow, number, text):
    with col.container(border=True):
        st.markdown(f"<div class='insight'><div class='eyebrow'>{html.escape(eyebrow)}</div>"
                    f"<div class='num'>{html.escape(number)}</div><div class='txt'>{html.escape(text)}</div></div>",
                    unsafe_allow_html=True)


def selection_note():
    fmts = " and ".join(FMT_NAME[f].lower() for f in FMT_NAME if f in pick_fmt)
    dss = " and ".join(DS_NAME[d] for d in DS_NAME if d in pick_ds)
    n = SEL.label.nunique()
    return f"{n} model version{'s' if n != 1 else ''} · {dss} · {fmts}"


# ------------------------------------------------------------------ app shell
st.set_page_config(page_title="Position Bias Explorer", page_icon=":material/bar_chart:", layout="wide",
                   initial_sidebar_state="expanded")
st.markdown("""<style>
.block-container {max-width: 1240px; padding-top: 2.4rem; padding-bottom: 4rem;}
.eyebrow {text-transform: uppercase; letter-spacing: .08em; font-size: .72rem; font-weight: 600; opacity: .62;}
h1 {letter-spacing: -.01em;}
.lede {font-size: 1.06rem; line-height: 1.6; opacity: .82; max-width: 60rem; margin-top: -.4rem;}
.sub {opacity: .74; font-size: .95rem; line-height: 1.5; max-width: 64rem; margin-top: -.6rem;}
.insight {min-height: 8.2rem; display: flex; flex-direction: column; gap: .35rem;}
.insight .num {font-size: 2rem; font-weight: 650; letter-spacing: -.02em; line-height: 1.1;}
.insight .txt {font-size: .9rem; line-height: 1.45; opacity: .8;}
[data-testid="stMetricValue"] {font-weight: 600;}
</style>""", unsafe_allow_html=True)

df, lvn = load()
P = palette()
for k, v in DEFAULTS.items():
    st.session_state.setdefault(k, v)
all_labels = sorted(df.label.unique(), key=lambda s: (company(s), s.lower()))


def reset_filters():
    for k in ("group", "custom", "ds", "fmt"):
        st.session_state[k] = DEFAULTS[k]


with st.sidebar:
    st.markdown("### Filters")
    st.caption("These apply to every page.")
    group = st.pills("Models", GROUPS, key="group") or "All models"
    if group == "Choose…":
        st.multiselect("Choose models", all_labels, key="custom", placeholder="Pick one or more")
    pick_ds = st.segmented_control("Dataset", list(DS_NAME), format_func=DS_NAME.get, selection_mode="multi", key="ds") or list(DS_NAME)
    pick_fmt = st.segmented_control("Answer labels", list(FMT_NAME), format_func=FMT_NAME.get, selection_mode="multi",
                                    key="fmt") or ["letter"]
    st.button("Reset filters", icon=":material/restart_alt:", on_click=reset_filters, width="stretch")
    st.divider()
    st.caption("Corrected runs of 2026. Code and data: [github.com/amernajdawi/LLM_Bias](https://github.com/amernajdawi/LLM_Bias)")

keep = {
    "All models": lambda r: True,
    "Large (7B+)": is_large,
    "Small (≤2B)": is_small,
    "Read from probabilities": lambda r: r.read == "Probabilities",
    "Choose…": lambda r: r.label in (st.session_state.custom or all_labels),
}[group]
SEL = df[pd.Series([keep(r) for r in df.itertuples()], index=df.index) & df.dataset.isin(pick_ds) & df.fmt.isin(pick_fmt)]
both_ds, both_fmt = SEL.dataset.nunique() > 1, SEL.fmt.nunique() > 1


def guard():
    if SEL.empty:
        st.info("No results match these filters. Use **Reset filters** in the side panel.", icon=":material/filter_alt_off:")
        st.stop()


# ------------------------------------------------------------------ pages
def page_overview():
    header("Multiple-choice QA · ARC-Challenge and OpenBookQA", "Position Bias Explorer",
           "Do language models prefer an answer because of <em>where</em> it is placed? Every question was asked four "
           "times with the options rotated, so the correct answer sits once in every position. Answers were read "
           "strictly, and every fix for the bias was tested on the same answers.")
    with st.container(horizontal=True):
        st.badge(f"{df.label.nunique()} model versions", icon=":material/neurology:", color="gray")
        st.badge(f"{df.company.nunique()} companies", icon=":material/apartment:", color="gray")
        st.badge("2 datasets · 200 questions each", icon=":material/quiz:", color="gray")
        st.badge(f"{int(df.n.sum()):,} answers", icon=":material/fact_check:", color="gray")
        with st.popover("How to read this", icon=":material/help:"):
            st.markdown("""
- **Position gap:** accuracy when the correct answer is in the model's best position minus its worst. 0 means no bias.
- **Rotation:** the options move one place each time, so every option visits every position once.
- **Asked once:** one answer per prompt, with no fix.
- **Read from probabilities:** the answer is the label the model gives the highest probability (models we run ourselves). Other models are read from their text reply.
- **Points:** differences between two percentages, e.g. 40% → 46% is +6 points.
""")

    letters = df[df.fmt == "letter"]
    c = st.columns(4)
    olds = letters[(letters.read == "Text") & letters.earlier.notna()]
    if len(olds):
        j = max(olds.itertuples(), key=lambda r: r.acc - r.earlier["acc"])
        insight(c[0], "Measurement fixed", f"{j.earlier['acc']:.0%} → {j.acc:.0%}",
                f"{j.label} on {DS_NAME[j.dataset]}: the old answer reader made it look biased; read correctly it is accurate.")
    large = letters[[is_large(r) and r.read == "Text" for r in letters.itertuples()]]
    if len(large):
        insight(c[1], "Large models (7B and up)", f"≤ {large.gap.max() * 100:.1f} pts",
                f"Largest gap between best and worst answer position. Accuracy {large.acc.min():.0%}–{large.acc.max():.0%}.")
    small = letters[[is_small(r) for r in letters.itertuples()]]
    if len(small):
        w = small.loc[small.gap.idxmax()]
        insight(c[2], "Small models (2B and below)", f"{w.gap * 100:.1f} pts",
                f"{w.label} on {DS_NAME[w.dataset]} picks {w.top} for {w.top_share:.0%} of its answers.")
    fixes = [(r, r.methods["avg_logprob"]) for r in letters.itertuples() if "avg_logprob" in r.methods]
    if fixes:
        r, m = max(fixes, key=lambda t: t[1]["accuracy"] - t[0].methods["single"]["accuracy"])
        insight(c[3], "Strongest fix", f"+{(m['accuracy'] - r.methods['single']['accuracy']) * 100:.1f} pts",
                f"Averaging over the four rotations: {r.label} on {DS_NAME[r.dataset]}, "
                f"{r.methods['single']['accuracy']:.0%} → {m['accuracy']:.0%}.")

    guard()
    st.divider()
    section("Current selection", selection_note())
    k1, k2, k3, k4 = st.columns(4)
    k1.metric("Models", SEL.label.nunique(), border=True)
    k2.metric("Answers", f"{int(SEL.n.sum()):,}", border=True)
    k3.metric("Median accuracy, asked once", pct(SEL.acc.median()), border=True)
    w = SEL.loc[SEL.gap.idxmax()]
    k4.metric("Largest position gap", f"{w.gap * 100:.1f} pts", border=True,
              help=f"{w.label} on {DS_NAME[w.dataset]} ({FMT_NAME[w.fmt].lower()}).")

    st.write("")
    left, right = st.columns([3, 1], vertical_alignment="bottom")
    with left:
        section("Accuracy for each position of the correct answer",
                "Each row is one model. An even row means no position bias; one dark cell means the model is right "
                "mostly when the answer happens to sit in that position.")
    right.segmented_control("Sort rows by", ["Accuracy", "Position gap", "Name"], key="sort")
    s = SEL.copy()
    s["row"] = [r.label + (f" · {'letters' if r.fmt == 'letter' else 'numbers'}" if both_fmt else "") for r in s.itertuples()]
    by = st.session_state.sort or "Accuracy"
    order = (s.groupby("row").acc.mean().sort_values().index.tolist() if by == "Accuracy" else
             s.groupby("row").gap.max().sort_values().index.tolist() if by == "Position gap" else
             sorted(s.row.unique(), key=str.lower, reverse=True))
    dss = [d for d in DS_NAME if d in set(s.dataset)]
    fig = make_subplots(rows=1, cols=len(dss), shared_yaxes=True, horizontal_spacing=0.025,
                        subplot_titles=[DS_NAME[d] for d in dss])
    xl = position_labels(set(s.fmt))
    for i, d in enumerate(dss):
        m = s[s.dataset == d].set_index("row")
        z = [[m.loc[r, "acc_pos"][j] if r in m.index else None for j in range(4)] for r in order]
        fig.add_trace(go.Heatmap(
            z=z, x=xl, y=order, zmin=0, zmax=1, colorscale=scale(P["seq"]), xgap=2, ygap=2, showscale=i == len(dss) - 1,
            texttemplate="%{z:.0%}", textfont=dict(size=12),
            hovertemplate="<b>%{z:.1%}</b> correct<br>%{y}<br>correct answer at %{x}<extra>" + DS_NAME[d] + "</extra>",
            colorbar=dict(title=dict(text="Accuracy", side="top"), tickformat=".0%", thickness=10, lenmode="pixels",
                          len=max(90, min(220, 26 * len(order))), y=1, yanchor="top", outlinewidth=0,
                          tickfont=dict(color=P["ink2"]))), row=1, col=i + 1)
    fig.update_layout(**layout(P, 90 + 30 * len(order), margin=dict(t=40)))
    fig.update_xaxes(showgrid=False, side="top", tickfont=dict(color=P["ink2"]))
    fig.update_yaxes(showgrid=False, tickfont=dict(color=P["ink"]))
    fig.update_annotations(font=dict(color=P["ink"], size=14), yshift=18)
    show(fig, "overview_heatmap")

    section("All models", "Click a column header to sort. The small bars show accuracy for positions 1–4 of the correct answer.")
    tbl = []
    for (label, fmt), g in SEL.groupby(["label", "fmt"]):
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
    t = pd.DataFrame(tbl).sort_values(f"Accuracy {DS_SHORT[dss[0]]}", ascending=False)
    t = t[[c for c in t.columns if not (c.endswith(("ARC", "OBQA")) and t[c].isna().all())]]
    st.dataframe(t, hide_index=True, width="stretch", column_config={
        **{f"Accuracy {v}": st.column_config.ProgressColumn(f"Accuracy {v}", format="%.1f%%", min_value=0, max_value=100) for v in DS_SHORT.values()},
        **{f"By position {v}": st.column_config.BarChartColumn(f"By position {v}", y_min=0, y_max=1) for v in DS_SHORT.values()},
        **{f"Gap {v}": st.column_config.NumberColumn(f"Gap {v} (pts)", format="%.1f", help="Best minus worst position accuracy") for v in DS_SHORT.values()},
    })


def step_model(options, d):
    cur = st.session_state.get("dd_model")
    i = options.index(cur) if cur in options else 0
    st.session_state["dd_model"] = options[(i + d) % len(options)]


def page_model():
    guard()
    header("Model explorer", "One model in detail",
           "Where a model puts its answers, how often it is right in each position, and how that compares with the earlier run.")
    labels_here = [x for x in all_labels if x in set(SEL.label)]
    if st.session_state.get("dd_model") not in labels_here:
        st.session_state["dd_model"] = "gpt-oss:120b" if "gpt-oss:120b" in labels_here else labels_here[0]
    c1, c2, c3, c4, c5 = st.columns([2.6, 0.9, 0.9, 1.7, 1.7], vertical_alignment="bottom")
    model = c1.selectbox("Model", labels_here, key="dd_model")
    c2.button("Previous", icon=":material/chevron_left:", on_click=step_model, args=(labels_here, -1), width="stretch")
    c3.button("Next", icon=":material/chevron_right:", icon_position="right", on_click=step_model, args=(labels_here, 1), width="stretch")
    ds = c4.segmented_control("Dataset", list(DS_NAME), format_func=DS_NAME.get, key="dd_ds") or "allenai_ai2_arc"
    fmt = c5.segmented_control("Answer labels", list(FMT_NAME), format_func=FMT_NAME.get, key="dd_fmt") or "letter"
    hit = df[(df.label == model) & (df.dataset == ds) & (df.fmt == fmt)]
    if hit.empty:
        st.info("This model has no run for that dataset and label format.", icon=":material/info:")
        return
    r = hit.iloc[0]
    L, e = r.letters, r.earlier

    with st.container(horizontal=True):
        st.badge(r.company, icon=":material/apartment:", color="gray")
        st.badge(f"Size {r['size']}" if r["size"] != "—" else "Size not stated", icon=":material/straighten:", color="gray")
        st.badge(r["where"], icon=":material/dns:", color="gray")
        st.badge(f"Read from {r.read.lower()}", icon=":material/data_object:", color="blue")
        if e:
            st.badge("Earlier run available", icon=":material/history:", color="orange")

    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Accuracy, asked once", pct(r.acc), border=True,
              delta=f"{(r.acc - e['acc']) * 100:+.1f} pts vs earlier run" if e else None)
    m2.metric("Position gap", f"{r.gap * 100:.1f} pts", border=True, delta_color="inverse",
              delta=f"{(r.gap - e['gap']) * 100:+.1f} pts vs earlier run" if e else None,
              help="Best minus worst accuracy over the four positions of the correct answer.")
    m3.metric("Most-picked position", f"{r.top} · {r.top_share:.0%}", border=True,
              help="Share of all answers that chose this position. A fair model picks each position about 25% of the time.")
    m4.metric("Wrong answers", f"{r.n_err} of {r.n}", border=True)

    acc = dict(zip(L, r.acc_pos))
    if r.gap < 0.05:
        lo, hi = f"{min(r.picks):.0%}", f"{max(r.picks):.0%}"
        say = (f"Almost no position bias: accuracy changes by only {r.gap * 100:.1f} points across the four positions, "
               f"and each position gets " + (f"about {lo} of the answers." if lo == hi else f"{lo}–{hi} of the answers."))
    else:
        rest = statistics.mean(acc[x] for x in L if x != r.top)
        say = (f"This model favours **{r.top}**: it gives that answer {r.top_share:.0%} of the time, where a fair share is 25%. "
               f"When the correct answer is at {r.top} it is right {acc[r.top]:.0%} of the time; in the other positions {rest:.0%}.")
    if e:
        et = max("ABCD", key=lambda x: e["picks"][x])
        say += (f" In the earlier run it chose **{et}** for {e['picks'][et]:.0%} of its answers and scored {e['acc']:.0%}"
                + (", because its replies were read with the old, faulty answer reader." if r.read == "Text"
                   else "; that run used an older prompt."))
    if r.fix_picks:
        say += f" After prior correction, no position gets more than {max(r.fix_picks.values()):.0%} of the answers."
    with st.container(border=True):
        st.markdown(f":material/insights: **What this shows** · {say}")

    g1, g2 = st.columns(2, gap="large")
    with g1:
        section("Where the model's answers go", "The line marks the fair share: 25% of answers in each position.")
        fig = go.Figure()
        if e:
            fig.add_bar(name="Earlier run", x=L, y=[e["picks"][x] for x in "ABCD"], marker_color=P["old"],
                        hovertemplate="<b>%{y:.1%}</b> of answers<br>earlier run, position %{x}<extra></extra>")
        fig.add_bar(name="New run, asked once", x=L, y=r.picks, marker_color=P["new"],
                    hovertemplate="<b>%{y:.1%}</b> of answers<br>new run, position %{x}<extra></extra>")
        if r.fix_picks:
            fig.add_bar(name="New run + prior correction", x=L, y=[r.fix_picks[x] for x in "ABCD"], marker_color=P["fix"],
                        hovertemplate="<b>%{y:.1%}</b> of answers<br>after prior correction, position %{x}<extra></extra>")
        fig.add_hline(y=0.25, line=dict(color=P["ink2"], width=1.2))
        fig.update_layout(**layout(P, 360, yaxis=dict(range=[0, 1], tickformat=".0%", title="Share of answers"),
                                   xaxis=dict(title="Position", showgrid=False), barmode="group"))
        show(fig, "dd_picks")
    with g2:
        section("Accuracy in each position", "Thin lines are 95% confidence intervals.")
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
        show(fig, "dd_acc")

    section("Failing cases", "Wrong answers only: which position they picked, and where the correct answer really was.")
    fc = pd.DataFrame({"Position": L,
                       "Picked by the wrong answers": [r.err_pred.get(x, 0) * 100 for x in L],
                       "Where the correct answer was": [r.err_gold.get(x, 0) * 100 for x in L]})
    cfg = {c: st.column_config.ProgressColumn(c, format="%.1f%%", min_value=0, max_value=100) for c in fc.columns[1:]}
    if e:
        fc["Picked (earlier run)"] = [e["err_pred"].get(x, 0) * 100 for x in "ABCD"]
        fc["Correct answer was (earlier run)"] = [e["err_gold"].get(x, 0) * 100 for x in "ABCD"]
        cfg.update({c: st.column_config.ProgressColumn(c, format="%.1f%%", min_value=0, max_value=100) for c in fc.columns[3:]})
    st.dataframe(fc, hide_index=True, width="stretch", column_config=cfg)
    if e:
        st.caption(f"Earlier run: {e['source']}.")


def page_fixes():
    guard()
    header("Fixing the bias", "Which fix helps, and by how much?",
           "Every fix was applied to the same answers. Gains are accuracy points over asking each question once with no fix.")
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
    s = SEL.copy()
    s["row"] = [row_label(r, both_ds, both_fmt) for r in s.itertuples()]
    gain = lambda r, k: (r.methods[k]["accuracy"] - r.methods["single"]["accuracy"]) * 100 if k in r.methods else None
    prob = [r for r in s.itertuples() if r.read == "Probabilities"]
    text = [r for r in s.itertuples() if r.read == "Text"]
    parts = []
    if prob:
        g = [gain(r, "avg_logprob") for r in prob]
        sig = sum((r.methods["avg_logprob"].get("vs_single") or {}).get("p_holm", 1) < 0.05 for r in prob)
        parts.append(f"Averaging raised accuracy in {sum(x > 0 for x in g)} of {len(g)} runs read from probabilities "
                     f"({min(g):+.1f} to {max(g):+.1f} points, significant in {sig}).")
        sm = [gain(r, "prior_heldout") for r in prob if is_small(r)]
        lg = [gain(r, "prior_heldout") for r in prob if is_large(r)]
        if sm and lg:
            parts.append(f"The one-call prior correction helped the small models ({min(sm):+.1f} to {max(sm):+.1f}) "
                         f"but barely changed the larger ones ({min(lg):+.1f} to {max(lg):+.1f}).")
    if text:
        g = [gain(r, "majority") for r in text]
        parts.append(f"For models read from text, majority voting changed accuracy by {min(g):+.1f} to {max(g):+.1f} points.")
    if parts:
        takeaway(" ".join(parts))

    section("Gain from each fix", "Blue cells improved accuracy, red cells lowered it. ✓ marks a significant gain after Holm "
            "correction (tested where answers come from probabilities). Empty cells: the fix needs probabilities the model does not give.")
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
                      len=max(90, min(220, 26 * len(rows))), y=1, yanchor="top", outlinewidth=0,
                      tickfont=dict(color=P["ink2"]))))
    fig.update_layout(**layout(P, 110 + 30 * len(rows), margin=dict(t=56), yaxis=dict(autorange="reversed")))
    fig.update_xaxes(side="top", showgrid=False, tickfont=dict(color=P["ink"]))
    fig.update_yaxes(showgrid=False, tickfont=dict(color=P["ink"]))
    show(fig, "fix_heatmap")

    if prob:
        section("Average gain per fix", f"Runs read from probabilities only, where every fix could be tested (n = {len(prob)}).")
        avg = sorted(((n, statistics.mean(gain(r, k) for r in prob)) for k, n, _ in METHODS), key=lambda t: t[1])
        fig = go.Figure(go.Bar(x=[a for _, a in avg], y=[n for n, _ in avg], orientation="h", marker_color=P["new"],
                               text=[f"{a:+.1f}" for _, a in avg], textposition="outside", cliponaxis=False,
                               textfont=dict(color=P["ink"], size=12),
                               hovertemplate="<b>%{x:+.1f} points</b> on average<br>%{y}<extra></extra>"))
        fig.update_layout(**layout(P, 60 + 42 * len(avg), bargap=0.5, showlegend=False, margin=dict(r=40),
                                   xaxis=dict(title="Average gain (points)", zeroline=True, zerolinecolor=P["ink2"]),
                                   yaxis=dict(showgrid=False, tickfont=dict(color=P["ink"]))))
        show(fig, "fix_avg")

    with st.expander("Show as a table (accuracy of every fix)", icon=":material/table:"):
        tt = pd.DataFrame([{"Run": r.row, "Asked once": r.methods.get("single", {}).get("accuracy", r.acc) * 100,
                            **{f"{n} ({c})": (r.methods[k]["accuracy"] * 100 if k in r.methods else None) for k, n, c in METHODS}}
                           for r in s.itertuples()])
        st.dataframe(tt, hide_index=True, width="stretch",
                     column_config={c: st.column_config.NumberColumn(c, format="%.1f%%") for c in tt.columns[1:]})


def page_old_new():
    guard()
    header("Old vs new", "The same model, earlier run and new run",
           "Earlier runs read answers with the old code (first A–D character, or \"A\" when unreadable); the open Hugging Face "
           "models used an older prompt. Letters A–D only, since the earlier runs used letters.")
    s = SEL[(SEL.fmt == "letter") & SEL.earlier.notna()].copy()
    if s.empty:
        st.info("No model in the selection has an earlier run. Include letters A–D in the filters.", icon=":material/info:")
        return
    s["row"] = [r.label + (f" · {DS_SHORT[r.dataset]}" if both_ds else "") for r in s.itertuples()]
    s["old"] = [e["acc"] for e in s.earlier]
    lopsided = sum(max(e["picks"].values()) > 0.8 for e in s.earlier)
    takeaway(f"Median accuracy rose from {s.old.median():.0%} in the earlier runs to {s.acc.median():.0%} in the new runs. "
             f"In {lopsided} of {len(s)} earlier runs, a single position received over 80% of all answers.")
    section("Accuracy before and after", "nemotron-3-nano is compared with the earlier nemotron3:33b (same family).")
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
    show(fig, "old_new")
    tt = pd.DataFrame([{"Model": r.label, "Dataset": DS_NAME[r.dataset], "Earlier run": r.earlier["source"],
                        "Earlier accuracy": r.old * 100, "New accuracy": r.acc * 100, "Change (pts)": (r.acc - r.old) * 100,
                        "Earlier gap (pts)": r.earlier["gap"] * 100, "New gap (pts)": r.gap * 100,
                        "Earlier most-picked": f"{max('ABCD', key=lambda x: r.earlier['picks'][x])} · "
                                               f"{max(r.earlier['picks'].values()):.0%}"}
                       for r in s.sort_values("acc", ascending=False).itertuples()])
    st.dataframe(tt, hide_index=True, width="stretch", column_config={
        "Earlier accuracy": st.column_config.ProgressColumn(format="%.1f%%", min_value=0, max_value=100),
        "New accuracy": st.column_config.ProgressColumn(format="%.1f%%", min_value=0, max_value=100),
        "Change (pts)": st.column_config.NumberColumn(format="%+.1f"),
        "Earlier gap (pts)": st.column_config.NumberColumn(format="%.1f"),
        "New gap (pts)": st.column_config.NumberColumn(format="%.1f")})


def page_lvn():
    guard()
    header("Letters vs numbers", "Numbers 1–4 instead of letters A–D",
           "The same questions and rotations, labelled (1)–(4) instead of A–D. Each answer is compared with its twin.")
    s = lvn[lvn.label.isin(set(SEL.label)) & lvn.dataset.isin(pick_ds)].copy()
    if s.empty:
        st.info("No model in the selection was run with both label formats.", icon=":material/info:")
        return
    sig = s[s.p < 0.05]
    takeaway(f"Numbers did better in {int((s['diff'] > 0).sum())} of {len(s)} runs and letters in {int((s['diff'] < 0).sum())}. "
             f"{len(sig)} differences are significant"
             + (f", {int((sig['diff'] < 0).sum())} of them in favour of letters." if len(sig) else ".")
             + f" The largest gap is {s['diff'].abs().max() * 100:.1f} points.")
    section("Numbers minus letters", "Bars to the right: numbers did better. Labelled bars are significant (exact McNemar test, p < 0.05).")
    s["row"] = [r.label + (f" · {DS_SHORT[r.dataset]}" if both_ds else "") for r in s.itertuples()]
    s = s.sort_values("diff")
    fig = go.Figure(go.Bar(
        x=(s["diff"] * 100).tolist(), y=s.row.tolist(), orientation="h",
        marker_color=[P["pos"] if d > 0 else P["neg"] for d in s["diff"]],
        text=[f"{d * 100:+.1f}" if p < 0.05 else "" for d, p in zip(s["diff"], s.p)], textposition="outside",
        textfont=dict(color=P["ink"], size=12), cliponaxis=False,
        customdata=[list(t) for t in zip(s.letters, s.numbers, s.p)],
        hovertemplate="<b>%{x:+.1f} points</b><br>letters %{customdata[0]:.1%} · numbers %{customdata[1]:.1%}"
                      "<br>McNemar p = %{customdata[2]:.3g}<br>%{y}<extra></extra>"))
    span = max(5, s["diff"].abs().max() * 100 + 3)
    fig.update_layout(**layout(P, 100 + 30 * len(s), bargap=0.45, showlegend=False,
                               xaxis=dict(range=[-span, span], title="Numbers minus letters (points)", zeroline=True,
                                          zerolinecolor=P["ink2"], zerolinewidth=1),
                               yaxis=dict(showgrid=False, tickfont=dict(color=P["ink"]))))
    show(fig, "lvn")
    st.dataframe(pd.DataFrame([{"Model": r.label, "Dataset": DS_NAME[r.dataset], "Letters": r.letters * 100,
                                "Numbers": r.numbers * 100, "Difference (pts)": r.diff * 100,
                                "Only letters right": r.letters_only, "Only numbers right": r.numbers_only,
                                "McNemar p": r.p} for r in s.sort_values("diff").itertuples()]),
                 hide_index=True, width="stretch", column_config={
                     "Letters": st.column_config.NumberColumn(format="%.1f%%"),
                     "Numbers": st.column_config.NumberColumn(format="%.1f%%"),
                     "Difference (pts)": st.column_config.NumberColumn(format="%+.1f"),
                     "McNemar p": st.column_config.NumberColumn(format="%.3g")})


def page_stats():
    guard()
    header("Statistics", "Significance tests", "The tests behind the charts, for every run in the current selection.")
    s = SEL.copy()
    s["row"] = [row_label(r, both_ds, both_fmt) for r in s.itertuples()]
    section("Is the choice of position different from chance?",
            "Chi-square test of the share of answers per position against 25% each (asked once).")
    st.dataframe(pd.DataFrame([{"Run": r.row, "χ²": r.chi["chi2"], "p-value (approx.)": r.chi["p_value"],
                                "Significant": "Yes" if r.chi["is_significant"] else "No"} for r in s.itertuples()]),
                 hide_index=True, width="stretch", column_config={"χ²": st.column_config.NumberColumn(format="%.1f")})
    section("First position against last position",
            "Two-proportion z-test: accuracy when the correct answer is first (A / 1) against last (D / 4).")
    st.dataframe(pd.DataFrame([{"Run": r.row, "Accuracy first": r.acc_pos[0] * 100, "Accuracy last": r.acc_pos[-1] * 100,
                                "z": r.z_first_last.get("z_score", 0.0), "p-value": r.z_first_last.get("p_value", 1.0),
                                "Significant": "Yes" if r.z_first_last.get("is_significant") else "No"} for r in s.itertuples()]),
                 hide_index=True, width="stretch", column_config={
                     "Accuracy first": st.column_config.NumberColumn(format="%.1f%%"),
                     "Accuracy last": st.column_config.NumberColumn(format="%.1f%%"),
                     "z": st.column_config.NumberColumn(format="%.2f"), "p-value": st.column_config.NumberColumn(format="%.3g")})
    section("Accuracy by position with 95% confidence intervals", "Wilson score intervals, 200 answers per position.")
    st.dataframe(pd.DataFrame([{"Run": r.row, "Position": x, "Accuracy": r.ci[x]["mean"] * 100,
                                "Lower": r.ci[x]["lower"] * 100, "Upper": r.ci[x]["upper"] * 100}
                               for r in s.itertuples() for x in r.letters]),
                 hide_index=True, width="stretch", height=420, column_config={
                     c: st.column_config.NumberColumn(format="%.1f%%") for c in ("Accuracy", "Lower", "Upper")})


def page_about():
    header("About", "How the results were made")
    a, b = st.columns([1.15, 1], gap="large")
    with a:
        st.markdown("""
- **Questions:** 200 from ARC-Challenge and 200 from OpenBookQA, four options each.
- **Rotations:** every question is asked four times with the options rotated, so each option, and the correct
  answer, appears once in every position.
- **Prompt:** the question, the options, then *"Reply with only the letter of the correct option (A, B, C, or D)."*
  The number version uses (1)–(4).
- **Reading the answer:** a text reply counts only as a clear label, "Answer: X", "option X" or the full text of one
  option. Anything else is invalid and asked once more. An unreadable reply is never counted as "A".
- **Probabilities:** for models we run ourselves, the answer is the label with the highest probability, and those
  probabilities are kept so every fix can be tested.
- **Statistics:** paired tests over questions (bootstrap intervals, sign-flip tests with Holm correction), exact
  McNemar tests for letters vs numbers, and Wilson intervals for accuracy.
""")
        st.markdown("**References:** Zheng et al., ICLR 2024 (PriDe); Zhou et al., ICLR 2024 (Batch Calibration); "
                    "Pezeshkpour & Hruschka, NAACL Findings 2024; Wei et al., 2024.")
    with b:
        st.markdown("**Models**")
        mt = (df.drop_duplicates("label")[["label", "company", "size", "read", "where"]]
              .rename(columns={"label": "Model", "company": "Company", "size": "Size", "read": "Read from", "where": "Ran on"})
              .sort_values(["Company", "Model"]))
        st.dataframe(mt, hide_index=True, width="stretch", height=600)


pg = st.navigation([
    st.Page(page_overview, title="Overview", icon=":material/space_dashboard:", default=True),
    st.Page(page_model, title="Model explorer", icon=":material/search:", url_path="model"),
    st.Page(page_fixes, title="Fixing the bias", icon=":material/build:", url_path="fixes"),
    st.Page(page_old_new, title="Old vs new", icon=":material/compare_arrows:", url_path="old-vs-new"),
    st.Page(page_lvn, title="Letters vs numbers", icon=":material/text_fields:", url_path="letters-vs-numbers"),
    st.Page(page_stats, title="Statistics", icon=":material/functions:", url_path="statistics"),
    st.Page(page_about, title="About", icon=":material/info:", url_path="about"),
], position="top")
pg.run()
