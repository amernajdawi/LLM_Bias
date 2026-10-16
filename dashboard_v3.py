"""
Position Bias Explorer: Streamlit dashboard for the corrected (v3) results.
Reads dashboard_data/v3_dashboard.json (no database needed).

    .venv/bin/python3 -m streamlit run dashboard_v3.py
Refresh the data: bash experiments/v3_corrected/update_results.sh && .venv/bin/python3 experiments/v3_corrected/export_dashboard.py
"""
import json
import math
import re
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
FONT = "Source Sans 3, Source Sans Pro, -apple-system, BlinkMacSystemFont, Segoe UI, Roboto, sans-serif"
# Validated palette (blue / orange / aqua pass the colour-blindness and contrast checks in both modes).
PALETTE = {
    "light": dict(ink="#0b0b0b", ink2="#52514e", muted="#7c7b76", grid="#e8e7e3", surface="#fcfcfb",
                  new="#2a78d6", old="#eb6834", fix="#1baf7a", neg="#e34948", mid="#f0efec", pos="#256abf",
                  seq=["#cde2fb", "#9ec5f4", "#6da7ec", "#3987e5", "#256abf", "#184f95", "#0d366b"]),
    "dark": dict(ink="#f0efec", ink2="#c3c2b7", muted="#9a998f", grid="#2f2f2d", surface="#1a1a19",
                 new="#3987e5", old="#d95926", fix="#199e70", neg="#e66767", mid="#383835", pos="#5598e7",
                 seq=["#0d366b", "#104281", "#184f95", "#256abf", "#3987e5", "#6da7ec", "#b7d3f6"]),
}
PLOT_CONFIG = {"displaylogo": False, "toImageButtonOptions": {"format": "png", "scale": 2},
               "modeBarButtonsToRemove": ["zoom2d", "pan2d", "select2d", "lasso2d", "zoomIn2d", "zoomOut2d",
                                          "autoScale2d", "resetScale2d"]}


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


# ------------------------------------------------------------------ chart helpers
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


# ------------------------------------------------------------------ page
st.set_page_config(page_title="Position Bias Explorer", page_icon=":material/bar_chart:", layout="wide")
st.markdown("""<style>
.block-container {padding-top: 2.2rem; padding-bottom: 3rem; max-width: 1280px;}
h1 {font-weight: 700; letter-spacing: -0.01em;}
[data-testid="stMetricValue"] {font-weight: 600;}
.lede {font-size: 1.05rem; max-width: 62rem; opacity: 0.85; margin-top: -0.6rem;}
</style>""", unsafe_allow_html=True)

df, lvn = load()
P = palette()

st.title("Position Bias Explorer")
st.markdown('<p class="lede">How much do language models prefer an answer because of <em>where</em> it is placed? '
            "Every question from ARC-Challenge and OpenBookQA (200 each) was asked four times with the options rotated, "
            "so the correct answer sits once in every position. Answers were read strictly, and every fix for the bias "
            "was tested on the same answers.</p>", unsafe_allow_html=True)

# filters: one row, scoping everything below
f1, f2, f3 = st.columns([3, 1.6, 1.6], vertical_alignment="bottom")
all_labels = sorted(df.label.unique(), key=lambda s: (company(s), s.lower()))
pick_models = f1.multiselect("Models", all_labels, placeholder="All models", key="models")
pick_ds = f2.segmented_control("Dataset", list(DS_NAME), format_func=DS_NAME.get, selection_mode="multi",
                               default=list(DS_NAME), key="ds") or list(DS_NAME)
pick_fmt = f3.segmented_control("Answer labels", list(FMT_NAME), format_func=FMT_NAME.get, selection_mode="multi",
                                default=["letter"], key="fmt") or ["letter"]
sel = df[(df.label.isin(pick_models or all_labels)) & df.dataset.isin(pick_ds) & df.fmt.isin(pick_fmt)]
if sel.empty:
    st.info("No results match these filters.")
    st.stop()
both_ds, both_fmt = sel.dataset.nunique() > 1, sel.fmt.nunique() > 1

k1, k2, k3, k4 = st.columns(4)
k1.metric("Models", sel.label.nunique(), border=True, help="Model versions in the current selection.")
k2.metric("Answers analysed", f"{int(sel.n.sum()):,}", border=True)
k3.metric("Median accuracy, asked once", pct(sel.acc.median()), border=True)
worst = sel.loc[sel.gap.idxmax()]
k4.metric("Largest position gap", f"{worst.gap * 100:.1f} pts", border=True,
          help=f"Best minus worst accuracy over the four answer positions. Largest for {worst.label} on "
               f"{DS_NAME[worst.dataset]} ({FMT_NAME[worst.fmt].lower()}).")

tabs = st.tabs(["Overview", "Position bias", "Fixing the bias", "Old vs new", "Letters vs numbers", "Statistics", "About"])

# ------------------------------------------------------------------ overview
with tabs[0]:
    letters = df[df.fmt == "letter"]
    text_runs = letters[letters.read == "Text"]
    large = text_runs[text_runs.size_num.isna() | (text_runs.size_num >= 7)]
    small = letters[letters.size_num.notna() & (letters.size_num <= 2)]
    gains = [m["avg_logprob"]["vs_single"]["diff"] for m in letters.methods if "avg_logprob" in m]
    with_old = letters[letters.earlier.notna() & (letters.read == "Text")]
    with st.container(border=True):
        st.markdown("**Key findings** · all models, letters A–D")
        lines = []
        if len(with_old):
            olds = [e["acc"] for e in with_old.earlier]
            lines.append(f"The earlier runs were misread: the same models scored {pct(min(olds), 0)}–{pct(max(olds), 0)} "
                         f"before and score {pct(with_old.acc.min(), 0)}–{pct(with_old.acc.max(), 0)} with answers read correctly.")
        if len(large):
            lines.append(f"Large models (7B and up) answer {pct(large.acc.min(), 0)}–{pct(large.acc.max(), 0)} correctly, "
                         f"and their best and worst positions differ by at most {large.gap.max() * 100:.1f} points.")
        if len(small):
            lines.append(f"Small models (2B and below) are strongly biased: asked once they reach {pct(small.acc.min(), 0)}–"
                         f"{pct(small.acc.max(), 0)}, with gaps of up to {small.gap.max() * 100:.1f} points.")
        if gains:
            lines.append(f"Averaging each option's probability over the four rotations is the strongest fix: "
                         f"{min(gains):+.1f} to {max(gains):+.1f} points on every model read from probabilities.")
        st.markdown("\n".join(f"- {x}" for x in lines))

    st.subheader("Accuracy for each position of the correct answer")
    st.caption("Each row is one model. An even row means no position bias; one dark cell means the model is right "
               "mostly when the answer happens to sit in that position. Rows are sorted by average accuracy.")
    s = sel.copy()
    s["row"] = [r.label + (f" · {'letters' if r.fmt == 'letter' else 'numbers'}" if both_fmt else "") for r in s.itertuples()]
    order = s.groupby("row").acc.mean().sort_values().index.tolist()
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
            colorbar=dict(title=dict(text="Accuracy", side="top"), tickformat=".0%", thickness=10, len=0.5, y=1,
                          yanchor="top", outlinewidth=0, tickfont=dict(color=P["ink2"]))), row=1, col=i + 1)
    fig.update_layout(**layout(P, 90 + 30 * len(order), margin=dict(t=40), xaxis=dict(gridcolor="rgba(0,0,0,0)"),
                               yaxis=dict(gridcolor="rgba(0,0,0,0)")))
    fig.update_xaxes(showgrid=False, side="top", tickfont=dict(color=P["ink2"]))
    fig.update_yaxes(showgrid=False, tickfont=dict(color=P["ink"]))
    fig.update_annotations(font=dict(color=P["ink"], size=14), yshift=18)
    show(fig, "overview_heatmap")

    st.subheader("All models")
    tbl = []
    for (label, fmt), g in sel.groupby(["label", "fmt"]):
        r0 = g.iloc[0]
        row = {"Model": label, "Company": r0.company, "Size": r0["size"], "Answers read from": r0.read, "Where it ran": r0["where"]}
        if both_fmt:
            row["Labels"] = FMT_NAME[fmt]
        for d in DS_NAME:
            x = g[g.dataset == d]
            row[f"Accuracy {DS_SHORT[d]}"] = None if x.empty else x.acc.iloc[0] * 100
            row[f"Gap {DS_SHORT[d]}"] = None if x.empty else x.gap.iloc[0] * 100
        tbl.append(row)
    t = pd.DataFrame(tbl).sort_values(f"Accuracy {DS_SHORT[dss[0]]}", ascending=False)
    t = t[[c for c in t.columns if not (c.endswith(("ARC", "OBQA")) and t[c].isna().all())]]
    st.dataframe(t, hide_index=True, width="stretch", column_config={
        **{f"Accuracy {v}": st.column_config.ProgressColumn(f"Accuracy {v}", format="%.1f%%", min_value=0, max_value=100) for v in DS_SHORT.values()},
        **{f"Gap {v}": st.column_config.NumberColumn(f"Gap {v} (pts)", format="%.1f", help="Best minus worst position accuracy") for v in DS_SHORT.values()},
    })

# ------------------------------------------------------------------ position bias (one run)
with tabs[1]:
    st.subheader("One model in detail")
    c1, c2, c3 = st.columns([2.4, 1.6, 1.6], vertical_alignment="bottom")
    labels_here = [x for x in all_labels if x in set(sel.label)]
    default = labels_here.index("gpt-oss:120b") if "gpt-oss:120b" in labels_here else 0
    model = c1.selectbox("Model", labels_here, index=default, key="dd_model")
    ds = c2.segmented_control("Dataset", list(DS_NAME), format_func=DS_NAME.get, default="allenai_ai2_arc", key="dd_ds") or "allenai_ai2_arc"
    fmt = c3.segmented_control("Answer labels", list(FMT_NAME), format_func=FMT_NAME.get, default="letter", key="dd_fmt") or "letter"
    hit = df[(df.label == model) & (df.dataset == ds) & (df.fmt == fmt)]
    if hit.empty:
        st.info("This model has no run for that dataset and label format.")
    else:
        r = hit.iloc[0]
        L, e = r.letters, r.earlier
        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Accuracy, asked once", pct(r.acc), border=True,
                  delta=f"{(r.acc - e['acc']) * 100:+.1f} pts vs earlier run" if e else None)
        m2.metric("Position gap", f"{r.gap * 100:.1f} pts", border=True, delta_color="inverse",
                  delta=f"{(r.gap - e['gap']) * 100:+.1f} pts vs earlier run" if e else None,
                  help="Best minus worst accuracy over the four positions of the correct answer.")
        m3.metric("Most-picked position", f"{r.top} · {r.top_share:.0%}", border=True,
                  help="Share of all answers that chose this position. A fair model picks each position about 25% of the time.")
        m4.metric("Wrong answers", f"{r.n_err} of {r.n}", border=True)

        g1, g2 = st.columns(2)
        with g1:
            st.markdown("**Where the model's answers go**  \n<span style='opacity:.7;font-size:.9rem'>The line marks "
                        "the fair share: 25% of answers in each position.</span>", unsafe_allow_html=True)
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
            st.markdown("**Accuracy when the correct answer is in each position**  \n<span style='opacity:.7;"
                        "font-size:.9rem'>Thin lines are 95% confidence intervals.</span>", unsafe_allow_html=True)
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
        if e:
            st.caption(f"Earlier run: {e['source']}, read with the old code.")

        st.markdown("**Failing cases (wrong answers only)**")
        fc = pd.DataFrame({"Position": L,
                           "Picked by the wrong answers": [r.err_pred.get(x, 0) * 100 for x in L],
                           "Where the correct answer was": [r.err_gold.get(x, 0) * 100 for x in L]})
        cfg = {c: st.column_config.ProgressColumn(c, format="%.1f%%", min_value=0, max_value=100) for c in fc.columns[1:]}
        if e:
            fc["Picked (earlier run)"] = [e["err_pred"].get(x, 0) * 100 for x in "ABCD"]
            fc["Correct answer was (earlier run)"] = [e["err_gold"].get(x, 0) * 100 for x in "ABCD"]
            cfg.update({c: st.column_config.ProgressColumn(c, format="%.1f%%", min_value=0, max_value=100) for c in fc.columns[3:]})
        st.dataframe(fc, hide_index=True, width="stretch", column_config=cfg)

# ------------------------------------------------------------------ fixing the bias
with tabs[2]:
    st.subheader("Gain from each fix, in accuracy points over asking once")
    st.caption("Blue cells improved accuracy, red cells lowered it. ✓ marks a significant gain after Holm correction "
               "(tested only where the answers come from probabilities). Prior correction, PriDe and averaging need the "
               "model's probability for each label, so they are empty for models that only return text.")
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
            gain = (m["accuracy"] - single) * 100
            sig = (m.get("vs_single") or {}).get("p_holm", 1) < 0.05
            zr.append(gain); tr.append(f"{gain:+.1f}{' ✓' if sig else ''}")
            hr.append(f"{pct(m['accuracy'])} with {name.lower()} ({calls})<br>{pct(single)} asked once")
        rows.append(r.row); z.append(zr); txt.append(tr); hov.append(hr)
    zmax = max(5.0, max((abs(v) for zr in z for v in zr if v is not None), default=5))
    fig = go.Figure(go.Heatmap(
        z=z, x=[f"{n}<br><span style='font-size:11px'>{c}</span>" for _, n, c in METHODS], y=rows, text=txt,
        customdata=hov, texttemplate="%{text}", textfont=dict(size=12), zmid=0, zmin=-zmax, zmax=zmax, xgap=2, ygap=2,
        colorscale=[[0, P["neg"]], [0.5, P["mid"]], [1, P["pos"]]],
        hovertemplate="<b>%{z:+.1f} points</b><br>%{customdata}<br>%{y}<extra></extra>",
        colorbar=dict(title=dict(text="Points", side="top"), thickness=10, len=0.5, y=1, yanchor="top", outlinewidth=0,
                      tickfont=dict(color=P["ink2"]))))
    fig.update_layout(**layout(P, 110 + 30 * len(rows), margin=dict(t=56), yaxis=dict(autorange="reversed")))
    fig.update_xaxes(side="top", showgrid=False, tickfont=dict(color=P["ink"]))
    fig.update_yaxes(showgrid=False, tickfont=dict(color=P["ink"]))
    show(fig, "fix_heatmap")
    with st.expander("Show as a table (accuracy of every method)"):
        tt = [{"Run": r.row, "Asked once": r.methods.get("single", {}).get("accuracy", r.acc) * 100,
               **{f"{n} ({c})": (r.methods[k]["accuracy"] * 100 if k in r.methods else None) for k, n, c in METHODS}}
              for r in s.itertuples()]
        st.dataframe(pd.DataFrame(tt), hide_index=True, width="stretch",
                     column_config={c: st.column_config.NumberColumn(c, format="%.1f%%") for c in pd.DataFrame(tt).columns[1:]})

# ------------------------------------------------------------------ old vs new
with tabs[3]:
    st.subheader("The same model, earlier run and new run")
    st.caption("Earlier runs read answers with the old code (first A–D character, or \"A\" when unreadable); the open "
               "Hugging Face models used an older prompt. nemotron-3-nano is compared with the earlier nemotron3:33b "
               "(same family). Letters A–D only, since earlier runs used letters.")
    s = sel[(sel.fmt == "letter") & sel.earlier.notna()].copy()
    if s.empty:
        st.info("No model in the selection has an earlier run. Include letters A–D in the filters.")
    else:
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
        show(fig, "old_new")
        tt = pd.DataFrame([{"Model": r.label, "Dataset": DS_NAME[r.dataset], "Earlier run": r.earlier["source"],
                            "Earlier accuracy": r.old * 100, "New accuracy": r.acc * 100, "Change (pts)": (r.acc - r.old) * 100,
                            "Earlier gap (pts)": r.earlier["gap"] * 100, "New gap (pts)": r.gap * 100,
                            "Earlier most-picked": f"{max('ABCD', key=lambda x: r.earlier['picks'][x])} · "
                                                   f"{max(r.earlier['picks'].values()):.0%}"}
                           for r in s.sort_values("acc", ascending=False).itertuples()])
        st.dataframe(tt, hide_index=True, width="stretch", column_config={
            "Earlier accuracy": st.column_config.NumberColumn(format="%.1f%%"),
            "New accuracy": st.column_config.NumberColumn(format="%.1f%%"),
            **{c: st.column_config.NumberColumn(format="%+.1f" if c.startswith("Change") else "%.1f")
               for c in ("Change (pts)", "Earlier gap (pts)", "New gap (pts)")}})

# ------------------------------------------------------------------ letters vs numbers
with tabs[4]:
    st.subheader("Numbers 1–4 instead of letters A–D")
    st.caption("Accuracy with numbers minus accuracy with letters, on the same questions and rotations. Bars to the right "
               "mean numbers did better. Labelled bars are significant (exact McNemar test, p < 0.05).")
    s = lvn[lvn.label.isin(set(sel.label)) & lvn.dataset.isin(pick_ds)].copy()
    if s.empty:
        st.info("No model in the selection was run with both label formats.")
    else:
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

# ------------------------------------------------------------------ statistics
with tabs[5]:
    s = sel.copy()
    s["row"] = [row_label(r, both_ds, both_fmt) for r in s.itertuples()]
    st.subheader("Is the choice of position different from chance?")
    st.caption("Chi-square test of the share of answers per position against 25% each (asked once).")
    st.dataframe(pd.DataFrame([{"Run": r.row, "χ²": r.chi["chi2"], "p-value (approx.)": r.chi["p_value"],
                                "Significant": "Yes" if r.chi["is_significant"] else "No"} for r in s.itertuples()]),
                 hide_index=True, width="stretch", column_config={"χ²": st.column_config.NumberColumn(format="%.1f")})
    st.subheader("First position against last position")
    st.caption("Two-proportion z-test: accuracy when the correct answer is first (A / 1) against last (D / 4).")
    st.dataframe(pd.DataFrame([{"Run": r.row, "Accuracy first": r.acc_pos[0] * 100, "Accuracy last": r.acc_pos[-1] * 100,
                                "z": r.z_first_last.get("z_score", 0.0), "p-value": r.z_first_last.get("p_value", 1.0),
                                "Significant": "Yes" if r.z_first_last.get("is_significant") else "No"} for r in s.itertuples()]),
                 hide_index=True, width="stretch", column_config={
                     "Accuracy first": st.column_config.NumberColumn(format="%.1f%%"),
                     "Accuracy last": st.column_config.NumberColumn(format="%.1f%%"),
                     "z": st.column_config.NumberColumn(format="%.2f"), "p-value": st.column_config.NumberColumn(format="%.3g")})
    st.subheader("Accuracy by position with 95% confidence intervals")
    st.caption("Wilson score intervals, 200 answers per position.")
    st.dataframe(pd.DataFrame([{"Run": r.row, "Position": x, "Accuracy": r.ci[x]["mean"] * 100,
                                "Lower": r.ci[x]["lower"] * 100, "Upper": r.ci[x]["upper"] * 100}
                               for r in s.itertuples() for x in r.letters]),
                 hide_index=True, width="stretch", height=420, column_config={
                     c: st.column_config.NumberColumn(format="%.1f%%") for c in ("Accuracy", "Lower", "Upper")})

# ------------------------------------------------------------------ about
with tabs[6]:
    a, b = st.columns([1.15, 1])
    with a:
        st.subheader("How the results were made")
        st.markdown("""
- **Questions:** 200 from ARC-Challenge and 200 from OpenBookQA, four options each.
- **Rotations:** every question is asked four times with the options rotated, so each option, and the correct
  answer, appears once in every position.
- **Prompt:** the question, the options, then *"Reply with only the letter of the correct option (A, B, C, or D)."*
  The same prompt with numbers (1)–(4) is the number version.
- **Reading the answer:** text replies are accepted only as a clear label, "Answer: X", "option X" or the full text
  of one option. Anything else is invalid and asked once more. An unreadable reply is never counted as "A".
- **Probabilities:** for models we run ourselves, the answer is the label with the highest probability, and those
  probabilities are kept so every fix can be tested.
""")
        st.subheader("The fixes")
        st.markdown("""
| Fix | Calls | What it does |
|---|---|---|
| Prior correction | 1 | Learns, without answer keys, how much the model favours each label and divides that out |
| PriDe | 1 | The same idea, with the preference learned from 5% of the questions asked in all rotations (Zheng et al., ICLR 2024) |
| Majority vote | 4 | Asks all four rotations and keeps the option chosen most often |
| PAV | 4 | Majority vote that weighs each vote by the model's learned position preference |
| PAV-adaptive | up to 4 | PAV that stops asking once it is 95% sure |
| Averaging | 4 | Adds up each option's log-probability over the four rotations; a fixed pull toward one position cancels out |
""")
    with b:
        st.subheader("Models")
        mt = (df.drop_duplicates("label")[["label", "company", "size", "read", "where"]]
              .rename(columns={"label": "Model", "company": "Company", "size": "Size", "read": "Answers read from",
                               "where": "Where it ran"}).sort_values(["Company", "Model"]))
        st.dataframe(mt, hide_index=True, width="stretch", height=600)
    st.caption("Code and data: github.com/amernajdawi/LLM_Bias · experiments/v3_corrected")
