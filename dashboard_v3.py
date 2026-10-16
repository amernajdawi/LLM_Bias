"""
Streamlit dashboard for the corrected (v3) position-bias results. Reads
dashboard_data/v3_dashboard.json (no database needed).

    .venv/bin/python3 -m streamlit run dashboard_v3.py
Refresh the data: bash experiments/v3_corrected/update_results.sh && .venv/bin/python3 experiments/v3_corrected/export_dashboard.py
"""
import json
import sys
from pathlib import Path

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
from src.analysis.metrics import (  # noqa: E402  same metrics as the old dashboard
    accuracy_by_position,
    accuracy_confidence_interval,
    accuracy_position_significance_test,
    anchored_bias_frequency,
    chi_square_option_proportion,
    cohens_d_sensitivity_gap,
    error_correct_position_proportion,
    error_prediction_proportion,
    ground_truth_position_proportion,
    option_proportion,
    overall_accuracy,
    prob_correct_by_position,
    sensitivity_gap,
)

DS_NAME = {"allenai_ai2_arc": "ARC-Challenge", "allenai_openbookqa": "OpenBookQA"}
SOURCE = {"cloud": "Ollama Cloud (text)", "local": "local Ollama (text)", "openai": "OpenAI API (text)",
          "probabilities": "label probabilities"}
METHODS = [("single", "Ask once", "1"), ("prior_heldout", "Prior correction", "1"), ("pride", "PriDe", "1"),
           ("majority", "Majority vote", "4"), ("pav", "PAV", "4"), ("pav_adaptive", "PAV-adaptive", "≤4"),
           ("avg_logprob", "Log-prob. average", "4")]


@st.cache_data
def load():
    return json.loads((ROOT / "dashboard_data" / "v3_dashboard.json").read_text())


def summary(rows, letters):
    acc = accuracy_by_position(rows, letters)
    out = {
        "n_samples": len(rows),
        "n_errors": sum(1 for r in rows if r.get("correct") != 1),
        "overall_accuracy": overall_accuracy(rows),
        "accuracy_by_position": acc,
        "sensitivity_gap": sensitivity_gap(acc),
        "sensitivity_gap_cohens_d": cohens_d_sensitivity_gap(acc),
        "anchored_bias_frequency": anchored_bias_frequency(rows, letters),
        "option_proportion_pred": option_proportion(rows, letters),
        "ground_truth_position_proportion": ground_truth_position_proportion(rows, letters),
        "error_prediction_proportion": error_prediction_proportion(rows, letters),
        "error_correct_position_proportion": error_correct_position_proportion(rows, letters),
        "chi_square_option_proportion": chi_square_option_proportion(rows, letters),
        "accuracy_confidence_intervals": accuracy_confidence_interval(rows, letters),
        "accuracy_significance_first_vs_last": accuracy_position_significance_test(rows, letters, letters[0], letters[-1]),
    }
    if any("prob_correct" in r for r in rows):
        out["prob_correct_by_position"] = prob_correct_by_position(rows, letters)
    return out


def label(run):
    fmt = "letters A–D" if run["format"] == "letter" else "numbers 1–4"
    return f"{run['model']} · {DS_NAME[run['dataset']]} · {fmt}"


LAYOUT = dict(margin=dict(l=50, r=30, t=40, b=50), yaxis=dict(range=[0, 1.05]), height=330,
              legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1))

# ------------------------------------------------------------------ page
st.set_page_config(page_title="Position Bias — Corrected Results", page_icon="📊", layout="wide",
                   initial_sidebar_state="expanded")
st.markdown("""<style>
.main-title { font-size: 2rem; font-weight: 700; margin-bottom: 0.25rem; }
.sub-title { font-size: 0.9rem; color: #6c757d; margin-bottom: 1.5rem; }
.block-container { padding-top: 1.5rem; padding-bottom: 1.5rem; }
</style>""", unsafe_allow_html=True)
st.markdown('<div class="main-title">📊 Position Bias in Multiple-Choice QA — corrected results</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-title">Every question asked in all four rotations, answers read strictly (no default "A"), '
            'letters and numbers, and every bias-reduction method compared on the same answers.</div>', unsafe_allow_html=True)

data = load()
runs, earlier = data["runs"], data["earlier"]
S = {k: summary(r["rows"], r["letters"]) for k, r in runs.items()}

# sidebar filters
st.sidebar.header("Filters")
models = sorted({r["model"] for r in runs.values()})
pick_models = st.sidebar.multiselect("Models", models, default=models, key="models")
pick_ds = st.sidebar.multiselect("Datasets", list(DS_NAME), default=list(DS_NAME), format_func=DS_NAME.get, key="ds")
pick_fmt = st.sidebar.multiselect("Labels", ["letter", "number"], default=["letter"],
                                  format_func={"letter": "Letters A–D", "number": "Numbers 1–4"}.get, key="fmt")
selected = [k for k, r in sorted(runs.items()) if r["model"] in pick_models and r["dataset"] in pick_ds and r["format"] in pick_fmt]
if not selected:
    st.warning("No run matches the filters.")
    st.stop()

c1, c2, c3, c4 = st.columns(4)
c1.metric("Runs", len(selected))
c2.metric("Total answers", f"{sum(S[k]['n_samples'] for k in selected):,}")
c3.metric("Avg accuracy (asked once)", f"{sum(S[k]['overall_accuracy'] for k in selected) / len(selected):.1%}")
c4.metric("Max sensitivity gap", f"{max(S[k]['sensitivity_gap'] for k in selected):.1%}")
st.divider()

tab1, tab2, tab3, tab4, tab5 = st.tabs(["📋 Overview", "📈 Option bias & failing cases", "📐 Statistical tests",
                                        "🛠 Fixing the bias", "⏮ Old vs new"])

with tab1:
    st.subheader("Summary (model × dataset × labels), asked once")
    rows = []
    for k in selected:
        s, run, L = S[k], runs[k], runs[k]["letters"]
        d = s["sensitivity_gap_cohens_d"]
        rows.append({"Run": label(run), "Answers read from": SOURCE.get(run["source"], run["source"]), "N": s["n_samples"],
                     "Errors": s["n_errors"], "Overall Acc": f"{s['overall_accuracy']:.1%}",
                     "Sensitivity gap": f"{s['sensitivity_gap']:.1%}",
                     "Anchored bias": f"{s['anchored_bias_frequency']:.1%}",
                     "Effect size": "Small" if abs(d) < 0.2 else ("Medium" if abs(d) < 0.8 else "Large"),
                     **{f"Acc(pos {i + 1})": f"{s['accuracy_by_position'][x]:.1%}" for i, x in enumerate(L)}})
    st.dataframe(pd.DataFrame(rows), width="stretch", hide_index=True)
    st.caption("Positions are numbered 1–4 so letter and number runs line up: position 1 is A or (1), and so on.")

with tab2:
    st.caption("Grey: ground truth (always 25%, every question is asked in all four rotations). Blue: where the model "
               "answers. Orange: the earlier run of the same model (old answer reader). Green: after the label-free "
               "prior correction (only for runs read from probabilities).")
    for k in selected:
        s, run, L = S[k], runs[k], runs[k]["letters"]
        with st.expander(f"**{label(run)}** (n={s['n_samples']})", expanded=len(selected) == 1):
            old = earlier.get(k)
            so = summary(old["rows"], ["A", "B", "C", "D"]) if old else None
            bars = [go.Bar(name="Ground truth", x=L, y=[s["ground_truth_position_proportion"][x] for x in L], marker_color="gray")]
            if so:
                bars.append(go.Bar(name=f"Earlier run ({old['source']})", x=L, y=[so["option_proportion_pred"][x] for x in "ABCD"],
                                   marker_color="#eb6834"))
            bars.append(go.Bar(name="New run, asked once", x=L, y=[s["option_proportion_pred"][x] for x in L], marker_color="#2a78d6"))
            if run.get("prior_corrected_picks"):
                bars.append(go.Bar(name="New run + prior correction", x=L,
                                   y=[run["prior_corrected_picks"][x] for x in "ABCD"], marker_color="#1baf7a"))
            fig = go.Figure(bars)
            fig.update_layout(barmode="group", title=f"{label(run)} — option proportion", xaxis_title="Option (position)",
                              yaxis_title="Proportion", **LAYOUT)
            st.plotly_chart(fig, width="stretch", key=f"op_{k}")
            st.write("**Failing-case (errors only)**")
            cols = {"Position": L,
                    "% errors predicted at": [f"{s['error_prediction_proportion'][x]:.1%}" for x in L],
                    "% errors where correct at": [f"{s['error_correct_position_proportion'][x]:.1%}" for x in L]}
            if so:
                cols["% errors predicted at (earlier run)"] = [f"{so['error_prediction_proportion'][x]:.1%}" for x in "ABCD"]
                cols["% errors where correct at (earlier run)"] = [f"{so['error_correct_position_proportion'][x]:.1%}" for x in "ABCD"]
            st.dataframe(pd.DataFrame(cols), width="stretch", hide_index=True)

with tab3:
    st.subheader("Statistical tests")
    st.write("### Option proportion significance (Chi-square)")
    st.caption("Tests if the predicted option distribution differs from uniform (25% each).")
    st.dataframe(pd.DataFrame([{"Run": label(runs[k]), "χ²": f"{S[k]['chi_square_option_proportion']['chi2']:.2f}",
                                "p-value": f"{S[k]['chi_square_option_proportion']['p_value']:.3f}",
                                "Significant?": "Yes" if S[k]["chi_square_option_proportion"]["is_significant"] else "No"}
                               for k in selected]), width="stretch", hide_index=True)
    st.write("### Accuracy: first vs last position")
    st.caption("Two-proportion z-test comparing accuracy when the correct answer is first (A / 1) or last (D / 4).")
    st.dataframe(pd.DataFrame([{"Run": label(runs[k]),
                                "Acc(first)": f"{S[k]['accuracy_by_position'][runs[k]['letters'][0]]:.1%}",
                                "Acc(last)": f"{S[k]['accuracy_by_position'][runs[k]['letters'][-1]]:.1%}",
                                "z-score": f"{S[k]['accuracy_significance_first_vs_last'].get('z_score', 0):.2f}",
                                "p-value": f"{S[k]['accuracy_significance_first_vs_last'].get('p_value', 1):.3f}",
                                "Significant?": "Yes" if S[k]["accuracy_significance_first_vs_last"].get("is_significant") else "No"}
                               for k in selected]), width="stretch", hide_index=True)
    st.write("### Accuracy confidence intervals (95% CI)")
    for k in selected:
        ci, L = S[k]["accuracy_confidence_intervals"], runs[k]["letters"]
        with st.expander(f"**{label(runs[k])}**", expanded=len(selected) == 1):
            st.dataframe(pd.DataFrame([{"Position": x, "Accuracy": f"{ci[x]['mean']:.1%}", "95% CI lower": f"{ci[x]['lower']:.1%}",
                                        "95% CI upper": f"{ci[x]['upper']:.1%}"} for x in L]), width="stretch", hide_index=True)

with tab4:
    st.subheader("Fixing the bias: every method on the same answers")
    st.caption("Accuracy of each method; in brackets the gain over asking once in points, ✓ = significant after Holm "
               "correction (only computed for runs read from probabilities). Prior correction, PriDe and the "
               "log-probability average need the model's label probabilities, so they are empty for text-only runs.")
    rows = []
    for k in selected:
        m = runs[k]["methods"]
        row = {"Run": label(runs[k])}
        for key, name, calls in METHODS:
            x = m.get(key)
            if not x:
                row[f"{name} ({calls})"] = "—"
                continue
            t = x.get("vs_single")
            gain = f" ({t['diff']:+.1f}{' ✓' if t.get('p_holm', 1) < 0.05 else ''})" if t else (
                f" ({(x['accuracy'] - m['single']['accuracy']) * 100:+.1f})" if key != "single" and "single" in m else "")
            row[f"{name} ({calls})"] = f"{x['accuracy']:.1%}{gain}"
        rows.append(row)
    st.dataframe(pd.DataFrame(rows), width="stretch", hide_index=True)
    st.caption("Number in each column header = model calls per question.")
    show = [k for k in selected if runs[k]["methods"].get("avg_logprob")]
    if show:
        fig = go.Figure()
        for key, name, _ in METHODS:
            fig.add_trace(go.Bar(name=name, x=[label(runs[k]) for k in show],
                                 y=[(runs[k]["methods"].get(key) or {}).get("accuracy") for k in show]))
        fig.update_layout(barmode="group", title="Runs read from probabilities: accuracy by method", yaxis_title="Accuracy",
                          **{**LAYOUT, "height": 420})
        st.plotly_chart(fig, width="stretch", key="methods_chart")

with tab5:
    st.subheader("Old vs new: the same model, earlier run vs corrected run (letters)")
    st.caption("Earlier runs used the old answer reader (first A–D character, or 'A' when unreadable) or, for the "
               "Hugging Face models, an older prompt. nemotron-3-nano is compared with the earlier nemotron3:33b (same family).")
    rows = []
    for k in selected:
        if k not in earlier:
            continue
        so, s = summary(earlier[k]["rows"], ["A", "B", "C", "D"]), S[k]
        top = max("ABCD", key=lambda x: so["option_proportion_pred"][x])
        rows.append({"Run": label(runs[k]), "Earlier run": earlier[k]["source"],
                     "Earlier accuracy": f"{so['overall_accuracy']:.1%}", "New accuracy": f"{s['overall_accuracy']:.1%}",
                     "Earlier gap": f"{so['sensitivity_gap']:.1%}", "New gap": f"{s['sensitivity_gap']:.1%}",
                     "Earlier most-picked": f"{top} ({so['option_proportion_pred'][top]:.0%})"})
    if rows:
        st.dataframe(pd.DataFrame(rows), width="stretch", hide_index=True)
    else:
        st.info("No selected run has an earlier run. Choose letters A–D in the filters.")

st.sidebar.divider()
st.sidebar.caption("Data: dashboard_data/v3_dashboard.json")
st.sidebar.caption("Refresh: update_results.sh, then export_dashboard.py")
