"""
Streamlit dashboard for position-bias analysis results.
Run: uv run streamlit run streamlit_app.py
Fetches results from the database (DATABASE_URL). Figures are loaded from outputs/figures/ if present (run scripts/analyze.py to generate them).
"""
import os
import sys
from pathlib import Path

import pandas as pd
import streamlit as st
import yaml

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

try:
    from dotenv import load_dotenv
    # Override so project .env wins (e.g. port 5433 for Docker Postgres) over shell/system DATABASE_URL
    load_dotenv(ROOT / ".env", override=True)
except Exception:
    pass


def load_config():
    cfg_path = ROOT / "config.yaml"
    if not cfg_path.exists():
        return {}
    return yaml.safe_load(cfg_path.read_text()) or {}


def _summary_one(key, results, letters):
    """Build summary dict for one run (same logic as scripts/analyze.py)."""
    from src.analysis.attention_analysis import attn_to_correct_vs_others, mean_attn_by_position
    from src.analysis.layerwise import accuracy_by_layer, prob_correct_by_layer
    from src.analysis.metrics import (
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
    n = len(results)
    acc = accuracy_by_position(results, letters)
    gap = sensitivity_gap(acc)
    prob_pos = prob_correct_by_position(results, letters)
    out = {
        "n_samples": n,
        "n_errors": sum(1 for r in results if r.get("correct") != 1),
        "overall_accuracy": overall_accuracy(results),
        "accuracy_by_position": acc,
        "sensitivity_gap": gap,
        "sensitivity_gap_cohens_d": cohens_d_sensitivity_gap(acc),
        "anchored_bias_frequency": anchored_bias_frequency(results, letters),
        "prob_correct_by_position": prob_pos,
        "option_proportion_pred": option_proportion(results, letters),
        "ground_truth_position_proportion": ground_truth_position_proportion(results, letters),
        "error_prediction_proportion": error_prediction_proportion(results, letters),
        "error_correct_position_proportion": error_correct_position_proportion(results, letters),
        "chi_square_option_proportion": chi_square_option_proportion(results, letters),
        "accuracy_confidence_intervals": accuracy_confidence_interval(results, letters),
    }
    # Add significance test for A vs D (most common comparison)
    if len(letters) >= 2:
        pos_a, pos_d = letters[0], letters[-1]
        out["accuracy_significance_a_vs_d"] = accuracy_position_significance_test(results, letters, pos_a, pos_d)
    if any(r.get("attn_to_options") for r in results):
        out["attn_by_position"] = mean_attn_by_position(results, letters, layer=-1)
        out["attn_correct_minus_others"] = attn_to_correct_vs_others(results, letters, layer=-1)
    if any(r.get("probs_per_layer") for r in results):
        from src.analysis.layerwise import logit_difference_by_layer
        out["accuracy_by_layer"] = accuracy_by_layer(results, letters)
        out["prob_correct_by_layer"] = prob_correct_by_layer(results, letters)
        out["logit_difference_by_layer"] = logit_difference_by_layer(results, letters)
    return out


@st.cache_data(ttl=120)
def load_summary_from_db():
    """Fetch runs that have results and compute summary. Prefers the run with most results per (model, dataset)."""
    from src.db.client import create_schema, get_results, list_runs_with_result_counts, _conn
    conn = _conn()
    if not conn:
        return None, None
    try:
        create_schema(conn)
        cfg = load_config()
        letters = cfg.get("option_letters", ["A", "B", "C", "D"])
        runs = list_runs_with_result_counts(conn)
        # For each (model, dataset), keep the run that has the most results (so we show existing data)
        best_per_key = {}
        for r in runs:
            n = int(r.get("n_results") or 0)
            if n == 0:
                continue
            key = f"{r['model']}_{r['dataset']}"
            if key not in best_per_key or n > best_per_key[key][0]:
                best_per_key[key] = (n, r["run_id"])
        summary = {}
        for key, (_n, run_id) in best_per_key.items():
            results = get_results(conn, run_id=str(run_id))
            if results:
                summary[key] = _summary_one(key, results, letters)
        return summary, letters
    finally:
        conn.close()


def get_figures_dir():
    cfg = load_config()
    return ROOT / cfg.get("figures_dir", "outputs/figures")


def _draw_figure_plotly(key, S, letters, prefix):
    """Draw a Plotly figure from summary data. Returns (fig, True) or (None, False)."""
    import plotly.graph_objects as go
    letters = list(letters)
    layout_default = dict(
        margin=dict(l=50, r=30, t=40, b=50),
        xaxis=dict(tickfont=dict(size=11)),
        yaxis=dict(tickfont=dict(size=11), range=[0, 1.05]),
        showlegend=True,
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        height=320,
    )
    if prefix == "acc_by_pos":
        acc = S.get("accuracy_by_position")
        if not acc:
            return None, False
        fig = go.Figure(data=[go.Bar(x=list(acc.keys()), y=[acc.get(L, 0) for L in letters], marker_color="steelblue", marker_line_color="black", marker_line_width=1)])
        fig.update_layout(title=f"{key} (n={S.get('n_samples', '?')})", xaxis_title="Correct answer position", yaxis_title="Accuracy", **layout_default)
        return fig, True
    elif prefix == "prob_by_pos":
        prob = S.get("prob_correct_by_position")
        if not prob:
            return None, False
        fig = go.Figure(data=[go.Bar(x=list(prob.keys()), y=[prob.get(L, 0) for L in letters], marker_color="seagreen", marker_line_color="black", marker_line_width=1, opacity=0.9)])
        fig.update_layout(title=f"{key} — confidence by position", xaxis_title="Correct answer position", yaxis_title="Mean P(correct)", **layout_default)
        return fig, True
    elif prefix == "option_proportion":
        pred = S.get("option_proportion_pred")
        gt = S.get("ground_truth_position_proportion")
        if not pred or not gt:
            return None, False
        fig = go.Figure(data=[
            go.Bar(name="Ground truth", x=letters, y=[gt.get(L, 0) for L in letters], marker_color="gray", marker_line_color="black", marker_line_width=1),
            go.Bar(name="Model predictions", x=letters, y=[pred.get(L, 0) for L in letters], marker_color="coral", marker_line_color="black", marker_line_width=1, opacity=0.9),
        ])
        fig.update_layout(barmode="group", title=f"{key} — option proportion", xaxis_title="Option (position)", yaxis_title="Proportion", **layout_default)
        return fig, True
    elif prefix == "attn_heatmap":
        attn = S.get("attn_by_position")
        if not attn or not isinstance(attn, dict):
            return None, False
        import numpy as np
        mat = np.array([[attn.get(pos, {}).get(opt, 0) for opt in letters] for pos in letters])
        fig = go.Figure(data=go.Heatmap(z=mat, x=letters, y=letters, colorscale="Blues", text=[[f"{mat[i, j]:.2f}" for j in range(len(letters))] for i in range(len(letters))], texttemplate="%{text}", textfont={"size": 10}))
        fig.update_layout(title=f"{key} — attention (last layer)", xaxis_title="Attention to option", yaxis_title="Correct answer at position", height=340, margin=dict(l=60, r=40, t=40, b=50), yaxis=dict(autorange="reversed"))
        return fig, True
    elif prefix == "layer_wise":
        acc_ly = S.get("accuracy_by_layer")
        prob_ly = S.get("prob_correct_by_layer")
        if not acc_ly or not prob_ly or len(acc_ly) != len(prob_ly):
            return None, False
        layers = list(range(len(acc_ly)))
        mean_acc = [sum(acc_ly[i].get(L, 0) for L in letters) / len(letters) for i in layers]
        mean_prob = [sum(prob_ly[i].get(L, 0) for L in letters) / len(letters) for i in layers]
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=layers, y=mean_acc, name="Mean accuracy", line=dict(color="rgb(31, 119, 180)", width=2), yaxis="y"))
        fig.add_trace(go.Scatter(x=layers, y=mean_prob, name="Mean P(correct)", line=dict(color="rgb(44, 160, 44)", width=2, dash="dash"), yaxis="y2"))
        fig.update_layout(
            title=f"{key} — layer-wise",
            xaxis=dict(title="Layer"),
            yaxis=dict(title=dict(text="Accuracy", font=dict(color="rgb(31, 119, 180)")), tickfont=dict(color="rgb(31, 119, 180)"), range=[0, 1.05]),
            yaxis2=dict(title=dict(text="Mean P(correct)", font=dict(color="rgb(44, 160, 44)")), tickfont=dict(color="rgb(44, 160, 44)"), range=[0, 1.05], overlaying="y", side="right"),
            height=340,
            margin=dict(l=50, r=50, t=40, b=50),
            legend=dict(orientation="h", yanchor="bottom", y=1.02),
        )
        return fig, True
    elif prefix == "layer_bias":
        logit_diff = S.get("logit_difference_by_layer")
        if not logit_diff:
            return None, False
        layers = [d.get("layer", i) for i, d in enumerate(logit_diff)]
        diffs = [d.get("logit_diff_anchor_minus_correct", 0.0) for d in logit_diff]
        fig = go.Figure()
        fig.add_trace(
            go.Scatter(
                x=layers,
                y=diffs,
                mode="lines+markers",
                name="logit[anchor] - logit[correct]",
                line=dict(color="indianred", width=2),
            )
        )
        fig.add_hline(y=0.0, line=dict(color="gray", width=1, dash="dot"))
        fig.update_layout(
            title=f"{key} — layer-wise anchored bias",
            xaxis=dict(title="Layer"),
            yaxis=dict(
                title="Mean logit difference (anchor − correct)",
                zeroline=True,
                zerolinewidth=1,
                zerolinecolor="gray",
            ),
            height=340,
            margin=dict(l=50, r=50, t=40, b=50),
            legend=dict(orientation="h", yanchor="bottom", y=1.02),
        )
        return fig, True
    elif prefix == "attn_anchor":
        attn_diff = S.get("attn_correct_minus_others")
        if not attn_diff or not isinstance(attn_diff, dict):
            return None, False
        ys = [attn_diff.get(L, 0.0) for L in letters]
        fig = go.Figure(
            data=[
                go.Bar(
                    x=letters,
                    y=ys,
                    marker_color="mediumpurple",
                    marker_line_color="black",
                    marker_line_width=1,
                )
            ]
        )
        fig.add_hline(y=0.0, line=dict(color="gray", width=1, dash="dot"))
        fig.update_layout(
            title=f"{key} — attention bias (correct vs others, last layer)",
            xaxis_title="Correct answer position",
            yaxis_title="Mean attention(correct) − mean attention(others)",
            height=340,
            margin=dict(l=50, r=50, t=40, b=50),
        )
        return fig, True
    return None, False


st.set_page_config(
    page_title="Position Bias — Analysis Dashboard",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown(
    """
    <style>
    /* General typography */
    .main-title {
        font-size: 2rem;
        font-weight: 700;
        margin-bottom: 0.25rem;
    }
    .sub-title {
        font-size: 0.9rem;
        color: #6c757d;
        margin-bottom: 1.5rem;
    }
    /* Metric cards */
    .block-container {
        padding-top: 1.5rem;
        padding-bottom: 1.5rem;
    }
    /* Expander header weight */
    .streamlit-expanderHeader {
        font-weight: 600;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

st.markdown('<div class="main-title">📊 Position Bias in Multiple-Choice QA</div>', unsafe_allow_html=True)
st.markdown(
    '<div class="sub-title">Analysis of option-shuffling experiments across models and datasets, linking output bias to internal model behaviour.</div>',
    unsafe_allow_html=True,
)

if not os.environ.get("DATABASE_URL"):
    st.error("**DATABASE_URL** is not set. Set it in `.env` (e.g. `DATABASE_URL=postgresql://user:pass@host:port/dbname`) and ensure the app can reach the database.")
    st.stop()

summary, letters = load_summary_from_db()
if not summary:
    st.error("No runs found in the database.")
    with st.expander("Diagnose connection and data", expanded=True):
        from src.db.client import create_schema, get_results, list_run_keys
        import psycopg
        url = os.environ.get("DATABASE_URL", "")
        # Show host:port (hide password)
        try:
            from urllib.parse import urlparse
            p = urlparse(url)
            host_port = f"{p.hostname or '?'}:{p.port or '?'}"
        except Exception:
            host_port = "(could not parse URL)"
        st.code(f"DATABASE_URL host: {host_port}", language=None)
        conn = None
        err_msg = None
        try:
            from src.db.client import _conn
            conn = _conn()
        except Exception as e:
            err_msg = str(e)
        if not conn:
            if not err_msg:
                try:
                    conn = psycopg.connect(url, connect_timeout=3)
                    conn.close()
                except Exception as e:
                    err_msg = str(e)
            st.warning("Could not connect to the database.")
            if err_msg:
                st.caption(f"Error: {err_msg}")
            st.markdown("**Fix:**")
            st.markdown("- **Dashboard on your machine** (e.g. `uv run streamlit run streamlit_app.py`): in `.env` set `DATABASE_URL=postgresql://llmbias:llmbias@localhost:5433/llmbias` and run `docker compose up -d` so Postgres is on port 5433.")
            st.markdown("- **Dashboard inside Docker** (e.g. `docker compose exec app uv run streamlit run ...`): the container already gets the right URL. Ensure Postgres is up: `docker compose ps`. If your `.env` has `localhost` in DATABASE_URL, it can override the container env — remove or comment out `DATABASE_URL` in `.env` when using Docker, or run the dashboard on the host with the URL above.")
        else:
            st.success("Connected to the database.")
            try:
                create_schema(conn)
                keys = list_run_keys(conn)
                st.write(f"**Runs (model × dataset):** {len(keys)}")
                # Per-run result counts (in case results are under specific run_ids)
                with conn.cursor() as cur:
                    cur.execute("""
                        SELECT rn.id, rn.model, rn.dataset, rn.created_at, COUNT(r.id) AS n_results
                        FROM runs rn
                        LEFT JOIN results r ON r.run_id = rn.id
                        GROUP BY rn.id, rn.model, rn.dataset, rn.created_at
                        ORDER BY rn.created_at DESC
                    """)
                    run_rows = cur.fetchall()
                total_results = sum(r["n_results"] or 0 for r in run_rows)
                st.write(f"**Total rows in `results` table:** {total_results}")
                if keys:
                    for (model, dataset) in keys[:20]:
                        results = get_results(conn, model, dataset)
                        st.write(f"- `{model}` × `{dataset}` → **{len(results)}** results")
                    if len(keys) > 20:
                        st.caption(f"... and {len(keys) - 20} more runs")
                if total_results == 0:
                    st.warning("The **results** table is empty. Runs exist but no experiment results were saved. Re-run experiments inside Docker: `docker compose exec app uv run python scripts/run_experiments.py`")
                elif run_rows:
                    st.caption("Per run (latest first):")
                    for row in run_rows[:15]:
                        st.caption(f"  • {row.get('model')} × {row.get('dataset')} (run_id={str(row.get('id'))[:8]}…) → {row.get('n_results') or 0} results")
                else:
                    st.caption("The `runs` table is empty. Run experiments: `docker compose exec app uv run python scripts/run_experiments.py`.")
                st.caption("If you ran experiments inside Docker, use **DATABASE_URL=postgresql://llmbias:llmbias@localhost:5433/llmbias** when running this dashboard on the host.")
            finally:
                conn.close()
    st.stop()

fig_dir = get_figures_dir()
run_keys = sorted(summary.keys())

# Sidebar: run selection
st.sidebar.header("Filters")
selected_runs = st.sidebar.multiselect(
    "Model × Dataset",
    run_keys,
    default=run_keys[:1] if run_keys else [],
    help="Select one or more runs to focus on.",
)
if not selected_runs:
    selected_runs = run_keys

# KPIs (over selected or all)
total_samples = sum(summary[k].get("n_samples", 0) for k in selected_runs)
total_errors = sum(summary[k].get("n_errors", 0) for k in selected_runs)
accs = [summary[k].get("overall_accuracy") for k in selected_runs if summary[k].get("overall_accuracy") is not None]
gaps = [summary[k].get("sensitivity_gap") for k in selected_runs if summary[k].get("sensitivity_gap") is not None]

col1, col2, col3, col4 = st.columns(4)
with col1:
    st.metric("Runs", len(selected_runs))
with col2:
    st.metric("Total samples", f"{total_samples:,}")
with col3:
    avg_acc = (sum(accs) / len(accs) * 100) if accs else 0
    st.metric("Avg overall accuracy", f"{avg_acc:.1f}%")
with col4:
    max_gap = (max(gaps) * 100) if gaps else 0
    st.metric("Max sensitivity gap", f"{max_gap:.1f}%")

st.divider()

# Tabs: Overview | Option bias & failing cases | Statistical tests | Internals & figures
tab1, tab2, tab3, tab4 = st.tabs(
    [
        "📋 Overview",
        "📈 Option bias & failing cases",
        "📐 Statistical tests",
        "🧠 Internals & figures",
    ]
)

with tab1:
    st.subheader("Summary (model × dataset)")
    rows = []
    for key in selected_runs:
        S = summary[key]
        acc = S.get("accuracy_by_position") or {}
        d = S.get("sensitivity_gap_cohens_d", 0)
        effect = "Small" if abs(d) < 0.2 else ("Medium" if abs(d) < 0.8 else "Large")
        abf = S.get("anchored_bias_frequency")
        rows.append({
            "Run": key,
            "N": S.get("n_samples", "—"),
            "Errors": S.get("n_errors", "—"),
            "Overall Acc": f"{S.get('overall_accuracy', 0):.1%}" if S.get("overall_accuracy") is not None else "—",
            "Sensitivity gap": f"{S.get('sensitivity_gap', 0):.1%}" if S.get("sensitivity_gap") is not None else "—",
            "Anchored bias": f"{abf:.1%}" if abf is not None else "—",
            "Effect size": effect,
            **{f"Acc({L})": f"{acc.get(L, 0):.1%}" for L in letters},
        })
    df = pd.DataFrame(rows)
    st.dataframe(df, use_container_width=True, hide_index=True)

with tab2:
    for key in selected_runs:
        S = summary[key]
        with st.expander(f"**{key}** (n={S.get('n_samples', '?')})", expanded=(len(selected_runs) == 1)):
            pred = S.get("option_proportion_pred")
            gt = S.get("ground_truth_position_proportion")
            err_pred = S.get("error_prediction_proportion")
            err_correct = S.get("error_correct_position_proportion")
            if pred and gt:
                st.write("**Option proportion (predicted vs ground truth)**")
                fig, _ = _draw_figure_plotly(key, S, letters, "option_proportion")
                if fig:
                    st.plotly_chart(fig, use_container_width=True, key=f"option_proportion_{key.replace('/', '_')}")
            if err_pred and err_correct and S.get("n_errors", 0) > 0:
                st.write("**Failing-case (errors only)**")
                fail_df = pd.DataFrame({
                    "Position": letters,
                    "% errors predicted at": [f"{err_pred.get(L, 0):.1%}" for L in letters],
                    "% errors where correct at": [f"{err_correct.get(L, 0):.1%}" for L in letters],
                })
                st.dataframe(fail_df, use_container_width=True, hide_index=True)

with tab3:
    st.subheader("Statistical tests")
    st.caption("Chi-square, effect size (Cohen's d), significance tests, and confidence intervals.")
    
    # Chi-square test
    st.write("### Option proportion significance (Chi-square)")
    st.caption("Tests if predicted option distribution differs from uniform (25% each).")
    chi_rows = []
    for key in selected_runs:
        S = summary[key]
        chi2_data = S.get("chi_square_option_proportion", {})
        chi_rows.append({
            "Run": key,
            "χ²": f"{chi2_data.get('chi2', 0):.2f}",
            "p-value": f"{chi2_data.get('p_value', 1.0):.3f}",
            "Significant?": "Yes" if chi2_data.get("is_significant", False) else "No",
        })
    if chi_rows:
        st.dataframe(pd.DataFrame(chi_rows), use_container_width=True, hide_index=True)
    
    # Effect size
    st.write("### Sensitivity gap effect size (Cohen's d)")
    st.caption("Effect size: small (<0.2), medium (0.2–0.8), large (>0.8).")
    effect_rows = []
    for key in selected_runs:
        S = summary[key]
        gap = S.get("sensitivity_gap", 0)
        d = S.get("sensitivity_gap_cohens_d", 0)
        if abs(d) < 0.2:
            effect = "Small"
        elif abs(d) < 0.8:
            effect = "Medium"
        else:
            effect = "Large"
        effect_rows.append({
            "Run": key,
            "Sensitivity gap": f"{gap:.1%}",
            "Cohen's d": f"{d:.2f}",
            "Effect size": effect,
        })
    if effect_rows:
        st.dataframe(pd.DataFrame(effect_rows), use_container_width=True, hide_index=True)
    
    # A vs D significance
    st.write("### Accuracy: Position A vs D significance test")
    st.caption("Two-proportion z-test comparing accuracy when correct is at A vs D.")
    sig_rows = []
    for key in selected_runs:
        S = summary[key]
        acc = S.get("accuracy_by_position", {})
        acc_a = acc.get(letters[0], 0) if letters else 0
        acc_d = acc.get(letters[-1], 0) if len(letters) >= 2 else 0
        sig_data = S.get("accuracy_significance_a_vs_d", {})
        sig_rows.append({
            "Run": key,
            "Acc(A)": f"{acc_a:.1%}",
            "Acc(D)": f"{acc_d:.1%}",
            "z-score": f"{sig_data.get('z_score', 0):.2f}",
            "p-value": f"{sig_data.get('p_value', 1.0):.3f}",
            "Significant?": "Yes" if sig_data.get("is_significant", False) else "No",
        })
    if sig_rows:
        st.dataframe(pd.DataFrame(sig_rows), use_container_width=True, hide_index=True)
    
    # Confidence intervals
    st.write("### Accuracy confidence intervals (95% CI)")
    st.caption("Wilson score intervals for accuracy by position.")
    for key in selected_runs:
        S = summary[key]
        ci = S.get("accuracy_confidence_intervals", {})
        if not ci:
            continue
        with st.expander(f"**{key}**", expanded=(len(selected_runs) == 1)):
            ci_rows = []
            for L in letters:
                ci_l = ci.get(L, {})
                ci_rows.append({
                    "Position": L,
                    "Accuracy": f"{ci_l.get('mean', 0):.1%}",
                    "95% CI lower": f"{ci_l.get('lower', 0):.1%}",
                    "95% CI upper": f"{ci_l.get('upper', 0):.1%}",
                })
            st.dataframe(pd.DataFrame(ci_rows), use_container_width=True, hide_index=True)

with tab4:
    st.subheader("Internals & figures")
    st.caption("Interactive visualisations from the database and `outputs/figures/`, focusing on how output bias emerges inside the model.")
    st.info(
        "**Why don’t all models have the same plots?** Accuracy, P(correct), and option proportion are shown for every run. "
        "**Attention heatmap**, **layer-wise**, and **anchored bias internals** plots need internal model data (attention weights, per-layer logits). "
        "Those are only collected for **Hugging Face models** (LLaMA, Qwen). **OpenAI** (gpt-4.1-mini) is an API and does not expose attention or "
        "layer outputs, so those plots show “no data” for OpenAI runs."
    )
    for key in selected_runs:
        S = summary.get(key)
        st.write(f"**{key}**")
        fig_names = [
            ("acc_by_pos", "Accuracy by correct answer position"),
            ("prob_by_pos", "Mean P(correct) by position"),
            ("option_proportion", "Option proportion (pred vs GT)"),
            ("attn_heatmap", "Attention to options (last layer)"),
            ("layer_wise", "Layer-wise accuracy & P(correct)"),
            ("attn_anchor", "Attention bias: correct vs others"),
            ("layer_bias", "Layer-wise anchored bias (logits)"),
        ]
        cols = st.columns(2)
        for i, (prefix, label) in enumerate(fig_names):
            path = fig_dir / f"{prefix}_{key.replace('/', '_')}.png"
            with cols[i % 2]:
                if path.exists():
                    st.image(str(path), caption=label, use_container_width=True)
                elif S:
                    fig, ok = _draw_figure_plotly(key, S, letters, prefix)
                    if ok:
                        st.plotly_chart(fig, use_container_width=True, key=f"fig_{key.replace('/', '_')}_{prefix}")
                    else:
                        st.caption(f"{label}: no data")
                else:
                    st.caption(f"{label}: no data")
        st.divider()

st.sidebar.divider()
if st.sidebar.button("Refresh from DB", help="Refetch results from the database"):
    load_summary_from_db.clear()
    st.rerun()
st.sidebar.caption("Data: database (DATABASE_URL)")
st.sidebar.caption("Figures: `outputs/figures/` (run analyze.py to generate)")
