"""
Full analysis pipeline: load results (DB or JSON), compute all metrics,
save summary.json, generate publication-ready figures, and write analysis_report.md.
"""
import os
import sys
from pathlib import Path

import yaml
from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
load_dotenv(ROOT / ".env")

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
from src.analysis.layerwise import logit_difference_by_layer
from src.io import load_json, save_json


def _collect(conn):
    from src.db.client import get_results, list_run_keys

    for (model, ds) in list_run_keys(conn):
        results = get_results(conn, model, ds)
        if not results:
            continue
        yield f"{model}_{ds}", results


def _summary_one(key, results, letters):
    n = len(results)
    acc = accuracy_by_position(results, letters)
    gap = sensitivity_gap(acc)
    prob_pos = prob_correct_by_position(results, letters)
    errors = [r for r in results if r.get("correct") != 1]
    out = {
        "n_samples": n,
        "n_errors": len(errors),
        "overall_accuracy": overall_accuracy(results),
        "accuracy_by_position": acc,
        "sensitivity_gap": gap,
        "sensitivity_gap_cohens_d": cohens_d_sensitivity_gap(acc),
        "anchored_bias_frequency": anchored_bias_frequency(results, letters),  # ACL 2025: % pred='A' when correct≠'A'
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
        out["accuracy_by_layer"] = accuracy_by_layer(results, letters)
        out["prob_correct_by_layer"] = prob_correct_by_layer(results, letters)
        # ACL 2025: Logit difference by layer (logit[A] - logit[correct])
        out["logit_difference_by_layer"] = logit_difference_by_layer(results, letters)
    return out


def _figures(summary, letters, fig_dir):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np

    for style in ("seaborn-v0_8-whitegrid", "seaborn-whitegrid", "ggplot"):
        try:
            plt.style.use(style)
            break
        except Exception:
            pass
    fig_dir = Path(fig_dir)

    for key, S in summary.items():
        acc = S.get("accuracy_by_position")
        if not acc:
            continue

        # 1. Accuracy by correct answer position (A/B/C/D)
        fig, ax = plt.subplots(figsize=(5, 3.5))
        ax.bar(acc.keys(), [acc[k] for k in letters], color="steelblue", edgecolor="black")
        ax.set_xlabel("Correct answer position")
        ax.set_ylabel("Accuracy")
        ax.set_ylim(0, 1.05)
        ax.set_title(f"{key} (n={S.get('n_samples', '?')})")
        fig.tight_layout()
        fig.savefig(fig_dir / f"acc_by_pos_{key}.png", dpi=150)
        plt.close()

        # 2. Mean probability of correct answer by position
        prob_pos = S.get("prob_correct_by_position")
        if prob_pos:
            fig, ax = plt.subplots(figsize=(5, 3.5))
            ax.bar(prob_pos.keys(), [prob_pos.get(k, 0) for k in letters], color="seagreen", edgecolor="black", alpha=0.8)
            ax.set_xlabel("Correct answer position")
            ax.set_ylabel("Mean P(correct)")
            ax.set_ylim(0, 1.05)
            ax.set_title(f"{key} — confidence by position")
            fig.tight_layout()
            fig.savefig(fig_dir / f"prob_by_pos_{key}.png", dpi=150)
            plt.close()

        # 3. Option proportion: predicted vs ground-truth position
        pred_prop = S.get("option_proportion_pred")
        gt_prop = S.get("ground_truth_position_proportion")
        if pred_prop and gt_prop:
            x = np.arange(len(letters))
            w = 0.35
            fig, ax = plt.subplots(figsize=(5, 3.5))
            ax.bar(x - w/2, [gt_prop.get(k, 0) for k in letters], w, label="Ground truth", color="gray", edgecolor="black")
            ax.bar(x + w/2, [pred_prop.get(k, 0) for k in letters], w, label="Model predictions", color="coral", edgecolor="black", alpha=0.8)
            ax.set_xticks(x)
            ax.set_xticklabels(letters)
            ax.set_xlabel("Option (position)")
            ax.set_ylabel("Proportion")
            ax.set_ylim(0, 1.05)
            ax.legend()
            ax.set_title(f"{key} — option proportion (bias toward positions)")
            fig.tight_layout()
            fig.savefig(fig_dir / f"option_proportion_{key}.png", dpi=150)
            plt.close()

        # 4. Attention heatmap: rows = correct position, cols = option (mean attention to that option)
        attn_by_pos = S.get("attn_by_position")
        if attn_by_pos and isinstance(attn_by_pos, dict):
            mat = np.array([[attn_by_pos.get(pos, {}).get(opt, 0) for opt in letters] for pos in letters])
            fig, ax = plt.subplots(figsize=(4, 3.5))
            im = ax.imshow(mat, cmap="Blues", vmin=0, vmax=mat.max() if mat.size else 1)
            ax.set_xticks(range(len(letters)))
            ax.set_yticks(range(len(letters)))
            ax.set_xticklabels(letters)
            ax.set_yticklabels(letters)
            ax.set_xlabel("Attention to option")
            ax.set_ylabel("Correct answer at position")
            for i in range(len(letters)):
                for j in range(len(letters)):
                    ax.text(j, i, f"{mat[i, j]:.2f}", ha="center", va="center", color="black", fontsize=9)
            plt.colorbar(im, ax=ax, label="Mean attention")
            ax.set_title(f"{key} — attention to options (last layer)")
            fig.tight_layout()
            fig.savefig(fig_dir / f"attn_heatmap_{key}.png", dpi=150)
            plt.close()

        # 5. Layer-wise: mean accuracy and mean P(correct) vs layer (HF models only)
        acc_ly = S.get("accuracy_by_layer")
        prob_ly = S.get("prob_correct_by_layer")
        if acc_ly and prob_ly and len(acc_ly) == len(prob_ly):
            layers = range(len(acc_ly))
            # Mean accuracy over positions at each layer
            mean_acc = [sum(acc_ly[i].get(L, 0) for L in letters) / len(letters) for i in layers]
            mean_prob = [sum(prob_ly[i].get(L, 0) for L in letters) / len(letters) for i in layers]
            fig, ax1 = plt.subplots(figsize=(6, 3.5))
            ax1.plot(layers, mean_acc, "b-", label="Mean accuracy by position")
            ax1.set_xlabel("Layer")
            ax1.set_ylabel("Accuracy", color="b")
            ax1.set_ylim(0, 1.05)
            ax1.tick_params(axis="y", labelcolor="b")
            ax2 = ax1.twinx()
            ax2.plot(layers, mean_prob, "g--", label="Mean P(correct) by position")
            ax2.set_ylabel("Mean P(correct)", color="g")
            ax2.set_ylim(0, 1.05)
            ax2.tick_params(axis="y", labelcolor="g")
            ax1.set_title(f"{key} — layer-wise (positional bias emergence)")
            fig.tight_layout()
            fig.savefig(fig_dir / f"layer_wise_{key}.png", dpi=150)
            plt.close()

    return


def _write_report(summary, letters, res_dir, fig_dir):
    report_path = Path(res_dir) / "analysis_report.md"
    lines = [
        "# Analysis Report — Position Bias in Multiple-Choice QA",
        "",
        "## Methodology",
        "",
        "This analysis follows best-practice evaluation for positional bias in LLM multiple-choice QA:",
        "",
        "- **Accuracy by position (A/B/C/D)** and **sensitivity gap** (max − min accuracy over positions) quantify how much performance depends on where the correct answer appears (Pezeshkpour & Hruschka, NAACL 2024 Findings; [Large Language Models Sensitivity to The Order of Options in Multiple-Choice Questions](https://aclanthology.org/2024.findings-naacl.130/)).",
        "- **Option proportion** (predicted vs ground-truth) and **failing-case analysis** (where the model predicts when wrong, and where the correct answer was when wrong) capture anchored/positional bias in errors (Li & Gao, ACL 2025 Findings; [Anchored Answers: Unravelling Positional Bias in GPT-2's Multiple-Choice Questions](https://aclanthology.org/2025.findings-acl.124/)).",
        "- **Layer-wise accuracy and P(correct)** (logit-lens style) and **attention to options** support mechanistic analysis of where bias emerges (ACL 2025).",
        "",
        "---",
        "",
        "## 1. Summary table (all model × dataset)",
        "",
        "| Model × Dataset | N | Overall Acc | Sensitivity gap | Anchored bias | Acc(A) | Acc(B) | Acc(C) | Acc(D) |",
        "|-----------------|---|-------------|-----------------|---------------|--------|--------|--------|--------|",
    ]
    for key in sorted(summary.keys()):
        S = summary[key]
        n = S.get("n_samples", "?")
        oa = S.get("overall_accuracy")
        oa_str = f"{oa:.1%}" if oa is not None else "—"
        gap = S.get("sensitivity_gap")
        gap_str = f"{gap:.1%}" if gap is not None else "—"
        abf = S.get("anchored_bias_frequency")
        abf_str = f"{abf:.1%}" if abf is not None else "—"
        acc = S.get("accuracy_by_position") or {}
        acc_str = " | ".join(f"{acc.get(L, 0):.1%}" for L in letters)
        lines.append(f"| {key} | {n} | {oa_str} | {gap_str} | {abf_str} | {acc_str} |")

    lines.extend([
        "",
        "## 2. Anchored bias frequency (ACL 2025)",
        "",
        "**Anchored bias frequency**: % of samples where model predicts 'A' (first position) but correct answer is NOT 'A'.",
        "This directly measures the \"anchored bias\" phenomenon identified in Li & Gao (ACL 2025).",
        "",
        "| Model × Dataset | Anchored bias frequency | Interpretation |",
        "|-----------------|------------------------|----------------|",
    ])
    for key in sorted(summary.keys()):
        S = summary[key]
        abf = S.get("anchored_bias_frequency")
        if abf is not None:
            if abf > 0.5:
                interp = "Strong anchored bias (model heavily favors 'A')"
            elif abf > 0.25:
                interp = "Moderate anchored bias"
            elif abf > 0.1:
                interp = "Weak anchored bias"
            else:
                interp = "No significant anchored bias"
            lines.append(f"| {key} | {abf:.1%} | {interp} |")
        else:
            lines.append(f"| {key} | — | — |")
    
    lines.extend([
        "",
        "## 3. Option proportion (prediction bias)",
        "",
        "Ground truth is balanced (correct at A/B/C/D equally). Model predictions may favor certain positions.",
        "",
    ])
    for key in sorted(summary.keys()):
        S = summary[key]
        pred = S.get("option_proportion_pred")
        gt = S.get("ground_truth_position_proportion")
        if not pred or not gt:
            continue
        lines.append(f"### {key}")
        lines.append("| Position | Ground truth | Model pred |")
        lines.append("|----------|--------------|------------|")
        for L in letters:
            lines.append(f"| {L} | {gt.get(L, 0):.1%} | {pred.get(L, 0):.1%} |")
        lines.append("")

    lines.extend([
        "## 4. Statistical tests",
        "",
        "### 3.1 Option proportion significance (Chi-square test)",
        "",
        "Tests whether predicted option distribution differs significantly from uniform (25% each).",
        "",
        "| Model × Dataset | χ² | p-value | Significant? | Interpretation |",
        "|-----------------|----|---------|--------------|----------------|",
    ])
    for key in sorted(summary.keys()):
        S = summary[key]
        chi2_data = S.get("chi_square_option_proportion", {})
        chi2_val = chi2_data.get("chi2", 0)
        p_val = chi2_data.get("p_value", 1.0)
        sig = chi2_data.get("is_significant", False)
        interp = "Bias detected" if sig else "No significant bias"
        lines.append(f"| {key} | {chi2_val:.2f} | {p_val:.3f} | {'Yes' if sig else 'No'} | {interp} |")
    
    lines.extend([
        "",
        "### 3.2 Sensitivity gap effect size (Cohen's d)",
        "",
        "Effect size for sensitivity gap: small (<0.2), medium (0.2–0.8), large (>0.8).",
        "",
        "| Model × Dataset | Sensitivity gap | Cohen's d | Effect size |",
        "|-----------------|-----------------|-----------|-------------|",
    ])
    for key in sorted(summary.keys()):
        S = summary[key]
        gap = S.get("sensitivity_gap", 0)
        d = S.get("sensitivity_gap_cohens_d", 0)
        if abs(d) < 0.2:
            effect = "Small"
        elif abs(d) < 0.8:
            effect = "Medium"
        else:
            effect = "Large"
        lines.append(f"| {key} | {gap:.1%} | {d:.2f} | {effect} |")
    
    lines.extend([
        "",
        "### 3.3 Accuracy by position: A vs D significance test",
        "",
        "Two-proportion z-test comparing accuracy when correct is at A vs D.",
        "",
        "| Model × Dataset | Acc(A) | Acc(D) | z-score | p-value | Significant? |",
        "|-----------------|--------|--------|---------|---------|--------------|",
    ])
    for key in sorted(summary.keys()):
        S = summary[key]
        acc = S.get("accuracy_by_position", {})
        acc_a = acc.get(letters[0], 0) if letters else 0
        acc_d = acc.get(letters[-1], 0) if len(letters) >= 2 else 0
        sig_data = S.get("accuracy_significance_a_vs_d", {})
        z = sig_data.get("z_score", 0)
        p_val = sig_data.get("p_value", 1.0)
        sig = sig_data.get("is_significant", False)
        lines.append(f"| {key} | {acc_a:.1%} | {acc_d:.1%} | {z:.2f} | {p_val:.3f} | {'Yes' if sig else 'No'} |")
    
    lines.extend([
        "",
        "### 3.4 Accuracy confidence intervals (95% CI)",
        "",
        "Wilson score intervals for accuracy by position.",
        "",
    ])
    for key in sorted(summary.keys()):
        S = summary[key]
        ci = S.get("accuracy_confidence_intervals", {})
        if not ci:
            continue
        lines.append(f"### {key}")
        lines.append("| Position | Accuracy | 95% CI (lower) | 95% CI (upper) |")
        lines.append("|----------|----------|----------------|-----------------|")
        for L in letters:
            ci_l = ci.get(L, {})
            mean = ci_l.get("mean", 0)
            lower = ci_l.get("lower", 0)
            upper = ci_l.get("upper", 0)
            lines.append(f"| {L} | {mean:.1%} | {lower:.1%} | {upper:.1%} |")
        lines.append("")
    
    lines.extend([
        "## 5. Logit difference by layer (ACL 2025)",
        "",
        "**Logit difference**: logit['A'] - logit[correct] per layer, computed only for samples where correct ≠ 'A'.",
        "Positive values indicate bias toward 'A' at that layer. This identifies which layers contribute most to anchored bias.",
        "",
    ])
    for key in sorted(summary.keys()):
        S = summary[key]
        logit_diff = S.get("logit_difference_by_layer")
        if logit_diff and isinstance(logit_diff, list) and len(logit_diff) > 0:
            lines.append(f"### {key}")
            lines.append("| Layer | Logit diff (A - correct) | Interpretation |")
            lines.append("|-------|--------------------------|----------------|")
            for item in logit_diff[:10]:  # Show first 10 layers
                layer_idx = item.get("layer", "?")
                diff = item.get("logit_diff_anchor_minus_correct", 0)
                if diff > 2:
                    interp = "Strong bias toward 'A'"
                elif diff > 0.5:
                    interp = "Moderate bias toward 'A'"
                elif diff > 0:
                    interp = "Weak bias toward 'A'"
                else:
                    interp = "No bias (or bias toward correct)"
                lines.append(f"| {layer_idx} | {diff:.3f} | {interp} |")
            if len(logit_diff) > 10:
                lines.append(f"| ... | ({len(logit_diff) - 10} more layers) | |")
            lines.append("")
    
    lines.extend([
        "## 6. Failing-case analysis (anchored bias in errors)",
        "",
        "Among **errors only**: proportion of predictions at each position (model bias when wrong) and proportion where the correct answer was at each position (where the model fails most).",
        "",
    ])
    for key in sorted(summary.keys()):
        S = summary[key]
        err_pred = S.get("error_prediction_proportion")
        err_correct = S.get("error_correct_position_proportion")
        n_err = S.get("n_errors", 0)
        if not err_pred or not err_correct or n_err == 0:
            continue
        lines.append(f"### {key} (n_errors = {n_err})")
        lines.append("| Position | % of errors predicted at | % of errors where correct at |")
        lines.append("|----------|---------------------------|-------------------------------|")
        for L in letters:
            lines.append(f"| {L} | {err_pred.get(L, 0):.1%} | {err_correct.get(L, 0):.1%} |")
        lines.append("")

    lines.extend([
        "## 7. Figures",
        "",
        "All figures are in `outputs/figures/` (or `figures_dir` in config).",
        "",
    ])
    for key in sorted(summary.keys()):
        lines.append(f"- **{key}**")
        lines.append(f"  - `acc_by_pos_{key}.png` — Accuracy when correct answer is at A/B/C/D")
        lines.append(f"  - `prob_by_pos_{key}.png` — Mean P(correct) by position")
        lines.append(f"  - `option_proportion_{key}.png` — Predicted vs ground-truth option proportion")
        if summary[key].get("attn_by_position"):
            lines.append(f"  - `attn_heatmap_{key}.png` — Attention to options (last layer)")
        if summary[key].get("accuracy_by_layer"):
            lines.append(f"  - `layer_wise_{key}.png` — Layer-wise accuracy and P(correct)")

    Path(report_path).write_text("\n".join(lines), encoding="utf-8")
    return report_path


def main():
    cfg_path = ROOT / "config.yaml"
    cfg = yaml.safe_load(cfg_path.read_text())
    letters = cfg.get("option_letters", ["A", "B", "C", "D"])
    res_dir = Path(cfg.get("results_dir", "outputs/results"))
    fig_dir = Path(cfg.get("figures_dir", "outputs/figures"))
    fig_dir.mkdir(parents=True, exist_ok=True)

    summary = {}
    if os.environ.get("DATABASE_URL"):
        from src.db.client import _conn, create_schema

        conn = _conn()
        if conn:
            try:
                create_schema(conn)
                for key, results in _collect(conn):
                    summary[key] = _summary_one(key, results, letters)
            finally:
                conn.close()

    if not summary:
        for p in sorted(res_dir.glob("*.json")):
            if p.name == "summary.json":
                continue
            data = load_json(p)
            key = f"{data.get('model', p.stem.split('_')[0])}_{data.get('dataset', 'unknown')}"
            results = data.get("results", [])
            if not results:
                continue
            summary[key] = _summary_one(key, results, letters)

    save_json(summary, res_dir / "summary.json")

    try:
        _figures(summary, letters, fig_dir)
    except Exception as e:
        print(f"Figures failed: {e}", file=sys.stderr)

    report_path = _write_report(summary, letters, res_dir, fig_dir)
    print(f"Summary: {res_dir / 'summary.json'}")
    print(f"Figures: {fig_dir}")
    print(f"Report:  {report_path}")


if __name__ == "__main__":
    main()
