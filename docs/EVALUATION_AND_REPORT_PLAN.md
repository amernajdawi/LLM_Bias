# Evaluation & Report Plan — Response to Antonela's Feedback

## Summary of Antonela's Feedback

1. ✅ **Pipeline ready** — Good progress!
2. 🔍 **Evaluation needed**: How to compare runs and determine bias effectively
3. 📊 **Metrics**: Check which metrics papers use and how they reflect model internals
4. 📝 **Report**: Need to write a report for practical work
5. 🎯 **Future**: Once bias is confirmed, thesis can focus on intervention/mitigation

---

## 1. Evaluation: Metrics You Already Have vs. Papers

### ✅ What You Already Implemented (Aligned with Papers)

| Metric | Your Code | Papers | Purpose |
|--------|-----------|--------|---------|
| **Accuracy by position** (A/B/C/D) | `accuracy_by_position()` | NAACL 2024, ACL 2025 | Shows if performance depends on where correct answer appears |
| **Sensitivity gap** (max − min accuracy) | `sensitivity_gap()` | NAACL 2024 | Quantifies overall positional bias magnitude |
| **Mean P(correct) by position** | `prob_correct_by_position()` | ACL 2024, ACL 2025 | Confidence/uncertainty varies by position |
| **Option proportion** (predicted vs GT) | `option_proportion()` | ACL 2024 | Distribution bias: does model favor certain positions? |
| **Failing-case analysis** | `error_prediction_proportion()`, `error_correct_position_proportion()` | ACL 2025 | Where does model predict when wrong? Where does it fail most? |
| **Attention to options** (by position) | `mean_attn_by_position()`, `attn_to_correct_vs_others()` | ACL 2025 | Internal mechanism: attention patterns |
| **Layer-wise accuracy & P(correct)** | `accuracy_by_layer()`, `prob_correct_by_layer()` | ACL 2025 | Where in the model does bias emerge? |

### 📊 How to Compare Runs Effectively

**A. Cross-model comparison** (same dataset):
- Compare **sensitivity gap** across models (OpenAI vs LLaMA vs Qwen)
- Compare **option proportion** — which models favor position A?
- Compare **failing-case patterns** — do all models fail similarly when correct is at D?

**B. Cross-dataset comparison** (same model):
- Does bias magnitude (sensitivity gap) differ between ARC-Challenge and OpenBookQA?
- Is option proportion consistent across datasets?

**C. Model internals** (HF models only):
- **Attention**: Do models attend more to position A regardless of content?
- **Layer-wise**: At which layer does positional bias start to appear? (early vs late layers)

---

## 2. Additional Metrics to Add (From Papers)

### Statistical Significance & Effect Size

**NAACL 2024** uses:
- **Effect size** (Cohen's d or similar) for sensitivity gap
- **Confidence intervals** for accuracy by position
- **Chi-square test** for option proportion (predicted vs uniform distribution)

**ACL 2025** uses:
- **Correlation** between attention to position A and prediction of A
- **Layer-wise trends** (at which layer does bias emerge?)

### Suggested Additions

1. **Statistical tests**:
   - Chi-square: Is option proportion significantly different from uniform (25% each)?
   - T-test or Mann-Whitney: Is accuracy at position A significantly different from D?

2. **Effect size**:
   - Cohen's d for sensitivity gap (small: <0.2, medium: 0.2–0.8, large: >0.8)

3. **Correlation metrics**:
   - Spearman correlation: attention to position A vs. prediction of A
   - Layer-wise: correlation between layer depth and bias magnitude

---

## 3. Report Structure (Practical Work)

### Suggested Outline

**1. Introduction**
- Problem: Position bias in LLMs for multiple-choice QA
- Motivation: Fair evaluation, understanding model behavior
- Research questions: Do models show positional bias? How is it reflected in internals?

**2. Related Work**
- NAACL 2024 (Pezeshkpour & Hruschka): Option-order sensitivity
- ACL 2024 (Wei et al.): Token vs order sensitivity
- ACL 2025 (Li & Gao): Anchored bias in GPT-2, mechanistic analysis

**3. Methodology**
- **Task**: Multiple-choice QA with option shuffling (correct at A/B/C/D)
- **Datasets**: ARC-Challenge, OpenBookQA
- **Models**: OpenAI (gpt-4.1-mini), LLaMA (Llama-3.2-1B), Qwen (Qwen2.5-0.5B)
- **Data collection**: For each item, create 4 variants (correct at A, B, C, D)
- **Model internals** (HF only): Attention weights, per-layer logits (logit lens)
- **Metrics**: Accuracy by position, sensitivity gap, option proportion, attention patterns, layer-wise analysis

**4. Results**
- **Table**: Summary table (model × dataset): N, overall accuracy, sensitivity gap, Acc(A/B/C/D)
- **Figures**:
  - Accuracy by position (bar chart per model/dataset)
  - Option proportion (predicted vs ground truth)
  - Sensitivity gap comparison (across models)
  - Attention heatmap (HF models only)
  - Layer-wise accuracy/P(correct) (HF models only)
- **Key findings**:
  - Which models show bias? (sensitivity gap > X%)
  - Which positions are favored? (option proportion)
  - Where does bias emerge? (layer-wise analysis)
  - How does bias manifest in errors? (failing-case)

**5. Discussion**
- Comparison with papers (e.g., NAACL 2024 reports 13–85% sensitivity gap)
- Model internals: Attention patterns, layer-wise emergence
- Limitations: OpenAI API doesn't expose internals, small models (1B/0.5B)

**6. Conclusion & Future Work**
- Summary of findings
- Future: Intervention/mitigation strategies (for thesis)

---

## 4. Next Steps (Action Items)

### Immediate (This Week)

1. **Run analysis** (if not done):
   ```bash
   uv run python scripts/analyze.py
   ```
   This generates `outputs/results/summary.json` and `outputs/results/analysis_report.md`.

2. **Review results**:
   - Open `outputs/results/summary.json` and check sensitivity gap for each run
   - Use the Streamlit dashboard: `uv run streamlit run streamlit_app.py`
   - Identify which models/datasets show the strongest bias

3. **Add statistical tests** (optional but recommended):
   - Add chi-square test for option proportion
   - Add effect size (Cohen's d) for sensitivity gap
   - Add confidence intervals for accuracy by position

### Short-term (Next 1–2 Weeks)

4. **Write the report**:
   - Use the structure above
   - Include tables and figures from `summary.json` and `outputs/figures/`
   - Compare your findings to papers (NAACL 2024, ACL 2025)

5. **Cross-model/dataset analysis**:
   - Create comparison tables: sensitivity gap across models, option proportion patterns
   - Identify patterns: Do all models favor A? Is bias stronger in certain datasets?

### Medium-term (For Thesis)

6. **Intervention/mitigation** (once bias is confirmed):
   - Test calibration methods (majority vote over 4 positions, as in NAACL 2024)
   - Test attention recalibration (as in ACL 2025)
   - Test prompt engineering (e.g., "Consider all options equally")

---

## 5. How to Determine Bias Effectively

### Decision Criteria (Based on Papers)

**Strong bias** if:
- Sensitivity gap > 10% (NAACL 2024 reports up to 85%)
- Option proportion for position A > 40% (should be ~25% if uniform)
- Accuracy at position A significantly higher than D (statistical test)

**Moderate bias** if:
- Sensitivity gap 5–10%
- Option proportion for A: 30–40%

**Weak/no bias** if:
- Sensitivity gap < 5%
- Option proportion close to uniform (25% each)

### Model Internals (HF Models)

**Bias in attention**:
- Attention to position A > attention to other positions (regardless of correctness)
- Attention to correct answer is lower when correct is at D vs A

**Bias in layers**:
- Bias emerges early (first few layers) → positional encoding issue
- Bias emerges late (final layers) → decision-making issue

---

## 6. Response to Antonela

**Draft email response**:

> Dear Antonela,
>
> Thank you for the feedback! I've completed the pipeline and have results for 3 models × 2 datasets (4800 samples total). I've implemented the evaluation metrics from the papers:
>
> - **Accuracy by position** and **sensitivity gap** (NAACL 2024)
> - **Option proportion** and **failing-case analysis** (ACL 2024, ACL 2025)
> - **Attention patterns** and **layer-wise analysis** (ACL 2025) for HF models
>
> I'm now analyzing the results to determine bias magnitude and patterns. I'll write the practical work report following the methodology and results structure.
>
> Regarding GPU: I'm currently using Docker on my machine with smaller models (0.5B–1B) due to RAM constraints. If institute GPU time becomes available, I can scale up to larger models (3B–7B) for more robust results.
>
> For thesis registration: I'll check KUSSS for the registration process and get back to you.
>
> Best regards,  
> Amer

---

## 7. Quick Reference: Key Metrics to Report

### Summary Table (from `summary.json`)

| Model | Dataset | N | Overall Acc | Sensitivity Gap | Acc(A) | Acc(B) | Acc(C) | Acc(D) |
|-------|---------|---|-------------|-----------------|--------|--------|--------|--------|
| gpt-4.1-mini | ARC | 800 | X% | Y% | ... | ... | ... | ... |
| ... | ... | ... | ... | ... | ... | ... | ... | ... |

### Key Findings to Highlight

1. **Sensitivity gap**: Which models show the largest gap? (compare to NAACL 2024: 13–85%)
2. **Option proportion**: Do models favor position A? (should be ~25% each if uniform)
3. **Model internals** (HF only): At which layer does bias emerge? Attention patterns?
4. **Cross-dataset**: Is bias consistent across ARC-Challenge and OpenBookQA?

---

## 8. Files You Have for the Report

- **`outputs/results/summary.json`**: All metrics per run
- **`outputs/results/analysis_report.md`**: Auto-generated report (tables + figure list)
- **`outputs/figures/*.png`**: Publication-ready figures (if you ran `analyze.py`)
- **Streamlit dashboard**: Interactive exploration (`uv run streamlit run streamlit_app.py`)

Use these to write your report!
