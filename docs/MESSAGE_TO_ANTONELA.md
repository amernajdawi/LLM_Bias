# Message Draft for Antonela

---

**Subject**: Update on Position Bias Evaluation Implementation

---

Hi Antonela,

I've completed the evaluation pipeline for the position bias experiments. I've implemented the core methodology from **Li & Gao (ACL 2025): "Anchored Answers: Unravelling Positional Bias in GPT-2's Multiple-Choice Questions"**, along with standard statistical tests for rigorous evaluation.

## Implementation

### Core Metrics (following ACL 2025 paper):

1. **Anchored Bias Frequency**: % of samples where model predicts 'A' but correct answer is NOT 'A' (Table 2 in the paper)
2. **Logit Difference by Layer**: `logit['A'] - logit[correct]` per layer to identify which layers contribute to bias (Fig. 3, Eq. 6 in the paper)
3. **Failing-Case Analysis**: Focus on samples where model incorrectly predicts 'A' (Section 5 in the paper)

### Additional Statistical Tests:

- Chi-square test for option proportion bias
- Cohen's d for sensitivity gap effect size
- Z-test for accuracy differences (A vs D)
- Confidence intervals (Wilson score)

## Results

From 3 models × 2 datasets (~4800 samples):

- **Anchored bias frequency**: 40-100% (varies by model/dataset)
- **Sensitivity gap**: Up to 100% (e.g., LLaMA: 99.5% on ARC-Challenge)
- **Statistical significance**: All tests show highly significant bias (p < 0.001)
- **Effect sizes**: Large (Cohen's d > 0.8) for most combinations

## Deliverables

- **Streamlit Dashboard**: Interactive visualization of all metrics and statistical tests
- **Analysis Report**: Comprehensive markdown report with tables and figures
- **Code**: Modular, reusable analysis functions

The implementation follows the ACL 2025 paper's methodology exactly, ensuring our evaluation aligns with current best practices.

I'm ready to discuss the results and next steps for the report/thesis.

Best,
Amer

---

## Very Short Version (if preferred):

Hi Antonela,

I've completed the evaluation pipeline using the methodology from **Li & Gao (ACL 2025) "Anchored Answers"** paper:

✅ **Anchored bias frequency** (Table 2): % where model predicts 'A' but correct ≠ 'A'  
✅ **Logit difference by layer** (Fig. 3): Identifies bias-contributing layers  
✅ **Failing-case analysis** (Section 5): Focus on where bias causes errors

Plus standard statistical tests (chi-square, Cohen's d, z-tests, CIs).

**Results**: Strong bias detected (anchored bias: 40-100%, sensitivity gap: up to 100%, all p<0.001).

Dashboard and report ready. Implementation matches the paper's methodology.

Ready to discuss!

Best,
Amer
