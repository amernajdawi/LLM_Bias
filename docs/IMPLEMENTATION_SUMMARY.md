# Implementation Summary: ACL 2025 Paper Metrics

## What We Added

Based on the ACL 2025 paper "Anchored Answers: Unravelling Positional Bias in GPT-2's Multiple-Choice Questions" by Li & Gao, we've added the following metrics and analyses:

### 1. **Anchored Bias Frequency** ✅

**Location**: `src/analysis/metrics.py::anchored_bias_frequency()`

**What it does**: 
- Calculates the % of samples where model predicts 'A' (first position) but correct answer is NOT 'A'
- This is the paper's core metric for measuring "anchored bias"

**Example**: If 97.4% of samples where correct≠'A' are predicted as 'A', that's strong anchored bias.

**Usage**: Already integrated into analysis script and report.

---

### 2. **Logit Difference by Layer** ✅

**Location**: `src/analysis/layerwise.py::logit_difference_by_layer()`

**What it does**:
- For each layer, calculates: `logit['A'] - logit[correct]`
- Only computed for samples where correct ≠ 'A' (failing cases)
- Positive values indicate bias toward 'A' at that layer
- Identifies which layers contribute most to anchored bias

**How it works**:
- Uses `probs_per_layer` data (already collected)
- Converts probabilities to logits: `logit = log(prob)`
- Computes difference: `logit_anchor - logit_correct`
- Averages across all failing cases

**Usage**: Already integrated into analysis script and report (Section 5).

---

### 3. **Failing-Case Filter Function** ✅

**Location**: `src/analysis/metrics.py::failing_cases_where_predicted_anchor()`

**What it does**:
- Filters results to cases where model predicted 'A' but correct is not 'A'
- Aligns with paper's focus on analyzing failing cases

**Usage**: Available for future analysis (not yet integrated into main report).

---

## Updated Files

1. **`src/analysis/metrics.py`**
   - Added `anchored_bias_frequency()`
   - Added `failing_cases_where_predicted_anchor()`

2. **`src/analysis/layerwise.py`**
   - Added `logit_difference_by_layer()`

3. **`scripts/analyze.py`**
   - Updated `_summary_one()` to compute new metrics
   - Updated `_write_report()` to include:
     - Anchored bias frequency in summary table
     - New section "Anchored bias frequency (ACL 2025)"
     - New section "Logit difference by layer (ACL 2025)"

---

## How This Answers Antonela's Question

Antonela asked: *"How to compare runs and effectively determine whether there's bias or not, and how is that reflected on the model internals."*

**Now you can answer:**

1. **Determine bias**: 
   - **Anchored bias frequency** shows % of samples where model incorrectly favors 'A'
   - **Sensitivity gap** shows how much accuracy varies by position
   - **Chi-square test** shows if option distribution is significantly biased

2. **Compare runs**:
   - Summary table now includes anchored bias frequency for all model×dataset combinations
   - Can directly compare which models/datasets show strongest anchored bias

3. **Model internals**:
   - **Logit difference by layer** shows which layers contribute most to bias
   - **Layer-wise accuracy/probability** shows where bias emerges
   - **Attention to options** shows which positions the model attends to

---

## Next Steps

1. **Run analysis**: `uv run python scripts/analyze.py`
   - This will compute all new metrics and update the report

2. **Check the report**: `outputs/analysis_report.md`
   - Section 2: Anchored bias frequency
   - Section 5: Logit difference by layer

3. **Update dashboard** (optional):
   - Add anchored bias frequency to Streamlit dashboard
   - Add logit difference by layer visualization

---

## Paper Alignment

| Paper Metric | Our Implementation | Status |
|--------------|-------------------|--------|
| Anchored bias frequency (Table 2) | `anchored_bias_frequency()` | ✅ |
| Logit difference by layer (Fig. 3) | `logit_difference_by_layer()` | ✅ |
| Failing-case focus (§ 5) | `failing_cases_where_predicted_anchor()` | ✅ |
| MLP contribution (Eq. 7) | Not implemented (requires MLP weights) | ⚠️ |
| Attention per-head analysis (Fig. 4) | Partial (we aggregate across heads) | ⚠️ |

**Note**: MLP contribution and per-head attention analysis require deeper model internals access. These are advanced features that may not be necessary for your evaluation.

---

## Example Output

After running analysis, you'll see in the report:

```
## 2. Anchored bias frequency (ACL 2025)

| Model × Dataset | Anchored bias frequency | Interpretation |
|-----------------|------------------------|----------------|
| Qwen/Qwen2.5-0.5B-Instruct_allenai_ai2_arc | 45.2% | Moderate anchored bias |
| ... | ... | ... |

## 5. Logit difference by layer (ACL 2025)

### Qwen/Qwen2.5-0.5B-Instruct_allenai_ai2_arc
| Layer | Logit diff (A - correct) | Interpretation |
|-------|--------------------------|----------------|
| 0 | 0.123 | Weak bias toward 'A' |
| 1 | 0.456 | Moderate bias toward 'A' |
| ... | ... | ... |
```

This directly answers: **"Which layers contribute most to anchored bias?"**
