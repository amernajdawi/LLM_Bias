# Analysis of ACL 2025 Paper: "Anchored Answers: Unravelling Positional Bias in GPT-2's Multiple-Choice Questions"

## Paper Summary

**Authors**: Ruizhe Li, Yanjun Gao (ACL 2025 Findings)  
**Key Finding**: GPT-2 models exhibit "anchored bias" — they consistently favor the first choice 'A' regardless of where the correct answer is placed.

---

## Key Methodologies from the Paper

### 1. **Anchored Bias Frequency** (Table 2)
- **Metric**: % of samples where model predicts 'A' when correct answer is NOT 'A'
- **Purpose**: Quantifies how often the model anchors to position A
- **Example**: GPT2-Medium shows 97.4% anchored bias on IOI dataset

### 2. **Logit Lens Analysis** (§ 5)
- **MLP logit difference**: `logit_MLP[A] - logit_MLP[correct]` per layer
- **Attention logit difference**: `logit_attn[A] - logit_attn[correct]` per head
- **Purpose**: Identifies which layers/heads contribute most to anchored bias
- **Finding**: Specific layers (e.g., layer 9 in GPT2-Small, layer 20 in Medium) dominate

### 3. **MLP Contribution Analysis** (Eq. 7)
- **Formula**: `Contrib(v) = |k| * ||v||` where k is coefficient, v is value vector
- **Purpose**: Identifies specific value vectors in MLP that store anchored bias
- **Finding**: Specific vectors (e.g., v9,1853 in GPT2-Small) unembed to tokens like "A", "a", "The"

### 4. **Attention Pattern Analysis** (Fig. 4, 5)
- **Per-head analysis**: Which attention heads attend more to position 'A' vs correct position
- **Visualization**: Attention weights from final token to option positions
- **Finding**: Specific heads (e.g., L8H1, L10H8 in GPT2-Small) favor 'A'

### 5. **Full Circuit Visualization** (Fig. 6, 11-13)
- **Shows**: All MLP layers and attention heads contributing to anchored bias
- **Threshold**: Components with logit difference > 4
- **Purpose**: Complete picture of where bias emerges in the model

### 6. **Failing-Case Focus**
- **Key insight**: Paper focuses on FAILING cases (where model predicts 'A' but correct is B/C/D/E)
- **Rationale**: Success cases don't reveal bias; failures show where bias causes errors

---

## What We Already Have vs. What We Should Add

### ✅ What We Already Have

| Paper Metric | Our Implementation | Status |
|--------------|-------------------|--------|
| Accuracy by position | `accuracy_by_position()` | ✅ |
| Option proportion | `option_proportion()` | ✅ |
| Failing-case analysis | `error_prediction_proportion()` | ✅ |
| Attention to options | `mean_attn_by_position()` | ✅ |
| Layer-wise probs | `probs_per_layer` + `prob_correct_by_layer()` | ✅ |
| Layer-wise accuracy | `accuracy_by_layer()` | ✅ |

### ❌ What We're Missing (Should Add)

| Paper Metric | What It Does | Why Important |
|--------------|--------------|---------------|
| **Anchored bias frequency** | % samples where pred='A' but correct≠'A' | Direct measure of "anchored bias" (paper's main finding) |
| **Logit difference by layer** | `logit[A] - logit[correct]` per layer | Identifies which layers contribute most to bias |
| **Attention logit difference per head** | `logit_attn[A] - logit_attn[correct]` per head | Identifies specific attention heads causing bias |
| **MLP contribution ranking** | `|k| * ||v||` to rank value vectors | Identifies specific MLP dimensions storing bias |
| **Failing-case logit analysis** | Logit differences only for samples where model fails | Focuses analysis on where bias actually causes errors |

---

## Recommended Additions to Our Analysis

### 1. **Anchored Bias Frequency Metric**

```python
def anchored_bias_frequency(results: List[dict], letters: List[str]) -> float:
    """
    % of samples where model predicts 'A' (first position) 
    but correct answer is NOT 'A'.
    This is the paper's core metric for "anchored bias".
    """
    anchored_count = 0
    total_non_a = 0
    for r in results:
        correct = r.get("correct_label")
        pred = r.get("pred")
        if correct and correct != letters[0]:  # correct is NOT 'A'
            total_non_a += 1
            if pred == letters[0]:  # but model predicted 'A'
                anchored_count += 1
    return anchored_count / total_non_a if total_non_a > 0 else 0.0
```

### 2. **Logit Difference by Layer** (for HF models)

```python
def logit_difference_by_layer(
    results: List[dict], 
    letters: List[str],
    anchor_pos: str = "A"
) -> List[Dict[str, float]]:
    """
    For each layer, calculate: logit[anchor_pos] - logit[correct]
    averaged over samples where correct != anchor_pos.
    Positive values = bias toward anchor_pos.
    """
    # Use probs_per_layer, convert to logits, compute difference
    # Return: [{layer_idx: diff_value}, ...]
```

### 3. **Attention Logit Difference per Head**

We'd need per-head attention data. Currently we aggregate across heads. If available:
- Calculate logit difference per head: `logit_head[A] - logit_head[correct]`
- Identify heads with largest positive differences (bias toward 'A')

### 4. **Failing-Case Filtered Analysis**

Focus metrics on samples where:
- Model predicts 'A' but correct is B/C/D/E (anchored bias failures)
- This reveals where bias actually causes errors

---

## Implementation Priority

### **High Priority** (Core to paper's findings):

1. ✅ **Anchored bias frequency** — Add this metric (easy, uses existing data)
2. ✅ **Logit difference by layer** — Use `probs_per_layer` to compute (medium effort)
3. ✅ **Failing-case filtered metrics** — Filter results where pred='A' and correct≠'A' (easy)

### **Medium Priority** (Nice to have):

4. **Attention per-head analysis** — Requires per-head attention data (may need code changes)
5. **MLP contribution ranking** — Requires access to MLP weights (advanced, may not be feasible)

### **Low Priority** (Visualization):

6. **Full circuit visualization** — Show all contributing layers/heads (nice but not essential)

---

## How This Aligns with Antonela's Feedback

Antonela asked: *"How to compare runs and effectively determine whether there's bias or not, and how is that reflected on the model internals."*

**The paper answers this by:**

1. **Determining bias**: Anchored bias frequency (% predicts 'A' when shouldn't)
2. **Comparing runs**: Table 2 shows frequency across models/datasets
3. **Model internals**: Logit difference analysis shows which layers/heads contribute

**We should add:**
- Anchored bias frequency metric (easy, high value)
- Logit difference by layer visualization (medium effort, high value)
- Focus analysis on failing cases (easy, aligns with paper's approach)

---

## Next Steps

1. **Add anchored bias frequency** to `src/analysis/metrics.py`
2. **Add logit difference by layer** analysis (using `probs_per_layer`)
3. **Update dashboard/report** to show anchored bias frequency prominently
4. **Add failing-case filtered analysis** (optional but aligns with paper)
