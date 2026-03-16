# Analysis Report — Position Bias in Multiple-Choice QA

## Methodology

This analysis follows best-practice evaluation for positional bias in LLM multiple-choice QA:

- **Accuracy by position (A/B/C/D)** and **sensitivity gap** (max − min accuracy over positions) quantify how much performance depends on where the correct answer appears (Pezeshkpour & Hruschka, NAACL 2024 Findings; [Large Language Models Sensitivity to The Order of Options in Multiple-Choice Questions](https://aclanthology.org/2024.findings-naacl.130/)).
- **Option proportion** (predicted vs ground-truth) and **failing-case analysis** (where the model predicts when wrong, and where the correct answer was when wrong) capture anchored/positional bias in errors (Li & Gao, ACL 2025 Findings; [Anchored Answers: Unravelling Positional Bias in GPT-2's Multiple-Choice Questions](https://aclanthology.org/2025.findings-acl.124/)).
- **Layer-wise accuracy and P(correct)** (logit-lens style) and **attention to options** support mechanistic analysis of where bias emerges (ACL 2025).

---

## 1. Summary table (all model × dataset)

| Model × Dataset | N | Overall Acc | Sensitivity gap | Anchored bias | Acc(A) | Acc(B) | Acc(C) | Acc(D) |
|-----------------|---|-------------|-----------------|---------------|--------|--------|--------|--------|
| qwen_allenai_ai2_arc | 800 | 42.8% | 48.5% | 41.8% | 73.0% | 33.0% | 24.5% | 40.5% |
| qwen_allenai_openbookqa | 800 | 39.0% | 71.0% | 64.0% | 87.5% | 23.5% | 16.5% | 28.5% |

## 2. Anchored bias frequency (ACL 2025)

**Anchored bias frequency**: % of samples where model predicts 'A' (first position) but correct answer is NOT 'A'.
This directly measures the "anchored bias" phenomenon identified in Li & Gao (ACL 2025).

| Model × Dataset | Anchored bias frequency | Interpretation |
|-----------------|------------------------|----------------|
| qwen_allenai_ai2_arc | 41.8% | Moderate anchored bias |
| qwen_allenai_openbookqa | 64.0% | Strong anchored bias (model heavily favors 'A') |

## 3. Option proportion (prediction bias)

Ground truth is balanced (correct at A/B/C/D equally). Model predictions may favor certain positions.

### qwen_allenai_ai2_arc
| Position | Ground truth | Model pred |
|----------|--------------|------------|
| A | 25.0% | 49.6% |
| B | 25.0% | 15.6% |
| C | 25.0% | 11.0% |
| D | 25.0% | 23.8% |

### qwen_allenai_openbookqa
| Position | Ground truth | Model pred |
|----------|--------------|------------|
| A | 25.0% | 69.9% |
| B | 25.0% | 9.8% |
| C | 25.0% | 6.5% |
| D | 25.0% | 13.9% |

## 4. Statistical tests

### 3.1 Option proportion significance (Chi-square test)

Tests whether predicted option distribution differs significantly from uniform (25% each).

| Model × Dataset | χ² | p-value | Significant? | Interpretation |
|-----------------|----|---------|--------------|----------------|
| qwen_allenai_ai2_arc | 285.39 | 0.050 | Yes | Bias detected |
| qwen_allenai_openbookqa | 867.95 | 0.050 | Yes | Bias detected |

### 3.2 Sensitivity gap effect size (Cohen's d)

Effect size for sensitivity gap: small (<0.2), medium (0.2–0.8), large (>0.8).

| Model × Dataset | Sensitivity gap | Cohen's d | Effect size |
|-----------------|-----------------|-----------|-------------|
| qwen_allenai_ai2_arc | 48.5% | 0.98 | Large |
| qwen_allenai_openbookqa | 71.0% | 1.46 | Large |

### 3.3 Accuracy by position: A vs D significance test

Two-proportion z-test comparing accuracy when correct is at A vs D.

| Model × Dataset | Acc(A) | Acc(D) | z-score | p-value | Significant? |
|-----------------|--------|--------|---------|---------|--------------|
| qwen_allenai_ai2_arc | 73.0% | 40.5% | 6.56 | 0.050 | Yes |
| qwen_allenai_openbookqa | 87.5% | 28.5% | 11.95 | 0.050 | Yes |

### 3.4 Accuracy confidence intervals (95% CI)

Wilson score intervals for accuracy by position.

### qwen_allenai_ai2_arc
| Position | Accuracy | 95% CI (lower) | 95% CI (upper) |
|----------|----------|----------------|-----------------|
| A | 73.0% | 66.5% | 78.7% |
| B | 33.0% | 26.9% | 39.8% |
| C | 24.5% | 19.1% | 30.9% |
| D | 40.5% | 33.9% | 47.4% |

### qwen_allenai_openbookqa
| Position | Accuracy | 95% CI (lower) | 95% CI (upper) |
|----------|----------|----------------|-----------------|
| A | 87.5% | 82.2% | 91.4% |
| B | 23.5% | 18.2% | 29.8% |
| C | 16.5% | 12.0% | 22.3% |
| D | 28.5% | 22.7% | 35.1% |

## 5. Logit difference by layer (ACL 2025)

**Logit difference**: logit['A'] - logit[correct] per layer, computed only for samples where correct ≠ 'A'.
Positive values indicate bias toward 'A' at that layer. This identifies which layers contribute most to anchored bias.

### qwen_allenai_ai2_arc
| Layer | Logit diff (A - correct) | Interpretation |
|-------|--------------------------|----------------|
| 0 | 0.005 | Weak bias toward 'A' |
| 1 | 0.037 | Weak bias toward 'A' |
| 2 | 0.100 | Weak bias toward 'A' |
| 3 | 0.162 | Weak bias toward 'A' |
| 4 | 0.149 | Weak bias toward 'A' |
| 5 | 0.255 | Weak bias toward 'A' |
| 6 | 0.235 | Weak bias toward 'A' |
| 7 | 0.229 | Weak bias toward 'A' |
| 8 | 0.330 | Weak bias toward 'A' |
| 9 | 0.308 | Weak bias toward 'A' |
| ... | (15 more layers) | |

### qwen_allenai_openbookqa
| Layer | Logit diff (A - correct) | Interpretation |
|-------|--------------------------|----------------|
| 0 | 0.005 | Weak bias toward 'A' |
| 1 | 0.037 | Weak bias toward 'A' |
| 2 | 0.097 | Weak bias toward 'A' |
| 3 | 0.155 | Weak bias toward 'A' |
| 4 | 0.147 | Weak bias toward 'A' |
| 5 | 0.252 | Weak bias toward 'A' |
| 6 | 0.234 | Weak bias toward 'A' |
| 7 | 0.230 | Weak bias toward 'A' |
| 8 | 0.329 | Weak bias toward 'A' |
| 9 | 0.305 | Weak bias toward 'A' |
| ... | (15 more layers) | |

## 6. Failing-case analysis (anchored bias in errors)

Among **errors only**: proportion of predictions at each position (model bias when wrong) and proportion where the correct answer was at each position (where the model fails most).

### qwen_allenai_ai2_arc (n_errors = 458)
| Position | % of errors predicted at | % of errors where correct at |
|----------|---------------------------|-------------------------------|
| A | 54.8% | 11.8% |
| B | 12.9% | 29.3% |
| C | 8.5% | 33.0% |
| D | 23.8% | 26.0% |

### qwen_allenai_openbookqa (n_errors = 488)
| Position | % of errors predicted at | % of errors where correct at |
|----------|---------------------------|-------------------------------|
| A | 78.7% | 5.1% |
| B | 6.4% | 31.4% |
| C | 3.9% | 34.2% |
| D | 11.1% | 29.3% |

## 7. Figures

All figures are in `outputs/figures/` (or `figures_dir` in config).

- **qwen_allenai_ai2_arc**
  - `acc_by_pos_qwen_allenai_ai2_arc.png` — Accuracy when correct answer is at A/B/C/D
  - `prob_by_pos_qwen_allenai_ai2_arc.png` — Mean P(correct) by position
  - `option_proportion_qwen_allenai_ai2_arc.png` — Predicted vs ground-truth option proportion
  - `attn_heatmap_qwen_allenai_ai2_arc.png` — Attention to options (last layer)
  - `layer_wise_qwen_allenai_ai2_arc.png` — Layer-wise accuracy and P(correct)
- **qwen_allenai_openbookqa**
  - `acc_by_pos_qwen_allenai_openbookqa.png` — Accuracy when correct answer is at A/B/C/D
  - `prob_by_pos_qwen_allenai_openbookqa.png` — Mean P(correct) by position
  - `option_proportion_qwen_allenai_openbookqa.png` — Predicted vs ground-truth option proportion
  - `attn_heatmap_qwen_allenai_openbookqa.png` — Attention to options (last layer)
  - `layer_wise_qwen_allenai_openbookqa.png` — Layer-wise accuracy and P(correct)