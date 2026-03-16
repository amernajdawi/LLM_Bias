# Research status and next steps (Amer & Antonela)

This document summarizes what the LLM_Bias project already does (aligned with Antonela’s suggestions and the three papers) and what to do next.

---

## 1. What we already did (implemented)

### Task and data (as agreed with Antonela)

- **Task**: Multiple-choice QA with **option shuffling**.
- **Shuffling**: For each question, the correct answer is placed at A, B, C, and D in turn (`build_permutations` in `src/data/shuffle.py`). So we get 4 variants per item (correct at position A, B, C, D).
- **Datasets**: Two datasets as Antonela suggested (not dataset-specific):
  - **ARC-Challenge** (`allenai/ai2_arc`, config `ARC-Challenge`)
  - **OpenBookQA** (`allenai/openbookqa`)
- **Models**: Different model families:
  - **OpenAI** (gpt-4.1-mini) — API, limited internals
  - **LLaMA** (Llama-3.2-1B-Instruct) — open-source, full internals
  - **Qwen** (Qwen2.5-0.5B or 3B) — open-source, full internals

### Model internals we collect (open-source HF models only)

- **Logprobs** (option-level): `logprobs` per result (A/B/C/D).
- **Attention to options**: `attn_to_options` — attention from the last token to each option position (A/B/C/D), **per layer** (list of dicts).
- **Probs per layer** (logit-lens style, as in ACL 2025): `probs_per_layer` — option probabilities at each layer (list of {A,B,C,D} per layer).

So we already implement: **attention over options**, **logits/logprobs**, and **layer-wise probing** (probs by layer).

### Analysis already implemented

| What Antonela / papers ask for | Where it is in the code |
|--------------------------------|-------------------------|
| **Accuracy by answer position** (A/B/C/D) | `src/analysis/metrics.py`: `accuracy_by_position` |
| **Sensitivity gap** (max − min accuracy over positions; NAACL 2024) | `src/analysis/metrics.py`: `sensitivity_gap` |
| **Probability of correct by position** (confidence vs position) | `src/analysis/metrics.py`: `prob_correct_by_position` |
| **Attention weights over options** | `src/analysis/attention_analysis.py`: `mean_attn_by_position`, `attn_to_correct_vs_others` |
| **Layer-wise analysis** (where bias emerges; ACL 2025 style) | `src/analysis/layerwise.py`: `accuracy_by_layer`, `prob_correct_by_layer` |
| **Option proportion** (pred vs GT) | `src/analysis/metrics.py`: `option_proportion`, `ground_truth_position_proportion` |
| **Failing-case analysis** (anchored bias in errors; ACL 2025) | `src/analysis/metrics.py`: `error_prediction_proportion`, `error_correct_position_proportion` |
| **Aggregate summary + figures + report** | `scripts/analyze.py`: summary per (model, dataset), `summary.json`, figures (acc/prob by position, option proportion, attention heatmap, layer-wise), `analysis_report.md` with methodology (ACL 2025 + NAACL 2024) |

### Storage

- Results (including `logprobs`, `attn_to_options`, `probs_per_layer`) are stored in **PostgreSQL** and/or **JSON** under `outputs/results/`.
- Each result has: `correct_label` (position A/B/C/D), `pred`, `correct`, `prob_correct`, `logprobs`, `attn_to_options`, `probs_per_layer`.

---

## 2. Link to the three papers

- **NAACL 2024** (Pezeshkpour & Hruschka): Sensitivity to **order of options** in MCQs; sensitivity gap; patterns (first+last vs adjacent); calibration (majority vote, MEC).  
  → We have: **option shuffling**, **accuracy by position**, **sensitivity gap**. We could add majority vote over our 4 orders per item.

- **ACL 2024** (Wei et al.): **Token vs order vs both** sensitivity; fluctuation rate; option proportion (prediction distribution A/B/C/D); gray-box (probability weighting/calibration), black-box two-hop.  
  → We have: **order** sensitivity (position A/B/C/D). We could add **option proportion** (distribution of predictions by A/B/C/D) from existing results.

- **ACL 2025** (Li & Gao, [Anchored Answers](https://aclanthology.org/2025.findings-acl.124/)): **Anchored bias** in GPT-2; **logit lens**; MLP/attention localization; analysis of **failing cases** to localize bias.  
  → We have: **probs per layer** (logit-lens style), **attention to options** per layer, and **failing-case metrics** (error_prediction_proportion, error_correct_position_proportion) to analyze where the model predicts when wrong and where the correct answer was when wrong.

---

## 3. What to do next (concrete steps)

### Step 1: Ensure experiments are complete

- Run (or re-run) so that all desired (model × dataset) runs exist:
  - `docker compose up -d` (and `LOW_MEMORY=1` if using Qwen 0.5B on limited RAM).
  - `docker compose exec app uv run python -u scripts/run_experiments.py`
- If you use the DB: export to JSON if needed:  
  `DATABASE_URL="..." uv run python scripts/export_db_to_files.py`

### Step 2: Run the analysis pipeline

- From project root (with `DATABASE_URL` set if using DB):  
  `uv run python scripts/analyze.py`
- This produces:
  - `outputs/results/summary.json`: per (model, dataset): accuracy by position, sensitivity gap, prob by position, attention by position, accuracy/prob by layer.
  - `outputs/figures/acc_by_pos_<model>_<dataset>.png`: bar plots of accuracy by correct position (A/B/C/D).

### Step 3: Inspect and extend the analysis

- **Sensitivity gap**: In `summary.json`, read `sensitivity_gap` for each model/dataset. Compare magnitude to NAACL 2024 (e.g. ~13–85% depending on model and dataset).
- **Attention**: Already computed; add plots if useful (e.g. heatmap: rows = correct position A/B/C/D, cols = attention to A/B/C/D; or bar charts per layer).
- **Layer-wise**: Use `accuracy_by_layer` and `prob_correct_by_layer` in `summary.json` to plot **accuracy vs layer** and **prob(correct) vs layer**, optionally by position, to see **where** positional bias emerges (ACL 2025 style).
- **Logits**: Use `prob_correct_by_position` and stored `logprobs` to analyze how **confidence** (prob/logprob of correct) varies with **position** (A/B/C/D).
- **Option proportion** (ACL 2024 Table 2): From existing results, compute the distribution of **predicted** labels (A/B/C/D) and compare to ground-truth position distribution; implement as a small function and add to `scripts/analyze.py` or a separate notebook.

### Step 4: Optional — calibration (NAACL 2024)

- For each item we have 4 orders (correct at A, B, C, D). You can add **majority vote** over the 4 predictions per item and report accuracy with and without calibration (and compare to single-order).

### Step 5: Write-up for Antonela

- **Method**: Task (MCQ + shuffling), datasets (ARC-Challenge, OpenBookQA), models (OpenAI, LLaMA, Qwen), metrics (accuracy by position, sensitivity gap, prob by position, attention by position, accuracy/prob by layer).
- **Results**: Tables and figures from `summary.json` and any new plots (sensitivity gap, attention, layer-wise, option proportion).
- **Related work**: Cite the three papers and briefly state how your setup and metrics align (order sensitivity, sensitivity gap, logits, layer-wise analysis).

---

## 4. Short summary

| Done | Next |
|------|------|
| MCQ with option shuffling (correct at A/B/C/D) | Run experiments to completion; run `analyze.py` |
| Two datasets (ARC-Challenge, OpenBookQA) | Use summary + figures in the write-up |
| Three model families (OpenAI, LLaMA, Qwen) | Optional: add more models/datasets later |
| Logprobs, attention to options, probs per layer stored | Plot layer-wise curves; analyze where bias emerges |
| Accuracy by position, sensitivity gap, prob by position | Report sensitivity gap; compare to papers |
| Attention by position, correct vs others | Add attention heatmaps/plots if needed |
| Accuracy and prob by layer | Add layer-wise plots and short analysis |
| Summary JSON + accuracy-by-position figures | Option proportion; optional majority vote; write-up |

You are in a good position: the pipeline and analyses Antonela asked for (attention, logits, layer-wise) are implemented. The main next steps are: **run the pipeline**, **run the analysis**, **add a few plots/reports** (layer-wise, option proportion), and **write the method and results** for Antonela.
