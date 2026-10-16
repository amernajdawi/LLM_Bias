# Position bias in LLMs on multiple-choice QA

Measures how much LLMs prefer certain answer positions on ARC-Challenge and OpenBookQA (200 questions each),
and tests methods that reduce this bias. Every question is asked in all four rotations of its options,
with letters (A–D) and numbers (1–4), and answers are read strictly (an unreadable reply is never counted as "A").

## Layout

| Path | What it is |
|---|---|
| `experiments/v3_corrected/run_cloud.py` | Runs models through Ollama Cloud, a local Ollama server or the OpenAI API (text answers) |
| `experiments/v3_corrected/run_hf.py` | Runs Hugging Face models, answers read from label probabilities |
| `experiments/v3_corrected/run_ollama_probs.py` | Runs local Ollama models, answers read from label probabilities |
| `experiments/v3_corrected/analyze_v3.py`, `analyze_hf.py` | Bias metrics and every mitigation method, with significance tests |
| `experiments/v3_corrected/build_report.py` | HTML report |
| `experiments/v3_corrected/export_dashboard.py`, `dashboard_v3.py` | Dashboard data export and the Streamlit dashboard |
| `experiments/v3_corrected/raw/`, `raw_hf/` | Every saved answer |
| `experiments/v3_pav/offline_eval.py` | PAV (position-aware voting) |
| `results_and_runs/` | Earlier runs, used for the old-vs-new comparison |

## Use

```bash
cp .env.example .env            # add OLLAMA_API_KEY, OPENAI_API_KEY, HF_TOKEN as needed
uv sync

uv run python experiments/v3_corrected/run_cloud.py --check --models gpt-oss:20b      # is a model available?
uv run python experiments/v3_corrected/run_cloud.py --models gpt-oss:20b               # Ollama Cloud
uv run python experiments/v3_corrected/run_hf.py --model Qwen/Qwen2.5-0.5B-Instruct    # Hugging Face

bash experiments/v3_corrected/update_results.sh                 # analyses + report
uv run python experiments/v3_corrected/export_dashboard.py      # dashboard data
uv run streamlit run dashboard_v3.py
```

The `Dockerfile` runs the dashboard (used for the Railway deployment).
