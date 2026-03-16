# Position Bias in LLMs

Experiment pipeline: multiple-choice QA with shuffled option order (correct at A/B/C/D), then analysis of accuracy, logprobs, attention, and layer-wise behaviour. Uses **UV**, **PostgreSQL**, and **Docker**.

---

## What a clone gets (and what they don’t)

If you **push this repo** and someone **clones it**:

| In the repo (they get it) | Not in the repo (they don’t get it) |
|---------------------------|--------------------------------------|
| All code (`src/`, `scripts/`, `config.yaml`, `Dockerfile`, `docker-compose.yml`, `pyproject.toml`) | `.env` (secrets) |
| Pipeline logic and config | Your **database data** (runs/results live in Docker volume, not in git) |
| Same models/datasets in config | Your **result files** (`outputs/results/*.json`, `outputs/figures/*.png` are gitignored) |
| Scripts: run_experiments, analyze, export_db, import_json, etc. | Pre-computed results or DB dumps |

So after a clone they have:

- **Empty database** (Postgres starts fresh in Docker).
- **No result JSON/figures** (outputs are gitignored).
- **Same code and config** → same pipeline and same **structure** of results once they run it.

They must **run the pipeline themselves** (and set `.env`) to get results. Optionally, you can share exported data (see below) so they can load it without re-running experiments.

---

## Setup (for you or someone who cloned)

1. **Clone and go to the project**
   ```bash
   cd LLM_Bias
   ```

2. **Create `.env` from the example**
   ```bash
   cp .env.example .env
   ```
   Edit `.env` and set:
   - `OPENAI_API_KEY` – required for the OpenAI model.
   - `HF_TOKEN` – required for LLaMA/Qwen (and Mistral if enabled). Get it from [Hugging Face](https://huggingface.co/settings/tokens) and accept the model licenses.
   - `DATABASE_URL` is set by Docker for the app; you only need it if you run scripts on the host (e.g. `postgresql://llmbias:llmbias@localhost:5433/llmbias`).

3. **Start services**
   ```bash
   docker compose up -d
   ```

4. **Run experiments** (downloads data/models, runs all configured models/datasets, writes DB + JSON)
   ```bash
   docker compose exec app uv run python -u scripts/run_experiments.py
   ```

5. **Run analysis** (reads from DB or JSON, writes summary + figures)
   ```bash
   docker compose exec app uv run python scripts/analyze.py
   ```

Results appear in:

- **Database:** tables `runs` and `results` (see “DB” below).
- **Files:** `outputs/results/*.json`, `outputs/results/summary.json`, `outputs/figures/*.png`.

6. **View the dashboard** (after experiments and optionally after analysis for figures)

   **Option A – dashboard on your machine** (uses the same Postgres via port 5433)
   - In project root `.env`, set: `DATABASE_URL=postgresql://llmbias:llmbias@localhost:5433/llmbias`
   - Then run:
     ```bash
     uv run streamlit run streamlit_app.py
     ```
   - Open the URL shown (e.g. http://localhost:8501).

   **Option B – dashboard inside Docker** (same network as Postgres, no .env change)
   - With `docker compose up -d` running:
     ```bash
     docker compose exec app uv run streamlit run streamlit_app.py --server.address=0.0.0.0
     ```
   - Open http://localhost:8502 in your browser (port 8502 is used so it doesn’t clash with a Streamlit already on 8501).

   The dashboard reads from the database. For the “Figures” tab, run step 5 (analyze) first; figures are saved under `outputs/figures/`.

---

## Exporting DB so someone else can load it

If you want someone who cloned to **start from your results** without re-running experiments:

1. **Export DB to files** (run in the project that has the DB with data):
   ```bash
   docker compose exec app uv run python scripts/export_db_to_files.py
   ```
   This writes:
   - `outputs/db_export/runs.json`, `outputs/db_export/results.json`
   - `outputs/db_export/runs.csv`, `outputs/db_export/results.csv`

2. **Commit and push the export** (if you want it in the repo):
   - Either add `outputs/db_export/` to the repo (remove it from `.gitignore` for that folder), or  
   - Share the files another way (e.g. Zenodo, Drive).

3. **Other person: clone, then either**
   - **Option A – run pipeline:** set `.env`, `docker compose up -d`, then run `run_experiments.py` and `analyze.py` (they get their own results), or  
   - **Option B – load your export:** you’d need a small script that reads `runs.json`/`results.json` (or CSV) and inserts into their empty DB; we can add that if you want.

---

## DB (PostgreSQL)

- **From host:** `localhost:5433`, database `llmbias`, user/password `llmbias`.
- **From app container:** `postgres:5432` (set via `DATABASE_URL` in docker-compose).
- **Tables:** `runs` (id, model, dataset, created_at), `results` (run_id, item_id, correct_label, pred, correct, prob_correct, logprobs, attn_to_options, probs_per_layer).

---

## Config

- `config.yaml`: models (openai, llama, qwen, mistral), datasets (ARC-Challenge, OpenBookQA), `max_samples_per_dataset`, prompt template.
- **Scripts:** `run_experiments.py`, `analyze.py`, `export_db_to_files.py`, `import_json_to_db.py`, `check_db.py`, `dedupe_db.py`, `dedupe_runs.sql`.

---

## Summary

- **Push + clone:** they get the **same code and config**; **no** your DB or result files.
- **Results they get:** same **pipeline and result structure** after they set `.env`, run `docker compose up`, then `run_experiments.py` and `analyze.py`.
- **Optional:** you export DB to `outputs/db_export/` (JSON/CSV) and share that so they can load a copy of your results instead of re-running.
