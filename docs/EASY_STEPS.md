# What to do – local and Railway

## On your machine (local)

### 1. Use the dashboard with your local database
- In `.env` set: `DATABASE_URL=postgresql://llmbias:llmbias@localhost:5433/llmbias`
- Start Postgres (if you use Docker): `docker compose up -d`
- Run the dashboard: `uv run streamlit run streamlit_app.py`
- Open the URL shown (e.g. http://localhost:8501). You’ll see all runs and results from your local DB.

### 2. Export local DB to CSV (to send data to Railway)
- Make sure `.env` has your **local** `DATABASE_URL` (or leave it as above).
- In the project folder run:
  ```bash
  cd /Users/ameralnajdawi/Desktop/LLM_Bias
  uv run python -m scripts.export_results_to_csv --dir results_and_runs
  ```
- You’ll see something like: `Exported 15 runs and 12000 results to ... results_and_runs`

### 3. Import that CSV into Railway (so Railway has the same data)
- Get your **Railway** URL: Railway project → **PostgreSQL** service → **Variables** → copy **DATABASE_PUBLIC_URL**.
- In the **same** terminal, run these two commands (paste your real Railway URL in the first one):
  ```bash
  export DATABASE_URL="postgresql://postgres:YOUR_PASSWORD@HOST:PORT/railway"
  uv run python -m scripts.import_results_from_csv --dir results_and_runs
  ```
- Wait until it prints “Imported X runs and Y results.”

---

## On Railway

### 1. Deploy the app
- Push your code to GitHub. Railway will build and deploy from the connected repo.
- In the **LLM_Bias** app service, set **Variables** → add **DATABASE_URL** from the Postgres service (Reference or paste the value).

### 2. Make the app public
- In the app service → **Settings** → **Networking** → **Generate Domain**. Use the URL Railway gives you.

### 3. Put data in the database
- Data does **not** come from CSV on Railway. You put data in Railway’s Postgres by running the **import on your machine** (step 3 under “On your machine” above) with `DATABASE_URL` set to Railway’s URL.
- After the import finishes, **refresh** the Railway dashboard in your browser. You should see all the runs and results.

### 4. If you add more data later
- Export from local again (step 2 above), then run the import again with Railway’s `DATABASE_URL` (step 3 above). Re-running the import is safe; it skips rows that already exist.

---

## Quick reference

| Goal                         | Where        | What to run |
|-----------------------------|-------------|-------------|
| View local data             | Your machine | `uv run streamlit run streamlit_app.py` (with local `DATABASE_URL` in `.env`) |
| Export local DB → CSV       | Your machine | `uv run python -m scripts.export_results_to_csv --dir results_and_runs` |
| Import CSV → Railway DB     | Your machine | `export DATABASE_URL="<Railway URL>"` then `uv run python -m scripts.import_results_from_csv --dir results_and_runs` |
| View Railway data           | Browser      | Open your Railway app URL and refresh after importing |
