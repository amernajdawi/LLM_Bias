import argparse
import csv
import json
from pathlib import Path
from typing import Any, Dict, List

# Load .env so DATABASE_URL is set (project root, then cwd; shell export overrides)
ROOT = Path(__file__).resolve().parent.parent
try:
    from dotenv import load_dotenv
    load_dotenv(ROOT / ".env")
    load_dotenv(Path.cwd() / ".env")
except Exception:
    pass

from src.db.client import _conn, create_schema


def load_runs_csv(path: Path) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            if not row.get("id"):
                continue
            rows.append(
                {
                    "id": row["id"].strip().strip('"'),
                    "model": (row.get("model") or "").strip(),
                    "dataset": (row.get("dataset") or "").strip(),
                    "created_at": (row.get("created_at") or "").strip(),
                }
            )
    return rows


def load_results_csv(path: Path) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            if not row.get("id"):
                continue
            def _as_int(val: Any) -> int | None:
                if val is None or val == "":
                    return None
                try:
                    return int(val)
                except ValueError:
                    return None

            def _as_float(val: Any) -> float | None:
                if val is None or val == "":
                    return None
                try:
                    return float(val)
                except ValueError:
                    return None

            def _as_json(val: Any):
                if val is None or val == "":
                    return None
                s = str(val).strip()
                try:
                    return json.loads(s)
                except Exception:
                    return None

            rows.append(
                {
                    "id": row["id"].strip().strip('"'),
                    "run_id": (row.get("run_id") or "").strip(),
                    "item_id": (row.get("item_id") or "").strip(),
                    "correct_label": (row.get("correct_label") or "").strip(),
                    "correct_idx": _as_int(row.get("correct_idx")),
                    "pred": (row.get("pred") or "").strip(),
                    "correct": _as_int(row.get("correct")),
                    "prob_correct": _as_float(row.get("prob_correct")),
                    "logprobs": _as_json(row.get("logprobs")),
                    "attn_to_options": _as_json(row.get("attn_to_options")),
                    "probs_per_layer": _as_json(row.get("probs_per_layer")),
                }
            )
    return rows


def import_into_db(runs_csv: Path, results_csv: Path) -> None:
    import os
    conn = _conn()
    if not conn:
        url = os.environ.get("DATABASE_URL", "")
        if not url:
            raise SystemExit(
                "DATABASE_URL is not set. Add it to .env in the project root, or run:\n"
                '  export DATABASE_URL="postgresql://user:pass@host:port/dbname"\n'
                "For Railway: use DATABASE_PUBLIC_URL from the Postgres service Variables."
            )
        raise SystemExit(
            "Could not connect to the database. Check that Postgres is running and DATABASE_URL is correct."
        )
    try:
        create_schema(conn)
        with conn.cursor() as cur:
            # Ensure extension for gen_random_uuid exists (no-op if already present)
            cur.execute("CREATE EXTENSION IF NOT EXISTS pgcrypto;")
            conn.commit()

        runs = load_runs_csv(runs_csv)
        results = load_results_csv(results_csv)

        BATCH = 500  # commit every N results to avoid SSL/timeout on long transactions
        with conn.cursor() as cur:
            for r in runs:
                cur.execute(
                    """
                    INSERT INTO runs (id, model, dataset, created_at)
                    VALUES (%s, %s, %s, %s)
                    ON CONFLICT (id) DO NOTHING
                    """,
                    (r["id"], r["model"], r["dataset"], r["created_at"] or None),
                )
            conn.commit()

            for i, r in enumerate(results):
                cur.execute(
                    """
                    INSERT INTO results (
                        id,
                        run_id,
                        item_id,
                        correct_label,
                        correct_idx,
                        pred,
                        correct,
                        prob_correct,
                        logprobs,
                        attn_to_options,
                        probs_per_layer
                    )
                    VALUES (
                        %s, %s, %s, %s, %s,
                        %s, %s, %s,
                        %s::jsonb, %s::jsonb, %s::jsonb
                    )
                    ON CONFLICT (id) DO NOTHING
                    """,
                    (
                        r["id"],
                        r["run_id"],
                        r["item_id"] or None,
                        r["correct_label"] or None,
                        r["correct_idx"],
                        r["pred"] or None,
                        r["correct"],
                        r["prob_correct"],
                        json.dumps(r["logprobs"]) if r["logprobs"] is not None else None,
                        json.dumps(r["attn_to_options"]) if r["attn_to_options"] is not None else None,
                        json.dumps(r["probs_per_layer"]) if r["probs_per_layer"] is not None else None,
                    ),
                )
                if (i + 1) % BATCH == 0:
                    conn.commit()
        conn.commit()
        print(f"Imported {len(runs)} runs and {len(results)} results.")
    finally:
        conn.close()


def main() -> None:
    parser = argparse.ArgumentParser(description="Import results and runs from CSV into the Postgres database.")
    parser.add_argument(
        "--dir",
        type=str,
        default="results_and_runs",
        help="Directory containing runs_*.csv and results_*.csv files.",
    )
    args = parser.parse_args()

    base = Path(args.dir)
    if not base.is_absolute():
        base = Path.cwd() / base

    runs_files = sorted(base.glob("runs_*.csv"))
    results_files = sorted(base.glob("results_*.csv"))
    if not runs_files or not results_files:
        raise SystemExit(f"Could not find runs_*.csv or results_*.csv in {base}")

    # Use the latest by name (they have timestamps in filenames)
    runs_csv = runs_files[-1]
    results_csv = results_files[-1]
    print(f"Using runs CSV: {runs_csv}")
    print(f"Using results CSV: {results_csv}")

    import_into_db(runs_csv, results_csv)


if __name__ == "__main__":
    main()

