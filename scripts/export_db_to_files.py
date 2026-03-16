#!/usr/bin/env python3
"""Export database runs and results to JSON and CSV in the outputs folder."""
import csv
import json
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

try:
    from dotenv import load_dotenv
    load_dotenv(ROOT / ".env")
except Exception:
    pass  # use DATABASE_URL from environment if dotenv missing/broken

from src.db.client import _conn


def _row_to_dict(row):
    """Convert a DB row (with possible date/JSONB) to a JSON-serializable dict."""
    d = dict(row)
    for k, v in d.items():
        if hasattr(v, "isoformat"):
            d[k] = v.isoformat()
        elif v is not None and isinstance(v, (dict, list)):
            pass  # keep as is for JSON
    return d


def main():
    conn = _conn()
    if not conn:
        print("No DATABASE_URL. Cannot export.")
        return

    out_dir = ROOT / "outputs" / "db_export"
    out_dir.mkdir(parents=True, exist_ok=True)

    try:
        with conn.cursor() as cur:
            cur.execute("SELECT id, model, dataset, created_at FROM runs ORDER BY created_at")
            runs = [_row_to_dict(row) for row in cur.fetchall()]

            cur.execute("""
                SELECT id, run_id, item_id, correct_label, correct_idx, pred, correct, prob_correct,
                       logprobs, attn_to_options, probs_per_layer
                FROM results ORDER BY run_id, id
            """)
            results = [_row_to_dict(row) for row in cur.fetchall()]

        # JSON
        runs_path = out_dir / "runs.json"
        results_path = out_dir / "results.json"
        with open(runs_path, "w", encoding="utf-8") as f:
            json.dump(runs, f, indent=2, ensure_ascii=False, default=str)
        with open(results_path, "w", encoding="utf-8") as f:
            json.dump(results, f, indent=2, ensure_ascii=False, default=str)
        print(f"Saved {len(runs)} runs -> {runs_path}")
        print(f"Saved {len(results)} results -> {results_path}")

        # CSV (JSONB columns as JSON strings)
        runs_csv = out_dir / "runs.csv"
        if runs:
            with open(runs_csv, "w", newline="", encoding="utf-8") as f:
                w = csv.DictWriter(f, fieldnames=list(runs[0].keys()))
                w.writeheader()
                for r in runs:
                    w.writerow({k: (json.dumps(v) if isinstance(v, (dict, list)) else v) for k, v in r.items()})
            print(f"Saved -> {runs_csv}")

        results_csv = out_dir / "results.csv"
        if results:
            with open(results_csv, "w", newline="", encoding="utf-8") as f:
                w = csv.DictWriter(f, fieldnames=list(results[0].keys()))
                w.writeheader()
                for r in results:
                    row = {}
                    for k, v in r.items():
                        if isinstance(v, (dict, list)):
                            row[k] = json.dumps(v)
                        else:
                            row[k] = v
                    w.writerow(row)
            print(f"Saved -> {results_csv}")

        print(f"\nAll exports in: {out_dir}")
    finally:
        conn.close()


if __name__ == "__main__":
    main()
