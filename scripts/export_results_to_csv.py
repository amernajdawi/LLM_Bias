"""
Export all runs and results from the database (DATABASE_URL) to CSV files
in the same format that import_results_from_csv expects.
Use this to copy local DB data to Railway: export with local DATABASE_URL,
then import with Railway DATABASE_URL.
"""
import argparse
import csv
import json
from datetime import datetime, timezone
from pathlib import Path

# Load .env from project root so DATABASE_URL is set when running from CLI
ROOT = Path(__file__).resolve().parent.parent
try:
    from dotenv import load_dotenv
    load_dotenv(ROOT / ".env", override=True)
except Exception:
    pass

from src.db.client import _conn, create_schema


def export_to_csv(out_dir: Path) -> None:
    conn = _conn()
    if not conn:
        raise SystemExit("Could not connect to database. Set DATABASE_URL (e.g. your local Postgres).")
    try:
        create_schema(conn)
        out_dir = Path(out_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        ts = datetime.now(timezone.utc).strftime("%Y%m%d%H%M%S")
        runs_path = out_dir / f"runs_{ts}.csv"
        results_path = out_dir / f"results_{ts}.csv"

        with conn.cursor() as cur:
            cur.execute("SELECT id, model, dataset, created_at FROM runs ORDER BY created_at, id")
            runs = [dict(row) for row in cur.fetchall()]
            cur.execute(
                """SELECT id, run_id, item_id, correct_label, correct_idx, pred, correct, prob_correct,
                          logprobs, attn_to_options, probs_per_layer
                   FROM results ORDER BY run_id, id"""
            )
            results = [dict(row) for row in cur.fetchall()]

        def _json_col(val):
            if val is None:
                return ""
            if isinstance(val, (dict, list)):
                return json.dumps(val)
            return str(val)

        with runs_path.open("w", encoding="utf-8", newline="") as f:
            w = csv.writer(f)
            w.writerow(["id", "model", "dataset", "created_at"])
            for r in runs:
                w.writerow([
                    str(r.get("id") or ""),
                    r.get("model") or "",
                    r.get("dataset") or "",
                    r.get("created_at") or "",
                ])

        with results_path.open("w", encoding="utf-8", newline="") as f:
            w = csv.writer(f)
            w.writerow([
                "id", "run_id", "item_id", "correct_label", "correct_idx", "pred", "correct",
                "prob_correct", "logprobs", "attn_to_options", "probs_per_layer",
            ])
            for r in results:
                w.writerow([
                    str(r.get("id") or ""),
                    str(r.get("run_id") or ""),
                    r.get("item_id") or "",
                    r.get("correct_label") or "",
                    r.get("correct_idx") if r.get("correct_idx") is not None else "",
                    r.get("pred") or "",
                    r.get("correct") if r.get("correct") is not None else "",
                    r.get("prob_correct") if r.get("prob_correct") is not None else "",
                    _json_col(r.get("logprobs")),
                    _json_col(r.get("attn_to_options")),
                    _json_col(r.get("probs_per_layer")),
                ])

        print(f"Exported {len(runs)} runs and {len(results)} results to {out_dir}")
        print(f"  Runs:    {runs_path.name}")
        print(f"  Results: {results_path.name}")
        print()
        print("To load this into Railway, run:")
        print('  export DATABASE_URL="<your Railway DATABASE_PUBLIC_URL>"')
        print(f"  uv run python -m scripts.import_results_from_csv --dir {out_dir.resolve()}")
    finally:
        conn.close()


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Export all runs and results from the database to CSV (for importing into Railway or backup)."
    )
    parser.add_argument(
        "--dir",
        type=str,
        default="results_and_runs",
        help="Directory to write runs_<timestamp>.csv and results_<timestamp>.csv",
    )
    args = parser.parse_args()
    base = Path(args.dir)
    if not base.is_absolute():
        base = Path.cwd() / base
    export_to_csv(base)


if __name__ == "__main__":
    main()
