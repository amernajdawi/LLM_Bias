#!/usr/bin/env python3
"""Import existing JSON result files into the database."""
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from dotenv import load_dotenv
load_dotenv(ROOT / ".env")

from src.db.client import create_schema, insert_run, insert_results, _conn
from src.io import load_json


def main():
    conn = _conn()
    if not conn:
        print("No DATABASE_URL. Cannot import.")
        return

    try:
        create_schema(conn)
        results_dir = ROOT / "outputs" / "results"

        json_files = list(results_dir.glob("*.json"))
        if not json_files:
            print(f"No JSON files found in {results_dir}")
            return

        imported = 0
        skipped = 0

        for json_file in json_files:
            name = json_file.stem
            parts = name.split("_", 1)
            if len(parts) != 2:
                print(f"Skipping {json_file.name} (unexpected format)")
                skipped += 1
                continue

            model_key, dataset_name = parts

            with conn.cursor() as cur:
                cur.execute(
                    "SELECT COUNT(*) FROM results r JOIN runs rn ON r.run_id = rn.id WHERE rn.model = %s AND rn.dataset = %s",
                    (model_key, dataset_name),
                )
                existing = cur.fetchone()["count"]

            if existing > 0:
                print(f"Skipping {json_file.name} (already in DB: {existing} results)")
                skipped += 1
                continue

            try:
                data = load_json(json_file)
                results = data.get("results", [])
                if not results:
                    print(f"Skipping {json_file.name} (no results)")
                    skipped += 1
                    continue
            except Exception as e:
                print(f"Error loading {json_file.name}: {e}")
                skipped += 1
                continue

            try:
                run_id = insert_run(conn, model_key, dataset_name)
                insert_results(conn, run_id, results)
                print(f"Imported {json_file.name}: {len(results)} results")
                imported += 1
            except Exception as e:
                print(f"Error importing {json_file.name}: {e}")
                skipped += 1

        print(f"\nDone. Imported: {imported}, Skipped: {skipped}")
    finally:
        conn.close()


if __name__ == "__main__":
    main()
