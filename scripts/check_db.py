#!/usr/bin/env python3
"""Print database stats: total results, per model+dataset, and per run (to spot duplicates)."""
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from dotenv import load_dotenv
load_dotenv(ROOT / ".env")

from src.db.client import _conn

def main():
    conn = _conn()
    if not conn:
        print("No DATABASE_URL; cannot connect.")
        return
    try:
        with conn.cursor() as cur:
            cur.execute("SELECT COUNT(*) as n FROM results")
            total = cur.fetchone()["n"]
            print(f"Total results: {total}")

            cur.execute("""
                SELECT rn.model, rn.dataset, COUNT(*) as cnt
                FROM results r
                JOIN runs rn ON r.run_id = rn.id
                GROUP BY rn.model, rn.dataset
                ORDER BY rn.model, rn.dataset
            """)
            rows = cur.fetchall()
            print("\nPer model + dataset (sum may include duplicate runs):")
            for r in rows:
                print(f"  {r['model']} + {r['dataset']}: {r['cnt']}")

            cur.execute("""
                SELECT rn.id, rn.model, rn.dataset, rn.created_at, COUNT(r.id) as cnt
                FROM runs rn
                LEFT JOIN results r ON r.run_id = rn.id
                GROUP BY rn.id, rn.model, rn.dataset, rn.created_at
                ORDER BY rn.model, rn.dataset, rn.created_at
            """)
            runs = cur.fetchall()
            print("\nPer run (run_id, model, dataset, created_at, count):")
            for r in runs:
                print(f"  {r['id']}  {r['model']} + {r['dataset']}  {r['created_at']}  -> {r['cnt']} results")
    finally:
        conn.close()

if __name__ == "__main__":
    main()
