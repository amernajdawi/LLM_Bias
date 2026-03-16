#!/usr/bin/env python3
"""Remove duplicate runs: keep only the latest run per (model, dataset). Results cascade-delete."""
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
        print("No DATABASE_URL. Set it (e.g. postgresql://llmbias:llmbias@localhost:5433/llmbias).")
        return
    try:
        with conn.cursor() as cur:
            # Runs to keep: latest per (model, dataset)
            cur.execute("""
                SELECT DISTINCT ON (model, dataset) id, model, dataset, created_at
                FROM runs
                ORDER BY model, dataset, created_at DESC
            """)
            keep = cur.fetchall()
            keep_ids = [r["id"] for r in keep]
            if not keep_ids:
                print("No runs found.")
                return
            placeholders = ",".join("%s" for _ in keep_ids)
            cur.execute(f"DELETE FROM runs WHERE id NOT IN ({placeholders})", keep_ids)
            deleted = cur.rowcount
        conn.commit()
        print(f"Kept {len(keep)} runs (latest per model+dataset). Deleted {deleted} duplicate runs (their results were removed by cascade).")
        for r in keep:
            print(f"  {r['model']} + {r['dataset']}  (kept)")
    finally:
        conn.close()

if __name__ == "__main__":
    main()
