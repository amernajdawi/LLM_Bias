import json
import os
import time
from typing import Any, Dict, List, Optional

import psycopg

try:
    from psycopg.rows import dict_row
except ModuleNotFoundError:
    # Fallback when psycopg.rows is missing (e.g. incomplete venv)
    def dict_row(cursor):
        # cursor.description is sequence of (name, ...) per DB-API
        columns = [d[0] for d in cursor.description] if cursor.description else []

        def make_row(values):
            return dict(zip(columns, values)) if columns else {}

        return make_row

_SCHEMA = """
CREATE TABLE IF NOT EXISTS runs (
  id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
  model TEXT NOT NULL,
  dataset TEXT NOT NULL,
  created_at TIMESTAMPTZ DEFAULT now()
);

CREATE TABLE IF NOT EXISTS results (
  id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
  run_id UUID NOT NULL REFERENCES runs(id) ON DELETE CASCADE,
  item_id TEXT,
  correct_label TEXT,
  correct_idx INT,
  pred TEXT,
  correct INT,
  prob_correct FLOAT,
  logprobs JSONB,
  attn_to_options JSONB,
  probs_per_layer JSONB
);

CREATE INDEX IF NOT EXISTS idx_results_run_id ON results(run_id);
CREATE INDEX IF NOT EXISTS idx_runs_model_dataset ON runs(model, dataset);
"""


def _conn(max_attempts: int = 5):
    url = os.environ.get("DATABASE_URL")
    if not url:
        return None
    for _ in range(max_attempts):
        try:
            return psycopg.connect(url, row_factory=dict_row)
        except Exception:
            time.sleep(1.5)
    return None


def create_schema(conn) -> None:
    with conn.cursor() as cur:
        cur.execute(_SCHEMA)
    conn.commit()


def insert_run(conn, model: str, dataset: str) -> str:
    with conn.cursor() as cur:
        cur.execute(
            "INSERT INTO runs (model, dataset) VALUES (%s, %s) RETURNING id",
            (model, dataset),
        )
        return str(cur.fetchone()["id"])


def _j(v: Any) -> Optional[str]:
    if v is None:
        return None
    return json.dumps(v) if not isinstance(v, str) else v


def insert_results(conn, run_id: str, results: List[Dict[str, Any]]) -> None:
    with conn.cursor() as cur:
        for r in results:
            cur.execute(
                """INSERT INTO results (run_id, item_id, correct_label, correct_idx, pred, correct, prob_correct, logprobs, attn_to_options, probs_per_layer)
                   VALUES (%s, %s, %s, %s, %s, %s, %s, %s::jsonb, %s::jsonb, %s::jsonb)""",
                (
                    run_id,
                    r.get("item_id"),
                    r.get("correct_label"),
                    r.get("correct_idx"),
                    r.get("pred"),
                    r.get("correct"),
                    r.get("prob_correct"),
                    _j(r.get("logprobs")),
                    _j(r.get("attn_to_options")),
                    _j(r.get("probs_per_layer")),
                ),
            )
    conn.commit()


def count_results_for_model_dataset(conn, model: str, dataset: str) -> int:
    with conn.cursor() as cur:
        cur.execute(
            """SELECT COUNT(*) FROM results r
               JOIN runs rn ON r.run_id = rn.id
               WHERE rn.model = %s AND rn.dataset = %s
               AND rn.id = (SELECT id FROM runs WHERE model = %s AND dataset = %s ORDER BY created_at DESC LIMIT 1)""",
            (model, dataset, model, dataset),
        )
        return cur.fetchone()["count"] or 0


def count_results_for_model_dataset_sync(model: str, dataset: str) -> int:
    conn = _conn()
    if not conn:
        return 0
    try:
        return count_results_for_model_dataset(conn, model, dataset)
    finally:
        conn.close()


def run_with_db(model: str, dataset: str, results: List[Dict[str, Any]]) -> bool:
    conn = _conn()
    if not conn:
        return False
    try:
        create_schema(conn)
        rid = insert_run(conn, model, dataset)
        insert_results(conn, rid, results)
        return True
    finally:
        conn.close()


def get_results(
    conn,
    model: Optional[str] = None,
    dataset: Optional[str] = None,
    run_id: Optional[str] = None,
) -> List[Dict[str, Any]]:
    q = """
    SELECT rn.model, rn.dataset, r.item_id, r.correct_label, r.correct_idx, r.pred, r.correct, r.prob_correct, r.logprobs, r.attn_to_options, r.probs_per_layer
    FROM results r
    JOIN runs rn ON r.run_id = rn.id
    WHERE 1=1
    """
    params: list = []
    if model:
        q += " AND rn.model = %s"
        params.append(model)
    if dataset:
        q += " AND rn.dataset = %s"
        params.append(dataset)
    if run_id:
        q += " AND r.run_id = %s"
        params.append(run_id)
    q += " ORDER BY rn.created_at DESC, r.id"
    with conn.cursor() as cur:
        cur.execute(q, params)
        rows = cur.fetchall()
    out = []
    for row in rows:
        d = dict(row)
        d["logprobs"] = d["logprobs"] if isinstance(d.get("logprobs"), dict) else (d.get("logprobs") or {})
        d["attn_to_options"] = d["attn_to_options"] if isinstance(d.get("attn_to_options"), list) else (d.get("attn_to_options") or [])
        d["probs_per_layer"] = d["probs_per_layer"] if isinstance(d.get("probs_per_layer"), list) else (d.get("probs_per_layer") or [])
        out.append(d)
    return out


def list_run_keys(conn) -> List[tuple]:
    """Return list of (model, dataset) pairs that exist in runs."""
    with conn.cursor() as cur:
        cur.execute("SELECT DISTINCT model, dataset FROM runs ORDER BY model, dataset")
        return [(row["model"], row["dataset"]) for row in cur.fetchall()]


def list_runs_with_result_counts(conn) -> List[Dict[str, Any]]:
    """Return each run with its result count, newest first. Use to pick runs that have data."""
    with conn.cursor() as cur:
        cur.execute("""
            SELECT rn.id AS run_id, rn.model, rn.dataset, rn.created_at, COUNT(r.id) AS n_results
            FROM runs rn
            LEFT JOIN results r ON r.run_id = rn.id
            GROUP BY rn.id, rn.model, rn.dataset, rn.created_at
            ORDER BY rn.created_at DESC
        """)
        return [dict(row) for row in cur.fetchall()]
