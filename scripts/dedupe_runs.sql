-- Keep only the latest run per (model, dataset). Delete older runs (results cascade).
-- Run:  cat scripts/dedupe_runs.sql | docker compose exec -T postgres psql -U llmbias -d llmbias
DELETE FROM runs
WHERE id NOT IN (
  SELECT DISTINCT ON (model, dataset) id
  FROM runs
  ORDER BY model, dataset, created_at DESC
);
