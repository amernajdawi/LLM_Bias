-- Replace short model keys with full model IDs in runs table (case-insensitive, trimmed)
UPDATE runs SET model = 'gpt-4.1-mini-2025-04-14' WHERE LOWER(TRIM(model)) = 'openai';
UPDATE runs SET model = 'meta-llama/Llama-3.2-1B-Instruct' WHERE LOWER(TRIM(model)) = 'llama';
UPDATE runs SET model = 'Qwen/Qwen2.5-0.5B-Instruct' WHERE LOWER(TRIM(model)) = 'qwen';
