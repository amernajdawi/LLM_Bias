#!/usr/bin/env bash
# Start more models in the background with the corrected protocol (same questions, prompt and rotations).
# Safe to run again: answers already saved are kept and each run continues where it stopped.
#
#   bash experiments/v3_corrected/run_more_models.sh
#
# When every log ends with "ALL DONE":   bash experiments/v3_corrected/update_results.sh
set -euo pipefail
cd "$(dirname "$0")/../.."
D=experiments/v3_corrected

# Ollama Cloud models that are free on your plan (checked 2026-10-04 with: run_cloud.py --check --models ...)
CLOUD_MODELS="nemotron-3-super nemotron-3-ultra"
# Open-weight Hugging Face models, answers read from probabilities. Runs on this server's CPU (about 1 hour per 1B model).
HF_MODELS="meta-llama/Llama-3.2-1B-Instruct"
# PAID: OpenAI models, uses OPENAI_API_KEY from .env (about 3,200 short requests per model). Empty = skip.
OPENAI_MODELS=""

running() { pgrep -f "^\.venv/bin/python3 -u $D/$1" >/dev/null; }   # never start the same run twice

if [ -n "$CLOUD_MODELS" ]; then
  if running "run_cloud\.py --models $CLOUD_MODELS"; then echo "Cloud models are already running (log: $D/run_cloud_more.log)"; else
    nohup .venv/bin/python3 -u $D/run_cloud.py --models $CLOUD_MODELS > $D/run_cloud_more.log 2>&1 &
    echo "Cloud models started:  $CLOUD_MODELS   (log: $D/run_cloud_more.log)"
  fi
fi
if [ -n "$OPENAI_MODELS" ]; then
  if running "run_cloud\.py --provider openai"; then echo "OpenAI models are already running (log: $D/run_openai.log)"; else
    nohup .venv/bin/python3 -u $D/run_cloud.py --provider openai --models $OPENAI_MODELS > $D/run_openai.log 2>&1 &
    echo "OpenAI models started: $OPENAI_MODELS   (log: $D/run_openai.log)"
  fi
fi
if [ -n "$HF_MODELS" ]; then
  if running "run_hf\.py" || pgrep -f "for m in $HF_MODELS" >/dev/null; then echo "Open models are already running (log: $D/run_hf_more.log)"; else
    nohup bash -c "for m in $HF_MODELS; do nice -n 10 .venv/bin/python3 -u $D/run_hf.py --model \$m; done" > $D/run_hf_more.log 2>&1 &
    echo "Open models started:   $HF_MODELS   (log: $D/run_hf_more.log)"
  fi
fi
echo
echo "Watch progress:   tail -f $D/run_cloud_more.log $D/run_hf_more.log"
echo "When done:        bash $D/update_results.sh"
