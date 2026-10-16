#!/usr/bin/env bash
# Run more small open-source models (answers read from probabilities) one after another on this server's CPU.
# Starts only after any open-model run already going has finished, and never runs twice at the same time.
# Safe to run again: answers already saved are kept and each model continues where it stopped.
#
#   bash experiments/v3_corrected/run_open_models.sh
#
# Takes several hours in total (about 1.5-2.5 hours per model). Then: bash experiments/v3_corrected/update_results.sh
set -euo pipefail
cd "$(dirname "$0")/../.."
D=experiments/v3_corrected
LOCK=$D/.run_open_models.lock
RUN="nice -n 10 .venv/bin/python3 -u $D/run_hf.py"

if ! flock -n "$LOCK" true; then
  echo "The open models are already running (log: $D/run_open_models.log)."
  exit 0
fi
nohup flock -n "$LOCK" bash -c "
  while pgrep -f '^\.venv/bin/python3 -u $D/run_hf\.py' >/dev/null; do sleep 60; done
  $RUN --model google/gemma-3-1b-it                                  # Google
  $RUN --model allenai/OLMo-2-0425-1B-Instruct --formats letter      # AI2; letters only: with numbers it rarely answers with a label
  $RUN --model HuggingFaceTB/SmolLM2-1.7B-Instruct                   # Hugging Face
  $RUN --model Qwen/Qwen2.5-1.5B-Instruct                            # Alibaba, bigger sibling of Qwen2.5-0.5B
  echo 'ALL OPEN MODELS DONE'
" > $D/run_open_models.log 2>&1 &

echo "Started: gemma-3-1b-it, OLMo-2-1B, SmolLM2-1.7B, Qwen2.5-1.5B (one after another, after Llama-3.2-1B)."
echo "Log:   tail -f $D/run_open_models.log"
echo "When the log ends with ALL OPEN MODELS DONE:   bash $D/update_results.sh"
