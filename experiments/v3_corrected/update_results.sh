#!/usr/bin/env bash
# Re-analyse every saved run and rebuild the report. Unfinished runs are left out automatically.
#
#   bash experiments/v3_corrected/update_results.sh
set -euo pipefail
cd "$(dirname "$0")/../.."
D=experiments/v3_corrected
export OMP_NUM_THREADS=1

echo "Analysing cloud, local and OpenAI runs (raw/) ..."
.venv/bin/python3 $D/analyze_v3.py > $D/v3_analysis.txt
echo "Analysing open-weight runs (raw_hf/) ..."
.venv/bin/python3 $D/analyze_hf.py > $D/hf_analysis.txt
.venv/bin/python3 $D/build_report.py $D/report.html > /dev/null

echo
echo "Done."
echo "  Results, cloud models:  $D/v3_analysis.txt"
echo "  Results, open models:   $D/hf_analysis.txt"
echo "  Report:                 $D/report.html"
