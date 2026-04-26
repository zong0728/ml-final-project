#!/usr/bin/env bash
# One-shot post-data finalization. Run after H200 jobs complete and CSV is rsync'd back.
#
# Usage:
#   bash scripts/finalize_report.sh
#
# Prerequisites:
#   - results/runs_per_task/*.csv exist (rsync'd back from H200)
#   - results/experiment_runs.csv may already include earlier runs (will be merged)
set -euo pipefail
cd "$(dirname "$0")/.."

source /opt/homebrew/Caskroom/miniconda/base/etc/profile.d/conda.sh
conda activate outage

echo "==== 1. Merge per-task CSVs ===="
python -m scripts.merge_csvs

echo
echo "==== 2. Plot updated CV results ===="
python -m scripts.plot_results

echo
echo "==== 3. Run finalize: top-K ensemble + policy 5-county selection ===="
python -m scripts.finalize --topk 5 --seeds 42 43 --stability-weight 0.3 --strategy auto

echo
echo "==== 4. Refresh report tables (incl. policy section, rationale) ===="
python -m scripts.build_report_tables

echo
echo "==== 5. Recompile report PDF ===="
cd report && tectonic report.tex 2>&1 | tail -3
python -c "from PyPDF2 import PdfReader; print(f'Final pages: {len(PdfReader(\"report.pdf\").pages)}')"
cd ..

echo
echo "==== 6. Build submission bundle ===="
bash scripts/build_submission_archive.sh

echo
echo "DONE. Submission ready in: submission_bundle/"
ls -la submission_bundle/
