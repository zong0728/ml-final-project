#!/usr/bin/env bash
# Run AFTER all sbatch jobs finish. From the project root on the login node:
#   bash scripts/h200/finalize_all.sh
#
# Steps:
#   1. squeue -u $USER  (sanity check no jobs left running)
#   2. merge per-task CSVs -> single results/experiment_runs.csv
#   3. plot CV summary
#   4. retrain top-K on full data, build equal-weight ensemble, write
#      submission CSVs + 5-county policy selection
#   5. rebuild report tables + recompile report.pdf (if tectonic available)
set -euo pipefail

module load python/miniforge3_pytorch/2.10.0
source activate "$HOME/.conda/envs/outage"
cd /projects/bgyq/sguan/ml-final-outage

echo "==== 1. Pending SLURM jobs (should be empty) ===="
squeue -u "$USER"

echo
echo "==== 2. Merging per-task CSVs ===="
python -m scripts.merge_csvs

echo
echo "==== 3. CV summary plots ===="
python -m scripts.plot_results

echo
echo "==== 4. Final ensemble + 5-county policy ===="
python -m scripts.finalize --topk 5 --seeds 42 43

echo
echo "==== 5. Report tables + PDF ===="
python -m scripts.build_report_tables
if command -v tectonic >/dev/null 2>&1; then
  (cd report && tectonic report.tex)
else
  echo "(tectonic not installed on this node — recompile locally)"
fi

echo
echo "==== Outputs to retrieve from your laptop ===="
echo "  rsync -avz sguan@dtai-login.delta.ncsa.illinois.edu:/projects/bgyq/sguan/ml-final-outage/results/submissions/ ./results_h200/submissions/"
echo "  rsync -avz sguan@dtai-login.delta.ncsa.illinois.edu:/projects/bgyq/sguan/ml-final-outage/results/policy_selection.txt ./"
echo "  rsync -avz sguan@dtai-login.delta.ncsa.illinois.edu:/projects/bgyq/sguan/ml-final-outage/results/experiment_runs.csv ./results_h200/"
