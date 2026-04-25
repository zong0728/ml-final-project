#!/bin/bash
#SBATCH --job-name=OUT_extras
#SBATCH --account=bgyq-dtai-gh
#SBATCH --partition=ghx4
#SBATCH --time=03:00:00
#SBATCH --nodes=1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --output=scripts/h200/logs/OUT_extras_%j.out
#SBATCH --error=scripts/h200/logs/OUT_extras_%j.err

# Run the non-grid "advanced" models: AutoARIMA per-county + Quantile LGB (P50/P90).
# These don't have a hparam grid — single config each, one cell per fold/horizon.
#
# Submit: sbatch scripts/h200/run_extras.sbatch
set -euo pipefail

module load python/miniforge3_pytorch/2.10.0
source activate "$HOME/.conda/envs/outage"

cd "$SLURM_SUBMIT_DIR"
echo "[OUT_extras] host=$(hostname)  start=$(date)"

mkdir -p results/runs_per_task
export OUTAGE_RUNS_CSV="results/runs_per_task/extras_${SLURM_JOB_ID}.csv"
echo "[OUT_extras] writing to $OUTAGE_RUNS_CSV"

python -m scripts.run_extras

echo "[OUT_extras] done $(date)"
