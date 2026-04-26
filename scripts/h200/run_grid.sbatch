#!/bin/bash
#SBATCH --job-name=OUT_grid
#SBATCH --account=bgyq-dtai-gh
#SBATCH --partition=ghx4
#SBATCH --time=04:00:00
#SBATCH --nodes=1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --output=scripts/h200/logs/OUT_grid_%j.out
#SBATCH --error=scripts/h200/logs/OUT_grid_%j.err

# Run a hyperparameter grid for the outage forecasting task on a single H200.
# Job names are prefixed `OUT_` to keep diffusion-project jobs separate.
#
# Submit from /projects/bgyq/sguan/ml-final-outage/:
#     sbatch scripts/h200/run_grid.sbatch <grid_name>
# where <grid_name> is one of: lgb, xgb, cat, neural, all

set -euo pipefail
GRID_NAME="${1:-all}"
mkdir -p scripts/h200/logs

module load python/miniforge3_pytorch/2.10.0
source activate "$HOME/.conda/envs/outage"

cd "$SLURM_SUBMIT_DIR"
echo "[OUT_grid] host=$(hostname) gpu=$(nvidia-smi -L | head -1)"
echo "[OUT_grid] grid=$GRID_NAME  start=$(date)"

# Per-job CSV to avoid concurrent-write corruption.
mkdir -p results/runs_per_task
export OUTAGE_RUNS_CSV="results/runs_per_task/grid_${GRID_NAME}_${SLURM_JOB_ID}.csv"
echo "[OUT_grid] writing to $OUTAGE_RUNS_CSV"

python -m scripts.run_grid --grid "$GRID_NAME"

echo "[OUT_grid] done $(date)"
