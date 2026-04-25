#!/bin/bash
#SBATCH --job-name=OUT_grid2
#SBATCH --account=bgyq-dtai-gh
#SBATCH --partition=ghx4
#SBATCH --time=12:00:00
#SBATCH --nodes=1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=128G
#SBATCH --output=scripts/h200/logs/OUT_grid2_%j.out
#SBATCH --error=scripts/h200/logs/OUT_grid2_%j.err

# Re-run grid jobs with 12h walltime so the larger configs (n_estimators=1500,
# max_depth=8, num_leaves=127) actually finish. Resume support means already-
# completed (model, seed, horizon, fold) cells are skipped.
#
# Submit:
#   sbatch scripts/h200/run_grid_long.sbatch lgb       # 54 LightGBM configs
#   sbatch scripts/h200/run_grid_long.sbatch lgbtw     # 24 Tweedie
#   sbatch scripts/h200/run_grid_long.sbatch xgb       # 18 XGB
#   sbatch scripts/h200/run_grid_long.sbatch cat       # 18 CatBoost
set -euo pipefail
GRID_NAME="${1:-all}"
mkdir -p scripts/h200/logs

module load python/miniforge3_pytorch/2.10.0
source activate "$HOME/.conda/envs/outage"

cd "$SLURM_SUBMIT_DIR"
echo "[OUT_grid2] grid=$GRID_NAME host=$(hostname) start=$(date)"

mkdir -p results/runs_per_task
export OUTAGE_RUNS_CSV="results/runs_per_task/grid2_${GRID_NAME}_${SLURM_JOB_ID}.csv"
echo "[OUT_grid2] writing to $OUTAGE_RUNS_CSV"

python -m scripts.run_grid --grid "$GRID_NAME"

echo "[OUT_grid2] done $(date)"
