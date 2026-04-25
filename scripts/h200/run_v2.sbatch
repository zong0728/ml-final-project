#!/bin/bash
#SBATCH --job-name=OUT_v2
#SBATCH --account=bgyq-dtai-gh
#SBATCH --partition=ghx4
#SBATCH --time=04:00:00
#SBATCH --nodes=1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=128G
#SBATCH --output=scripts/h200/logs/OUT_v2_%j.out
#SBATCH --error=scripts/h200/logs/OUT_v2_%j.err

# v2 feature-set + loss-function exploration. ~10 models × 2 seeds × 6 folds × 2 horizons.
set -euo pipefail

module load python/miniforge3_pytorch/2.10.0
source activate "$HOME/.conda/envs/outage"

cd "$SLURM_SUBMIT_DIR"
echo "[OUT_v2] host=$(hostname) start=$(date)"

mkdir -p results/runs_per_task
export OUTAGE_RUNS_CSV="results/runs_per_task/v2_${SLURM_JOB_ID}.csv"
echo "[OUT_v2] writing to $OUTAGE_RUNS_CSV"

python -m scripts.run_v2_grid

echo "[OUT_v2] done $(date)"
