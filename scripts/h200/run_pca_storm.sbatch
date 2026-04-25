#!/bin/bash
#SBATCH --job-name=OUT_pca
#SBATCH --account=bgyq-dtai-gh
#SBATCH --partition=ghx4
#SBATCH --time=04:00:00
#SBATCH --nodes=1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=128G
#SBATCH --output=scripts/h200/logs/OUT_pca_%j.out
#SBATCH --error=scripts/h200/logs/OUT_pca_%j.err

# PCA-reduced weather + storm-expert MoE — last two ablation directions.
# 6 models x 2 seeds x 6 folds x 2 horizons = 144 cells.
set -euo pipefail

module load python/miniforge3_pytorch/2.10.0
source activate "$HOME/.conda/envs/outage"

cd "$SLURM_SUBMIT_DIR"
echo "[OUT_pca] start=$(date) host=$(hostname)"

mkdir -p results/runs_per_task
export OUTAGE_RUNS_CSV="results/runs_per_task/pca_storm_${SLURM_JOB_ID}.csv"
echo "[OUT_pca] writing to $OUTAGE_RUNS_CSV"

python -m scripts.run_pca_storm

echo "[OUT_pca] done $(date)"
