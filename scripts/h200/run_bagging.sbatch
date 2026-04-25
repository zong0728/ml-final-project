#!/bin/bash
#SBATCH --job-name=OUT_bag
#SBATCH --account=bgyq-dtai-gh
#SBATCH --partition=ghx4
#SBATCH --time=12:00:00
#SBATCH --nodes=1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=128G
#SBATCH --output=scripts/h200/logs/OUT_bag_%j.out
#SBATCH --error=scripts/h200/logs/OUT_bag_%j.err

# Heavy bagging — 10 seeds for each top config. Variance reduction
# from N independent draws is theoretically O(1/sqrt(N)).
set -euo pipefail

module load python/miniforge3_pytorch/2.10.0
source activate "$HOME/.conda/envs/outage"

cd "$SLURM_SUBMIT_DIR"
echo "[OUT_bag] host=$(hostname) start=$(date)"

mkdir -p results/runs_per_task
export OUTAGE_RUNS_CSV="results/runs_per_task/bag_${SLURM_JOB_ID}.csv"
echo "[OUT_bag] writing to $OUTAGE_RUNS_CSV"

python -m scripts.run_bagging --n-seeds 10

echo "[OUT_bag] done $(date)"
