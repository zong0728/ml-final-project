#!/bin/bash
#SBATCH --job-name=OUT_nb
#SBATCH --account=bgyq-dtai-gh
#SBATCH --partition=ghx4
#SBATCH --time=04:00:00
#SBATCH --nodes=1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --array=0-1
#SBATCH --output=scripts/h200/logs/OUT_nb_%A_%a.out
#SBATCH --error=scripts/h200/logs/OUT_nb_%A_%a.err

# NBeats + NHiTS array — 2 tasks. Submit AFTER the main neural array
# (which uses 0-9 for the original 10 archs) so we don't double-run.
set -euo pipefail

ARCHS=(nbeats nhits)
ARCH=${ARCHS[$SLURM_ARRAY_TASK_ID]}

module load python/miniforge3_pytorch/2.10.0
source activate "$HOME/.conda/envs/outage"

cd "$SLURM_SUBMIT_DIR"
echo "[OUT_nb] task=$SLURM_ARRAY_TASK_ID  arch=$ARCH  host=$(hostname)"
nvidia-smi -L
echo "[OUT_nb] start=$(date)"

mkdir -p results/runs_per_task
export OUTAGE_RUNS_CSV="results/runs_per_task/nn_${ARCH}_${SLURM_ARRAY_JOB_ID}_${SLURM_ARRAY_TASK_ID}.csv"
echo "[OUT_nb] writing to $OUTAGE_RUNS_CSV"

python -m scripts.run_neural_grid_arch --arch "$ARCH"

echo "[OUT_nb] done $(date)"
