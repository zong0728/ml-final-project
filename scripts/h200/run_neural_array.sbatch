#!/bin/bash
#SBATCH --job-name=OUT_nn
#SBATCH --account=bgyq-dtai-gh
#SBATCH --partition=ghx4
#SBATCH --time=04:00:00
#SBATCH --nodes=1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --array=0-11
#SBATCH --output=scripts/h200/logs/OUT_nn_%A_%a.out
#SBATCH --error=scripts/h200/logs/OUT_nn_%A_%a.err

# Array job: each task handles ONE neural architecture across all hparams
# and all folds. 12 archs => 12 array tasks => fully parallel on 12 GPUs.
#
# Submit: sbatch scripts/h200/run_neural_array.sbatch
set -euo pipefail

ARCHS=(gru lstm bilstm tcn mlp transformer nlinear dlinear patchtst itransformer nbeats nhits)
ARCH=${ARCHS[$SLURM_ARRAY_TASK_ID]}

module load python/miniforge3_pytorch/2.10.0
source activate "$HOME/.conda/envs/outage"

cd "$SLURM_SUBMIT_DIR"
echo "[OUT_nn] task=$SLURM_ARRAY_TASK_ID  arch=$ARCH  host=$(hostname)"
nvidia-smi -L
echo "[OUT_nn] start=$(date)"

# Per-task CSV — merged later via scripts/merge_csvs.py
mkdir -p results/runs_per_task
export OUTAGE_RUNS_CSV="results/runs_per_task/nn_${ARCH}_${SLURM_ARRAY_JOB_ID}_${SLURM_ARRAY_TASK_ID}.csv"
echo "[OUT_nn] writing to $OUTAGE_RUNS_CSV"

python -m scripts.run_neural_grid_arch --arch "$ARCH"

echo "[OUT_nn] done $(date)"
