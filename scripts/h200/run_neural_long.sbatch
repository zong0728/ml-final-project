#!/bin/bash
#SBATCH --job-name=OUT_nn2
#SBATCH --account=bgyq-dtai-gh
#SBATCH --partition=ghx4
#SBATCH --time=12:00:00
#SBATCH --nodes=1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --array=0-11
#SBATCH --output=scripts/h200/logs/OUT_nn2_%A_%a.out
#SBATCH --error=scripts/h200/logs/OUT_nn2_%A_%a.err

# Re-run NN array with 12h walltime so all 8 hparams x 2 seeds x 6 folds x 2
# horizons (192 cells / arch) actually complete. Previous 4h walltime gave
# us only ~25% completion per arch.
set -euo pipefail

ARCHS=(gru lstm bilstm tcn mlp transformer nlinear dlinear patchtst itransformer nbeats nhits)
ARCH=${ARCHS[$SLURM_ARRAY_TASK_ID]}

module load python/miniforge3_pytorch/2.10.0
source activate "$HOME/.conda/envs/outage"

cd "$SLURM_SUBMIT_DIR"
echo "[OUT_nn2] task=$SLURM_ARRAY_TASK_ID arch=$ARCH host=$(hostname) start=$(date)"
nvidia-smi -L

mkdir -p results/runs_per_task
export OUTAGE_RUNS_CSV="results/runs_per_task/nn2_${ARCH}_${SLURM_ARRAY_JOB_ID}_${SLURM_ARRAY_TASK_ID}.csv"
echo "[OUT_nn2] writing to $OUTAGE_RUNS_CSV"

python -m scripts.run_neural_grid_arch --arch "$ARCH"

echo "[OUT_nn2] done $(date)"
