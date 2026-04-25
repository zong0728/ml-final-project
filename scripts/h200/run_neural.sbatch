#!/bin/bash
#SBATCH --job-name=OUT_neural
#SBATCH --account=bgyq-dtai-gh
#SBATCH --partition=ghx4
#SBATCH --time=06:00:00
#SBATCH --nodes=1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --output=scripts/h200/logs/OUT_neural_%j.out
#SBATCH --error=scripts/h200/logs/OUT_neural_%j.err

# Neural model grid on H200 GPU. These models actually exercise the GPU,
# unlike LightGBM/CatBoost which are CPU-bound on aarch64.
#
# Submit: sbatch scripts/h200/run_neural.sbatch
set -euo pipefail

module load python/miniforge3_pytorch/2.10.0
source activate "$HOME/.conda/envs/outage"

cd "$SLURM_SUBMIT_DIR"
echo "[OUT_neural] host=$(hostname)"
nvidia-smi -L
echo "[OUT_neural] start=$(date)"

# Run all neural models on the rolling-origin CV folds. Multiple seeds for
# robustness — H200 makes this affordable.
python -m scripts.run_neural_grid

echo "[OUT_neural] done $(date)"
