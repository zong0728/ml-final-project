#!/bin/bash
#SBATCH --job-name=OUT_final
#SBATCH --account=bgyq-dtai-gh
#SBATCH --partition=ghx4
#SBATCH --time=02:00:00
#SBATCH --nodes=1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --output=scripts/h200/logs/OUT_final_%j.out
#SBATCH --error=scripts/h200/logs/OUT_final_%j.err

# Final consolidation job: merge CSVs, retrain top-K on full data, build
# ensemble, write submission CSVs + 5-county policy.
set -euo pipefail

module load python/miniforge3_pytorch/2.10.0
source activate "$HOME/.conda/envs/outage"

cd "$SLURM_SUBMIT_DIR"
echo "[OUT_final] start=$(date) host=$(hostname)"

# Don't overwrite the merged main csv during retrain (predict_full doesn't write
# to the runs_csv anyway, but be paranoid).

echo "==== 1. Merge per-task CSVs ===="
python -m scripts.merge_csvs

echo "==== 2. CV summary plots ===="
python -m scripts.plot_results

echo "==== 3. Hyperparameter sensitivity plots ===="
python -m scripts.plot_hparam_sensitivity

echo "==== 4. Final ensemble + 5-county policy ===="
python -m scripts.finalize --topk 5 --seeds 42 43 --stability-weight 0.3 --strategy auto

echo "==== 5. Report tables ===="
python -m scripts.build_report_tables

echo "[OUT_final] done $(date)"
