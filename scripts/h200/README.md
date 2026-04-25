# H200 Setup & Run Guide

This project must remain isolated from the diffusion project (jobs, env, paths).
Conventions enforced here:

| Item | Value |
|---|---|
| Project dir | `/projects/bgyq/sguan/ml-final-outage/` |
| Conda env | `outage` |
| SLURM partition | `ghx4` |
| SLURM job-name prefix | `OUT_` |
| Repo | `https://github.com/zong0728/ml-final-project` |

## One-time setup (login node)

```bash
# 1. Pull repo + create env + install deps
cd ~ && bash <(curl -s https://raw.githubusercontent.com/zong0728/ml-final-project/main/scripts/h200/setup_env.sh)
# ... or, after cloning manually:
cd /projects/bgyq/sguan/ml-final-outage
bash scripts/h200/setup_env.sh
```

## Copy dataset (from your laptop, one-time)

```bash
# From local: /Users/zong/Desktop/ml-final-project/
rsync -avz --progress dataset/data/*.nc \
    sguan@dtai-login.delta.ncsa.illinois.edu:/projects/bgyq/sguan/ml-final-outage/dataset/data/
```

The dataset is excluded from git (`.gitignore` has `dataset/`), so this rsync
is the only way it gets to H200.

## Submit grid-search jobs

From the project root on the login node:

```bash
# Sub-grids to keep individual jobs small:
sbatch scripts/h200/run_grid.sbatch lgb       # ~54 LightGBM configs
sbatch scripts/h200/run_grid.sbatch xgb       # ~18 XGBoost configs
sbatch scripts/h200/run_grid.sbatch cat       # ~18 CatBoost configs
sbatch scripts/h200/run_grid.sbatch lgbtw     # ~24 LightGBM-Tweedie configs

# Or all together (longer):
sbatch scripts/h200/run_grid.sbatch all
```

Job names will appear as `OUT_grid` in `squeue -u $USER`. Diffusion jobs
(`T1_extend`, `T2_ch384`, etc.) are unaffected.

## Pull results back to laptop

```bash
# Sync results (from local):
rsync -avz \
    sguan@dtai-login.delta.ncsa.illinois.edu:/projects/bgyq/sguan/ml-final-outage/results/ \
    results_h200/
```

Then on local: `python -m scripts.plot_results` will produce report figures
once `results/experiment_runs.csv` is unioned (manual concat ok).

## Final inference (after best models picked)

```bash
# On H200, after the grid is done:
python -m scripts.finalize --topk 5 --seeds 42 43 44

# Outputs to retrieve:
#   results/submissions/ensemble_pred_24h.csv
#   results/submissions/ensemble_pred_48h.csv
#   results/policy_selection.txt
```
