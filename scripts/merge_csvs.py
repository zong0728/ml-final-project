"""Merge per-task experiment CSVs into the main results/experiment_runs.csv.

Usage:
    python -m scripts.merge_csvs
"""
from pathlib import Path

import pandas as pd

from src import config


def main():
    main_csv = config.RUNS_CSV
    per_task_dir = config.RESULTS_DIR / "runs_per_task"
    if not per_task_dir.exists():
        print(f"No per-task dir at {per_task_dir} — nothing to merge.")
        return

    parts = sorted(per_task_dir.glob("*.csv"))
    if not parts:
        print(f"{per_task_dir} is empty.")
        return

    dfs = []
    if main_csv.exists():
        dfs.append(pd.read_csv(main_csv))
    for p in parts:
        try:
            dfs.append(pd.read_csv(p))
        except Exception as e:
            print(f"  skip {p.name}: {e}")
    merged = pd.concat(dfs, ignore_index=True, sort=False)

    # Dedup: keep latest row for each (model, seed, horizon, fold)
    if "fold" not in merged.columns:
        merged["fold"] = -1
    merged = merged.sort_values("timestamp").drop_duplicates(
        subset=["model", "seed", "horizon", "fold"], keep="last"
    )

    merged.to_csv(main_csv, index=False)
    print(f"Merged {len(parts)} per-task files into {main_csv}: total {len(merged)} rows")


if __name__ == "__main__":
    main()
