"""Heavy-bagging pass: take the current top-K models and run them with
many seeds on every fold, so the final ensemble can average over more
independent runs (variance reduction).

For LGB/Cat/XGB the random-seed effect comes from `subsample` + per-tree
column subsampling, so each seed gives a slightly different model.

Default: 10 seeds × top 8 configs × 6 folds × 2 horizons = 960 cells
At ~30s per cell on H200, ~8 GPU-hours total.

    python -m scripts.run_bagging --n-seeds 10
"""
from __future__ import annotations

import argparse

from src.runner import run_all
from src.training import summarize_runs


# Top configs identified from prior runs (calm-period RMSE ranked).
# These are the "champion" configs we want to bag aggressively.
BAG_CONFIGS = [
    "lgb_quantile_p50",
    "lgb_v2_log_corrected",
    "lgb__nl63_lr03_n800_ss7",       # top LGB grid
    "lgb__nl63_lr03_n400_ss7",
    "cat__d6_lr03_n400",              # top Cat grid
    "cat__d4_lr05_n800",
    "xgb__d8_lr03_n400",              # top XGB grid
    "two_stage_lgb",                   # diversity from hurdle
]


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--n-seeds", type=int, default=10)
    p.add_argument("--seed-start", type=int, default=42)
    args = p.parse_args()

    seeds = list(range(args.seed_start, args.seed_start + args.n_seeds))
    print(f"[Bag] {len(BAG_CONFIGS)} configs × {len(seeds)} seeds = "
           f"{len(BAG_CONFIGS) * len(seeds)} (model, seed) tuples")
    print(f"[Bag] seeds: {seeds}")

    # Need to register grid models since some BAG_CONFIGS are grid-generated names
    from scripts.run_grid import (register_lgb_grid, register_xgb_grid,
                                    register_cat_grid, register_lgb_tweedie_grid)
    register_lgb_grid()
    register_xgb_grid()
    register_cat_grid()
    register_lgb_tweedie_grid()

    run_all(model_names=BAG_CONFIGS, seeds=seeds, skip_on_error=True)

    print("\n========= Bagging summary =========")
    s = summarize_runs()
    sub = s[s["model"].isin(BAG_CONFIGS)]
    print(sub.sort_values(["horizon", "calm_mean"]).to_string(index=False))


if __name__ == "__main__":
    main()
