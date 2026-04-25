"""End-to-end finalization:

1. Identify the top-K models by mean fold RMSE (from results/experiment_runs.csv)
2. Retrain each on FULL train data and predict the test horizon (24h + 48h)
3. Build an equal-weight ensemble of the top-K predictions
4. Save submissions to results/submissions/
5. Compute the 5-county generator selection from the ensemble's predictions
6. Save selection to results/policy_selection.txt and a one-page rationale to
   report_assets/policy_rationale.md

Usage:
    python -m scripts.finalize --topk 5

Be careful: this trains every chosen model from scratch on the FULL train set.
It takes a few minutes but is idempotent — predictions are cached under
results/test_preds/<model>__h{H}.npy
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr

from src import config
from src.data import drop_zero_variance_features, load_raw, get_arrays
from src.evaluation import save_submission
from src.registry import MODEL_REGISTRY
from src.training import set_seed, summarize_runs

# Trigger model registration (side effect imports). Tree-only here; neural
# models are loaded lazily by the runner if needed.
from src import models_baselines      # noqa: F401
from src import models_classical      # noqa: F401
from src import models_zero_inflated  # noqa: F401
from src import models_ensemble       # noqa: F401


TEST_PRED_DIR = config.RESULTS_DIR / "test_preds"
TEST_PRED_DIR.mkdir(exist_ok=True)


def pick_topk(horizon: int, k: int,
              exclude_prefixes: tuple[str, ...] = ("zero", "persistence",
                                                    "seasonal_naive", "historical_mean",
                                                    "linreg")) -> list[str]:
    """Return top-K model names ranked by mean CV RMSE for a given horizon.

    We exclude weak baselines from the ensemble pool because mixing in a
    constant-zero predictor doesn't improve RMSE on storm folds.
    """
    s = summarize_runs()
    if s.empty:
        raise RuntimeError("No runs logged yet — run a training script first.")
    sub = s[s["horizon"] == horizon].copy()
    sub = sub[~sub["model"].str.startswith(exclude_prefixes)]
    sub = sub.sort_values("val_rmse_mean").head(k)
    return sub["model"].tolist()


def predict_full(model_name: str, horizon: int, seed: int = 42, force: bool = False) -> np.ndarray:
    """Train `model_name` on full train data and produce (horizon, L) test prediction.

    Caches numpy under results/test_preds/<model>__h{H}__seed{S}.npy
    """
    cache = TEST_PRED_DIR / f"{model_name}__h{horizon}__seed{seed}.npy"
    if cache.exists() and not force:
        return np.load(cache)

    ds_train, ds_test_24h, ds_test_48h = load_raw()
    ds_train_f = drop_zero_variance_features(ds_train)
    out_full, weather_full, locations, _, ts_full = get_arrays(ds_train_f)

    info = MODEL_REGISTRY[model_name]
    set_seed(seed)
    preds, meta = info.fn(
        out_fit=out_full, weather_fit=weather_full,
        timestamps_fit=ts_full, locations=locations,
        horizon=horizon, seed=seed,
    )
    if preds.shape != (horizon, len(locations)):
        raise ValueError(f"{model_name} returned {preds.shape}, expected ({horizon},{len(locations)})")
    np.save(cache, preds)
    return preds


def build_ensemble(model_names: list[str], horizon: int, seeds: list[int]) -> np.ndarray:
    """Equal-weight mean of (model × seed) predictions on the test horizon."""
    preds_list = []
    for m in model_names:
        info = MODEL_REGISTRY[m]
        run_seeds = seeds if info.stochastic else [seeds[0]]
        for s in run_seeds:
            p = predict_full(m, horizon, seed=s)
            preds_list.append(p)
    stack = np.stack(preds_list, axis=0)  # (N, horizon, L)
    return stack.mean(axis=0)


def select_5_counties(pred_48h: np.ndarray, locations: list[str], tracked_max: np.ndarray
                      ) -> tuple[list[str], pd.DataFrame]:
    """Pick 5 county allocations (with multiplicity) for backup generators.

    Each generator covers up to 1000 households. Strategy:
      * For each county c, predicted-served-households-c
        = sum_h min(pred[h, c], 1000)             (one generator's marginal benefit)
      * Greedy assignment: at each of 5 steps, pick the county whose marginal
        benefit (capped at remaining un-served outage) is highest. After
        assignment, subtract the served capacity (up to 1000) from all hours.
    Returns the list of 5 FIPS codes (with possible repeats) plus a rationale df.
    """
    H, L = pred_48h.shape
    remaining = pred_48h.copy()                     # (H, L) un-served outage
    selections = []
    rationale_rows = []
    for step in range(1, 6):
        # Marginal benefit of placing one generator in county c:
        # served = sum_h min(remaining[h, c], 1000)
        cap = np.minimum(remaining, 1000.0)         # (H, L)
        # Households-served *upper-bounded by tracked_max* (can't help more
        # households than exist in the county).
        county_cap_per_hour = np.minimum(1000.0, tracked_max[None, :])
        cap = np.minimum(cap, county_cap_per_hour)
        marginal = cap.sum(axis=0)                  # (L,)
        ci = int(np.argmax(marginal))
        selections.append(locations[ci])
        rationale_rows.append({
            "step": step,
            "county_fips": locations[ci],
            "marginal_served_household_hours": float(marginal[ci]),
            "tracked_max": int(tracked_max[ci]),
            "remaining_peak_at_pick_time": float(pred_48h[:, ci].max()),
        })
        # subtract capacity from this county's remaining
        remaining[:, ci] -= np.minimum(remaining[:, ci], 1000.0)
        remaining = np.clip(remaining, 0, None)
    rdf = pd.DataFrame(rationale_rows)
    return selections, rdf


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--topk", type=int, default=5,
                    help="number of best CV models to ensemble")
    ap.add_argument("--seeds", nargs="+", type=int, default=[42],
                    help="seed list for stochastic models (more seeds = more stable ensemble)")
    args = ap.parse_args()

    # ---- 1. pick top-K per horizon ----
    top24 = pick_topk(24, args.topk)
    top48 = pick_topk(48, args.topk)
    print(f"[Final] top-{args.topk} for h=24: {top24}")
    print(f"[Final] top-{args.topk} for h=48: {top48}")

    # ---- 2. predict + ensemble per horizon ----
    ds_train, ds_test_24h, ds_test_48h = load_raw()
    ds_train_f = drop_zero_variance_features(ds_train)
    _, _, locations, _, _ = get_arrays(ds_train_f)
    tracked_max = np.nanmax(
        ds_train.tracked.transpose("timestamp", "location").values, axis=0
    )

    ens_24 = build_ensemble(top24, horizon=24, seeds=args.seeds)
    ens_48 = build_ensemble(top48, horizon=48, seeds=args.seeds)

    # ---- 3. save submissions ----
    test_ts_24 = pd.to_datetime(ds_test_24h.timestamp.values)
    test_ts_48 = pd.to_datetime(ds_test_48h.timestamp.values)

    p24 = save_submission(ens_24, test_ts_24, locations, horizon=24, tag="ensemble")
    p48 = save_submission(ens_48, test_ts_48, locations, horizon=48, tag="ensemble")
    print(f"[Final] wrote {p24}\n[Final] wrote {p48}")

    # ---- 4. policy selection (uses 48h forecast) ----
    sel, rationale = select_5_counties(ens_48, locations, tracked_max)
    sel_path = config.RESULTS_DIR / "policy_selection.txt"
    sel_path.write_text("[" + ", ".join(sel) + "]\n")
    rationale_path = config.PROJECT_ROOT / "report_assets" / "policy_rationale.csv"
    rationale.to_csv(rationale_path, index=False)
    print(f"[Final] policy: {sel}")
    print(f"[Final] wrote {sel_path}")
    print(f"[Final] wrote {rationale_path}")


if __name__ == "__main__":
    main()
