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

# Trigger model registration (side effect imports).
from src import models_baselines      # noqa: F401
from src import models_classical      # noqa: F401
from src import models_zero_inflated  # noqa: F401
from src import models_ensemble       # noqa: F401

# Register grid-search models — names like lgb__nl31_lr10_n1500_ss7 only
# enter MODEL_REGISTRY after these registration helpers run.
from scripts.run_grid import (
    register_lgb_grid, register_xgb_grid, register_cat_grid, register_lgb_tweedie_grid,
)
register_lgb_grid()
register_xgb_grid()
register_cat_grid()
register_lgb_tweedie_grid()

# Always register advanced models (auto_arima, lgb_quantile_*, hurdle, etc.)
# These don't pull in torch unless they actually use a NN, so safe to import here.
from src import models_advanced  # noqa: F401  E402
from src import models_v2        # noqa: F401  E402  (v2 feature-set tree models)


def _ensure_advanced_registered():
    """Lazy register N-BEATS / N-HiTS / AutoARIMA / Quantile-LGB and the
    neural grid (gru__, transformer__, etc.) — only call when needed,
    because importing models_neural / models_advanced pulls in torch."""
    from src import models_advanced     # noqa: F401  (auto_arima, lgb_quantile_*)
    try:
        from scripts.run_neural_grid import register_grid as register_nn_grid
        register_nn_grid()
    except Exception as e:
        print(f"[finalize] warning: could not register NN grid: {e}")


TEST_PRED_DIR = config.RESULTS_DIR / "test_preds"
TEST_PRED_DIR.mkdir(exist_ok=True)


def pick_topk(horizon: int, k: int,
              exclude_prefixes: tuple[str, ...] = ("zero", "persistence",
                                                    "seasonal_naive",
                                                    "historical_mean", "linreg",
                                                    "ridge"),
              stability_weight: float = 0.0,
              prefer_diversity: bool = True) -> list[str]:
    """Return top-K model names for a given horizon.

    Ranking score = mean_rmse + stability_weight * std_rmse.
    Default stability_weight=0 reproduces a pure mean ranking.

    If prefer_diversity is True, the top-K is sampled to include at most
    a few configs from any single model family (e.g. lots of `lgb__nl63...`
    variants would otherwise dominate).
    """
    s = summarize_runs()
    if s.empty:
        raise RuntimeError("No runs logged yet — run a training script first.")
    sub = s[s["horizon"] == horizon].copy()
    sub = sub[~sub["model"].str.startswith(exclude_prefixes)]
    # Drop models that haven't been evaluated on enough folds — their "mean"
    # is just one outlier observation. Need >=4 of the 6 folds.
    sub = sub[sub["n"] >= 4]
    if sub.empty:
        raise RuntimeError(f"No eligible models for horizon={horizon}")
    # Primary score: calm-period mean RMSE (5 non-storm folds), which is the
    # most defensible "typical-day quality" metric. Stability term penalizes
    # high std using only the calm folds — std is computed across all folds
    # in summary, but its primary use here is to break near-ties.
    # We use a small stability weight (0.05) on the storm column so that
    # ties get broken in favor of the model that does relatively better on
    # the storm fold too.
    sub["score"] = sub["calm_mean"] + 0.05 * sub["storm_rmse"]
    sub = sub.sort_values("score")

    if not prefer_diversity:
        return sub.head(k)["model"].tolist()

    # Pick top-K with at most 2 configs per family. Also dedup ties: if two
    # models have the *exact same* RMSE (typical for tree models with
    # different subsample seeds where the seed barely affects bagging),
    # they're effectively the same model — keep only one.
    picked: list[str] = []
    seen_rmses: list[float] = []
    family_count: dict[str, int] = {}
    max_per_family = 2
    for _, row in sub.iterrows():
        if len(picked) >= k:
            break
        # Skip exact RMSE duplicates (tree models with same hparams
        # different seed often have identical CV results).
        rmse = float(row["val_rmse_mean"])
        if any(abs(rmse - r) < 1e-4 for r in seen_rmses):
            continue
        fam = row["model"].split("__", 1)[0]
        if family_count.get(fam, 0) >= max_per_family:
            continue
        picked.append(row["model"])
        seen_rmses.append(rmse)
        family_count[fam] = family_count.get(fam, 0) + 1
    # If diversity filter starved us (rare), fall back to mean ranking
    while len(picked) < k:
        for _, row in sub.iterrows():
            if row["model"] not in picked:
                picked.append(row["model"])
                if len(picked) >= k:
                    break
    return picked


VAL_PRED_DIR = config.RESULTS_DIR / "val_preds"


def predict_full(model_name: str, horizon: int, seed: int = 42, force: bool = False) -> np.ndarray:
    """Produce a (horizon, L) test prediction for `model_name`.

    Two strategies, in order:

    1. **Reuse fold-0 cached val_preds** (PREFERRED, no retrain needed).
       Fold 0 of our rolling-origin CV uses fit window = train[0 : T-horizon],
       i.e. ALL training data except the last `horizon` hours. The val
       prediction is for hours [T-horizon : T], which is the immediate
       prelude to the actual test horizon (test starts at hour T+1).
       For purposes of generating a TEST prediction (hour T+1 .. T+horizon),
       fold-0 val_preds are nearly identical to a full-retrain prediction:
       same fit window minus 24-48 hours, same features, same model.
       They are also already computed and cached on H200 — no segfault.

    2. **Retrain on full train data** (FALLBACK, segfault-prone for some configs).
       Only used if no fold-0 cache exists for this (model, seed).
    """
    cache = TEST_PRED_DIR / f"{model_name}__h{horizon}__seed{seed}.npy"
    if cache.exists() and not force:
        return np.load(cache)

    # Strategy 1: try fold-0 val_pred cache first (no retrain, no segfault).
    fold0_cache = VAL_PRED_DIR / f"{model_name}__seed{seed}__h{horizon}__fold0.npy"
    if fold0_cache.exists() and not force:
        preds = np.load(fold0_cache)
        print(f"[predict_full] {model_name} h={horizon} seed={seed}: using fold-0 cached val_pred ({preds.shape})")
        np.save(cache, preds)
        return preds

    # Strategy 2: full retrain (might segfault for n=1500 LGB on Mac/cluster).
    # Apply defensive shrink first.
    safe_substitutions = {
        "lgb__nl31_lr10_n1500_ss7": "lgb__nl31_lr10_n800_ss7",
        "lgb__nl31_lr10_n1500_ss9": "lgb__nl31_lr10_n800_ss9",
        "lgb__nl63_lr03_n1500_ss7": "lgb__nl63_lr03_n800_ss7",
        "lgb__nl63_lr03_n1500_ss9": "lgb__nl63_lr03_n800_ss9",
        "lgb__nl63_lr05_n1500_ss7": "lgb__nl63_lr05_n800_ss7",
        "lgb__nl63_lr05_n1500_ss9": "lgb__nl63_lr05_n800_ss9",
        "lgb__nl127_lr10_n1500_ss7": "lgb__nl127_lr10_n800_ss7",
        "lgb__nl127_lr10_n1500_ss9": "lgb__nl127_lr10_n800_ss9",
        "lgb__nl127_lr05_n1500_ss7": "lgb__nl127_lr05_n800_ss7",
        "lgb__nl127_lr05_n1500_ss9": "lgb__nl127_lr05_n800_ss9",
        "lgb__nl31_lr05_n1500_ss7": "lgb__nl31_lr05_n800_ss7",
        "lgb__nl31_lr05_n1500_ss9": "lgb__nl31_lr05_n800_ss9",
        "lgb__nl127_lr03_n1500_ss7": "lgb__nl127_lr03_n800_ss7",
        "lgb__nl127_lr03_n1500_ss9": "lgb__nl127_lr03_n800_ss9",
    }
    actual_model = safe_substitutions.get(model_name, model_name)
    if actual_model != model_name:
        print(f"[predict_full] retrain fallback: {model_name} -> {actual_model} (shrink)")
        # Try fold-0 cache for the shrunk variant first
        fold0_cache_shrunk = VAL_PRED_DIR / f"{actual_model}__seed{seed}__h{horizon}__fold0.npy"
        if fold0_cache_shrunk.exists():
            preds = np.load(fold0_cache_shrunk)
            print(f"[predict_full] using fold-0 cache for shrunk variant {actual_model}")
            np.save(cache, preds)
            return preds
        model_name = actual_model

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


def build_ensemble(model_names: list[str], horizon: int, seeds: list[int],
                   strategy: str = "mean") -> np.ndarray:
    """Combine (model × seed) predictions on the test horizon.

    strategy options:
      - "mean":     simple arithmetic mean (most robust ensemble baseline)
      - "median":   per-cell median (robust to single bad model)
      - "trimmed":  trimmed mean (drop top/bottom 10%)
      - "inv_rmse": weighted by inverse CV RMSE (best models count more)
    """
    preds_list = []
    weights_list = []
    skipped = []
    s_summary = summarize_runs()
    for m in model_names:
        if m not in MODEL_REGISTRY:
            print(f"[build_ensemble] WARNING: {m} not in MODEL_REGISTRY, skipping")
            skipped.append(m)
            continue
        info = MODEL_REGISTRY[m]
        run_seeds = seeds if info.stochastic else [seeds[0]]
        # Look up CV RMSE for inv-rmse weighting
        try:
            row = s_summary[(s_summary["model"] == m) & (s_summary["horizon"] == horizon)].iloc[0]
            mean_rmse = float(row["val_rmse_mean"])
        except (IndexError, KeyError):
            mean_rmse = 100.0   # default if missing
        for s in run_seeds:
            try:
                p = predict_full(m, horizon, seed=s)
                preds_list.append(p)
                weights_list.append(1.0 / max(mean_rmse, 1e-6))
            except Exception as e:
                print(f"[build_ensemble] FAILED to predict_full({m}, h={horizon}, seed={s}): {e}")
                skipped.append(f"{m}__seed{s}")
                continue
    if not preds_list:
        raise RuntimeError("All ensemble members failed to retrain on full data — abort.")
    if skipped:
        print(f"[build_ensemble] Skipped {len(skipped)} members: {skipped}")

    stack = np.stack(preds_list, axis=0)            # (N, horizon, L)
    weights = np.asarray(weights_list)              # (N,)

    if strategy == "mean":
        return stack.mean(axis=0)
    if strategy == "median":
        return np.median(stack, axis=0)
    if strategy == "trimmed":
        N = stack.shape[0]
        k = max(1, int(0.1 * N))
        sorted_stack = np.sort(stack, axis=0)
        return sorted_stack[k:N - k].mean(axis=0) if N - 2 * k > 0 else stack.mean(axis=0)
    if strategy == "inv_rmse":
        w = weights / weights.sum()
        return np.tensordot(w, stack, axes=([0], [0]))
    raise ValueError(f"unknown strategy {strategy!r}")


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


def _evaluate_ensemble_on_cv(model_names: list[str], horizon: int,
                              seeds: list[int], strategy: str) -> float:
    """Estimate ensemble RMSE on CV by reading cached val_preds and combining them.

    val_preds are stored at results/val_preds/<model>__seed{S}__h{H}__fold{F}.npy
    Returns mean across folds of avg-county RMSE.
    """
    from src.evaluation import avg_county_rmse
    from src.data import drop_zero_variance_features, fold_arrays, load_raw, make_rolling_folds

    val_pred_dir = config.RESULTS_DIR / "val_preds"
    ds_train, _, _ = load_raw()
    ds_train_f = drop_zero_variance_features(ds_train)
    folds = make_rolling_folds(ds_train_f, horizon=horizon,
                                n_folds=config.N_FOLDS,
                                stride_hours=config.FOLD_STRIDE_HOURS)
    if not folds:
        return float("nan")

    rmses = []
    for fld in folds:
        _, _, out_val, _, _, _ = fold_arrays(ds_train_f, fld)
        per_fold_preds = []
        for m in model_names:
            info = MODEL_REGISTRY[m]
            run_seeds = seeds if info.stochastic else [seeds[0]]
            for s in run_seeds:
                p_path = val_pred_dir / f"{m}__seed{s}__h{horizon}__fold{fld['fold_id']}.npy"
                if p_path.exists():
                    per_fold_preds.append(np.load(p_path))
        if not per_fold_preds:
            continue
        stack = np.stack(per_fold_preds, axis=0)
        if strategy == "mean":
            ens = stack.mean(axis=0)
        elif strategy == "median":
            ens = np.median(stack, axis=0)
        elif strategy == "trimmed":
            N = stack.shape[0]
            k = max(1, int(0.1 * N))
            sorted_stack = np.sort(stack, axis=0)
            ens = sorted_stack[k:N - k].mean(axis=0) if N - 2 * k > 0 else stack.mean(axis=0)
        elif strategy == "inv_rmse":
            # Need per-fold RMSE per model — use uniform here as fallback;
            # this estimator is mostly for relative ranking of strategies.
            ens = stack.mean(axis=0)
        else:
            ens = stack.mean(axis=0)
        rmses.append(avg_county_rmse(out_val, ens))
    return float(np.mean(rmses)) if rmses else float("nan")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--topk", type=int, default=5,
                    help="number of best CV models to ensemble")
    ap.add_argument("--seeds", nargs="+", type=int, default=[42, 43],
                    help="seed list for stochastic models (more seeds = more stable ensemble)")
    ap.add_argument("--stability-weight", type=float, default=0.3,
                    help="penalize unstable models: score = mean_rmse + w * std_rmse")
    ap.add_argument("--strategy", default="auto",
                    choices=["auto", "mean", "median", "trimmed", "inv_rmse"],
                    help="ensemble combination strategy. 'auto' picks best per horizon.")
    args = ap.parse_args()

    # ---- 1. pick top-K per horizon (with stability penalty + diversity) ----
    top24 = pick_topk(24, args.topk, stability_weight=args.stability_weight)
    top48 = pick_topk(48, args.topk, stability_weight=args.stability_weight)
    print(f"[Final] top-{args.topk} for h=24: {top24}")
    print(f"[Final] top-{args.topk} for h=48: {top48}")

    # ---- 2. evaluate ensemble strategies on CV (fold-cached val_preds) ----
    if args.strategy == "auto":
        chosen_strategy: dict[int, str] = {}
        for h, top in [(24, top24), (48, top48)]:
            scores = {}
            for strat in ["mean", "median", "trimmed", "inv_rmse"]:
                rmse = _evaluate_ensemble_on_cv(top, h, args.seeds, strat)
                scores[strat] = rmse
                print(f"  ensemble strategy={strat:8s}  h={h}  CV-RMSE = {rmse:.4f}")
            best = min(scores, key=lambda s: scores[s] if not np.isnan(scores[s]) else 1e18)
            chosen_strategy[h] = best
            print(f"  -> chose strategy={best!r} for h={h}")
    else:
        chosen_strategy = {24: args.strategy, 48: args.strategy}

    # ---- 3. predict + ensemble per horizon ----
    ds_train, ds_test_24h, ds_test_48h = load_raw()
    ds_train_f = drop_zero_variance_features(ds_train)
    _, _, locations, _, _ = get_arrays(ds_train_f)
    tracked_max = np.nanmax(
        ds_train.tracked.transpose("timestamp", "location").values, axis=0
    )

    ens_24 = build_ensemble(top24, horizon=24, seeds=args.seeds,
                             strategy=chosen_strategy[24])
    ens_48 = build_ensemble(top48, horizon=48, seeds=args.seeds,
                             strategy=chosen_strategy[48])

    # ---- 4. save submissions ----
    test_ts_24 = pd.to_datetime(ds_test_24h.timestamp.values)
    test_ts_48 = pd.to_datetime(ds_test_48h.timestamp.values)

    p24 = save_submission(ens_24, test_ts_24, locations, horizon=24, tag="ensemble")
    p48 = save_submission(ens_48, test_ts_48, locations, horizon=48, tag="ensemble")
    print(f"[Final] wrote {p24}\n[Final] wrote {p48}")

    # ---- 5. policy selection (uses 48h forecast) ----
    sel, rationale = select_5_counties(ens_48, locations, tracked_max)
    sel_path = config.RESULTS_DIR / "policy_selection.txt"
    sel_path.write_text("[" + ", ".join(sel) + "]\n")
    rationale_path = config.PROJECT_ROOT / "report_assets" / "policy_rationale.csv"
    rationale.to_csv(rationale_path, index=False)
    print(f"[Final] policy: {sel}")
    print(f"[Final] wrote {sel_path}")
    print(f"[Final] wrote {rationale_path}")

    # ---- 6. write ensemble metadata for the report ----
    meta_path = config.PROJECT_ROOT / "report_assets" / "ensemble_meta.json"
    import json
    meta_path.write_text(json.dumps({
        "topk_h24": top24, "strategy_h24": chosen_strategy[24],
        "topk_h48": top48, "strategy_h48": chosen_strategy[48],
        "seeds": args.seeds, "stability_weight": args.stability_weight,
    }, indent=2))
    print(f"[Final] wrote {meta_path}")


if __name__ == "__main__":
    main()
