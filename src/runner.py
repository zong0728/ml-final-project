"""Main orchestrator: run every (model × seed × horizon) and log results.

Key guarantees:
  * Resumable: if Colab disconnects, rerun — already-logged triplets are skipped.
  * Each run's val prediction is cached to results/val_preds/<model>_<seed>_<h>.npy
    so we can build ensembles later without re-training.
  * Progress is printed per run (+ cumulative wall clock) so you can follow along
    if you check in.
"""
from __future__ import annotations

import time
import traceback
from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr

from . import config
from .data import (
    drop_zero_variance_features,
    get_arrays,
    load_raw,
    split_train_val,
)
from .evaluation import avg_county_rmse, save_submission
from .registry import MODEL_REGISTRY, ModelInfo
from .training import already_ran, log_run, set_seed


# Import to trigger registration (side effect)
from . import models_baselines  # noqa: F401
from . import models_classical  # noqa: F401
from . import models_neural     # noqa: F401
from . import models_sota       # noqa: F401


VAL_PRED_DIR = config.RESULTS_DIR / "val_preds"
VAL_PRED_DIR.mkdir(exist_ok=True)


def run_all(
    model_names: list[str] | None = None,
    horizons: list[int] = config.HORIZONS,
    seeds: list[int] = config.SEEDS,
    skip_on_error: bool = True,
    dry_run: bool = False,
) -> pd.DataFrame:
    """Iterate over every (model, seed, horizon) combination and log to CSV.

    Parameters
    ----------
    model_names : list of str, optional
        Restrict to a subset of registered models. None means all.
    horizons : list of int
        Forecast horizons to evaluate (default: [24, 48]).
    seeds : list of int
        Seeds used for STOCHASTIC models. Deterministic models always run once.
    skip_on_error : bool
        If True, log the error and continue; else raise.
    dry_run : bool
        Print the schedule without running anything.
    """
    ds_train, _, _ = load_raw()
    ds_train_f = drop_zero_variance_features(ds_train)
    ds_fit, ds_val = split_train_val(ds_train_f, val_hours=config.VAL_HOURS)

    out_fit, weather_fit, locations, _, timestamps_fit = get_arrays(ds_fit)
    out_val, _, _, _, _ = get_arrays(ds_val)
    # Truth slices for each horizon
    val_truths = {h: out_val[:h] for h in horizons}

    if model_names is None:
        model_names = list(MODEL_REGISTRY.keys())

    # ---- Build the schedule ----
    schedule: list[tuple[str, int, int]] = []
    for name in model_names:
        info = MODEL_REGISTRY[name]
        run_seeds = seeds if info.stochastic else [seeds[0]]
        for s in run_seeds:
            for h in horizons:
                schedule.append((name, s, h))

    print(f"[Runner] Scheduled {len(schedule)} runs across {len(model_names)} models, "
          f"seeds={seeds}, horizons={horizons}")
    if dry_run:
        for n, s, h in schedule:
            tag = "STOCH" if MODEL_REGISTRY[n].stochastic else "DET"
            print(f"  {tag:5s}  model={n:16s}  seed={s}  horizon={h}")
        return pd.DataFrame(schedule, columns=["model", "seed", "horizon"])

    # ---- Execute ----
    total_start = time.time()
    completed = 0
    skipped = 0
    errors = 0
    for i, (name, seed, horizon) in enumerate(schedule, start=1):
        if already_ran(name, seed, horizon):
            skipped += 1
            continue

        info: ModelInfo = MODEL_REGISTRY[name]
        header = f"[{i:>3}/{len(schedule)}] model={name}  seed={seed}  horizon={horizon}h"
        print(f"\n{header}\n{'-' * len(header)}")
        set_seed(seed)

        t0 = time.time()
        try:
            preds, meta = info.fn(
                out_fit=out_fit,
                weather_fit=weather_fit,
                timestamps_fit=timestamps_fit,
                locations=locations,
                horizon=horizon,
                seed=seed,
            )
            elapsed = time.time() - t0

            # Validate shape
            if preds.shape != (horizon, len(locations)):
                raise ValueError(
                    f"{name} returned shape {preds.shape}, expected ({horizon}, {len(locations)})"
                )

            val_rmse = avg_county_rmse(val_truths[horizon], preds)

            # Cache val prediction
            np.save(VAL_PRED_DIR / f"{name}__seed{seed}__h{horizon}.npy", preds)

            log_run(
                model_name=name,
                seed=seed,
                horizon=horizon,
                val_rmse=val_rmse,
                train_time_s=elapsed,
                config_dict={**meta, "tier": info.tier},
            )
            completed += 1
            elapsed_total = time.time() - total_start
            print(f"  val_rmse={val_rmse:.4f}  time={elapsed:.1f}s  total={elapsed_total/60:.1f}min")
        except Exception as e:
            errors += 1
            print(f"  ERROR running {name}: {e}")
            if not skip_on_error:
                raise
            traceback.print_exc()

    elapsed_total = time.time() - total_start
    print(
        f"\n[Runner] Done. completed={completed} skipped={skipped} errors={errors} "
        f"total_wall={elapsed_total/60:.1f}min"
    )
    from .training import summarize_runs
    return summarize_runs()


def retrain_best_on_full_and_predict_test(
    model_name: str,
    horizon: int,
    seed: int = 42,
) -> np.ndarray:
    """After the comparison, retrain the best model on the FULL train set
    (including the held-out val window) and produce test predictions.

    Returns (horizon, L) predictions and also writes a submission CSV.
    """
    ds_train, ds_test_24h, ds_test_48h = load_raw()
    ds_train_f = drop_zero_variance_features(ds_train)

    out_full, weather_full, locations, _, timestamps_full = get_arrays(ds_train_f)

    info = MODEL_REGISTRY[model_name]
    set_seed(seed)
    t0 = time.time()
    preds, meta = info.fn(
        out_fit=out_full,
        weather_fit=weather_full,
        timestamps_fit=timestamps_full,
        locations=locations,
        horizon=horizon,
        seed=seed,
    )
    elapsed = time.time() - t0
    print(f"[Final] {model_name}@{horizon}h trained on full data in {elapsed:.1f}s — meta={meta}")

    # Align test timestamps
    if horizon == 24:
        test_ts = pd.to_datetime(ds_test_24h.timestamp.values)
    else:
        test_ts = pd.to_datetime(ds_test_48h.timestamp.values)

    # Save submission CSV
    out_path = save_submission(preds, test_ts, locations, horizon=horizon, tag=f"final_{model_name}")
    print(f"[Final] wrote submission → {out_path}")
    return preds
