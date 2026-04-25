"""Main orchestrator: run every (model × seed × horizon × fold) and log results.

Key design:
  * **Rolling-origin CV** — instead of one held-out window, we evaluate on
    `n_folds` folds whose val window matches the production setup
    (val = `horizon` hours immediately after fit, no future weather).
    Reported metric per (model, seed, horizon) is the mean across folds.
  * Resumable: already-logged (model, seed, horizon, fold) rows are skipped.
  * Each fold's val prediction cached as
        results/val_preds/<model>__seed{s}__h{H}__fold{F}.npy
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
    fold_arrays,
    load_raw,
    make_rolling_folds,
)
from .evaluation import avg_county_rmse, save_submission
from .registry import MODEL_REGISTRY, ModelInfo
from .training import already_ran_fold, log_run_fold, set_seed


# Import to trigger registration (side effect).
from . import models_baselines      # noqa: F401
from . import models_classical      # noqa: F401
from . import models_zero_inflated  # noqa: F401
from . import models_ensemble       # noqa: F401
from . import models_neural         # noqa: F401
from . import models_sota           # noqa: F401


VAL_PRED_DIR = config.RESULTS_DIR / "val_preds"
VAL_PRED_DIR.mkdir(exist_ok=True)


def run_all(
    model_names: list[str] | None = None,
    horizons: list[int] = config.HORIZONS,
    seeds: list[int] = config.SEEDS,
    n_folds: int = config.N_FOLDS,
    fold_stride_h: int = config.FOLD_STRIDE_HOURS,
    skip_on_error: bool = True,
    dry_run: bool = False,
) -> pd.DataFrame:
    """Iterate over every (model, seed, horizon, fold) combination.

    Parameters
    ----------
    n_folds, fold_stride_h
        Rolling-origin CV controls. Default 6 folds, stride 72h.
    """
    ds_train, _, _ = load_raw()
    ds_train_f = drop_zero_variance_features(ds_train)

    # Build fold list per horizon (fold structure depends on horizon size).
    folds_by_h: dict[int, list[dict]] = {}
    for h in horizons:
        f = make_rolling_folds(ds_train_f, horizon=h, n_folds=n_folds,
                               stride_hours=fold_stride_h, require_nonzero=True)
        if not f:
            raise RuntimeError(f"No valid folds for horizon={h}")
        folds_by_h[h] = f

    print(f"[Runner] CV setup: n_folds={n_folds}, stride={fold_stride_h}h")
    for h in horizons:
        print(f"  horizon={h}h: {len(folds_by_h[h])} folds:")
        for fld in folds_by_h[h]:
            print(f"    fold {fld['fold_id']}: val={fld['val_start_ts']} → {fld['val_end_ts']}  "
                  f"(sum={fld['val_sum']:.0f}, peak={fld['val_peak']:.0f})")

    if model_names is None:
        model_names = list(MODEL_REGISTRY.keys())

    # Build flat schedule: (model, seed, horizon, fold_id)
    schedule: list[tuple[str, int, int, int]] = []
    for name in model_names:
        info = MODEL_REGISTRY[name]
        run_seeds = seeds if info.stochastic else [seeds[0]]
        for s in run_seeds:
            for h in horizons:
                for fld in folds_by_h[h]:
                    schedule.append((name, s, h, fld["fold_id"]))

    print(f"\n[Runner] Scheduled {len(schedule)} runs across {len(model_names)} models, "
          f"seeds={seeds}, horizons={horizons}, n_folds={n_folds}")
    if dry_run:
        return pd.DataFrame(schedule, columns=["model", "seed", "horizon", "fold"])

    # ---- Execute ----
    total_start = time.time()
    completed = skipped = errors = 0
    for i, (name, seed, horizon, fold_id) in enumerate(schedule, start=1):
        if already_ran_fold(name, seed, horizon, fold_id):
            skipped += 1
            continue

        info: ModelInfo = MODEL_REGISTRY[name]
        fld = folds_by_h[horizon][fold_id]
        out_fit, weather_fit, out_val, locations, _, ts_fit = fold_arrays(ds_train_f, fld)

        header = (f"[{i:>4}/{len(schedule)}] {name}  seed={seed}  h={horizon}  "
                  f"fold={fold_id}  fit→{ts_fit[-1].strftime('%m-%d %H')}h")
        print(f"\n{header}")
        set_seed(seed)
        t0 = time.time()
        try:
            preds, meta = info.fn(
                out_fit=out_fit,
                weather_fit=weather_fit,
                timestamps_fit=ts_fit,
                locations=locations,
                horizon=horizon,
                seed=seed,
            )
            elapsed = time.time() - t0
            if preds.shape != (horizon, len(locations)):
                raise ValueError(f"{name} returned shape {preds.shape}, expected ({horizon}, {len(locations)})")
            val_rmse = avg_county_rmse(out_val, preds)
            np.save(VAL_PRED_DIR / f"{name}__seed{seed}__h{horizon}__fold{fold_id}.npy", preds)
            log_run_fold(
                model_name=name, seed=seed, horizon=horizon, fold=fold_id,
                val_rmse=val_rmse, train_time_s=elapsed,
                config_dict={**meta, "tier": info.tier,
                             "val_start": str(fld["val_start_ts"]),
                             "val_peak": fld["val_peak"]},
            )
            completed += 1
            elapsed_total = time.time() - total_start
            print(f"  rmse={val_rmse:.4f}  t={elapsed:.1f}s  total={elapsed_total/60:.1f}min")
        except Exception as e:
            errors += 1
            print(f"  ERROR {name}: {e}")
            if not skip_on_error:
                raise
            traceback.print_exc()

    elapsed_total = time.time() - total_start
    print(f"\n[Runner] Done. completed={completed} skipped={skipped} errors={errors} "
          f"total_wall={elapsed_total/60:.1f}min")
    from .training import summarize_runs
    return summarize_runs()


def retrain_best_on_full_and_predict_test(
    model_name: str,
    horizon: int,
    seed: int = 42,
) -> np.ndarray:
    """After CV, retrain the model on the FULL train set (no held-out window)
    and produce test predictions. Writes a submission CSV."""
    ds_train, ds_test_24h, ds_test_48h = load_raw()
    ds_train_f = drop_zero_variance_features(ds_train)
    out_full, weather_full, locations, _, ts_full = _arr(ds_train_f)

    info = MODEL_REGISTRY[model_name]
    set_seed(seed)
    t0 = time.time()
    preds, meta = info.fn(
        out_fit=out_full, weather_fit=weather_full,
        timestamps_fit=ts_full, locations=locations,
        horizon=horizon, seed=seed,
    )
    elapsed = time.time() - t0
    print(f"[Final] {model_name}@{horizon}h trained on full data in {elapsed:.1f}s — meta={meta}")

    test_ts = pd.to_datetime((ds_test_24h if horizon == 24 else ds_test_48h).timestamp.values)
    out_path = save_submission(preds, test_ts, locations, horizon=horizon, tag=f"final_{model_name}")
    print(f"[Final] wrote submission → {out_path}")
    return preds


def _arr(ds):
    from .data import get_arrays
    return get_arrays(ds)
