"""Pre-compute & cache the lag-feature matrices for every (fold, horizon) once,
so each grid run reuses them instead of re-building from scratch.

Saves to:
    cache/fold{F}_h{H}/X_tr.npy
    cache/fold{F}_h{H}/y_tr.npy
    cache/fold{F}_h{H}/X_inf.npy
    cache/fold{F}_h{H}/loc_ids.npy
    cache/fold{F}_h{H}/feature_names.json

Run once on the login node (no GPU needed):
    python -m scripts.h200.cache_folds
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from src import config
from src.data import build_lag_table, drop_zero_variance_features, load_raw, fold_arrays, make_rolling_folds


CACHE_DIR = config.PROJECT_ROOT / "cache"
ORIGIN_STRIDE = 8


def main():
    ds_train, _, _ = load_raw()
    ds_train_f = drop_zero_variance_features(ds_train)
    horizons = [24, 48]

    for h in horizons:
        folds = make_rolling_folds(ds_train_f, horizon=h,
                                   n_folds=config.N_FOLDS,
                                   stride_hours=config.FOLD_STRIDE_HOURS)
        for fld in folds:
            out_dir = CACHE_DIR / f"fold{fld['fold_id']}_h{h}"
            if (out_dir / "X_tr.npy").exists():
                print(f"[cache] skip {out_dir.name} (exists)")
                continue
            out_dir.mkdir(parents=True, exist_ok=True)
            out_fit, weather_fit, out_val, locations, _, ts_fit = fold_arrays(ds_train_f, fld)
            X_tr, y_tr, X_inf, loc_ids, feat_names = build_lag_table(
                out=out_fit, weather=weather_fit, timestamps=ts_fit,
                horizon=h, origin_stride=ORIGIN_STRIDE,
            )
            np.save(out_dir / "X_tr.npy", X_tr)
            np.save(out_dir / "y_tr.npy", y_tr)
            np.save(out_dir / "X_inf.npy", X_inf)
            np.save(out_dir / "loc_ids.npy", loc_ids)
            np.save(out_dir / "out_val.npy", out_val)
            (out_dir / "feature_names.json").write_text(json.dumps(feat_names))
            (out_dir / "meta.json").write_text(json.dumps({
                "fold_id": fld["fold_id"], "horizon": h,
                "origin_idx": fld["origin_idx"],
                "val_start": str(fld["val_start_ts"]),
                "val_end": str(fld["val_end_ts"]),
                "val_peak": fld["val_peak"], "val_sum": fld["val_sum"],
                "X_tr_shape": list(X_tr.shape), "y_tr_shape": list(y_tr.shape),
                "X_inf_shape": list(X_inf.shape),
                "locations": locations,
            }))
            print(f"[cache] wrote {out_dir.name}: X_tr={X_tr.shape}  X_inf={X_inf.shape}")

    # Also cache full-train (for finalize)
    full_dir = CACHE_DIR / "full_h24"
    if not (full_dir / "X_tr.npy").exists():
        from src.data import get_arrays
        out_full, weather_full, locations, _, ts_full = get_arrays(ds_train_f)
        for h in horizons:
            full_dir = CACHE_DIR / f"full_h{h}"
            full_dir.mkdir(parents=True, exist_ok=True)
            X_tr, y_tr, X_inf, loc_ids, feat_names = build_lag_table(
                out=out_full, weather=weather_full, timestamps=ts_full,
                horizon=h, origin_stride=ORIGIN_STRIDE,
            )
            np.save(full_dir / "X_tr.npy", X_tr)
            np.save(full_dir / "y_tr.npy", y_tr)
            np.save(full_dir / "X_inf.npy", X_inf)
            np.save(full_dir / "loc_ids.npy", loc_ids)
            (full_dir / "feature_names.json").write_text(json.dumps(feat_names))
            (full_dir / "meta.json").write_text(json.dumps({
                "scope": "full_train", "horizon": h,
                "X_tr_shape": list(X_tr.shape),
                "locations": locations,
            }))
            print(f"[cache] wrote {full_dir.name}: X_tr={X_tr.shape}")


if __name__ == "__main__":
    main()
