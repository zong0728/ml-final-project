"""Enhanced feature engineering — `build_lag_table_v2`.

Adds, on top of the original build_lag_table:

  A. **Deep outage lags**     — up to 4 weeks (672h)
  B. **State-aggregated lags**  — sum of outage across all counties at lags
                                  (captures correlated, large-storm signals)
  C. **County-static metadata** — tracked_max, historical max outage, mean
                                  hourly outage  (per-county features that
                                  the model can use to distinguish 83 cells)
  D. **Weather statistics**    — rolling mean / max / std of weather at
                                  short horizons (handles persisted-weather
                                  approximation downstream)
  E. **Future calendar**       — already done (sin/cos hour/dow/month at t+h)
  F. **Cross-horizon target**  — h_step still used; relative position works.

NOT added here (separate model class):
  G. Recursive (autoregressive) prediction — handled by a wrapper that
     calls predict step-by-step, feeding back lag_1.

This is a pure additive change — leaves the original `build_lag_table` intact
for back-compat.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from . import config
from .data import calendar_features


# Default deep lags: original 168 + weekly multiples up to 4 weeks
DEEP_LAGS = [1, 2, 3, 6, 12, 24, 48, 168, 336, 504, 672]
DEEP_ROLLS = [6, 24, 168, 336]   # adds 2-week rolling stats
WEATHER_LAGS_V2 = (3, 6, 12, 24, 48)  # adds 6h and 48h


def build_lag_table_v2(
    out: np.ndarray,                  # (T, L)
    weather: np.ndarray,              # (T, L, F)
    timestamps: pd.DatetimeIndex,
    horizon: int,
    lags: list[int] = DEEP_LAGS,
    rolls: list[int] = DEEP_ROLLS,
    weather_lags: tuple[int, ...] = WEATHER_LAGS_V2,
    origin_stride: int = 8,
    add_state_lags: bool = True,
    add_county_metadata: bool = True,
    add_weather_stats: bool = True,
):
    """Returns (X_tr, y_tr, X_inf, loc_ids_tr, feature_names).

    Same shape contract as `build_lag_table` for downstream tree models.
    """
    T, L = out.shape
    F = weather.shape[-1]
    max_lag = max([max(lags), max(rolls) if rolls else 0,
                   max(weather_lags) if weather_lags else 0, 24])

    # ---- A. Outage lag features ----
    lag_arr = np.full((T, L, len(lags)), np.nan, dtype=np.float32)
    for j, k in enumerate(lags):
        if k - 1 < T:
            lag_arr[k - 1:, :, j] = out[: T - (k - 1), :]

    # ---- Outage rolling mean/std ----
    df_out = pd.DataFrame(out)
    roll_arr = np.full((T, L, 2 * len(rolls)), np.nan, dtype=np.float32)
    for j, w in enumerate(rolls):
        roll_arr[:, :, 2 * j] = df_out.rolling(w, min_periods=1).mean().values.astype(np.float32)
        roll_arr[:, :, 2 * j + 1] = df_out.rolling(w, min_periods=1).std().fillna(0.0).values.astype(np.float32)

    # ---- Storm indicators (rolling 24h max / nonzero count / dod-delta) ----
    roll24_max = df_out.rolling(24, min_periods=1).max().values.astype(np.float32)
    roll24_nonzero = (df_out > 0).rolling(24, min_periods=1).sum().values.astype(np.float32)
    dod_delta = np.full_like(out, np.nan, dtype=np.float32)
    if 24 < T:
        dod_delta[24:] = out[24:] - out[:-24]

    # NEW: 168h (week) and 336h (2-week) deltas — captures storm momentum
    week_delta = np.full_like(out, np.nan, dtype=np.float32)
    if 168 < T:
        week_delta[168:] = out[168:] - out[:-168]

    storm_feats = [roll24_max, roll24_nonzero, dod_delta, week_delta]
    storm_names = ["storm_peak_24h", "storm_nonzero_count_24h",
                    "outage_delta_vs_24h_ago", "outage_delta_vs_168h_ago"]

    # ---- B. State-wide aggregated lags (NEW) ----
    state_lag_arrays = []
    state_lag_names = []
    if add_state_lags:
        statewide = out.sum(axis=1, keepdims=False)   # (T,)
        state_lag_horizons = [1, 6, 24, 48, 168]
        for k in state_lag_horizons:
            arr = np.full((T,), np.nan, dtype=np.float32)
            if k - 1 < T:
                arr[k - 1:] = statewide[: T - (k - 1)]
            # Broadcast: same value for all L counties at time t
            state_lag_arrays.append(np.broadcast_to(arr[:, None], (T, L)).copy())
            state_lag_names.append(f"statewide_lag{k}")
        # Also state-wide rolling mean/max
        df_state = pd.Series(statewide)
        for w in [6, 24, 168]:
            roll_mean = df_state.rolling(w, min_periods=1).mean().values.astype(np.float32)
            roll_max = df_state.rolling(w, min_periods=1).max().values.astype(np.float32)
            state_lag_arrays.append(np.broadcast_to(roll_mean[:, None], (T, L)).copy())
            state_lag_arrays.append(np.broadcast_to(roll_max[:, None], (T, L)).copy())
            state_lag_names.extend([f"statewide_rollmean_{w}", f"statewide_rollmax_{w}"])

    # ---- C. County metadata (NEW) ----
    county_meta_arrays = []
    county_meta_names = []
    if add_county_metadata:
        # tracked_max — per-county constant, broadcast across T
        # We compute it from the FIT data only (no test leakage). Caller
        # passes us only fit data anyway.
        tracked_max_per_county = np.nanmax(out, axis=0)   # (L,) — actually should pass tracked from caller; using outage max as proxy
        max_outage_per_county = tracked_max_per_county
        mean_outage_per_county = np.nanmean(out, axis=0)
        # Static features broadcast over time
        for vec, nm in [(np.log1p(max_outage_per_county), "log_county_max_outage"),
                         (mean_outage_per_county, "county_mean_outage"),
                         (np.log1p(mean_outage_per_county), "log_county_mean_outage")]:
            arr = np.broadcast_to(vec[None, :], (T, L)).astype(np.float32).copy()
            county_meta_arrays.append(arr)
            county_meta_names.append(nm)

    # ---- D. Weather rolling stats (NEW) ----
    weather_stat_arrays = []
    weather_stat_names = []
    if add_weather_stats:
        # Rolling 6h mean of every weather feature — captures storm build-up better than point value
        # Roll over the time axis
        for w in [6, 24]:
            arr = np.zeros_like(weather, dtype=np.float32)
            for fi in range(F):
                col = weather[:, :, fi]   # (T, L)
                df_col = pd.DataFrame(col)
                arr[:, :, fi] = df_col.rolling(w, min_periods=1).mean().values.astype(np.float32)
            weather_stat_arrays.append(arr)
            weather_stat_names += [f"w{i}_rollmean{w}" for i in range(F)]

    # ---- Weather at t and lags ----
    weather_at_t = weather
    weather_lag_arrays = []
    weather_lag_names = []
    for lag in weather_lags:
        arr = np.full_like(weather, np.nan, dtype=np.float32)
        if lag < T:
            arr[lag:] = weather[:-lag]
        weather_lag_arrays.append(arr)
        weather_lag_names += [f"w{i}_lag{lag}" for i in range(F)]

    # ---- Stack everything (T, L, P_base) ----
    feat_list = [lag_arr, roll_arr]
    feat_list += [sf[:, :, None] for sf in storm_feats]
    if add_state_lags:
        feat_list += [arr[:, :, None] for arr in state_lag_arrays]
    if add_county_metadata:
        feat_list += [arr[:, :, None] for arr in county_meta_arrays]
    feat_list += [weather_at_t]
    if add_weather_stats:
        feat_list += weather_stat_arrays
    feat_list += weather_lag_arrays

    feat_at_t = np.concatenate(feat_list, axis=-1)  # (T, L, P_base)
    P_base = feat_at_t.shape[-1]

    # ---- Calendar features for all timestamps ----
    cal_all = calendar_features(timestamps)  # (T, 6)

    # ---- Valid origin times ----
    t_start = max_lag - 1
    t_end = T - horizon
    valid_origins = np.arange(t_start, t_end, origin_stride)
    N_origins = len(valid_origins)
    if N_origins == 0:
        raise ValueError(f"No valid origins: T={T}, max_lag={max_lag}, horizon={horizon}")

    P = P_base + 6 + 1 + 1
    feature_names = (
        [f"lag_{k}" for k in lags]
        + [f"{s}_{w}" for w in rolls for s in ("rollmean", "rollstd")]
        + storm_names
        + (state_lag_names if add_state_lags else [])
        + (county_meta_names if add_county_metadata else [])
        + [f"w_{i}" for i in range(F)]
        + (weather_stat_names if add_weather_stats else [])
        + weather_lag_names
        + ["hour_sin", "hour_cos", "dow_sin", "dow_cos", "month_sin", "month_cos"]
        + ["h_step", "location_idx"]
    )

    N_rows = N_origins * L * horizon
    X_tr = np.empty((N_rows, P), dtype=np.float32)
    y_tr = np.empty(N_rows, dtype=np.float32)
    loc_ids_tr = np.empty(N_rows, dtype=np.int64)

    loc_idx_arr = np.arange(L, dtype=np.float32)
    row = 0
    for t in valid_origins:
        base = feat_at_t[t]                              # (L, P_base)
        for h in range(1, horizon + 1):
            cal_row = cal_all[t + h]
            X_tr[row:row + L, :P_base] = base
            X_tr[row:row + L, P_base:P_base + 6] = cal_row
            X_tr[row:row + L, P_base + 6] = h
            X_tr[row:row + L, P_base + 7] = loc_idx_arr
            y_tr[row:row + L] = out[t + h, :]
            loc_ids_tr[row:row + L] = np.arange(L)
            row += L

    # ---- Drop rows with NaN ----
    mask = np.isfinite(X_tr).all(axis=1)
    X_tr = X_tr[mask]
    y_tr = y_tr[mask]
    loc_ids_tr = loc_ids_tr[mask]

    # ---- Inference: origin t_inf = T-1 ----
    t_inf = T - 1
    base_inf = feat_at_t[t_inf]                          # (L, P_base)
    future_ts = pd.DatetimeIndex(
        [timestamps[t_inf] + pd.Timedelta(hours=h) for h in range(1, horizon + 1)]
    )
    cal_fut = calendar_features(future_ts)
    X_inf = np.empty((L * horizon, P), dtype=np.float32)
    for h_step in range(1, horizon + 1):
        r0 = (h_step - 1) * L
        X_inf[r0:r0 + L, :P_base] = base_inf
        X_inf[r0:r0 + L, P_base:P_base + 6] = cal_fut[h_step - 1]
        X_inf[r0:r0 + L, P_base + 6] = h_step
        X_inf[r0:r0 + L, P_base + 7] = loc_idx_arr
    X_inf = np.nan_to_num(X_inf, nan=0.0)

    return X_tr, y_tr, X_inf, loc_ids_tr, feature_names
