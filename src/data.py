"""Data loading, cleaning, scaling, and sliding-window construction.

Key design choices:
  * Future weather during the forecast horizon is NEVER used as a feature —
    only the history up to and including the forecast origin is visible.
  * Val split: last `VAL_HOURS` of training are held out as the outer val set.
  * All z-scoring statistics are fit on the training-only portion.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd
import xarray as xr

from . import config


# ============================================================================
# Raw loaders
# ============================================================================

def load_raw() -> tuple[xr.Dataset, xr.Dataset, xr.Dataset]:
    """Open the three .nc datasets."""
    ds_train = xr.open_dataset(config.TRAIN_PATH)
    ds_test_24h = xr.open_dataset(config.TEST_24H_PATH)
    ds_test_48h = xr.open_dataset(config.TEST_48H_PATH)
    return ds_train, ds_test_24h, ds_test_48h


def drop_zero_variance_features(ds: xr.Dataset, ref: xr.Dataset | None = None) -> xr.Dataset:
    """Remove weather features with ~zero variance in `ref` (defaults to ds itself)."""
    ref = ref if ref is not None else ds
    w = ref.weather.values
    F = w.shape[-1]
    stds = np.nanstd(w.reshape(-1, F), axis=0)
    valid = stds > 1e-6
    keep = [f for f, v in zip(list(ref.feature.values), valid) if v]
    return ds.sel(feature=keep)


def split_train_val(ds_train: xr.Dataset, val_hours: int = config.VAL_HOURS):
    """Hold out the last `val_hours` of training as outer validation."""
    T = len(ds_train.timestamp)
    ds_fit = ds_train.isel(timestamp=slice(0, T - val_hours))
    ds_val = ds_train.isel(timestamp=slice(T - val_hours, T))
    return ds_fit, ds_val


# ============================================================================
# Rolling-origin CV folds
#
# The test setup (per project description) is: forecast 24h or 48h immediately
# after the last training hour, with NO future weather. Our CV must mirror that
# exactly. So each fold is:
#
#     fit  = ds_train.isel(timestamp = 0 .. origin)
#     val  = ds_train.isel(timestamp = origin .. origin + horizon)
#
# Picking origins:
#   * The natural last origin == T - horizon, identical to the prior single split.
#   * To get more validation signal we slide back N more origins, each spaced
#     by `stride` hours. This gives N+1 folds per horizon.
#   * An origin is REJECTED if its val window has all-zero outage everywhere
#     (boring period that would be dominated by 0 = degenerate RMSE).
# ============================================================================


def make_rolling_folds(
    ds_train: xr.Dataset,
    horizon: int,
    n_folds: int = 6,
    stride_hours: int = 72,
    require_nonzero: bool = True,
) -> list[dict]:
    """Build rolling-origin folds that mirror the test-time setup.

    Returns a list of dicts:
      {fold_id, origin_idx, fit_slice, val_slice, val_start_ts, val_end_ts}
    where slices are Python slice objects suitable for ds.isel(timestamp=slice).

    The most-recent fold (fold_id=0) ends at T-1 (i.e. val == last `horizon`
    hours of train). Subsequent folds slide backward by `stride_hours`.
    """
    T = len(ds_train.timestamp)
    timestamps = pd.to_datetime(ds_train.timestamp.values)
    out = ds_train.out.transpose("timestamp", "location").values  # (T, L)

    folds: list[dict] = []
    fid = 0
    candidate_origin = T - horizon   # newest origin first
    while len(folds) < n_folds and candidate_origin > horizon * 4:
        val_lo = candidate_origin
        val_hi = candidate_origin + horizon
        if val_hi > T:
            candidate_origin -= stride_hours
            continue
        val_window = out[val_lo:val_hi]
        if require_nonzero and val_window.sum() < 1e-6:
            candidate_origin -= stride_hours
            continue
        folds.append(dict(
            fold_id=fid,
            origin_idx=int(val_lo),
            fit_slice=slice(0, val_lo),
            val_slice=slice(val_lo, val_hi),
            val_start_ts=timestamps[val_lo],
            val_end_ts=timestamps[val_hi - 1],
            val_sum=float(val_window.sum()),
            val_peak=float(val_window.sum(axis=1).max()),
        ))
        fid += 1
        candidate_origin -= stride_hours
    return folds


def fold_arrays(ds_train: xr.Dataset, fold: dict
                ) -> tuple[np.ndarray, np.ndarray, np.ndarray, list[str], list[str], pd.DatetimeIndex]:
    """Materialize (out_fit, weather_fit, out_val, locations, features, timestamps_fit) for a fold."""
    ds_fit = ds_train.isel(timestamp=fold["fit_slice"])
    ds_val = ds_train.isel(timestamp=fold["val_slice"])
    out_fit, weather_fit, locations, features, timestamps_fit = get_arrays(ds_fit)
    out_val, _, _, _, _ = get_arrays(ds_val)
    return out_fit, weather_fit, out_val, locations, features, timestamps_fit


# ============================================================================
# Core matrices — outage (T, L) and weather (T, L, F)
# ============================================================================

def get_arrays(ds: xr.Dataset) -> tuple[np.ndarray, np.ndarray, list[str], list[str], pd.DatetimeIndex]:
    """Returns (out, weather, locations, features, timestamps)."""
    out = ds.out.transpose("timestamp", "location").values.astype(np.float32)
    weather = ds.weather.transpose("timestamp", "location", "feature").values.astype(np.float32)
    locations = [str(x) for x in ds.location.values]
    features = [str(x) for x in ds.feature.values]
    timestamps = pd.to_datetime(ds.timestamp.values)
    return out, weather, locations, features, timestamps


# ============================================================================
# Scalers
# ============================================================================

@dataclass
class Scalers:
    """Training-only z-score statistics used by NN models."""
    y_mu: float
    y_sd: float
    w_mu: np.ndarray  # (F,)
    w_sd: np.ndarray  # (F,)

    def scale_y(self, y: np.ndarray) -> np.ndarray:
        return (y - self.y_mu) / self.y_sd

    def inv_y(self, y: np.ndarray) -> np.ndarray:
        return y * self.y_sd + self.y_mu

    def scale_w(self, w: np.ndarray) -> np.ndarray:
        return (w - self.w_mu) / self.w_sd


def fit_scalers(out: np.ndarray, weather: np.ndarray) -> Scalers:
    y_mu = float(np.nanmean(out))
    y_sd = float(max(np.nanstd(out), 1e-6))
    F = weather.shape[-1]
    flat = weather.reshape(-1, F)
    w_mu = np.nanmean(flat, axis=0).astype(np.float32)
    w_sd = np.nanstd(flat, axis=0).astype(np.float32)
    w_sd = np.where(w_sd < 1e-6, 1.0, w_sd)
    return Scalers(y_mu=y_mu, y_sd=y_sd, w_mu=w_mu, w_sd=w_sd)


# ============================================================================
# Calendar features (leak-free: only uses timestamp)
# ============================================================================

def calendar_features(timestamps: np.ndarray | pd.DatetimeIndex) -> np.ndarray:
    """Cyclical encoding: sin/cos of hour-of-day, day-of-week, month."""
    ts = pd.to_datetime(timestamps)
    hour = ts.hour.to_numpy()
    dow = ts.dayofweek.to_numpy()
    month = ts.month.to_numpy()
    feats = np.stack(
        [
            np.sin(2 * np.pi * hour / 24),
            np.cos(2 * np.pi * hour / 24),
            np.sin(2 * np.pi * dow / 7),
            np.cos(2 * np.pi * dow / 7),
            np.sin(2 * np.pi * (month - 1) / 12),
            np.cos(2 * np.pi * (month - 1) / 12),
        ],
        axis=1,
    )
    return feats.astype(np.float32)


# ============================================================================
# Sliding-window construction for NN models
# ============================================================================

def build_nn_windows(
    out: np.ndarray,                 # (T, L) raw outage
    weather: np.ndarray,             # (T, L, F) raw weather
    scalers: Scalers,
    seq_len: int,
    horizon: int,
    include_calendar: bool = True,
    timestamps: np.ndarray | pd.DatetimeIndex | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Build sliding windows for every (location, time) pair.

    Returns:
      X_train: (N, seq_len, D) — features for training
      Y_train: (N, horizon)   — target SCALED outage for training
      X_last:  (L, seq_len, D) — last available window per location for inference

    D = 1 (outage) + F (weather) + [6 (calendar) if include_calendar]
    Target y is scaled with scalers; inverse via scalers.inv_y at inference.
    """
    T, L = out.shape
    F = weather.shape[-1]

    y_s = scalers.scale_y(out.astype(np.float32))           # (T, L)
    w_s = scalers.scale_w(weather.astype(np.float32))       # (T, L, F)

    # Calendar features are the same across locations at a given time.
    if include_calendar:
        assert timestamps is not None, "timestamps required when include_calendar=True"
        cal = calendar_features(timestamps)                  # (T, 6)
        cal_bcast = np.broadcast_to(cal[:, None, :], (T, L, cal.shape[1]))
        x_per_step = np.concatenate([y_s[..., None], w_s, cal_bcast], axis=-1)
    else:
        x_per_step = np.concatenate([y_s[..., None], w_s], axis=-1)
    # x_per_step: (T, L, D)

    D = x_per_step.shape[-1]
    N_per_county = T - seq_len - horizon + 1
    if N_per_county <= 0:
        raise ValueError(f"Not enough time steps: T={T}, seq_len={seq_len}, horizon={horizon}")

    # Build arrays — vectorize across counties, loop over origin time.
    X = np.empty((N_per_county * L, seq_len, D), dtype=np.float32)
    Y = np.empty((N_per_county * L, horizon), dtype=np.float32)
    for i in range(N_per_county):
        # x_per_step[i:i+seq_len, :, :] has shape (seq_len, L, D)
        X[i * L:(i + 1) * L] = x_per_step[i:i + seq_len].transpose(1, 0, 2)
        Y[i * L:(i + 1) * L] = y_s[i + seq_len:i + seq_len + horizon].T  # (L, horizon)

    X_last = x_per_step[T - seq_len:T].transpose(1, 0, 2).copy()  # (L, seq_len, D)

    return X, Y, X_last


# ============================================================================
# Tabular lag matrix for tree / linear models — numpy-only, no pandas DataFrame.
#
# For each origin time t ∈ [t_start, T-horizon) and each county l, we create
# `horizon` training rows with features:
#   * Outage lags:    out[t], out[t-1], ..., out[t - max_lag]
#   * Outage rolling: mean/std over the K hours ending at t
#   * Weather at t:   w[t, l, :]            (no weather lags → keep feature count small)
#   * Calendar at target (t+h): 6 cyclical features
#   * Horizon step h: one scalar
#   * Location index: one categorical (as ordinal — tree models can split on it)
#
# The rows for (origin t, county l, horizon h) are packed as
#   row index = (t_rel * L + l) * horizon + (h - 1)
# i.e. origin-major → county → horizon-step, to keep target alignment simple.
# ============================================================================

def build_lag_table(
    out: np.ndarray,                      # (T, L)
    weather: np.ndarray,                  # (T, L, F)
    timestamps: pd.DatetimeIndex,
    horizon: int,
    lags: list[int] = config.LAG_FEATURES,
    rolls: list[int] = config.ROLLING_WINDOWS,
    weather_lags: tuple[int, ...] = (3, 12, 24),    # NEW: lags for weather features (storm build-up)
    extra_storm_features: bool = True,              # NEW: rolling max / nonzero count / day-over-day delta
    origin_stride: int = 1,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, list[str]]:
    """Build train (X, y) and inference X_inf for tree/linear models.

    Feature groups produced (all leak-free: use only data up to time t):
      * Outage lags            — `lags` offsets of outage history
      * Outage rolling stats   — mean/std over `rolls` windows
      * Outage storm indicators (if `extra_storm_features`):
          · rolling max over 24h (peak of recent storm)
          · count of nonzero-outage hours over 24h (storm duration)
          · day-over-day delta  out[t] − out[t-24] (momentum signal)
      * Weather at t
      * Weather lags (if `weather_lags` not empty) — 3/12/24h ago by default
      * Calendar at target time t+h
      * Horizon step h
      * Location index

    Returns
    -------
    X_tr : (N, P) float32
    y_tr : (N,)   float32
    X_inf : (L * horizon, P) float32
    loc_ids_tr : (N,) int64
    feature_names : list[str]
    """
    T, L = out.shape
    F = weather.shape[-1]
    max_lag = max([max(lags), max(rolls) if rolls else 0,
                   max(weather_lags) if weather_lags else 0, 24])

    # ---- Outage lag features: shape (T, L, len(lags)) ----
    lag_arr = np.full((T, L, len(lags)), np.nan, dtype=np.float32)
    for j, k in enumerate(lags):
        if k - 1 < T:
            lag_arr[k - 1:, :, j] = out[: T - (k - 1), :]

    # ---- Outage rolling mean/std: shape (T, L, 2*len(rolls)) ----
    roll_arr = np.full((T, L, 2 * len(rolls)), np.nan, dtype=np.float32)
    df_out = pd.DataFrame(out)
    for j, w in enumerate(rolls):
        roll_arr[:, :, 2 * j] = df_out.rolling(w, min_periods=1).mean().values.astype(np.float32)
        roll_arr[:, :, 2 * j + 1] = df_out.rolling(w, min_periods=1).std().fillna(0.0).values.astype(np.float32)

    # ---- Storm indicator features (over past 24h) ----
    storm_feats = []
    storm_names = []
    if extra_storm_features:
        # Rolling max over past 24h (peak of recent storm)
        roll24_max = df_out.rolling(24, min_periods=1).max().values.astype(np.float32)
        # Count of hours with outage > 0 over past 24h (storm duration)
        roll24_nonzero = (df_out > 0).rolling(24, min_periods=1).sum().values.astype(np.float32)
        # Day-over-day delta: out[t] - out[t-24]
        dod_delta = np.full_like(out, np.nan, dtype=np.float32)
        if 24 < T:
            dod_delta[24:] = out[24:] - out[:-24]
        storm_feats = [roll24_max, roll24_nonzero, dod_delta]
        storm_names = ["storm_peak_24h", "storm_nonzero_count_24h", "outage_delta_vs_24h_ago"]

    # ---- Weather at t: shape (T, L, F) ----
    weather_at_t = weather  # alias

    # ---- Weather lags: (T, L, F * len(weather_lags)) ----
    weather_lag_arrays = []
    weather_lag_names = []
    for lag in weather_lags:
        arr = np.full_like(weather, np.nan, dtype=np.float32)
        if lag < T:
            arr[lag:] = weather[:-lag]
        weather_lag_arrays.append(arr)
        weather_lag_names += [f"w{i}_lag{lag}" for i in range(F)]

    # ---- Stack per-step features (excluding calendar-at-target and h_step) ----
    feat_list = [lag_arr, roll_arr]
    if extra_storm_features:
        # storm features are (T, L) each — need (T, L, 1) to stack
        feat_list += [sf[:, :, None] for sf in storm_feats]
    feat_list += [weather_at_t] + weather_lag_arrays

    feat_at_t = np.concatenate(feat_list, axis=-1)  # (T, L, P_base)
    P_base = feat_at_t.shape[-1]

    # ---- Calendar features for all timestamps ----
    cal_all = calendar_features(timestamps)  # (T, 6)

    # ---- Valid origin times ----
    t_start = max_lag - 1
    t_end = T - horizon        # last valid origin yields targets up to t+horizon
    valid_origins = np.arange(t_start, t_end, origin_stride)
    N_origins = len(valid_origins)
    if N_origins == 0:
        raise ValueError(f"No valid origins: T={T}, max_lag={max_lag}, horizon={horizon}")

    P = P_base + 6 + 1 + 1  # + calendar(6) + h_step + location_idx
    feature_names = (
        [f"lag_{k}" for k in lags]
        + [f"{s}_{w}" for w in rolls for s in ("rollmean", "rollstd")]
        + (storm_names if extra_storm_features else [])
        + [f"w_{i}" for i in range(F)]
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
            cal_row = cal_all[t + h]                     # (6,)
            X_tr[row:row + L, :P_base] = base
            X_tr[row:row + L, P_base:P_base + 6] = cal_row
            X_tr[row:row + L, P_base + 6] = h
            X_tr[row:row + L, P_base + 7] = loc_idx_arr
            y_tr[row:row + L] = out[t + h, :]
            loc_ids_tr[row:row + L] = np.arange(L)
            row += L

    # ---- Drop rows with any NaN (early origins missing deep lags / rolling) ----
    mask = np.isfinite(X_tr).all(axis=1)
    X_tr = X_tr[mask]
    y_tr = y_tr[mask]
    loc_ids_tr = loc_ids_tr[mask]

    # ---- Inference: origin t_inf = T-1, all counties, all horizons ----
    t_inf = T - 1
    base_inf = feat_at_t[t_inf]                          # (L, P_base)
    future_ts = pd.DatetimeIndex(
        [timestamps[t_inf] + pd.Timedelta(hours=h) for h in range(1, horizon + 1)]
    )
    cal_fut = calendar_features(future_ts)               # (horizon, 6)
    X_inf = np.empty((L * horizon, P), dtype=np.float32)
    for h_step in range(1, horizon + 1):
        r0 = (h_step - 1) * L
        X_inf[r0:r0 + L, :P_base] = base_inf
        X_inf[r0:r0 + L, P_base:P_base + 6] = cal_fut[h_step - 1]
        X_inf[r0:r0 + L, P_base + 6] = h_step
        X_inf[r0:r0 + L, P_base + 7] = loc_idx_arr
    X_inf = np.nan_to_num(X_inf, nan=0.0)

    return X_tr, y_tr, X_inf, loc_ids_tr, feature_names
