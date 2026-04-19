"""Non-parametric baselines. All deterministic — seed is ignored."""
from __future__ import annotations

import numpy as np
import pandas as pd

from .registry import register


@register(
    "zero",
    tier="baseline",
    stochastic=False,
    description="Predict zero for every county-hour.",
)
def zero_model(out_fit, weather_fit, timestamps_fit, locations, horizon, seed):
    T, L = out_fit.shape
    return np.zeros((horizon, L), dtype=np.float32), {}


@register(
    "persistence",
    tier="baseline",
    stochastic=False,
    description="Repeat the last observed value for the whole horizon.",
)
def persistence(out_fit, weather_fit, timestamps_fit, locations, horizon, seed):
    last = out_fit[-1]                              # (L,)
    preds = np.broadcast_to(last[None, :], (horizon, out_fit.shape[1])).copy()
    return preds.astype(np.float32), {}


@register(
    "seasonal_naive_24",
    tier="baseline",
    stochastic=False,
    description="ŷ[t+h] = y[t+h-24] — repeat yesterday's same hour.",
)
def seasonal_naive_24(out_fit, weather_fit, timestamps_fit, locations, horizon, seed):
    T, L = out_fit.shape
    preds = np.zeros((horizon, L), dtype=np.float32)
    for h in range(1, horizon + 1):
        lag = h - 24
        if lag <= 0:
            idx = T + lag - 1   # e.g., h=1 → T-24; h=24 → T-1
            preds[h - 1] = out_fit[idx]
        else:
            # beyond 24h horizon, cycle
            preds[h - 1] = preds[h - 1 - 24]
    return preds, {}


@register(
    "seasonal_naive_168",
    tier="baseline",
    stochastic=False,
    description="ŷ[t+h] = y[t+h-168] — repeat same hour last week.",
)
def seasonal_naive_168(out_fit, weather_fit, timestamps_fit, locations, horizon, seed):
    T, L = out_fit.shape
    if T < 168:
        # Fallback to seasonal_naive_24
        return seasonal_naive_24(out_fit, weather_fit, timestamps_fit, locations, horizon, seed)
    preds = np.zeros((horizon, L), dtype=np.float32)
    for h in range(1, horizon + 1):
        idx = T - 168 + h - 1
        preds[h - 1] = out_fit[idx]
    return preds, {}


@register(
    "historical_mean",
    tier="baseline",
    stochastic=False,
    description="Per-(county, hour-of-day, day-of-week) historical mean.",
)
def historical_mean(out_fit, weather_fit, timestamps_fit, locations, horizon, seed):
    T, L = out_fit.shape
    ts_fit = pd.to_datetime(timestamps_fit)
    hour = ts_fit.hour.to_numpy()
    dow = ts_fit.dayofweek.to_numpy()

    # Build lookup table: mean outage for each (county, hour, dow)
    # Shape (L, 24, 7)
    table = np.zeros((L, 24, 7), dtype=np.float64)
    counts = np.zeros((L, 24, 7), dtype=np.int64)
    for t in range(T):
        h, d = hour[t], dow[t]
        table[:, h, d] += out_fit[t]
        counts[:, h, d] += 1
    counts = np.where(counts == 0, 1, counts)
    table = table / counts

    # Future timestamps
    future = pd.date_range(
        start=ts_fit[-1] + pd.Timedelta(hours=1),
        periods=horizon,
        freq="h",
    )
    fh = future.hour.to_numpy()
    fd = future.dayofweek.to_numpy()

    preds = np.zeros((horizon, L), dtype=np.float32)
    for h in range(horizon):
        preds[h] = table[:, fh[h], fd[h]]
    return preds, {}
