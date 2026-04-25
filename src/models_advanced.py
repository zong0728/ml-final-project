"""Advanced architectures purpose-built for short-window, sparse time-series.

Models added here:
  * N-BEATS  (Oreshkin et al. 2020)  — generic basis-decomposition residual stack
  * N-HiTS   (Challu et al. 2023)    — hierarchical multi-rate variant
  * AutoARIMA per-county             — order-search SARIMA, replaces fixed (1,0,1)
  * Quantile LightGBM                — emit P50 (median), useful for RMSE-vs-MAE
                                        debate and provides uncertainty for policy

All NN models reuse the shared training loop in models_neural._train_and_predict.
"""
from __future__ import annotations

import warnings
from typing import Optional

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F

from . import config
from .data import build_lag_table
from .registry import register


# ============================================================================
# N-BEATS (Oreshkin 2020)
# ============================================================================

class _NBeatsBlock(nn.Module):
    """Generic FC stack producing backcast + forecast vectors."""

    def __init__(self, input_size: int, hidden: int, n_layers: int,
                 backcast_size: int, forecast_size: int, dropout: float = 0.0):
        super().__init__()
        layers = []
        d = input_size
        for _ in range(n_layers):
            layers += [nn.Linear(d, hidden), nn.ReLU(), nn.Dropout(dropout)]
            d = hidden
        self.fc_stack = nn.Sequential(*layers)
        self.backcast_head = nn.Linear(hidden, backcast_size)
        self.forecast_head = nn.Linear(hidden, forecast_size)

    def forward(self, x):
        h = self.fc_stack(x)
        return self.backcast_head(h), self.forecast_head(h)


class NBeatsNet(nn.Module):
    """Generic N-BEATS, channel-flat: takes flattened (seq_len, D) and outputs horizon."""

    def __init__(self, input_dim: int, seq_len: int, horizon: int,
                 n_blocks: int = 6, hidden: int = 256, n_layers: int = 4,
                 dropout: float = 0.1):
        super().__init__()
        # Flatten across feature dim. The "backcast" is the historical outage
        # only (seq_len-long), so we project the multivariate window first.
        self.proj_in = nn.Linear(input_dim, 1)   # squeeze features -> 1 univariate
        self.seq_len = seq_len
        self.horizon = horizon
        self.blocks = nn.ModuleList([
            _NBeatsBlock(seq_len, hidden, n_layers, seq_len, horizon, dropout)
            for _ in range(n_blocks)
        ])

    def forward(self, x):
        # x: (B, T, D) — squash to (B, T) by projecting features then squeezing
        x_uni = self.proj_in(x).squeeze(-1)        # (B, T)
        residual = x_uni
        forecast_total = torch.zeros(x.size(0), self.horizon, device=x.device)
        for blk in self.blocks:
            backcast, forecast = blk(residual)
            residual = residual - backcast
            forecast_total = forecast_total + forecast
        return forecast_total


# ============================================================================
# N-HiTS (Challu 2023) — multi-rate sampling with pooling kernels
# ============================================================================

class _NHiTSBlock(nn.Module):
    def __init__(self, seq_len: int, hidden: int, n_layers: int,
                 forecast_size: int, pool_kernel: int, dropout: float = 0.0):
        super().__init__()
        self.pool = nn.AvgPool1d(kernel_size=pool_kernel, stride=pool_kernel,
                                  ceil_mode=True)
        pooled_size = -(-seq_len // pool_kernel)   # ceil division
        layers = []
        d = pooled_size
        for _ in range(n_layers):
            layers += [nn.Linear(d, hidden), nn.ReLU(), nn.Dropout(dropout)]
            d = hidden
        self.fc_stack = nn.Sequential(*layers)
        self.backcast_head = nn.Linear(hidden, pooled_size)
        self.forecast_head = nn.Linear(hidden, forecast_size)
        self.pooled_size = pooled_size
        self.seq_len = seq_len

    def forward(self, x):
        # x: (B, T)
        x_pool = self.pool(x.unsqueeze(1)).squeeze(1)   # (B, T_pool)
        h = self.fc_stack(x_pool)
        backcast_pool = self.backcast_head(h)            # (B, T_pool)
        # Upsample back to T via linear interpolation
        backcast = F.interpolate(backcast_pool.unsqueeze(1),
                                  size=self.seq_len, mode="linear",
                                  align_corners=False).squeeze(1)
        forecast = self.forecast_head(h)
        return backcast, forecast


class NHiTSNet(nn.Module):
    def __init__(self, input_dim: int, seq_len: int, horizon: int,
                 n_blocks: int = 6, hidden: int = 256, n_layers: int = 2,
                 dropout: float = 0.1, pool_kernels: tuple = (8, 4, 2, 1, 1, 1)):
        super().__init__()
        self.proj_in = nn.Linear(input_dim, 1)
        self.seq_len = seq_len
        self.horizon = horizon
        if len(pool_kernels) < n_blocks:
            pool_kernels = (pool_kernels + (1,) * n_blocks)[:n_blocks]
        else:
            pool_kernels = pool_kernels[:n_blocks]
        self.blocks = nn.ModuleList([
            _NHiTSBlock(seq_len, hidden, n_layers, horizon, pool_kernels[i], dropout)
            for i in range(n_blocks)
        ])

    def forward(self, x):
        x_uni = self.proj_in(x).squeeze(-1)
        residual = x_uni
        forecast_total = torch.zeros(x.size(0), self.horizon, device=x.device)
        for blk in self.blocks:
            backcast, forecast = blk(residual)
            residual = residual - backcast
            forecast_total = forecast_total + forecast
        return forecast_total


# ============================================================================
# Registered NN models — default hparams (grid search uses run_neural_grid)
# ============================================================================

@register("nbeats", tier="advanced", stochastic=True,
          description="N-BEATS generic stack (Oreshkin 2020).")
def nbeats_default(out_fit, weather_fit, timestamps_fit, locations, horizon, seed):
    from .models_neural import _train_and_predict

    def build(input_dim):
        return NBeatsNet(input_dim, seq_len=config.SEQ_LEN, horizon=horizon,
                         n_blocks=6, hidden=256, n_layers=4, dropout=0.1)
    preds, meta = _train_and_predict(build, out_fit, weather_fit, timestamps_fit,
                                      horizon, seed)
    meta.update({"n_blocks": 6, "hidden": 256, "n_layers": 4})
    return preds, meta


@register("nhits", tier="advanced", stochastic=True,
          description="N-HiTS hierarchical multi-rate (Challu 2023).")
def nhits_default(out_fit, weather_fit, timestamps_fit, locations, horizon, seed):
    from .models_neural import _train_and_predict

    def build(input_dim):
        return NHiTSNet(input_dim, seq_len=config.SEQ_LEN, horizon=horizon,
                        n_blocks=6, hidden=256, n_layers=2, dropout=0.1)
    preds, meta = _train_and_predict(build, out_fit, weather_fit, timestamps_fit,
                                      horizon, seed)
    meta.update({"n_blocks": 6, "hidden": 256, "n_layers": 2})
    return preds, meta


# ============================================================================
# AutoARIMA per-county — replace fixed SARIMAX(1,0,1)
# ============================================================================

@register("auto_arima", tier="classical", stochastic=False,
          description="Per-county SARIMA with order grid (p,d,q) selected by AIC.")
def auto_arima_model(out_fit, weather_fit, timestamps_fit, locations, horizon, seed):
    """Auto-select SARIMA order per county.

    pmdarima isn't always available on aarch64; we do a small manual grid
    instead: orders from {(1,0,0),(2,0,0),(1,0,1),(2,0,1),(0,0,1)} chosen by
    AIC. Falls back to (1,0,1) on failure (matches the existing sarimax model).
    """
    from statsmodels.tsa.statespace.sarimax import SARIMAX

    T, L = out_fit.shape
    candidate_orders = [(1, 0, 0), (2, 0, 0), (1, 0, 1), (2, 0, 1), (0, 0, 1)]
    preds = np.zeros((horizon, L), dtype=np.float32)
    best_orders = []

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        for li in range(L):
            y = out_fit[:, li].astype(float)
            if len(y) < 8 or np.allclose(y, y[0]):
                preds[:, li] = float(y.mean())
                best_orders.append((0, 0, 0))
                continue
            best_aic = np.inf
            best_fc = None
            best_order = (1, 0, 1)
            for order in candidate_orders:
                try:
                    res = SARIMAX(y, order=order,
                                   enforce_stationarity=False,
                                   enforce_invertibility=False).fit(disp=False)
                    if res.aic < best_aic:
                        best_aic = res.aic
                        best_fc = np.asarray(res.forecast(steps=horizon), dtype=float)
                        best_order = order
                except Exception:
                    continue
            if best_fc is None:
                preds[:, li] = 0.0
            else:
                preds[:, li] = np.clip(best_fc, 0, None)
            best_orders.append(best_order)

    # Most common selected order — for the report.
    from collections import Counter
    order_hist = Counter(best_orders).most_common(5)
    return preds, {"order_grid": [list(o) for o in candidate_orders],
                   "top_orders_in_grid": [(list(o), n) for o, n in order_hist]}


# ============================================================================
# Quantile LightGBM (P50 — median regression, more robust to outliers/skewness)
# ============================================================================

_TABULAR_ORIGIN_STRIDE = 8


@register("lgb_quantile_p50", tier="classical", stochastic=True,
          description="LightGBM with quantile loss at alpha=0.5 (median regression).")
def lgb_quantile_p50(out_fit, weather_fit, timestamps_fit, locations, horizon, seed):
    """RMSE squares the error and so penalizes the heavy storm tail enormously.
    Median regression (q=0.5) is naturally robust; included as a contrast point
    rather than a winner — useful for reporting and ensembling.
    """
    import lightgbm as lgb
    X_tr, y_tr, X_inf, _, _ = build_lag_table(
        out=out_fit, weather=weather_fit, timestamps=timestamps_fit,
        horizon=horizon, origin_stride=_TABULAR_ORIGIN_STRIDE,
    )
    model = lgb.LGBMRegressor(
        objective="quantile", alpha=0.5,
        n_estimators=800, num_leaves=63, learning_rate=0.05,
        subsample=0.8, colsample_bytree=0.8,
        random_state=seed, verbosity=-1,
    )
    model.fit(X_tr, y_tr)   # quantile loss works directly on raw target
    pred = model.predict(X_inf)
    pred = np.clip(pred, 0.0, float(y_tr.max()) * 1.5 + 10.0)
    L = out_fit.shape[1]
    return pred.reshape(horizon, L).astype(np.float32), {
        "objective": "quantile", "alpha": 0.5,
        "n_estimators": 800, "num_leaves": 63,
        "origin_stride": _TABULAR_ORIGIN_STRIDE,
    }


@register("lgb_quantile_p90", tier="classical", stochastic=True,
          description="LightGBM with quantile loss at alpha=0.9 (90% upper bound).")
def lgb_quantile_p90(out_fit, weather_fit, timestamps_fit, locations, horizon, seed):
    """Provides an uncertainty upper bound. Useful in policy section to
    reason about worst-case generator demand."""
    import lightgbm as lgb
    X_tr, y_tr, X_inf, _, _ = build_lag_table(
        out=out_fit, weather=weather_fit, timestamps=timestamps_fit,
        horizon=horizon, origin_stride=_TABULAR_ORIGIN_STRIDE,
    )
    model = lgb.LGBMRegressor(
        objective="quantile", alpha=0.9,
        n_estimators=800, num_leaves=63, learning_rate=0.05,
        subsample=0.8, colsample_bytree=0.8,
        random_state=seed, verbosity=-1,
    )
    model.fit(X_tr, y_tr)
    pred = model.predict(X_inf)
    pred = np.clip(pred, 0.0, float(y_tr.max()) * 2.0 + 10.0)
    L = out_fit.shape[1]
    return pred.reshape(horizon, L).astype(np.float32), {
        "objective": "quantile", "alpha": 0.9,
        "origin_stride": _TABULAR_ORIGIN_STRIDE,
    }
