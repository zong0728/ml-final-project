"""Classical ML models: Linear/Ridge regression, XGBoost, LightGBM, SARIMAX.

All of these use the leak-free lag table built in ``data.build_lag_table``.
"""
from __future__ import annotations

import warnings

import numpy as np
import pandas as pd

from . import config
from .data import build_lag_table
from .registry import register


def _fit_predict_tabular(
    sklearn_style_model,
    out_fit, weather_fit, timestamps_fit, locations, horizon,
    log_target: bool = True,
    origin_stride: int = 1,
) -> np.ndarray:
    """Shared fit/predict flow for any sklearn-compatible model."""
    X_tr, y_tr, X_inf, _, _ = build_lag_table(
        out=out_fit,
        weather=weather_fit,
        timestamps=timestamps_fit,
        horizon=horizon,
        origin_stride=origin_stride,
    )
    if log_target:
        y_tr_used = np.log1p(np.clip(y_tr, 0, None))
    else:
        y_tr_used = y_tr

    sklearn_style_model.fit(X_tr, y_tr_used)

    pred_flat = sklearn_style_model.predict(X_inf)
    if log_target:
        # Clip the log-prediction before exponentiating. Max historical target
        # sets a reasonable ceiling; add a little headroom (2 nats) but cap.
        max_log = float(np.log1p(y_tr.max())) + 2.0
        pred_flat = np.expm1(np.clip(pred_flat, None, max_log))
    # Also cap the real-space prediction at 1.5× historical max to stay sane
    # if a model insists on extrapolating far beyond training range.
    max_y = float(y_tr.max()) * 1.5 + 10.0
    pred_flat = np.clip(pred_flat, 0.0, max_y)

    L = out_fit.shape[1]
    # X_inf ordering: for h in 1..H, for l in 0..L-1  (see build_lag_table)
    preds = pred_flat.reshape(horizon, L).astype(np.float32)
    return preds


# ============================================================================
# Linear / Ridge
# ============================================================================

_TABULAR_ORIGIN_STRIDE = 4  # subsample origin hours to keep the feature matrix manageable


@register(
    "linreg_lag",
    tier="classical",
    stochastic=False,
    description="LinearRegression on lag/rolling/weather features (log1p target).",
)
def linreg_lag(out_fit, weather_fit, timestamps_fit, locations, horizon, seed):
    from sklearn.linear_model import LinearRegression
    model = LinearRegression()
    preds = _fit_predict_tabular(
        model, out_fit, weather_fit, timestamps_fit, locations, horizon,
        log_target=True, origin_stride=_TABULAR_ORIGIN_STRIDE,
    )
    return preds, {"log_target": True, "origin_stride": _TABULAR_ORIGIN_STRIDE}


@register(
    "ridge_lag",
    tier="classical",
    stochastic=False,
    description="Ridge(α=1.0) on lag/rolling/weather features (log1p target).",
)
def ridge_lag(out_fit, weather_fit, timestamps_fit, locations, horizon, seed):
    from sklearn.linear_model import Ridge
    model = Ridge(alpha=1.0, random_state=seed)
    preds = _fit_predict_tabular(
        model, out_fit, weather_fit, timestamps_fit, locations, horizon,
        log_target=True, origin_stride=_TABULAR_ORIGIN_STRIDE,
    )
    return preds, {"alpha": 1.0, "log_target": True, "origin_stride": _TABULAR_ORIGIN_STRIDE}


# ============================================================================
# Gradient boosting
# ============================================================================

@register(
    "xgboost",
    tier="classical",
    stochastic=True,
    description="XGBoost regressor on lag features (log1p target).",
)
def xgboost_lag(out_fit, weather_fit, timestamps_fit, locations, horizon, seed):
    try:
        import xgboost as xgb
    except ImportError as e:
        raise RuntimeError("xgboost not installed — run `pip install xgboost`") from e
    # Opt into GPU only via `device` (XGBoost 2.x). Fall back silently if rejected.
    base_kwargs = dict(
        n_estimators=500,
        max_depth=6,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=seed,
        tree_method="hist",
        verbosity=0,
    )
    try:
        if _torch_cuda_available():
            model = xgb.XGBRegressor(**base_kwargs, device="cuda")
        else:
            model = xgb.XGBRegressor(**base_kwargs)
    except TypeError:
        model = xgb.XGBRegressor(**base_kwargs)
    preds = _fit_predict_tabular(
        model, out_fit, weather_fit, timestamps_fit, locations, horizon,
        log_target=True, origin_stride=_TABULAR_ORIGIN_STRIDE,
    )
    return preds, {"n_estimators": 500, "max_depth": 6, "lr": 0.05, "log_target": True,
                   "origin_stride": _TABULAR_ORIGIN_STRIDE}


@register(
    "lightgbm",
    tier="classical",
    stochastic=True,
    description="LightGBM regressor on lag features (log1p target).",
)
def lightgbm_lag(out_fit, weather_fit, timestamps_fit, locations, horizon, seed):
    try:
        import lightgbm as lgb
    except ImportError as e:
        raise RuntimeError("lightgbm not installed — run `pip install lightgbm`") from e
    model = lgb.LGBMRegressor(
        n_estimators=800,
        num_leaves=63,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=seed,
        verbosity=-1,
    )
    preds = _fit_predict_tabular(
        model, out_fit, weather_fit, timestamps_fit, locations, horizon,
        log_target=True, origin_stride=_TABULAR_ORIGIN_STRIDE,
    )
    return preds, {"n_estimators": 800, "num_leaves": 63, "lr": 0.05, "log_target": True,
                   "origin_stride": _TABULAR_ORIGIN_STRIDE}


# ============================================================================
# SARIMAX (per-county)
# ============================================================================

@register(
    "sarimax",
    tier="classical",
    stochastic=False,
    description="Per-county SARIMAX(1,0,1) — classical time-series baseline.",
)
def sarimax_model(out_fit, weather_fit, timestamps_fit, locations, horizon, seed):
    try:
        from statsmodels.tsa.statespace.sarimax import SARIMAX
    except ImportError as e:
        raise RuntimeError("statsmodels not installed") from e

    T, L = out_fit.shape
    preds = np.zeros((horizon, L), dtype=np.float32)
    failures = 0
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        for li in range(L):
            y = out_fit[:, li].astype(float)
            if len(y) < 8 or np.allclose(y, y[0]):
                continue
            try:
                res = SARIMAX(
                    y, order=(1, 0, 1),
                    enforce_stationarity=False, enforce_invertibility=False,
                ).fit(disp=False)
                fc = np.asarray(res.forecast(steps=horizon), dtype=float)
                preds[:, li] = np.clip(fc, 0, None)
            except Exception:
                failures += 1
                continue
    return preds, {"order": [1, 0, 1], "failed_counties": failures}


# ============================================================================
# Helpers
# ============================================================================

def _torch_cuda_available() -> bool:
    try:
        import torch
        return torch.cuda.is_available()
    except Exception:
        return False
