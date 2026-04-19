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


@register(
    "lightgbm_tweedie",
    tier="classical",
    stochastic=True,
    description="LightGBM with Tweedie loss — native handling of zero-inflated positive data.",
)
def lightgbm_tweedie(out_fit, weather_fit, timestamps_fit, locations, horizon, seed):
    """Compound-Poisson-Gamma (Tweedie) objective is designed exactly for the
    mixture of zero mass + continuous positive tail we see in outage counts."""
    try:
        import lightgbm as lgb
    except ImportError as e:
        raise RuntimeError("lightgbm not installed") from e
    model = lgb.LGBMRegressor(
        objective="tweedie",
        tweedie_variance_power=1.5,   # 1 = Poisson, 2 = Gamma, 1.5 = in-between
        n_estimators=800,
        num_leaves=63,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=seed,
        verbosity=-1,
    )
    # NOTE: Tweedie works directly on raw (non-negative) target, no log needed.
    preds = _fit_predict_tabular(
        model, out_fit, weather_fit, timestamps_fit, locations, horizon,
        log_target=False, origin_stride=_TABULAR_ORIGIN_STRIDE,
    )
    return preds, {"objective": "tweedie", "tweedie_variance_power": 1.5,
                   "n_estimators": 800, "num_leaves": 63,
                   "origin_stride": _TABULAR_ORIGIN_STRIDE}


# ============================================================================
# CatBoost (third gradient-boosting family for ensembling)
# ============================================================================

@register(
    "catboost",
    tier="classical",
    stochastic=True,
    description="CatBoost regressor — handles categorical features & NaN natively.",
)
def catboost_lag(out_fit, weather_fit, timestamps_fit, locations, horizon, seed):
    try:
        from catboost import CatBoostRegressor
    except ImportError as e:
        raise RuntimeError("catboost not installed — run `pip install catboost`") from e
    # CatBoost can natively treat `location_idx` as categorical, which removes
    # the ordinal bias that xgboost/lightgbm suffer from on FIPS indices.
    # Our lag table appends location_idx as the LAST column (index P-1).
    from .data import build_lag_table
    X_tr, y_tr, X_inf, _, feature_names = build_lag_table(
        out=out_fit, weather=weather_fit, timestamps=timestamps_fit,
        horizon=horizon, origin_stride=_TABULAR_ORIGIN_STRIDE,
    )
    cat_idx = [feature_names.index("location_idx")]  # treat as categorical

    y_tr_log = np.log1p(np.clip(y_tr, 0, None))

    model = CatBoostRegressor(
        iterations=800,
        depth=6,
        learning_rate=0.05,
        l2_leaf_reg=3.0,
        subsample=0.8,
        bootstrap_type="Bernoulli",
        random_seed=seed,
        verbose=False,
        cat_features=cat_idx,
    )
    # CatBoost requires categorical columns as int
    X_tr_mix = X_tr.astype(object)
    X_tr_mix[:, cat_idx[0]] = X_tr[:, cat_idx[0]].astype(int)
    X_inf_mix = X_inf.astype(object)
    X_inf_mix[:, cat_idx[0]] = X_inf[:, cat_idx[0]].astype(int)

    model.fit(X_tr_mix, y_tr_log)
    pred_log = model.predict(X_inf_mix)
    max_log = float(np.log1p(y_tr.max())) + 2.0
    pred = np.expm1(np.clip(pred_log, None, max_log))
    max_y = float(y_tr.max()) * 1.5 + 10.0
    pred = np.clip(pred, 0.0, max_y)

    L = out_fit.shape[1]
    preds = pred.reshape(horizon, L).astype(np.float32)
    return preds, {"iterations": 800, "depth": 6, "lr": 0.05, "log_target": True,
                   "cat_features": ["location_idx"],
                   "origin_stride": _TABULAR_ORIGIN_STRIDE}


# ============================================================================
# Ensemble of xgboost + lightgbm + catboost (simple arithmetic mean)
# ============================================================================

@register(
    "ensemble_tree",
    tier="classical",
    stochastic=True,
    description="Simple average of xgboost + lightgbm + catboost predictions.",
)
def ensemble_tree(out_fit, weather_fit, timestamps_fit, locations, horizon, seed):
    """Averaging prediction diversity across GBT families — Kaggle-standard
    tabular ensembling. Each component is trained from scratch with the same
    seed so differences come from model family, not randomness."""
    p_xgb, _ = xgboost_lag(out_fit, weather_fit, timestamps_fit, locations, horizon, seed)
    p_lgb, _ = lightgbm_lag(out_fit, weather_fit, timestamps_fit, locations, horizon, seed)
    p_cat, _ = catboost_lag(out_fit, weather_fit, timestamps_fit, locations, horizon, seed)
    preds = (p_xgb + p_lgb + p_cat) / 3.0
    return preds.astype(np.float32), {
        "components": ["xgboost", "lightgbm", "catboost"],
        "weights": [1/3, 1/3, 1/3],
    }


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
