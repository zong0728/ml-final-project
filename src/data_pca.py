"""PCA-reduced weather features.

Educator slide 13 hints: 'Could use techniques like PCA to reduce dimensionality'.

The 109 weather features (after dropping zero-variance, ~91) are highly
correlated (cape ~ cape_1; tcc ~ lcc; multiple wind components). PCA on the
training-only weather cube reduces this to k orthogonal components that
explain >95% of the variance, then the lag table is built on the PCA features
instead of raw.

The expected effect on GBDT RMSE is small (gain-importance already filters).
But this is a clean ablation that documents we tested the educator's hint.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from . import config
from .data import build_lag_table
from .registry import register


_ORIGIN_STRIDE = 8


def _pca_fit_transform(weather: np.ndarray, k: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Fit PCA on (T, L, F) -> reduce F to k. Returns (W_pca, mean_F, components)."""
    T, L, F = weather.shape
    flat = weather.reshape(-1, F)            # (T*L, F)
    mean = np.nanmean(flat, axis=0)
    flat_c = flat - mean
    # SVD: flat_c = U S V^T  ->  components = V[:k]
    # Use np.linalg.svd with full_matrices=False
    flat_c = np.nan_to_num(flat_c, nan=0.0)
    _, _, Vt = np.linalg.svd(flat_c, full_matrices=False)
    components = Vt[:k]                       # (k, F)
    proj = flat_c @ components.T              # (T*L, k)
    return proj.reshape(T, L, k), mean, components


def _pca_apply(weather: np.ndarray, mean: np.ndarray, components: np.ndarray) -> np.ndarray:
    """Apply pre-fit PCA to a new weather cube."""
    T, L, F = weather.shape
    k = components.shape[0]
    flat = weather.reshape(-1, F) - mean
    flat = np.nan_to_num(flat, nan=0.0)
    proj = flat @ components.T                # (T*L, k)
    return proj.reshape(T, L, k)


def _fit_predict_pca(out_fit, weather_fit, ts_fit, locations, horizon, seed,
                     model, k: int = 20, log_target: bool = True):
    """Reduce weather to k PCs (fit on FIT data only), then run the standard
    lag table with reduced weather."""
    weather_pca, w_mean, w_comp = _pca_fit_transform(weather_fit, k)

    X_tr, y_tr, X_inf, _, _ = build_lag_table(
        out=out_fit, weather=weather_pca, timestamps=ts_fit,
        horizon=horizon, origin_stride=_ORIGIN_STRIDE,
        weather_lags=(3, 12, 24),
    )
    if log_target:
        y = np.log1p(np.clip(y_tr, 0, None))
    else:
        y = y_tr
    model.fit(X_tr, y)
    pred = model.predict(X_inf)
    if log_target:
        cap = float(np.log1p(y_tr.max())) + 2.0
        pred = np.expm1(np.clip(pred, None, cap))
    cap_y = float(y_tr.max()) * 1.5 + 10.0
    pred = np.clip(pred, 0.0, cap_y)
    L = out_fit.shape[1]
    return pred.reshape(horizon, L).astype(np.float32), {
        "feature_set": "pca",
        "n_pca_components": int(k),
        "n_features": int(X_tr.shape[1]),
        "origin_stride": _ORIGIN_STRIDE,
        "log_target": log_target,
    }


@register("lgb_pca20", tier="pca", stochastic=True,
          description="LightGBM on PCA-20 weather features (educator slide 13 hint).")
def lgb_pca20(out_fit, weather_fit, timestamps_fit, locations, horizon, seed):
    import lightgbm as lgb
    model = lgb.LGBMRegressor(
        n_estimators=1500, num_leaves=63, learning_rate=0.03,
        subsample=0.8, colsample_bytree=0.8, min_child_samples=30,
        random_state=seed, verbosity=-1,
    )
    return _fit_predict_pca(out_fit, weather_fit, timestamps_fit, locations,
                              horizon, seed, model, k=20)


@register("lgb_pca30", tier="pca", stochastic=True,
          description="LightGBM on PCA-30 weather features.")
def lgb_pca30(out_fit, weather_fit, timestamps_fit, locations, horizon, seed):
    import lightgbm as lgb
    model = lgb.LGBMRegressor(
        n_estimators=1500, num_leaves=63, learning_rate=0.03,
        subsample=0.8, colsample_bytree=0.8, min_child_samples=30,
        random_state=seed, verbosity=-1,
    )
    return _fit_predict_pca(out_fit, weather_fit, timestamps_fit, locations,
                              horizon, seed, model, k=30)


@register("xgb_pca20", tier="pca", stochastic=True,
          description="XGBoost on PCA-20 weather features.")
def xgb_pca20(out_fit, weather_fit, timestamps_fit, locations, horizon, seed):
    import xgboost as xgb
    base_kwargs = dict(
        n_estimators=1000, max_depth=6, learning_rate=0.03,
        subsample=0.8, colsample_bytree=0.7, min_child_weight=5,
        random_state=seed, tree_method="hist", verbosity=0,
    )
    try:
        import torch
        if torch.cuda.is_available():
            base_kwargs["device"] = "cuda"
    except Exception:
        pass
    model = xgb.XGBRegressor(**base_kwargs)
    return _fit_predict_pca(out_fit, weather_fit, timestamps_fit, locations,
                              horizon, seed, model, k=20)


@register("cat_pca20", tier="pca", stochastic=True,
          description="CatBoost on PCA-20 weather features.")
def cat_pca20(out_fit, weather_fit, timestamps_fit, locations, horizon, seed):
    from catboost import CatBoostRegressor
    model = CatBoostRegressor(
        iterations=1500, depth=6, learning_rate=0.03,
        l2_leaf_reg=3.0, subsample=0.8, bootstrap_type="Bernoulli",
        random_seed=seed, verbose=False,
    )
    return _fit_predict_pca(out_fit, weather_fit, timestamps_fit, locations,
                              horizon, seed, model, k=20)
