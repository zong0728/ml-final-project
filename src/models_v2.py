"""Models that use the enhanced v2 lag table — registered as `*_v2`."""
from __future__ import annotations

import numpy as np

from .data_v2 import build_lag_table_v2
from .registry import register


_ORIGIN_STRIDE = 8


def _fit_predict_v2(out_fit, weather_fit, ts_fit, locations, horizon, seed,
                     model, log_target=True, **kwargs):
    """Shared pipeline using the v2 feature builder."""
    X_tr, y_tr, X_inf, _, _ = build_lag_table_v2(
        out=out_fit, weather=weather_fit, timestamps=ts_fit,
        horizon=horizon, origin_stride=_ORIGIN_STRIDE,
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
        "feature_set": "v2",
        "n_features": int(X_tr.shape[1]),
        "n_train_rows": int(X_tr.shape[0]),
        "origin_stride": _ORIGIN_STRIDE,
        "log_target": log_target,
        **kwargs,
    }


@register("lgb_v2", tier="v2", stochastic=True,
          description="LightGBM on v2 features (deep lags + statewide + county metadata + weather stats).")
def lgb_v2(out_fit, weather_fit, timestamps_fit, locations, horizon, seed):
    import lightgbm as lgb
    model = lgb.LGBMRegressor(
        n_estimators=1500, num_leaves=63, learning_rate=0.03,
        subsample=0.8, colsample_bytree=0.7,
        min_child_samples=30,
        random_state=seed, verbosity=-1,
    )
    return _fit_predict_v2(out_fit, weather_fit, timestamps_fit, locations,
                             horizon, seed, model)


@register("lgb_v2_deep", tier="v2", stochastic=True,
          description="LightGBM on v2 features, deeper trees / more leaves.")
def lgb_v2_deep(out_fit, weather_fit, timestamps_fit, locations, horizon, seed):
    import lightgbm as lgb
    model = lgb.LGBMRegressor(
        n_estimators=2000, num_leaves=127, learning_rate=0.03,
        subsample=0.8, colsample_bytree=0.6,
        min_child_samples=20,
        random_state=seed, verbosity=-1,
    )
    return _fit_predict_v2(out_fit, weather_fit, timestamps_fit, locations,
                             horizon, seed, model)


@register("xgb_v2", tier="v2", stochastic=True,
          description="XGBoost on v2 features.")
def xgb_v2(out_fit, weather_fit, timestamps_fit, locations, horizon, seed):
    import xgboost as xgb
    base_kwargs = dict(
        n_estimators=1000, max_depth=6, learning_rate=0.03,
        subsample=0.8, colsample_bytree=0.7,
        min_child_weight=5,
        random_state=seed, tree_method="hist", verbosity=0,
    )
    try:
        import torch
        if torch.cuda.is_available():
            base_kwargs["device"] = "cuda"
    except Exception:
        pass
    model = xgb.XGBRegressor(**base_kwargs)
    return _fit_predict_v2(out_fit, weather_fit, timestamps_fit, locations,
                             horizon, seed, model)


@register("cat_v2", tier="v2", stochastic=True,
          description="CatBoost on v2 features.")
def cat_v2(out_fit, weather_fit, timestamps_fit, locations, horizon, seed):
    from catboost import CatBoostRegressor
    model = CatBoostRegressor(
        iterations=1500, depth=6, learning_rate=0.03,
        l2_leaf_reg=3.0, subsample=0.8, bootstrap_type="Bernoulli",
        random_seed=seed, verbose=False,
    )
    return _fit_predict_v2(out_fit, weather_fit, timestamps_fit, locations,
                             horizon, seed, model)


@register("lgb_tweedie_v2", tier="v2", stochastic=True,
          description="LightGBM with Tweedie loss on v2 features.")
def lgb_tweedie_v2(out_fit, weather_fit, timestamps_fit, locations, horizon, seed):
    import lightgbm as lgb
    model = lgb.LGBMRegressor(
        objective="tweedie", tweedie_variance_power=1.5,
        n_estimators=1500, num_leaves=63, learning_rate=0.03,
        subsample=0.8, colsample_bytree=0.7,
        min_child_samples=30,
        random_state=seed, verbosity=-1,
    )
    return _fit_predict_v2(out_fit, weather_fit, timestamps_fit, locations,
                             horizon, seed, model, log_target=False)


@register("lgb_v2_mse", tier="v2", stochastic=True,
          description="LightGBM v2 features + DIRECT MSE (no log target). "
                       "RMSE evaluation rewards predicting big values when they happen.")
def lgb_v2_mse(out_fit, weather_fit, timestamps_fit, locations, horizon, seed):
    """Train directly on RMSE/MSE objective rather than log-space.
    Log-space training underestimates large outages because:
        E[exp(X)] != exp(E[X])  (Jensen's inequality)
    A direct-MSE LGB will be more aggressive on large values.
    """
    import lightgbm as lgb
    model = lgb.LGBMRegressor(
        objective="regression",   # MSE
        n_estimators=2000, num_leaves=63, learning_rate=0.03,
        subsample=0.8, colsample_bytree=0.7,
        min_child_samples=20,
        random_state=seed, verbosity=-1,
    )
    return _fit_predict_v2(out_fit, weather_fit, timestamps_fit, locations,
                             horizon, seed, model, log_target=False)


@register("lgb_v2_huber", tier="v2", stochastic=True,
          description="LightGBM v2 features + Huber loss (less sensitive to large outliers).")
def lgb_v2_huber(out_fit, weather_fit, timestamps_fit, locations, horizon, seed):
    import lightgbm as lgb
    model = lgb.LGBMRegressor(
        objective="huber", alpha=0.95,
        n_estimators=2000, num_leaves=63, learning_rate=0.03,
        subsample=0.8, colsample_bytree=0.7,
        min_child_samples=20,
        random_state=seed, verbosity=-1,
    )
    return _fit_predict_v2(out_fit, weather_fit, timestamps_fit, locations,
                             horizon, seed, model, log_target=False)


@register("lgb_v2_weighted", tier="v2", stochastic=True,
          description="LightGBM v2 with sample weights upweighting non-zero samples.")
def lgb_v2_weighted(out_fit, weather_fit, timestamps_fit, locations, horizon, seed):
    """Sample weights ∝ log1p(y) — non-zero samples get more weight.

    This counteracts the model's natural bias toward predicting 0 (because
    70% of training samples have y=0). When the storm hits, we want the
    model to commit to non-zero predictions.
    """
    import lightgbm as lgb
    X_tr, y_tr, X_inf, _, _ = build_lag_table_v2(
        out=out_fit, weather=weather_fit, timestamps=timestamps_fit,
        horizon=horizon, origin_stride=_ORIGIN_STRIDE,
    )
    weights = 1.0 + np.log1p(np.clip(y_tr, 0, None))   # weight = 1 for y=0, ~7 for y=1000
    model = lgb.LGBMRegressor(
        objective="regression",
        n_estimators=2000, num_leaves=63, learning_rate=0.03,
        subsample=0.8, colsample_bytree=0.7,
        min_child_samples=20,
        random_state=seed, verbosity=-1,
    )
    model.fit(X_tr, y_tr, sample_weight=weights)
    pred = model.predict(X_inf)
    cap = float(y_tr.max()) * 1.5 + 10.0
    pred = np.clip(pred, 0.0, cap).astype(np.float32)
    L = out_fit.shape[1]
    return pred.reshape(horizon, L), {
        "feature_set": "v2", "objective": "weighted_MSE",
        "weight_formula": "1 + log1p(y)",
    }


@register("lgb_v2_log_corrected", tier="v2", stochastic=True,
          description="LightGBM v2 log-target with bias correction E[exp(X)] = exp(mu + sigma^2/2).")
def lgb_v2_log_corrected(out_fit, weather_fit, timestamps_fit, locations, horizon, seed):
    """Apply Smearing / Duan's correction when transforming back from log space.
    Standard predict log -> exp gives the median of y, NOT the mean.
    For RMSE we want the mean, which under log-normal noise is
        E[y] = exp(mu) * E[exp(eps)]
    where eps has variance sigma^2. Estimate sigma^2 from training residuals
    and apply correction factor.
    """
    import lightgbm as lgb
    X_tr, y_tr, X_inf, _, _ = build_lag_table_v2(
        out=out_fit, weather=weather_fit, timestamps=timestamps_fit,
        horizon=horizon, origin_stride=_ORIGIN_STRIDE,
    )
    y_log = np.log1p(np.clip(y_tr, 0, None))
    model = lgb.LGBMRegressor(
        n_estimators=2000, num_leaves=63, learning_rate=0.03,
        subsample=0.8, colsample_bytree=0.7,
        min_child_samples=20,
        random_state=seed, verbosity=-1,
    )
    model.fit(X_tr, y_log)
    # Estimate residual variance on training set (same set; could use OOB)
    train_pred_log = model.predict(X_tr)
    sigma2 = float(np.var(y_log - train_pred_log))
    # Inference
    pred_log = model.predict(X_inf)
    cap = float(np.log1p(y_tr.max())) + 2.0
    pred = np.expm1(np.clip(pred_log, None, cap))
    # Smearing correction: multiply by exp(sigma^2/2)
    pred = pred * np.exp(0.5 * sigma2)
    cap_y = float(y_tr.max()) * 1.5 + 10.0
    pred = np.clip(pred, 0.0, cap_y).astype(np.float32)
    L = out_fit.shape[1]
    return pred.reshape(horizon, L), {
        "feature_set": "v2", "log_target": True,
        "smearing_sigma2": float(sigma2),
        "smearing_factor": float(np.exp(0.5 * sigma2)),
    }


@register("two_stage_lgb_v2", tier="v2", stochastic=True,
          description="Two-stage LightGBM hurdle on v2 features.")
def two_stage_lgb_v2(out_fit, weather_fit, timestamps_fit, locations, horizon, seed):
    import lightgbm as lgb
    X_tr, y_tr, X_inf, _, _ = build_lag_table_v2(
        out=out_fit, weather=weather_fit, timestamps=timestamps_fit,
        horizon=horizon, origin_stride=_ORIGIN_STRIDE,
    )
    # Stage 1: classifier
    y_bin = (y_tr > 0).astype(int)
    clf = lgb.LGBMClassifier(
        n_estimators=800, num_leaves=63, learning_rate=0.03,
        subsample=0.8, colsample_bytree=0.7, min_child_samples=30,
        random_state=seed, verbosity=-1,
    )
    clf.fit(X_tr, y_bin)
    p_pos = clf.predict_proba(X_inf)[:, 1]

    # Stage 2: regressor on positives
    pos_mask = y_tr > 0
    if pos_mask.sum() < 50:
        pred = p_pos * float(y_tr.mean())
    else:
        X_pos = X_tr[pos_mask]; y_pos = y_tr[pos_mask]
        y_pos_log = np.log1p(y_pos)
        reg = lgb.LGBMRegressor(
            n_estimators=1500, num_leaves=63, learning_rate=0.03,
            subsample=0.8, colsample_bytree=0.7, min_child_samples=20,
            random_state=seed, verbosity=-1,
        )
        reg.fit(X_pos, y_pos_log)
        mu_log = reg.predict(X_inf)
        max_log = float(np.log1p(y_pos.max())) + 2.0
        mu = np.expm1(np.clip(mu_log, None, max_log))
        pred = p_pos * mu
    cap = float(y_tr.max()) * 1.5 + 10.0
    pred = np.clip(pred, 0.0, cap).astype(np.float32)
    L = out_fit.shape[1]
    return pred.reshape(horizon, L), {
        "feature_set": "v2", "log_target": "stage2_only",
        "pos_fraction": float(pos_mask.mean()),
        "origin_stride": _ORIGIN_STRIDE,
    }
