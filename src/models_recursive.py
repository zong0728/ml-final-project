"""Recursive (auto-regressive) forecasting wrappers.

Standard `build_lag_table` predicts all horizon steps simultaneously, with
features fixed at origin t. This means lag_1 (predicting t+1) and lag_1
(predicting t+48) both use out[t] — but for the t+48 prediction lag_1 should
ideally be y[t+47], which we don't have.

Recursive forecasting closes this loop:
  step 1: predict y[t+1] using lag_1=out[t], lag_2=out[t-1], ...
  step 2: predict y[t+2] using lag_1=ŷ[t+1] (predicted), lag_2=out[t], ...
  step h: predict y[t+h] using lag_1=ŷ[t+h-1], etc.

This re-uses recent predictions as inputs. For horizons > 1 hour and
strongly autoregressive series (which our data are — diurnal autocorr > 0.5),
this can reduce RMSE significantly.

Implementation strategy:
  1. Train ONE model that maps (features, h_step=1) -> y[t+1].  In other
     words, train only on horizon=1 targets.
  2. At inference, iterate: feed the current prediction back as the next
     lag_1 for h_step=2, etc.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from . import config
from .data import build_lag_table, calendar_features
from .registry import register


_ORIGIN_STRIDE = 4   # finer for recursive: more 1-step training pairs


def _train_one_step_model(out_fit, weather_fit, ts_fit, model_factory, seed):
    """Train a model on horizon=1 targets only."""
    X_tr, y_tr, _, _, feature_names = build_lag_table(
        out=out_fit, weather=weather_fit, timestamps=ts_fit,
        horizon=1, origin_stride=_ORIGIN_STRIDE,
    )
    y = np.log1p(np.clip(y_tr, 0, None))
    model = model_factory(seed)
    model.fit(X_tr, y)
    return model, feature_names, float(y_tr.max())


def _recursive_predict(model, out_fit, weather_fit, ts_fit, horizon,
                        feature_names, max_y):
    """Roll out horizon steps recursively.

    We mutate a copy of `out_fit`'s last few hours by appending each new
    prediction, then re-call build_lag_table with horizon=1 to get the
    next-step inference row. This is correct but slower than vectorizing.

    For test-time only (horizon predictions per origin) so the cost is OK.
    """
    L = out_fit.shape[1]
    out_extended = out_fit.copy().astype(np.float32)         # (T_fit, L)
    weather_extended = weather_fit.copy().astype(np.float32) # (T_fit, L, F)
    ts_extended = list(pd.to_datetime(ts_fit))

    # Persisted-weather assumption: future weather = last observed weather.
    # This is the realistic deployment assumption (utilities don't have
    # ground-truth future weather either at decision time).
    last_weather = weather_fit[-1:].copy()                   # (1, L, F)

    preds = np.zeros((horizon, L), dtype=np.float32)
    for step in range(horizon):
        # Build a fake outage row at the next timestamp (will be filled in
        # AFTER we predict it). For now extend with NaN.
        next_ts = ts_extended[-1] + pd.Timedelta(hours=1)

        # Append weather (persisted-future)
        weather_extended = np.concatenate([weather_extended, last_weather], axis=0)
        # Use the build_lag_table to get inference features at this new timestamp
        # Simplest: build with horizon=1 and take X_inf
        ts_now = pd.DatetimeIndex(ts_extended + [next_ts])
        # Pad out_extended with a placeholder zero (will be overwritten by prediction)
        out_padded = np.concatenate([out_extended, np.zeros((1, L), dtype=np.float32)], axis=0)
        # Need horizon arg = 1 just to use the inference path
        _, _, X_inf_step, _, _ = build_lag_table(
            out=out_padded, weather=weather_extended, timestamps=ts_now,
            horizon=1, origin_stride=_ORIGIN_STRIDE,
        )
        # X_inf_step is (L, P) — predict
        pred_log = model.predict(X_inf_step)
        max_log = float(np.log1p(max_y)) + 2.0
        pred_real = np.expm1(np.clip(pred_log, None, max_log))
        pred_real = np.clip(pred_real, 0.0, max_y * 1.5 + 10.0).astype(np.float32)
        preds[step] = pred_real

        # Append the prediction to out_extended for the next iteration
        out_extended = np.concatenate([out_extended, pred_real[None, :]], axis=0)
        ts_extended.append(next_ts)

    return preds


@register("lgb_recursive", tier="recursive", stochastic=True,
          description="LightGBM recursive (autoregressive) forecasting with persisted-future weather.")
def lgb_recursive(out_fit, weather_fit, timestamps_fit, locations, horizon, seed):
    import lightgbm as lgb

    def factory(s):
        return lgb.LGBMRegressor(
            n_estimators=1500, num_leaves=63, learning_rate=0.03,
            subsample=0.8, colsample_bytree=0.7, min_child_samples=20,
            random_state=s, verbosity=-1,
        )

    model, feat_names, max_y = _train_one_step_model(out_fit, weather_fit,
                                                       timestamps_fit, factory, seed)
    preds = _recursive_predict(model, out_fit, weather_fit, timestamps_fit,
                                 horizon, feat_names, max_y)
    return preds, {
        "method": "recursive_1step",
        "weather_assumption": "persisted_last_observed",
        "origin_stride": _ORIGIN_STRIDE,
    }


@register("xgb_recursive", tier="recursive", stochastic=True,
          description="XGBoost recursive forecasting with persisted-future weather.")
def xgb_recursive(out_fit, weather_fit, timestamps_fit, locations, horizon, seed):
    import xgboost as xgb

    def factory(s):
        kw = dict(
            n_estimators=800, max_depth=6, learning_rate=0.03,
            subsample=0.8, colsample_bytree=0.7, min_child_weight=5,
            random_state=s, tree_method="hist", verbosity=0,
        )
        try:
            import torch
            if torch.cuda.is_available():
                kw["device"] = "cuda"
        except Exception:
            pass
        return xgb.XGBRegressor(**kw)

    model, feat_names, max_y = _train_one_step_model(out_fit, weather_fit,
                                                       timestamps_fit, factory, seed)
    preds = _recursive_predict(model, out_fit, weather_fit, timestamps_fit,
                                 horizon, feat_names, max_y)
    return preds, {"method": "recursive_1step",
                   "weather_assumption": "persisted_last_observed"}
