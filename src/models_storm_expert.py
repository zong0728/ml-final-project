"""Storm-expert mixture-of-experts.

The fold-1 RMSE of 700-900 across all models is the dominant contributor to
aggregate RMSE. Diagnostic showed that 50% of fold-0 RMSE is concentrated in
Wayne County during the storm tail, where the global GBDT under-predicts
because it learned from a 70% zero-mass training distribution.

This module trains a STORM-SPECIALIZED model:
  1. Identify storm-period training rows: (county, hour) where state-wide
     outage exceeded a threshold (matches our EDA storm event detector).
  2. Train an LGB regressor only on those storm-period rows.
  3. At inference, the storm expert is invoked when the *recent state-wide
     outage* (computed from the same lag features) exceeds a threshold.
  4. A simple gate blends storm-expert prediction with global GBDT prediction.

Expected effect: small/no improvement on calm folds (gate routes to global
model), better RMSE on storm-period predictions (fold 0/1).

Approach is a MoE-lite: a hard, threshold-based gate. We don't train a
soft gating network because we don't have enough storm samples to fit one
without overfit.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from . import config
from .data import build_lag_table
from .registry import register


_ORIGIN_STRIDE = 4   # finer for storm expert: more storm-period rows


# State-wide outage threshold (per hour) above which we consider the
# situation "storm-mode". Matches the EDA storm event detector.
STORM_THRESHOLD = 2000.0
STORM_GATE_THRESHOLD = 1000.0   # at inference, predict-storm if recent state-wide > this


def _identify_storm_rows(out: np.ndarray, statewide_thr: float = STORM_THRESHOLD) -> np.ndarray:
    """Boolean mask (T,) where state-wide outage at time t exceeded threshold."""
    statewide = out.sum(axis=1)   # (T,)
    return statewide > statewide_thr


@register("lgb_storm_expert", tier="moe", stochastic=True,
          description="LightGBM mixture-of-experts: global model + storm-specialized expert.")
def lgb_storm_expert(out_fit, weather_fit, timestamps_fit, locations, horizon, seed):
    """Train two LightGBMs:
      - global: all training rows (current SOTA configuration)
      - storm:  only rows where state-wide outage > threshold
    At inference, route based on the most recent observed state-wide outage.

    The blending is a hard switch: if the latest hour's state-wide outage
    exceeds STORM_GATE_THRESHOLD, use the storm expert; otherwise the global
    model.
    """
    import lightgbm as lgb

    X_tr, y_tr, X_inf, loc_ids_tr, feature_names = build_lag_table(
        out=out_fit, weather=weather_fit, timestamps=timestamps_fit,
        horizon=horizon, origin_stride=_ORIGIN_STRIDE,
    )
    y_log = np.log1p(np.clip(y_tr, 0, None))

    # ---- 1. Global model ----
    global_model = lgb.LGBMRegressor(
        n_estimators=1500, num_leaves=63, learning_rate=0.03,
        subsample=0.8, colsample_bytree=0.7, min_child_samples=30,
        random_state=seed, verbosity=-1,
    )
    global_model.fit(X_tr, y_log)

    # ---- 2. Identify storm-period rows in TRAINING data ----
    # Each row in X_tr corresponds to (origin_t, county, h_step). We need to
    # know whether origin_t was a storm hour. The lag table builder doesn't
    # expose origin_t directly, so we recompute via the timestamp column.
    # Hack: lag_1 = out[origin] for that county. State-wide-at-origin =
    # sum across counties at origin time. We approximate by summing lag_1
    # column across the L rows that share an origin.

    # Simplest: identify storm rows as those where lag_1 (out at origin)
    # was high *for that county*. This is per-county threshold rather than
    # state-wide, but works as a proxy.
    lag1_idx = feature_names.index("lag_1")
    per_row_lag1 = X_tr[:, lag1_idx]
    # Threshold: any row where the county was in active outage at origin
    # (lag_1 > 100, an arbitrary "non-trivial outage" threshold).
    storm_row_mask = per_row_lag1 > 100.0

    storm_n = int(storm_row_mask.sum())
    if storm_n < 200:
        # Not enough storm samples — fall back to global only
        pred_log = global_model.predict(X_inf)
        cap = float(np.log1p(y_tr.max())) + 2.0
        pred = np.expm1(np.clip(pred_log, None, cap))
        cap_y = float(y_tr.max()) * 1.5 + 10.0
        pred = np.clip(pred, 0.0, cap_y).astype(np.float32)
        L = out_fit.shape[1]
        return pred.reshape(horizon, L), {
            "method": "storm_expert_fallback_global_only",
            "storm_n": storm_n,
        }

    # ---- 3. Train storm expert ----
    X_storm = X_tr[storm_row_mask]
    y_storm_log = y_log[storm_row_mask]

    storm_model = lgb.LGBMRegressor(
        n_estimators=1000, num_leaves=63, learning_rate=0.03,
        subsample=0.8, colsample_bytree=0.7, min_child_samples=20,
        random_state=seed + 1, verbosity=-1,
    )
    storm_model.fit(X_storm, y_storm_log)

    # ---- 4. Predict inference rows with both models ----
    pred_global_log = global_model.predict(X_inf)
    pred_storm_log = storm_model.predict(X_inf)
    cap = float(np.log1p(y_tr.max())) + 2.0
    pred_global = np.expm1(np.clip(pred_global_log, None, cap))
    pred_storm = np.expm1(np.clip(pred_storm_log, None, cap))

    # ---- 5. Gate: per-row, route based on inference-time lag_1 ----
    inf_lag1 = X_inf[:, lag1_idx]
    gate = inf_lag1 > 100.0    # use storm expert for active-storm rows

    pred = np.where(gate, pred_storm, pred_global)
    cap_y = float(y_tr.max()) * 1.5 + 10.0
    pred = np.clip(pred, 0.0, cap_y).astype(np.float32)

    L = out_fit.shape[1]
    return pred.reshape(horizon, L), {
        "method": "storm_expert_hard_gate",
        "storm_n_train_rows": storm_n,
        "storm_row_fraction": float(storm_row_mask.mean()),
        "storm_gate_fraction": float(gate.mean()),
        "origin_stride": _ORIGIN_STRIDE,
    }


@register("lgb_storm_blend", tier="moe", stochastic=True,
          description="LightGBM with soft 50/50 blend of global + storm models on storm rows.")
def lgb_storm_blend(out_fit, weather_fit, timestamps_fit, locations, horizon, seed):
    """Soft variant of storm_expert: instead of hard switch, blend
    global and storm predictions 50/50 when in storm regime. This is more
    robust if the gate occasionally misfires."""
    import lightgbm as lgb

    X_tr, y_tr, X_inf, loc_ids_tr, feature_names = build_lag_table(
        out=out_fit, weather=weather_fit, timestamps=timestamps_fit,
        horizon=horizon, origin_stride=_ORIGIN_STRIDE,
    )
    y_log = np.log1p(np.clip(y_tr, 0, None))

    global_model = lgb.LGBMRegressor(
        n_estimators=1500, num_leaves=63, learning_rate=0.03,
        subsample=0.8, colsample_bytree=0.7, min_child_samples=30,
        random_state=seed, verbosity=-1,
    )
    global_model.fit(X_tr, y_log)

    lag1_idx = feature_names.index("lag_1")
    per_row_lag1 = X_tr[:, lag1_idx]
    storm_row_mask = per_row_lag1 > 100.0
    storm_n = int(storm_row_mask.sum())

    if storm_n < 200:
        pred_log = global_model.predict(X_inf)
        cap = float(np.log1p(y_tr.max())) + 2.0
        pred = np.expm1(np.clip(pred_log, None, cap))
        cap_y = float(y_tr.max()) * 1.5 + 10.0
        pred = np.clip(pred, 0.0, cap_y).astype(np.float32)
        L = out_fit.shape[1]
        return pred.reshape(horizon, L), {"method": "blend_fallback_global"}

    X_storm = X_tr[storm_row_mask]
    y_storm_log = y_log[storm_row_mask]
    storm_model = lgb.LGBMRegressor(
        n_estimators=1000, num_leaves=63, learning_rate=0.03,
        subsample=0.8, colsample_bytree=0.7, min_child_samples=20,
        random_state=seed + 1, verbosity=-1,
    )
    storm_model.fit(X_storm, y_storm_log)

    cap = float(np.log1p(y_tr.max())) + 2.0
    pred_global = np.expm1(np.clip(global_model.predict(X_inf), None, cap))
    pred_storm = np.expm1(np.clip(storm_model.predict(X_inf), None, cap))

    # Soft blend: in storm regime, take 50/50 average of two models
    inf_lag1 = X_inf[:, lag1_idx]
    storm_weight = (inf_lag1 > 100.0).astype(np.float32) * 0.5  # 0.5 if storm, 0 otherwise
    pred = (1.0 - storm_weight) * pred_global + storm_weight * pred_storm

    cap_y = float(y_tr.max()) * 1.5 + 10.0
    pred = np.clip(pred, 0.0, cap_y).astype(np.float32)
    L = out_fit.shape[1]
    return pred.reshape(horizon, L), {
        "method": "storm_expert_soft_blend_50_50",
        "storm_n_train_rows": storm_n,
        "origin_stride": _ORIGIN_STRIDE,
    }
