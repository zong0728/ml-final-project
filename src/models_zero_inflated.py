"""Zero-inflated regression models — textbook treatment for sparse count data.

Our target distribution has ~70.5% zeros, creating a **compound distribution**:
    y = 0           with probability  (1 - π(x))
    y ~ positive    with probability  π(x)

A single MSE-trained regressor over-predicts zeros (it "gives up" on spikes
to minimize average squared error on the abundant zero mass). Two-stage
decomposition handles this explicitly:

    E[y | x] = P(y > 0 | x) · E[y | y > 0, x]
             = π̂(x)        · μ̂(x)

  * π̂(x): binary classifier, trained on all samples with label [y > 0].
  * μ̂(x): regressor, trained ONLY on samples where y > 0 (the "positive subset").
           Target log1p(y) for numerical stability.

Final prediction = π̂(x) × expm1(μ̂(x)), clipped to [0, max_y].

References
----------
* Cragg (1971) "Some Statistical Models for Limited Dependent Variables
  with Application to the Demand for Durable Goods" — hurdle models.
* Lambert (1992) "Zero-Inflated Poisson Regression" — ZIP models.
"""
from __future__ import annotations

import numpy as np

from . import config
from .data import build_lag_table
from .registry import register

_TABULAR_ORIGIN_STRIDE = 4


def _two_stage_predict(
    out_fit, weather_fit, timestamps_fit, horizon, seed,
    clf_factory, reg_factory,
    positive_threshold: float = 0.0,
):
    """Shared two-stage fit/predict pipeline.

    Parameters
    ----------
    clf_factory : callable returning an sklearn-compatible binary classifier
                  (must expose `.fit(X, y)` and `.predict_proba(X)`)
    reg_factory : callable returning an sklearn-compatible regressor
                  (trained on log1p target on the positive subset)
    positive_threshold : float
        Sample counts as "positive" when `y > positive_threshold`. Default 0.0
        makes the classifier predict "will any household lose power?".
    """
    X_tr, y_tr, X_inf, _, _ = build_lag_table(
        out=out_fit, weather=weather_fit, timestamps=timestamps_fit,
        horizon=horizon, origin_stride=_TABULAR_ORIGIN_STRIDE,
    )

    # ---- Stage 1: classifier over all samples ----
    y_binary = (y_tr > positive_threshold).astype(int)
    clf = clf_factory()
    clf.fit(X_tr, y_binary)
    p_pos = clf.predict_proba(X_inf)[:, 1]          # P(y > 0 | x) at inference

    # ---- Stage 2: regressor on the positive subset ----
    pos_mask = y_tr > positive_threshold
    if pos_mask.sum() < 50:
        # Not enough positive samples — fall back to classifier-only (rare)
        pred = p_pos * np.mean(y_tr[pos_mask]) if pos_mask.any() else np.zeros_like(p_pos)
    else:
        X_pos = X_tr[pos_mask]
        y_pos = y_tr[pos_mask]
        y_pos_log = np.log1p(y_pos)
        reg = reg_factory()
        reg.fit(X_pos, y_pos_log)
        mu_log = reg.predict(X_inf)
        max_log = float(np.log1p(y_pos.max())) + 2.0
        mu = np.expm1(np.clip(mu_log, None, max_log))
        pred = p_pos * mu

    # Safety clip
    max_y = float(y_tr.max()) * 1.5 + 10.0
    pred = np.clip(pred, 0.0, max_y).astype(np.float32)

    L = out_fit.shape[1]
    return pred.reshape(horizon, L), {
        "classifier": clf.__class__.__name__,
        "regressor": reg.__class__.__name__ if pos_mask.sum() >= 50 else "fallback_mean",
        "pos_fraction": float(pos_mask.mean()),
        "positive_threshold": positive_threshold,
        "origin_stride": _TABULAR_ORIGIN_STRIDE,
    }


@register(
    "two_stage_xgb",
    tier="zero_inflated",
    stochastic=True,
    description="Two-stage model: XGBoost classifier (y>0) × XGBoost regressor on positives.",
)
def two_stage_xgb(out_fit, weather_fit, timestamps_fit, locations, horizon, seed):
    try:
        import xgboost as xgb
    except ImportError as e:
        raise RuntimeError("xgboost not installed") from e

    def clf_factory():
        return xgb.XGBClassifier(
            n_estimators=400,
            max_depth=6,
            learning_rate=0.05,
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=seed,
            tree_method="hist",
            eval_metric="logloss",
            verbosity=0,
        )

    def reg_factory():
        return xgb.XGBRegressor(
            n_estimators=500,
            max_depth=6,
            learning_rate=0.05,
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=seed,
            tree_method="hist",
            verbosity=0,
        )

    return _two_stage_predict(
        out_fit, weather_fit, timestamps_fit, horizon, seed,
        clf_factory=clf_factory, reg_factory=reg_factory,
    )


@register(
    "two_stage_lgb",
    tier="zero_inflated",
    stochastic=True,
    description="Two-stage model: LightGBM classifier × LightGBM regressor on positives.",
)
def two_stage_lgb(out_fit, weather_fit, timestamps_fit, locations, horizon, seed):
    try:
        import lightgbm as lgb
    except ImportError as e:
        raise RuntimeError("lightgbm not installed") from e

    def clf_factory():
        return lgb.LGBMClassifier(
            n_estimators=600,
            num_leaves=63,
            learning_rate=0.05,
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=seed,
            verbosity=-1,
        )

    def reg_factory():
        return lgb.LGBMRegressor(
            n_estimators=800,
            num_leaves=63,
            learning_rate=0.05,
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=seed,
            verbosity=-1,
        )

    return _two_stage_predict(
        out_fit, weather_fit, timestamps_fit, horizon, seed,
        clf_factory=clf_factory, reg_factory=reg_factory,
    )
