"""High-diversity ensembles combining multiple gradient-boosting families and
zero-inflated treatments.

Ensembling works because errors from different model families are partially
decorrelated. Combining:
  * XGBoost (standard log-target)
  * LightGBM (standard log-target)
  * CatBoost (native categorical FIPS handling)
  * Two-stage LightGBM (hurdle decomposition for zero-inflation)
  * LightGBM Tweedie (compound Poisson-Gamma likelihood)

gives three *algorithmic* families × three *target-handling* strategies,
a typical setup for tabular Kaggle-style forecasting tasks.
"""
from __future__ import annotations

import numpy as np

from .models_classical import (
    catboost_lag,
    lightgbm_lag,
    lightgbm_tweedie,
    xgboost_lag,
)
from .models_zero_inflated import two_stage_lgb, two_stage_xgb
from .registry import register


@register(
    "ensemble_5way",
    tier="classical",
    stochastic=True,
    description="Mean of xgboost + lightgbm + catboost + two_stage_lgb + lightgbm_tweedie.",
)
def ensemble_5way(out_fit, weather_fit, timestamps_fit, locations, horizon, seed):
    """Five-way arithmetic mean. Each component is trained from scratch with
    the same seed, so diversity comes from model family and target handling."""
    p_xgb, _ = xgboost_lag(out_fit, weather_fit, timestamps_fit, locations, horizon, seed)
    p_lgb, _ = lightgbm_lag(out_fit, weather_fit, timestamps_fit, locations, horizon, seed)
    p_cat, _ = catboost_lag(out_fit, weather_fit, timestamps_fit, locations, horizon, seed)
    p_tsl, _ = two_stage_lgb(out_fit, weather_fit, timestamps_fit, locations, horizon, seed)
    p_twd, _ = lightgbm_tweedie(out_fit, weather_fit, timestamps_fit, locations, horizon, seed)

    preds = (p_xgb + p_lgb + p_cat + p_tsl + p_twd) / 5.0
    return preds.astype(np.float32), {
        "components": ["xgboost", "lightgbm", "catboost", "two_stage_lgb", "lightgbm_tweedie"],
        "weights": [0.2] * 5,
        "strategy": "arithmetic mean",
    }


@register(
    "ensemble_6way",
    tier="classical",
    stochastic=True,
    description="Mean of 5-way + two_stage_xgb (adds second hurdle base-learner).",
)
def ensemble_6way(out_fit, weather_fit, timestamps_fit, locations, horizon, seed):
    """Adds a second hurdle decomposition (XGBoost-based) on top of 5-way. A touch
    more algorithmic diversity in the zero-inflated treatment."""
    p_xgb, _ = xgboost_lag(out_fit, weather_fit, timestamps_fit, locations, horizon, seed)
    p_lgb, _ = lightgbm_lag(out_fit, weather_fit, timestamps_fit, locations, horizon, seed)
    p_cat, _ = catboost_lag(out_fit, weather_fit, timestamps_fit, locations, horizon, seed)
    p_tsl, _ = two_stage_lgb(out_fit, weather_fit, timestamps_fit, locations, horizon, seed)
    p_tsx, _ = two_stage_xgb(out_fit, weather_fit, timestamps_fit, locations, horizon, seed)
    p_twd, _ = lightgbm_tweedie(out_fit, weather_fit, timestamps_fit, locations, horizon, seed)

    preds = (p_xgb + p_lgb + p_cat + p_tsl + p_tsx + p_twd) / 6.0
    return preds.astype(np.float32), {
        "components": ["xgboost", "lightgbm", "catboost", "two_stage_lgb",
                       "two_stage_xgb", "lightgbm_tweedie"],
        "weights": [1 / 6] * 6,
        "strategy": "arithmetic mean",
    }
