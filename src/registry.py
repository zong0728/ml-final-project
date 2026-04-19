"""Model registry — a single source of truth for all architectures.

Every model is a callable with signature:

    fit_predict(
        out_fit:      (T, L) np.ndarray,       # raw training outages
        weather_fit:  (T, L, F) np.ndarray,    # raw training weather
        timestamps_fit: pd.DatetimeIndex,      # length T
        locations: list[str],                  # length L
        horizon: int,                          # 24 or 48
        seed: int,
    ) -> tuple[np.ndarray, dict]

Return:
    * preds: (horizon, L) non-negative forecast starting at timestamps_fit[-1] + 1 hour
    * config_dict: JSON-serializable hyperparameters (logged with the run)

Any model can ignore fields it doesn't use (e.g., baselines don't look at weather).
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import numpy as np
import pandas as pd


FitPredict = Callable[..., tuple[np.ndarray, dict]]


@dataclass
class ModelInfo:
    fn: FitPredict
    tier: str                # 'baseline' | 'classical' | 'neural' | 'sota'
    stochastic: bool         # whether to run once per SEED
    description: str = ""


MODEL_REGISTRY: dict[str, ModelInfo] = {}


def register(name: str, tier: str, stochastic: bool = False, description: str = ""):
    def decorator(fn: FitPredict) -> FitPredict:
        if name in MODEL_REGISTRY:
            raise KeyError(f"Model {name!r} already registered")
        MODEL_REGISTRY[name] = ModelInfo(fn=fn, tier=tier, stochastic=stochastic, description=description)
        return fn
    return decorator


def list_models() -> list[str]:
    return list(MODEL_REGISTRY.keys())
