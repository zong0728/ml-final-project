"""RMSE metric + prediction-to-submission helpers.

The grading metric is: mean across counties of per-county RMSE over the
forecast horizon. This matches the demo's `evaluate_per_county` approach.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from . import config


def rmse(y_true, y_pred) -> float:
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    return float(np.sqrt(np.mean((y_true - y_pred) ** 2)))


def avg_county_rmse(truth: np.ndarray, pred: np.ndarray) -> float:
    """truth, pred both shape (T, L). Return mean of per-county RMSEs."""
    if truth.shape != pred.shape:
        raise ValueError(f"Shape mismatch: truth {truth.shape} vs pred {pred.shape}")
    T, L = truth.shape
    rmses = np.sqrt(np.mean((truth - pred) ** 2, axis=0))  # per-county RMSE
    return float(np.nanmean(rmses))


def per_county_rmse(truth: np.ndarray, pred: np.ndarray) -> np.ndarray:
    return np.sqrt(np.mean((truth - pred) ** 2, axis=0))


# ============================================================================
# Submission CSV I/O — must match the template row order (location-major).
# ============================================================================

def pred_to_submission_df(
    pred: np.ndarray,                # (T, L) non-negative prediction
    timestamps,                      # length T
    locations: list[str],            # length L
) -> pd.DataFrame:
    """Build a long-format DataFrame in the submission template order."""
    T, L = pred.shape
    assert T == len(timestamps), f"pred T={T} vs timestamps len={len(timestamps)}"
    assert L == len(locations), f"pred L={L} vs locations len={len(locations)}"
    rows = []
    for i, loc in enumerate(locations):
        rows.append(
            pd.DataFrame({
                "timestamp": np.asarray(timestamps),
                "location": str(loc),
                "pred": np.clip(pred[:, i], 0.0, None).astype(float),
            })
        )
    return pd.concat(rows, ignore_index=True)


def save_submission(pred: np.ndarray, timestamps, locations, horizon: int, tag: str) -> Path:
    """Write <tag>_pred_<horizon>h.csv to results/submissions/ and return the path."""
    df = pred_to_submission_df(pred, timestamps, locations)
    out = config.SUBMISSIONS_DIR / f"{tag}_pred_{horizon}h.csv"
    df.to_csv(out, index=False)
    return out
