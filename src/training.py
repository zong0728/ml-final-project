"""Seed control, append-only run logging, resume support, summary tables."""
from __future__ import annotations

import json
import random
from pathlib import Path

import numpy as np
import pandas as pd

from . import config


# ============================================================================
# Seeds
# ============================================================================

def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
    except ImportError:
        pass


# ============================================================================
# Append-only run log
# ============================================================================

_LOG_COLS = [
    "timestamp", "model", "seed", "horizon",
    "val_rmse", "train_time_s", "config",
]


def log_run(
    model_name: str,
    seed: int,
    horizon: int,
    val_rmse: float,
    train_time_s: float,
    config_dict: dict | None = None,
    csv_path: Path = config.RUNS_CSV,
) -> None:
    """Append one run to the CSV (creates file with header if missing)."""
    row = {
        "timestamp": pd.Timestamp.now().isoformat(timespec="seconds"),
        "model": model_name,
        "seed": seed,
        "horizon": horizon,
        "val_rmse": round(float(val_rmse), 4),
        "train_time_s": round(float(train_time_s), 2),
        "config": json.dumps(config_dict or {}, default=str),
    }
    df = pd.DataFrame([row], columns=_LOG_COLS)
    write_header = not csv_path.exists()
    df.to_csv(csv_path, mode="a", header=write_header, index=False)


def load_runs(csv_path: Path = config.RUNS_CSV) -> pd.DataFrame:
    if not csv_path.exists():
        return pd.DataFrame(columns=_LOG_COLS)
    return pd.read_csv(csv_path)


def already_ran(model: str, seed: int, horizon: int, csv_path: Path = config.RUNS_CSV) -> bool:
    """True if this (model, seed, horizon) triplet is already logged."""
    df = load_runs(csv_path)
    if df.empty:
        return False
    mask = (df["model"] == model) & (df["seed"].astype(int) == int(seed)) & (df["horizon"].astype(int) == int(horizon))
    return bool(mask.any())


# ============================================================================
# Summary
# ============================================================================

def summarize_runs(csv_path: Path = config.RUNS_CSV) -> pd.DataFrame:
    """Aggregate per (model, horizon): mean / std / min / count of val_rmse."""
    df = load_runs(csv_path)
    if df.empty:
        return df
    agg = (
        df.groupby(["model", "horizon"])["val_rmse"]
        .agg(mean="mean", std="std", min="min", max="max", n="count")
        .reset_index()
    )
    agg["val_rmse_mean"] = agg["mean"].round(4)
    agg["val_rmse_std"] = agg["std"].fillna(0.0).round(4)
    agg["val_rmse_min"] = agg["min"].round(4)
    agg["val_rmse_max"] = agg["max"].round(4)
    agg = agg[["model", "horizon", "val_rmse_mean", "val_rmse_std", "val_rmse_min", "val_rmse_max", "n"]]
    # Sort: by horizon then by mean ascending
    return agg.sort_values(["horizon", "val_rmse_mean"]).reset_index(drop=True)


def best_model_per_horizon(csv_path: Path = config.RUNS_CSV) -> dict[int, str]:
    """Return {horizon: best_model_name} based on mean val_rmse."""
    summary = summarize_runs(csv_path)
    if summary.empty:
        return {}
    best: dict[int, str] = {}
    for h in sorted(summary["horizon"].unique()):
        sub = summary[summary["horizon"] == h]
        best[int(h)] = str(sub.iloc[0]["model"])
    return best
