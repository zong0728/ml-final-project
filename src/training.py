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
# Append-only run log (fold-aware)
# ============================================================================

_LOG_COLS = [
    "timestamp", "model", "seed", "horizon", "fold",
    "val_rmse", "train_time_s", "config",
]


def log_run_fold(
    model_name: str,
    seed: int,
    horizon: int,
    fold: int,
    val_rmse: float,
    train_time_s: float,
    config_dict: dict | None = None,
    csv_path: Path = config.RUNS_CSV,
) -> None:
    """Append one (model, seed, horizon, fold) row."""
    row = {
        "timestamp": pd.Timestamp.now().isoformat(timespec="seconds"),
        "model": model_name,
        "seed": seed,
        "horizon": horizon,
        "fold": fold,
        "val_rmse": round(float(val_rmse), 4),
        "train_time_s": round(float(train_time_s), 2),
        "config": json.dumps(config_dict or {}, default=str),
    }
    df = pd.DataFrame([row], columns=_LOG_COLS)
    write_header = not csv_path.exists()
    df.to_csv(csv_path, mode="a", header=write_header, index=False)


# Back-compat: pre-fold rows (no `fold` col). Treated as fold=-1.
def log_run(
    model_name: str, seed: int, horizon: int,
    val_rmse: float, train_time_s: float,
    config_dict: dict | None = None,
    csv_path: Path = config.RUNS_CSV,
) -> None:
    log_run_fold(model_name, seed, horizon, fold=-1,
                 val_rmse=val_rmse, train_time_s=train_time_s,
                 config_dict=config_dict, csv_path=csv_path)


def load_runs(csv_path: Path = config.RUNS_CSV) -> pd.DataFrame:
    if not csv_path.exists():
        return pd.DataFrame(columns=_LOG_COLS)
    df = pd.read_csv(csv_path)
    if "fold" not in df.columns:
        df["fold"] = -1
    return df


def already_ran(model: str, seed: int, horizon: int, csv_path: Path = config.RUNS_CSV) -> bool:
    df = load_runs(csv_path)
    if df.empty:
        return False
    mask = ((df["model"] == model)
            & (df["seed"].astype(int) == int(seed))
            & (df["horizon"].astype(int) == int(horizon)))
    return bool(mask.any())


def already_ran_fold(model: str, seed: int, horizon: int, fold: int,
                     csv_path: Path = config.RUNS_CSV) -> bool:
    df = load_runs(csv_path)
    if df.empty:
        return False
    mask = ((df["model"] == model)
            & (df["seed"].astype(int) == int(seed))
            & (df["horizon"].astype(int) == int(horizon))
            & (df["fold"].astype(int) == int(fold)))
    return bool(mask.any())


# ============================================================================
# Summary
# ============================================================================

def summarize_runs(csv_path: Path = config.RUNS_CSV) -> pd.DataFrame:
    """Aggregate per (model, horizon) across folds and seeds.

    Reports mean/std/min/max RMSE plus number of (fold,seed) cells N.
    """
    df = load_runs(csv_path)
    if df.empty:
        return df
    # First average over seeds within each (model, horizon, fold), then aggregate folds.
    per_fold = (
        df.groupby(["model", "horizon", "fold"])["val_rmse"]
        .mean().reset_index()
    )
    agg = (
        per_fold.groupby(["model", "horizon"])["val_rmse"]
        .agg(mean="mean", std="std", min="min", max="max", n="count")
        .reset_index()
    )
    agg["val_rmse_mean"] = agg["mean"].round(4)
    agg["val_rmse_std"] = agg["std"].fillna(0.0).round(4)
    agg["val_rmse_min"] = agg["min"].round(4)
    agg["val_rmse_max"] = agg["max"].round(4)
    agg = agg[["model", "horizon", "val_rmse_mean", "val_rmse_std",
               "val_rmse_min", "val_rmse_max", "n"]]
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
