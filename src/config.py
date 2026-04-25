"""Global config: paths, seeds, default hyperparameters.

Paths auto-detect Colab vs. local by searching upward for a `dataset/` directory.
Upload the whole project folder to Colab and the paths resolve automatically.
"""
from __future__ import annotations

import os
from pathlib import Path


def _detect_project_root() -> Path:
    """Walk up from this file until we find `dataset/`; fall back to cwd."""
    p = Path(__file__).resolve().parent
    for _ in range(6):
        if (p / "dataset").exists():
            return p
        p = p.parent
    return Path.cwd()


PROJECT_ROOT: Path = _detect_project_root()
DATA_DIR: Path = PROJECT_ROOT / "dataset" / "data"
RESULTS_DIR: Path = PROJECT_ROOT / "results"
CHECKPOINT_DIR: Path = RESULTS_DIR / "checkpoints"
FIGURES_DIR: Path = RESULTS_DIR / "figures"
SUBMISSIONS_DIR: Path = RESULTS_DIR / "submissions"
for d in (RESULTS_DIR, CHECKPOINT_DIR, FIGURES_DIR, SUBMISSIONS_DIR):
    d.mkdir(parents=True, exist_ok=True)

TRAIN_PATH: Path = DATA_DIR / "train.nc"
TEST_24H_PATH: Path = DATA_DIR / "test_24h_demo.nc"
TEST_48H_PATH: Path = DATA_DIR / "test_48h_demo.nc"

SUBMISSION_TEMPLATE_24H: Path = PROJECT_ROOT / "dataset" / "submission_template_24h.csv"
SUBMISSION_TEMPLATE_48H: Path = PROJECT_ROOT / "dataset" / "submission_template_48h.csv"

RUNS_CSV: Path = RESULTS_DIR / "experiment_runs.csv"

# ---------- Data split ----------
VAL_HOURS: int = 48              # legacy single-window split (kept for back-compat)
HORIZONS: list[int] = [24, 48]   # evaluate both forecasting horizons

# Rolling-origin CV: each fold mirrors test setup (val = horizon hours
# immediately after fit). Folds slide back by FOLD_STRIDE_HOURS.
N_FOLDS: int = 6
FOLD_STRIDE_HOURS: int = 72

# ---------- Seeds ----------
SEEDS: list[int] = [42, 43, 44]  # stochastic models run once per seed

# ---------- NN defaults (overridable per model via kwargs) ----------
SEQ_LEN: int = 48                # lookback window for neural models
DEFAULT_EPOCHS: int = 20
DEFAULT_BATCH_SIZE: int = 256
DEFAULT_LR: float = 1e-3
DEFAULT_HIDDEN: int = 128
DEFAULT_NUM_LAYERS: int = 2
DEFAULT_DROPOUT: float = 0.1
EARLY_STOP_PATIENCE: int = 5     # epochs without val improvement

# ---------- Classical ML defaults ----------
LAG_FEATURES: list[int] = [1, 2, 3, 6, 12, 24, 48, 168]  # outage lags for tree/LR models
ROLLING_WINDOWS: list[int] = [6, 24, 168]                # rolling-mean window sizes

# ---------- Misc ----------
VERBOSE: bool = True

# Torch device is resolved lazily to avoid importing torch at config time.
def get_device() -> str:
    import torch
    return "cuda" if torch.cuda.is_available() else "cpu"
