"""Standard neural architectures for sequence forecasting.

All models share the same I/O contract:
  * Input windows: (B, seq_len, D) with D = 1 (outage) + F (weather) + 6 (calendar).
  * Output: (B, horizon) — scaled future outage for the query county.
Inputs & targets are z-scored using training-only statistics; the wrapper
inverse-scales + clips at inference.
"""
from __future__ import annotations

import math
import time
import warnings
from contextlib import nullcontext
from typing import Callable

warnings.filterwarnings("ignore", category=FutureWarning, message=r".*torch\.cuda\.amp.*")

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from . import config
from .data import Scalers, build_nn_windows, fit_scalers
from .registry import register
from .training import set_seed


def _make_scaler(enabled: bool):
    """Return a GradScaler using the new API if available, else the old one."""
    if not enabled:
        return torch.cuda.amp.GradScaler(enabled=False)
    try:
        return torch.amp.GradScaler("cuda", enabled=True)  # torch >= 2.3
    except (AttributeError, TypeError):
        return torch.cuda.amp.GradScaler(enabled=True)


def _autocast(enabled: bool):
    if not enabled:
        return nullcontext()
    try:
        return torch.amp.autocast("cuda", enabled=True)   # torch >= 2.3
    except (AttributeError, TypeError):
        return torch.cuda.amp.autocast(enabled=True)


# ============================================================================
# Architectures
# ============================================================================

class MLPNet(nn.Module):
    def __init__(self, input_dim: int, seq_len: int, hidden_dim: int, horizon: int, dropout: float = 0.1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Flatten(),
            nn.Linear(input_dim * seq_len, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, horizon),
        )

    def forward(self, x):
        return self.net(x)


class LSTMNet(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int, num_layers: int, horizon: int,
                 bidirectional: bool = False, dropout: float = 0.1):
        super().__init__()
        self.lstm = nn.LSTM(
            input_dim, hidden_dim, num_layers,
            batch_first=True,
            bidirectional=bidirectional,
            dropout=dropout if num_layers > 1 else 0.0,
        )
        out_dim = hidden_dim * (2 if bidirectional else 1)
        self.head = nn.Linear(out_dim, horizon)

    def forward(self, x):
        out, _ = self.lstm(x)
        return self.head(out[:, -1])


class GRUNet(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int, num_layers: int, horizon: int, dropout: float = 0.1):
        super().__init__()
        self.gru = nn.GRU(
            input_dim, hidden_dim, num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )
        self.head = nn.Linear(hidden_dim, horizon)

    def forward(self, x):
        out, _ = self.gru(x)
        return self.head(out[:, -1])


class TCNBlock(nn.Module):
    def __init__(self, in_c: int, out_c: int, kernel: int, dilation: int, dropout: float):
        super().__init__()
        pad = (kernel - 1) * dilation
        self.conv1 = nn.Conv1d(in_c, out_c, kernel, padding=pad, dilation=dilation)
        self.conv2 = nn.Conv1d(out_c, out_c, kernel, padding=pad, dilation=dilation)
        self.drop = nn.Dropout(dropout)
        self.res = nn.Conv1d(in_c, out_c, 1) if in_c != out_c else nn.Identity()
        self.pad = pad

    def forward(self, x):
        # causal: trim right-padding to keep length
        y = self.conv1(x)[:, :, :-self.pad] if self.pad > 0 else self.conv1(x)
        y = torch.relu(y)
        y = self.drop(y)
        y = self.conv2(y)[:, :, :-self.pad] if self.pad > 0 else self.conv2(y)
        y = torch.relu(y)
        y = self.drop(y)
        return torch.relu(y + self.res(x))


class TCNNet(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int, horizon: int,
                 num_blocks: int = 4, kernel: int = 3, dropout: float = 0.1):
        super().__init__()
        layers = []
        c = input_dim
        for i in range(num_blocks):
            layers.append(TCNBlock(c, hidden_dim, kernel, dilation=2 ** i, dropout=dropout))
            c = hidden_dim
        self.blocks = nn.Sequential(*layers)
        self.head = nn.Linear(hidden_dim, horizon)

    def forward(self, x):
        x = x.transpose(1, 2)                  # (B, D, T)
        x = self.blocks(x)
        return self.head(x[:, :, -1])          # last time step


class TransformerEncoderNet(nn.Module):
    def __init__(self, input_dim: int, seq_len: int, d_model: int,
                 nhead: int, num_layers: int, horizon: int, dropout: float = 0.1):
        super().__init__()
        self.input_proj = nn.Linear(input_dim, d_model)
        self.pos = nn.Parameter(torch.randn(1, seq_len, d_model) * 0.02)
        layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=nhead, dim_feedforward=d_model * 4,
            dropout=dropout, batch_first=True, norm_first=True, activation="gelu",
        )
        self.encoder = nn.TransformerEncoder(layer, num_layers=num_layers)
        self.head = nn.Linear(d_model, horizon)

    def forward(self, x):
        x = self.input_proj(x) + self.pos[:, :x.size(1)]
        x = self.encoder(x)
        return self.head(x[:, -1])


# ============================================================================
# Shared training loop
# ============================================================================

def _train_and_predict(
    build_model: Callable[[int], nn.Module],   # build_model(input_dim) -> nn.Module
    out_fit: np.ndarray,
    weather_fit: np.ndarray,
    timestamps_fit: pd.DatetimeIndex,
    horizon: int,
    seed: int,
    epochs: int = config.DEFAULT_EPOCHS,
    batch_size: int = config.DEFAULT_BATCH_SIZE,
    lr: float = config.DEFAULT_LR,
    seq_len: int = config.SEQ_LEN,
    weight_decay: float = 1e-5,
    use_amp: bool = True,
) -> tuple[np.ndarray, dict]:
    """Shared training + inference routine for all NN models."""
    set_seed(seed)
    device = config.get_device()

    # --- Scalers + windows ---
    scalers = fit_scalers(out_fit, weather_fit)
    X_all, Y_all, X_last = build_nn_windows(
        out=out_fit, weather=weather_fit, scalers=scalers,
        seq_len=seq_len, horizon=horizon,
        include_calendar=True, timestamps=timestamps_fit,
    )
    input_dim = X_all.shape[-1]

    # Inner val split: last ~48h of origin times (per location) for early stopping.
    # Our window order is origin-major (i*L + l), so the last block of size L is the
    # most recent origin. We reserve the last 48 origin-blocks as inner val.
    T = out_fit.shape[0]
    L = out_fit.shape[1]
    N_origins = T - seq_len - horizon + 1
    inner_val_origins = min(48, max(1, N_origins // 10))
    n_val = inner_val_origins * L
    n_tr = X_all.shape[0] - n_val
    X_tr, Y_tr = X_all[:n_tr], Y_all[:n_tr]
    X_val, Y_val = X_all[n_tr:], Y_all[n_tr:]

    # --- Dataloaders ---
    tr_ds = TensorDataset(torch.from_numpy(X_tr), torch.from_numpy(Y_tr))
    tr_dl = DataLoader(tr_ds, batch_size=batch_size, shuffle=True,
                       num_workers=2, pin_memory=(device == "cuda"))
    X_val_t = torch.from_numpy(X_val).to(device, non_blocking=True)
    Y_val_t = torch.from_numpy(Y_val).to(device, non_blocking=True)

    # --- Model + optimizer ---
    model = build_model(input_dim).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)
    crit = nn.MSELoss()
    amp_on = bool(use_amp and device == "cuda")
    scaler_amp = _make_scaler(amp_on)

    best_val = float("inf")
    best_state = None
    patience_left = config.EARLY_STOP_PATIENCE

    for ep in range(epochs):
        model.train()
        for xb, yb in tr_dl:
            xb = xb.to(device, non_blocking=True)
            yb = yb.to(device, non_blocking=True)
            opt.zero_grad(set_to_none=True)
            with _autocast(amp_on):
                pred = model(xb)
                loss = crit(pred, yb)
            scaler_amp.scale(loss).backward()
            scaler_amp.unscale_(opt)
            nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            scaler_amp.step(opt)
            scaler_amp.update()
        scheduler.step()

        model.eval()
        with torch.no_grad():
            with _autocast(amp_on):
                val_pred = model(X_val_t)
                val_loss = crit(val_pred, Y_val_t).item()

        if val_loss < best_val - 1e-6:
            best_val = val_loss
            best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
            patience_left = config.EARLY_STOP_PATIENCE
        else:
            patience_left -= 1
            if patience_left <= 0:
                break

    if best_state is not None:
        model.load_state_dict(best_state)

    # --- Inference: one forward per location, using X_last ---
    model.eval()
    X_inf = torch.from_numpy(X_last).to(device)
    with torch.no_grad():
        with _autocast(amp_on):
            pred_scaled = model(X_inf).cpu().float().numpy()  # (L, horizon)
    pred = scalers.inv_y(pred_scaled)                         # unscale
    pred = np.clip(pred, 0.0, None).astype(np.float32)
    preds_T_first = pred.T                                    # (horizon, L)
    meta = {
        "epochs_run": ep + 1,
        "best_inner_val_loss": round(float(best_val), 6),
        "input_dim": int(input_dim),
        "seq_len": int(seq_len),
    }
    return preds_T_first, meta


# ============================================================================
# Registered models
# ============================================================================

@register("mlp", tier="neural", stochastic=True,
          description="Flat MLP over flattened (seq_len × D) window.")
def mlp_model(out_fit, weather_fit, timestamps_fit, locations, horizon, seed):
    def build(input_dim):
        return MLPNet(
            input_dim=input_dim, seq_len=config.SEQ_LEN,
            hidden_dim=config.DEFAULT_HIDDEN, horizon=horizon,
            dropout=config.DEFAULT_DROPOUT,
        )
    preds, meta = _train_and_predict(build, out_fit, weather_fit, timestamps_fit, horizon, seed)
    meta.update({"hidden_dim": config.DEFAULT_HIDDEN})
    return preds, meta


@register("lstm", tier="neural", stochastic=True,
          description="LSTM encoder + linear head (demo's model, deeper).")
def lstm_model(out_fit, weather_fit, timestamps_fit, locations, horizon, seed):
    def build(input_dim):
        return LSTMNet(
            input_dim=input_dim, hidden_dim=config.DEFAULT_HIDDEN,
            num_layers=config.DEFAULT_NUM_LAYERS, horizon=horizon,
            bidirectional=False, dropout=config.DEFAULT_DROPOUT,
        )
    preds, meta = _train_and_predict(build, out_fit, weather_fit, timestamps_fit, horizon, seed)
    meta.update({"hidden_dim": config.DEFAULT_HIDDEN, "num_layers": config.DEFAULT_NUM_LAYERS})
    return preds, meta


@register("gru", tier="neural", stochastic=True,
          description="GRU encoder + linear head.")
def gru_model(out_fit, weather_fit, timestamps_fit, locations, horizon, seed):
    def build(input_dim):
        return GRUNet(
            input_dim=input_dim, hidden_dim=config.DEFAULT_HIDDEN,
            num_layers=config.DEFAULT_NUM_LAYERS, horizon=horizon,
            dropout=config.DEFAULT_DROPOUT,
        )
    return _train_and_predict(build, out_fit, weather_fit, timestamps_fit, horizon, seed)


@register("bilstm", tier="neural", stochastic=True,
          description="Bidirectional LSTM — looks both ways within the history window.")
def bilstm_model(out_fit, weather_fit, timestamps_fit, locations, horizon, seed):
    def build(input_dim):
        return LSTMNet(
            input_dim=input_dim, hidden_dim=config.DEFAULT_HIDDEN,
            num_layers=config.DEFAULT_NUM_LAYERS, horizon=horizon,
            bidirectional=True, dropout=config.DEFAULT_DROPOUT,
        )
    preds, meta = _train_and_predict(build, out_fit, weather_fit, timestamps_fit, horizon, seed)
    meta.update({"bidirectional": True})
    return preds, meta


@register("tcn", tier="neural", stochastic=True,
          description="Dilated causal 1D-CNN (TCN, Bai et al. 2018).")
def tcn_model(out_fit, weather_fit, timestamps_fit, locations, horizon, seed):
    def build(input_dim):
        return TCNNet(
            input_dim=input_dim, hidden_dim=config.DEFAULT_HIDDEN,
            horizon=horizon, num_blocks=4, kernel=3,
            dropout=config.DEFAULT_DROPOUT,
        )
    preds, meta = _train_and_predict(build, out_fit, weather_fit, timestamps_fit, horizon, seed)
    meta.update({"num_blocks": 4, "kernel": 3})
    return preds, meta


@register("transformer", tier="neural", stochastic=True,
          description="Vanilla Transformer encoder + last-token head.")
def transformer_model(out_fit, weather_fit, timestamps_fit, locations, horizon, seed):
    def build(input_dim):
        d_model = config.DEFAULT_HIDDEN
        return TransformerEncoderNet(
            input_dim=input_dim, seq_len=config.SEQ_LEN,
            d_model=d_model, nhead=8,
            num_layers=config.DEFAULT_NUM_LAYERS, horizon=horizon,
            dropout=config.DEFAULT_DROPOUT,
        )
    preds, meta = _train_and_predict(build, out_fit, weather_fit, timestamps_fit, horizon, seed)
    meta.update({"d_model": config.DEFAULT_HIDDEN, "nhead": 8})
    return preds, meta
