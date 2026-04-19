"""Popular recent architectures for time-series forecasting.

All use the same window-based fit/predict pipeline as neural models. Implementations
are intentionally compact — faithful to the core ideas of the papers, not the
feature-complete public libraries.

References:
  * NLinear / DLinear ... Zeng et al. 2022 "Are Transformers Effective for Time
    Series Forecasting?" (AAAI 2023).
  * PatchTST ............ Nie et al. 2023 "A Time Series is Worth 64 Words" (ICLR 2023).
  * iTransformer ......... Liu et al. 2024 "iTransformer: Inverted Transformers"
    (ICLR 2024).
"""
from __future__ import annotations

import math

import numpy as np
import pandas as pd
import torch
import torch.nn as nn

from . import config
from .models_neural import _train_and_predict
from .registry import register


# ============================================================================
# DLinear / NLinear (Zeng et al. 2022) — channel-mixing variants.
# Our input has D channels (1 outage + F weather + 6 calendar). Output is the
# single outage channel over `horizon` steps.
# ============================================================================

class NLinearNet(nn.Module):
    """Subtract last-step value, apply Linear, add it back. Predicts outage only."""

    def __init__(self, input_dim: int, seq_len: int, horizon: int):
        super().__init__()
        self.input_dim = input_dim
        self.seq_len = seq_len
        self.horizon = horizon
        self.linear = nn.Linear(seq_len * input_dim, horizon)

    def forward(self, x):
        # x: (B, T, D). Outage is channel 0.
        last_outage = x[:, -1:, 0]                  # (B, 1)
        x_centered = x.clone()
        x_centered[:, :, 0] = x_centered[:, :, 0] - last_outage
        flat = x_centered.reshape(x.size(0), -1)
        out = self.linear(flat)
        return out + last_outage                    # (B, horizon)


class MovingAvg(nn.Module):
    """Average-pool with reflect padding — used for DLinear trend/seasonal split."""
    def __init__(self, kernel_size: int):
        super().__init__()
        self.kernel_size = kernel_size
        self.avg = nn.AvgPool1d(kernel_size=kernel_size, stride=1, padding=0)

    def forward(self, x):
        # x: (B, T, C) -> pad ends, smooth -> (B, T, C)
        pad_left = (self.kernel_size - 1) // 2
        pad_right = self.kernel_size - 1 - pad_left
        front = x[:, :1, :].repeat(1, pad_left, 1)
        back = x[:, -1:, :].repeat(1, pad_right, 1)
        padded = torch.cat([front, x, back], dim=1)
        return self.avg(padded.transpose(1, 2)).transpose(1, 2)


class DLinearNet(nn.Module):
    """DLinear: trend + seasonal decomposition, two linear maps per component."""

    def __init__(self, input_dim: int, seq_len: int, horizon: int, ma_kernel: int = 25):
        super().__init__()
        self.input_dim = input_dim
        self.seq_len = seq_len
        self.horizon = horizon
        self.decomp = MovingAvg(kernel_size=ma_kernel if ma_kernel <= seq_len else seq_len)
        self.linear_trend = nn.Linear(seq_len * input_dim, horizon)
        self.linear_season = nn.Linear(seq_len * input_dim, horizon)

    def forward(self, x):
        trend = self.decomp(x)
        season = x - trend
        t_flat = trend.reshape(x.size(0), -1)
        s_flat = season.reshape(x.size(0), -1)
        return self.linear_trend(t_flat) + self.linear_season(s_flat)


# ============================================================================
# PatchTST (Nie et al. 2023) — channel-independent transformer over patches.
# We follow the multivariate channel-independent setup: every channel becomes
# its own sequence of patch tokens; the transformer shares weights across
# channels. At the end we read out only the outage channel (channel 0).
# ============================================================================

class PatchTSTNet(nn.Module):
    def __init__(
        self,
        input_dim: int,
        seq_len: int,
        horizon: int,
        patch_len: int = 16,
        stride: int = 8,
        d_model: int = 128,
        nhead: int = 8,
        num_layers: int = 2,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.input_dim = input_dim
        self.seq_len = seq_len
        self.horizon = horizon
        self.patch_len = min(patch_len, seq_len)
        self.stride = min(stride, self.patch_len)
        self.num_patches = max(1, (seq_len - self.patch_len) // self.stride + 1)

        self.patch_proj = nn.Linear(self.patch_len, d_model)
        self.pos = nn.Parameter(torch.randn(1, self.num_patches, d_model) * 0.02)

        enc_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=nhead, dim_feedforward=d_model * 4,
            dropout=dropout, batch_first=True, norm_first=True, activation="gelu",
        )
        self.encoder = nn.TransformerEncoder(enc_layer, num_layers=num_layers)
        self.head = nn.Linear(d_model * self.num_patches, horizon)

    def _patchify(self, x_channel):
        """x_channel: (B, T) -> (B, num_patches, patch_len)"""
        B, T = x_channel.shape
        P, S, N = self.patch_len, self.stride, self.num_patches
        idx = torch.arange(N, device=x_channel.device) * S
        patches = torch.stack([x_channel[:, i:i + P] for i in idx.tolist()], dim=1)
        return patches

    def forward(self, x):
        # x: (B, T, D) — channel-independent: loop over channels (weight-shared transformer)
        B, T, D = x.shape
        # Stack all channels along batch for shared-weight processing
        x_perm = x.permute(0, 2, 1).contiguous()              # (B, D, T)
        x_flat = x_perm.reshape(B * D, T)                      # (B*D, T)

        patches = self._patchify(x_flat)                       # (B*D, N, P)
        tokens = self.patch_proj(patches) + self.pos           # (B*D, N, d_model)
        enc = self.encoder(tokens)                             # (B*D, N, d_model)
        flat = enc.reshape(B * D, -1)
        per_channel = self.head(flat).reshape(B, D, self.horizon)  # (B, D, horizon)

        # Read out outage channel only
        return per_channel[:, 0, :]                            # (B, horizon)


# ============================================================================
# iTransformer (Liu et al. 2024) — invert the attention: each VARIATE becomes a
# token, attending across variates; the temporal sequence is embedded into each
# token. This flips the usual token axis and excels at multivariate forecasting.
# ============================================================================

class ITransformerNet(nn.Module):
    def __init__(
        self,
        input_dim: int,
        seq_len: int,
        horizon: int,
        d_model: int = 128,
        nhead: int = 8,
        num_layers: int = 2,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.input_dim = input_dim
        self.seq_len = seq_len
        self.horizon = horizon
        # Each variate (channel) → d_model embedding from its seq_len history
        self.var_embed = nn.Linear(seq_len, d_model)
        enc_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=nhead, dim_feedforward=d_model * 4,
            dropout=dropout, batch_first=True, norm_first=True, activation="gelu",
        )
        self.encoder = nn.TransformerEncoder(enc_layer, num_layers=num_layers)
        self.head = nn.Linear(d_model, horizon)

    def forward(self, x):
        # x: (B, T, D) — transpose to (B, D, T); each variate becomes one token.
        tokens = self.var_embed(x.permute(0, 2, 1))            # (B, D, d_model)
        enc = self.encoder(tokens)                             # (B, D, d_model)
        per_variate = self.head(enc)                           # (B, D, horizon)
        return per_variate[:, 0, :]                            # outage channel only


# ============================================================================
# Registered model functions
# ============================================================================

@register("nlinear", tier="sota", stochastic=True,
          description="NLinear (Zeng 2022): last-value normalize + linear.")
def nlinear_model(out_fit, weather_fit, timestamps_fit, locations, horizon, seed):
    def build(input_dim):
        return NLinearNet(input_dim=input_dim, seq_len=config.SEQ_LEN, horizon=horizon)
    preds, meta = _train_and_predict(build, out_fit, weather_fit, timestamps_fit, horizon, seed)
    return preds, meta


@register("dlinear", tier="sota", stochastic=True,
          description="DLinear (Zeng 2022): trend/seasonal decomposition + two linears.")
def dlinear_model(out_fit, weather_fit, timestamps_fit, locations, horizon, seed):
    def build(input_dim):
        return DLinearNet(input_dim=input_dim, seq_len=config.SEQ_LEN, horizon=horizon, ma_kernel=25)
    preds, meta = _train_and_predict(build, out_fit, weather_fit, timestamps_fit, horizon, seed)
    meta.update({"ma_kernel": 25})
    return preds, meta


@register("patchtst", tier="sota", stochastic=True,
          description="PatchTST (Nie 2023): channel-independent transformer over patches.")
def patchtst_model(out_fit, weather_fit, timestamps_fit, locations, horizon, seed):
    def build(input_dim):
        return PatchTSTNet(
            input_dim=input_dim, seq_len=config.SEQ_LEN, horizon=horizon,
            patch_len=16, stride=8,
            d_model=config.DEFAULT_HIDDEN, nhead=8,
            num_layers=config.DEFAULT_NUM_LAYERS, dropout=config.DEFAULT_DROPOUT,
        )
    preds, meta = _train_and_predict(build, out_fit, weather_fit, timestamps_fit, horizon, seed)
    meta.update({"patch_len": 16, "stride": 8, "d_model": config.DEFAULT_HIDDEN})
    return preds, meta


@register("itransformer", tier="sota", stochastic=True,
          description="iTransformer (Liu 2024): variates-as-tokens attention.")
def itransformer_model(out_fit, weather_fit, timestamps_fit, locations, horizon, seed):
    def build(input_dim):
        return ITransformerNet(
            input_dim=input_dim, seq_len=config.SEQ_LEN, horizon=horizon,
            d_model=config.DEFAULT_HIDDEN, nhead=8,
            num_layers=config.DEFAULT_NUM_LAYERS, dropout=config.DEFAULT_DROPOUT,
        )
    preds, meta = _train_and_predict(build, out_fit, weather_fit, timestamps_fit, horizon, seed)
    meta.update({"d_model": config.DEFAULT_HIDDEN})
    return preds, meta
