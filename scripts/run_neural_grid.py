"""Neural-net grid search on H200 (deep models, transformer family, SOTA).

Sweeps:
  archs:    GRU, LSTM, BiLSTM, TCN, MLP, Transformer, NLinear, DLinear,
            PatchTST, iTransformer
  hparams:  hidden_dim, num_layers, dropout, lr, seq_len  (8 combinations)

Total runs (default): 10 archs × 8 hparams × 6 folds × 2 horizons × 2 seeds ≈ 1920
On H200, ~30-60s per run → ~24-32 GPU-hours total. With 4-hour SLURM jobs we
schedule a *few* in parallel, but a single 06:00 job will get a representative
portion done. Resume support means re-submitting just continues.

    python -m scripts.run_neural_grid
"""
from __future__ import annotations

import numpy as np

from src import config
from src.registry import register, MODEL_REGISTRY
from src.runner import run_all
from src.training import summarize_runs
from src import models_neural as MN     # registers basic NN archs as side effect
from src import models_sota as MS       # registers nlinear/dlinear/patchtst/itransformer
from src import models_advanced as MA   # registers nbeats / nhits / auto_arima / quantile-LGB


# ----------------------------------------------------------------------------
# Hyperparameter grid
# ----------------------------------------------------------------------------

HPARAM_GRID = [
    dict(hidden=64,  layers=1, drop=0.1,  lr=1e-3, sl=48),
    dict(hidden=128, layers=1, drop=0.1,  lr=1e-3, sl=48),
    dict(hidden=128, layers=2, drop=0.1,  lr=1e-3, sl=48),
    dict(hidden=256, layers=2, drop=0.1,  lr=1e-3, sl=48),
    dict(hidden=128, layers=2, drop=0.2,  lr=5e-4, sl=72),
    dict(hidden=128, layers=2, drop=0.0,  lr=2e-3, sl=48),
    dict(hidden=128, layers=3, drop=0.1,  lr=1e-3, sl=48),
    dict(hidden=192, layers=2, drop=0.15, lr=1e-3, sl=96),
]

# Architecture catalogue. Each entry maps name -> nn.Module factory taking
# (input_dim, horizon, hp).
def _build(arch: str, input_dim: int, horizon: int, hp: dict):
    if arch == "gru":
        return MN.GRUNet(input_dim, hp["hidden"], hp["layers"], horizon, hp["drop"])
    if arch == "lstm":
        return MN.LSTMNet(input_dim, hp["hidden"], hp["layers"], horizon, False, hp["drop"])
    if arch == "bilstm":
        return MN.LSTMNet(input_dim, hp["hidden"], hp["layers"], horizon, True, hp["drop"])
    if arch == "tcn":
        return MN.TCNNet(input_dim, hp["hidden"], horizon, hp["layers"] + 2, 3, hp["drop"])
    if arch == "mlp":
        return MN.MLPNet(input_dim, hp["sl"], hp["hidden"], horizon, hp["drop"])
    if arch == "transformer":
        # nhead must divide d_model
        nhead = 8 if hp["hidden"] % 8 == 0 else 4
        return MN.TransformerEncoderNet(input_dim, hp["sl"], hp["hidden"], nhead,
                                         hp["layers"], horizon, hp["drop"])
    if arch == "nlinear":
        return MS.NLinearNet(input_dim, hp["sl"], horizon)
    if arch == "dlinear":
        return MS.DLinearNet(input_dim, hp["sl"], horizon, ma_kernel=25)
    if arch == "patchtst":
        nhead = 8 if hp["hidden"] % 8 == 0 else 4
        return MS.PatchTSTNet(input_dim, hp["sl"], horizon,
                               patch_len=16, stride=8,
                               d_model=hp["hidden"], nhead=nhead,
                               num_layers=hp["layers"], dropout=hp["drop"])
    if arch == "itransformer":
        nhead = 8 if hp["hidden"] % 8 == 0 else 4
        return MS.ITransformerNet(input_dim, hp["sl"], horizon,
                                    d_model=hp["hidden"], nhead=nhead,
                                    num_layers=hp["layers"], dropout=hp["drop"])
    if arch == "nbeats":
        return MA.NBeatsNet(input_dim, hp["sl"], horizon,
                            n_blocks=max(2, hp["layers"] + 2),
                            hidden=hp["hidden"], n_layers=4, dropout=hp["drop"])
    if arch == "nhits":
        return MA.NHiTSNet(input_dim, hp["sl"], horizon,
                           n_blocks=max(2, hp["layers"] + 2),
                           hidden=hp["hidden"], n_layers=2, dropout=hp["drop"])
    raise KeyError(arch)


ARCHS = ["gru", "lstm", "bilstm", "tcn", "mlp", "transformer",
         "nlinear", "dlinear", "patchtst", "itransformer",
         "nbeats", "nhits"]


def _short_name(arch: str, hp: dict) -> str:
    return (
        f"{arch}__h{hp['hidden']}_l{hp['layers']}_d{int(hp['drop']*100):02d}"
        f"_lr{int(hp['lr']*1e4):04d}_sl{hp['sl']}"
    )


def _make_fit_predict(arch: str, hp: dict):
    """Returns a fit_predict closure that the runner can call."""
    def fit_predict(out_fit, weather_fit, timestamps_fit, locations, horizon, seed):
        def build(input_dim, _arch=arch, _hp=hp, _h=horizon):
            return _build(_arch, input_dim, _h, _hp)
        preds, meta = MN._train_and_predict(
            build, out_fit, weather_fit, timestamps_fit, horizon, seed,
            seq_len=hp["sl"], lr=hp["lr"],
        )
        meta.update(hp)
        meta["arch"] = arch
        return preds, meta
    return fit_predict


def register_grid() -> list[str]:
    names = []
    tier = "neural_grid"
    for arch in ARCHS:
        for hp in HPARAM_GRID:
            nm = _short_name(arch, hp)
            if nm in MODEL_REGISTRY:
                names.append(nm); continue
            register(nm, tier=tier, stochastic=True,
                     description=str(hp))(_make_fit_predict(arch, hp))
            names.append(nm)
    return names


def main():
    names = register_grid()
    print(f"[NeuralGrid] registered {len(names)} models")
    # 2 seeds for stability
    run_all(model_names=names, seeds=[42, 43], skip_on_error=True)
    print("\n========= Top 20 by mean RMSE =========")
    print(summarize_runs().head(20).to_string(index=False))


if __name__ == "__main__":
    main()
