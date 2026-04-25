"""Generate hyperparameter-sensitivity plots from the grid search results.

For each model family (lgb, xgb, cat, gru, transformer, patchtst, ...), parses
the model name (e.g. lgb__nl63_lr05_n800_ss7) into hparams and plots RMSE vs
each hparam separately.

Output: results/figures/hparam_sensitivity_<family>_h<H>.png
"""
from __future__ import annotations

import re
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src import config
from src.training import summarize_runs


# Per-family parser: model_name -> dict of hparams
def _parse_lgb(nm: str) -> dict | None:
    m = re.match(r"lgb__nl(\d+)_lr(\d+)_n(\d+)_ss(\d+)", nm)
    if not m: return None
    nl, lr, n, ss = m.groups()
    return dict(num_leaves=int(nl), learning_rate=int(lr) / 100,
                n_estimators=int(n), subsample=int(ss) / 10)


def _parse_xgb(nm: str) -> dict | None:
    m = re.match(r"xgb__d(\d+)_lr(\d+)_n(\d+)", nm)
    if not m: return None
    return dict(max_depth=int(m.group(1)), learning_rate=int(m.group(2)) / 100,
                n_estimators=int(m.group(3)))


def _parse_cat(nm: str) -> dict | None:
    m = re.match(r"cat__d(\d+)_lr(\d+)_n(\d+)", nm)
    if not m: return None
    return dict(depth=int(m.group(1)), learning_rate=int(m.group(2)) / 100,
                iterations=int(m.group(3)))


def _parse_lgbtw(nm: str) -> dict | None:
    m = re.match(r"lgbtw__nl(\d+)_lr(\d+)_n(\d+)_vp(\d+)", nm)
    if not m: return None
    nl, lr, n, vp = m.groups()
    return dict(num_leaves=int(nl), learning_rate=int(lr) / 100,
                n_estimators=int(n), variance_power=int(vp) / 10)


def _parse_nn(nm: str, family: str) -> dict | None:
    m = re.match(rf"{family}__h(\d+)_l(\d+)_d(\d+)_lr(\d+)_sl(\d+)", nm)
    if not m: return None
    h, l, d, lr, sl = m.groups()
    return dict(hidden=int(h), layers=int(l), dropout=int(d) / 100,
                lr=int(lr) / 1e4, seq_len=int(sl))


PARSERS = {
    "lgb": _parse_lgb,
    "xgb": _parse_xgb,
    "cat": _parse_cat,
    "lgbtw": _parse_lgbtw,
}
NN_FAMILIES = ["gru", "lstm", "bilstm", "tcn", "mlp", "transformer",
               "nlinear", "dlinear", "patchtst", "itransformer", "nbeats", "nhits"]


def main():
    fig_dir = config.FIGURES_DIR

    summary = summarize_runs()
    if summary.empty:
        print("No runs to plot.")
        return

    # Process each family
    for family, parser in {**PARSERS, **{f: lambda nm, f=f: _parse_nn(nm, f) for f in NN_FAMILIES}}.items():
        sub = summary[summary["model"].str.startswith(f"{family}__")].copy()
        if sub.empty:
            continue

        # Parse hparams from the name
        records = []
        for _, r in sub.iterrows():
            hp = parser(r["model"])
            if hp is None:
                continue
            records.append({**hp, "horizon": r["horizon"],
                             "rmse_mean": r["val_rmse_mean"],
                             "rmse_std": r["val_rmse_std"],
                             "model": r["model"]})
        if not records:
            continue
        df = pd.DataFrame(records)

        # Plot per horizon
        for h in [24, 48]:
            dh = df[df["horizon"] == h]
            if dh.empty:
                continue
            hparams = [c for c in dh.columns
                       if c not in ("horizon", "rmse_mean", "rmse_std", "model")]
            ncol = min(4, len(hparams))
            nrow = (len(hparams) + ncol - 1) // ncol
            fig, axes = plt.subplots(nrow, ncol, figsize=(4 * ncol, 3 * nrow), squeeze=False)
            for ax, hp in zip(axes.flat, hparams):
                # Box plot or scatter — use scatter+jitter for ordinal hparams
                vals = dh[hp].values
                rmses = dh["rmse_mean"].values
                ax.scatter(vals, rmses, alpha=0.5, s=20, color="steelblue")
                # Overlay mean per unique value
                grouped = dh.groupby(hp)["rmse_mean"].agg(["mean", "std"]).reset_index()
                ax.errorbar(grouped[hp], grouped["mean"], yerr=grouped["std"],
                             fmt="o-", color="firebrick", capsize=3, lw=1.5,
                             label="mean over rest of grid")
                ax.set_xlabel(hp)
                ax.set_ylabel("CV RMSE (mean across folds)")
                ax.set_title(f"{family} h={h}: sensitivity to {hp}")
                ax.grid(alpha=0.3)
            for ax in axes.flat[len(hparams):]:
                ax.axis("off")
            fig.tight_layout()
            out = fig_dir / f"hparam_sensitivity_{family}_h{h}.png"
            fig.savefig(out, dpi=110)
            plt.close(fig)
            print(f"saved {out}")


if __name__ == "__main__":
    main()
