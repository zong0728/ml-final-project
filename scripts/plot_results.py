"""Generate report-ready plots from results/experiment_runs.csv.

    python -m scripts.plot_results
"""
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src import config
from src.training import load_runs, summarize_runs


def main():
    fig_dir = config.FIGURES_DIR
    asset_dir = Path(config.PROJECT_ROOT / "report_assets")

    df = load_runs()
    if df.empty:
        print("No runs yet."); return

    summary = summarize_runs()
    summary.to_csv(asset_dir / "cv_summary.csv", index=False)
    print("CV summary saved to report_assets/cv_summary.csv")
    print(summary.to_string(index=False))

    # ---- Bar plot per horizon ----
    for h in [24, 48]:
        sub = summary[summary["horizon"] == h].copy()
        if sub.empty: continue
        sub = sub.sort_values("val_rmse_mean")
        fig, ax = plt.subplots(figsize=(11, 0.35 * len(sub) + 1))
        y = np.arange(len(sub))
        ax.errorbar(sub["val_rmse_mean"], y, xerr=sub["val_rmse_std"],
                    fmt="o", color="steelblue", ecolor="gray", capsize=3)
        ax.set_yticks(y); ax.set_yticklabels(sub["model"].tolist())
        ax.invert_yaxis()
        ax.set_xlabel(f"{h}h CV RMSE (mean ± std across folds)")
        ax.set_title(f"Per-model {h}-hour rolling-origin CV RMSE")
        ax.grid(axis="x", alpha=0.3)
        fig.tight_layout()
        out = fig_dir / f"cv_results_h{h}.png"
        fig.savefig(out, dpi=120)
        print(f"saved {out}")

    # ---- Per-fold heatmap (model × fold), one per horizon ----
    for h in [24, 48]:
        sub = df[df["horizon"] == h].copy()
        if sub.empty: continue
        # Average over seeds within (model, fold)
        m = sub.groupby(["model", "fold"])["val_rmse"].mean().unstack(fill_value=np.nan)
        # Sort rows by mean RMSE
        m = m.assign(_mean=m.mean(axis=1)).sort_values("_mean").drop(columns="_mean")
        fig, ax = plt.subplots(figsize=(8, 0.3 * len(m) + 1.2))
        im = ax.imshow(m.values, aspect="auto", cmap="RdYlGn_r")
        ax.set_yticks(range(len(m))); ax.set_yticklabels(m.index)
        ax.set_xticks(range(m.shape[1])); ax.set_xticklabels([f"f{c}" for c in m.columns])
        for i in range(m.shape[0]):
            for j in range(m.shape[1]):
                v = m.values[i, j]
                if np.isfinite(v):
                    ax.text(j, i, f"{v:.0f}", ha="center", va="center", fontsize=7,
                            color="white" if v > np.nanmedian(m.values) else "black")
        ax.set_title(f"{h}-h RMSE per (model, fold) — fold 1 ≈ big-storm, others ≈ medium")
        plt.colorbar(im, ax=ax, label="RMSE")
        fig.tight_layout()
        out = fig_dir / f"cv_heatmap_h{h}.png"
        fig.savefig(out, dpi=120)
        print(f"saved {out}")


if __name__ == "__main__":
    main()
