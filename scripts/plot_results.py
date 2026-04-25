"""Generate report-ready plots from results/experiment_runs.csv.

Uses MEDIAN (or per-fold breakdown) — never mean — for ranking, because
fold 1 is a 700-900 RMSE outlier that masks all model differences in mean.

    python -m scripts.plot_results
"""
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src import config
from src.training import load_runs, summarize_runs


def _family_of(name: str) -> str:
    return name.split("__", 1)[0]


def main():
    fig_dir = config.FIGURES_DIR
    asset_dir = Path(config.PROJECT_ROOT / "report_assets")

    df = load_runs()
    if df.empty:
        print("No runs yet."); return
    df = df[df["fold"] >= 0].copy()  # drop legacy fold=-1

    summary = summarize_runs()
    summary.to_csv(asset_dir / "cv_summary.csv", index=False)
    print(f"CV summary saved (n={len(summary)} model-horizon combos)")

    # ====================================================================
    # 1. The headline figure: PER-FAMILY × per-fold heatmap.
    #    Replaces the prior 15-model heatmap which had no visible variance.
    #    Each row = one model family (best config), each column = one fold.
    # ====================================================================
    for h in [24, 48]:
        sub = summary[(summary["horizon"] == h) & (summary["n"] >= 4)].copy()
        if sub.empty: continue
        sub = sub[~sub["model"].str.startswith(("ridge_lag", "linreg_lag"))]
        sub["family"] = sub["model"].apply(_family_of)
        # Best config per family by calm_mean
        best_per_fam = sub.loc[sub.groupby("family")["calm_mean"].idxmin()].copy()
        best_per_fam = best_per_fam.sort_values("calm_mean")

        df_h = df[df["horizon"] == h]
        # Build per-(family, fold) matrix using each family's best model
        m = df_h.groupby(["model", "fold"])["val_rmse"].mean().unstack()
        m_fam = m.loc[best_per_fam["model"].tolist()]
        m_fam.index = best_per_fam["family"].tolist()
        # Drop legacy fold=-1
        if -1 in m_fam.columns:
            m_fam = m_fam.drop(columns=[-1])

        fig, axes = plt.subplots(1, 2, figsize=(12, 0.35 * len(m_fam) + 1.5),
                                  sharey=True, gridspec_kw={"width_ratios": [3, 1]})

        # LEFT: 5 calm folds. Color scale tuned to show family-level differences:
        # GBDT cluster ~40-45, NNs ~50-110, baselines ~50-80.
        calm_folds = [c for c in m_fam.columns if c != 1]
        calm_order = sorted(calm_folds, reverse=True)
        m_calm = m_fam[calm_order]

        cmap = plt.get_cmap("RdYlGn_r")
        # Color range: 15 (best fold-4) to 250 (worst NLinear fold-0). Log-ish.
        norm = mcolors.SymLogNorm(linthresh=30, vmin=15,
                                    vmax=float(np.nanmax(m_calm.values)))
        axes[0].imshow(m_calm.values, aspect="auto", cmap=cmap, norm=norm)
        axes[0].set_xticks(range(len(calm_order)))
        axes[0].set_xticklabels([f"f{c}\n(calm)" for c in calm_order], fontsize=8)
        axes[0].set_yticks(range(len(m_fam)))
        axes[0].set_yticklabels(m_fam.index, fontsize=9)
        for i in range(m_calm.shape[0]):
            for j in range(m_calm.shape[1]):
                v = m_calm.values[i, j]
                if np.isfinite(v):
                    fontcolor = "white" if v > 100 else "black"
                    axes[0].text(j, i, f"{v:.0f}", ha="center", va="center",
                                  fontsize=7, color=fontcolor)
        axes[0].set_title(f"Calm folds — model family differences are visible "
                            f"({h}h horizon)")

        # RIGHT: storm fold separately
        if 1 in m_fam.columns:
            m_storm = m_fam[[1]]
            norm_storm = mcolors.Normalize(vmin=550, vmax=1000)
            axes[1].imshow(m_storm.values, aspect="auto", cmap=cmap, norm=norm_storm)
            axes[1].set_xticks([0])
            axes[1].set_xticklabels(["f1\n(storm)"], fontsize=8)
            for i in range(m_storm.shape[0]):
                v = m_storm.values[i, 0]
                if np.isfinite(v):
                    axes[1].text(0, i, f"{v:.0f}", ha="center", va="center",
                                  fontsize=7, color="white")
            axes[1].set_title("Storm fold\n(separate scale)")

        fig.suptitle(f"Per-family × per-fold RMSE — best config per family on {h}h horizon",
                      y=1.02, fontsize=10)
        fig.tight_layout()
        out = fig_dir / f"cv_family_fold_h{h}.png"
        fig.savefig(out, dpi=130, bbox_inches="tight")
        plt.close(fig)
        print(f"saved {out}")
        # Remove old file if present
        old_path = fig_dir / f"cv_per_fold_h{h}.png"
        if old_path.exists():
            old_path.unlink()
            print(f"removed {old_path}")

    # ====================================================================
    # 2. Best per family — TWO panels side by side: calm vs storm
    # ====================================================================
    for h in [24, 48]:
        sub = summary[(summary["horizon"] == h) & (summary["n"] >= 4)].copy()
        if sub.empty: continue
        # ridge_lag is technically valid but the linear model with log target
        # extrapolates wildly on storm-adjacent fold 0 (RMSE ~1200) — its
        # calm_mean of ~270 dominates the x-axis and obscures the GBDT cluster.
        # Same reason linreg_lag was excluded earlier.
        sub = sub[~sub["model"].str.startswith(("ridge_lag", "linreg_lag"))]
        sub["family"] = sub["model"].apply(_family_of)
        best_per_fam = sub.loc[sub.groupby("family")["val_rmse_median"].idxmin()].copy()
        best_per_fam = best_per_fam.sort_values("calm_mean")

        fig, axes = plt.subplots(1, 2, figsize=(13, 0.4 * len(best_per_fam) + 1.4),
                                  sharey=True)
        y = np.arange(len(best_per_fam))
        labels = best_per_fam["family"].tolist()

        # Left: calm-period RMSE — meaningful comparison
        bars_left = axes[0].barh(y, best_per_fam["calm_mean"], color="steelblue",
                                   edgecolor="black", linewidth=0.5)
        for i, v in enumerate(best_per_fam["calm_mean"]):
            axes[0].text(v + 0.5, i, f"{v:.1f}", va="center", fontsize=8)
        axes[0].set_yticks(y); axes[0].set_yticklabels(labels, fontsize=9)
        axes[0].invert_yaxis()
        axes[0].set_xlabel("Calm-period RMSE\n(mean over 5 non-storm folds, lower=better)")
        axes[0].set_title(f"Typical-day performance — {h}h")
        axes[0].grid(axis="x", alpha=0.3)
        # Highlight the best
        bars_left[0].set_color("forestgreen")

        # Right: storm fold RMSE (everyone fails)
        axes[1].barh(y, best_per_fam["storm_rmse"], color="firebrick",
                      edgecolor="black", linewidth=0.5)
        for i, v in enumerate(best_per_fam["storm_rmse"]):
            axes[1].text(v + 5, i, f"{v:.0f}", va="center", fontsize=8)
        axes[1].set_xlabel("Storm-fold RMSE\n(fold 1, Jun 25-27 peak 86k outages)")
        axes[1].set_title(f"Out-of-distribution storm — {h}h")
        axes[1].grid(axis="x", alpha=0.3)

        fig.suptitle(f"Best configuration per model family — {h}h horizon", y=1.02, fontsize=11)
        fig.tight_layout()
        out = fig_dir / f"cv_per_family_h{h}.png"
        fig.savefig(out, dpi=130, bbox_inches="tight")
        plt.close(fig)
        print(f"saved {out}")

    # ====================================================================
    # 3. Top-30 ranking by median (compact dot plot)
    # ====================================================================
    for h in [24, 48]:
        sub = summary[(summary["horizon"] == h) & (summary["n"] >= 4)].copy()
        if sub.empty: continue
        sub = sub.sort_values("val_rmse_median").head(30)

        families = sub["model"].apply(_family_of)
        unique_fams = list(dict.fromkeys(families.tolist()))
        cmap = plt.get_cmap("tab20")
        fam_color = {f: cmap(i % 20) for i, f in enumerate(unique_fams)}
        colors = [fam_color[f] for f in families]

        fig, ax = plt.subplots(figsize=(10, 0.32 * len(sub) + 1.2))
        y = np.arange(len(sub))
        # Plot calm-period RMSE — the meaningful number
        ax.scatter(sub["calm_mean"], y, c=colors, s=60, zorder=3,
                    edgecolors="black", linewidths=0.4)
        ax.set_yticks(y); ax.set_yticklabels(sub["model"].tolist(), fontsize=8)
        ax.invert_yaxis()
        ax.set_xlabel(f"Calm-period RMSE (5-fold mean excl. storm fold)")
        ax.set_title(f"Top-30 models — {h}h, ranked by MEDIAN RMSE\n"
                      f"(calm-period mean is the meaningful comparison; "
                      f"all models fail similarly on storm fold)")
        ax.grid(axis="x", alpha=0.3)
        from matplotlib.patches import Patch
        handles = [Patch(facecolor=fam_color[f], label=f) for f in unique_fams[:12]]
        ax.legend(handles=handles, loc="lower right", fontsize=7, ncol=2,
                   title="model family", title_fontsize=8)
        fig.tight_layout()
        out = fig_dir / f"cv_results_h{h}.png"
        fig.savefig(out, dpi=130)
        plt.close(fig)
        print(f"saved {out}")


if __name__ == "__main__":
    main()
