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
    # 1. The headline figure: PER-FOLD RMSE for top-K models (no mean!)
    #    Each model gets a row; each fold a column. Storm fold separated.
    # ====================================================================
    for h in [24, 48]:
        sub = summary[(summary["horizon"] == h) & (summary["n"] >= 4)].copy()
        if sub.empty: continue
        sub = sub.sort_values("val_rmse_median").head(15)
        top_models = sub["model"].tolist()

        # Get per-fold RMSE matrix for these models
        df_h = df[df["horizon"] == h]
        m = df_h.groupby(["model", "fold"])["val_rmse"].mean().unstack()
        m = m.loc[top_models]   # preserve top-K order

        fig, axes = plt.subplots(1, 2, figsize=(12, 0.4 * len(m) + 1.5),
                                  sharey=True, gridspec_kw={"width_ratios": [3, 1]})

        # LEFT: 5 calm folds (0,2,3,4,5) on a linear scale
        calm_folds = [c for c in m.columns if c != 1]
        m_calm = m[calm_folds]
        # Sort columns by date order (fold 5 = oldest, fold 0 = newest)
        calm_order = sorted(calm_folds, reverse=True)
        m_calm = m_calm[calm_order]

        cmap = plt.get_cmap("RdYlGn_r")
        norm = mcolors.Normalize(vmin=15, vmax=90)
        im_left = axes[0].imshow(m_calm.values, aspect="auto", cmap=cmap, norm=norm)
        axes[0].set_xticks(range(len(calm_order)))
        axes[0].set_xticklabels([f"f{c}\n(calm)" for c in calm_order], fontsize=8)
        axes[0].set_yticks(range(len(m)))
        axes[0].set_yticklabels(m.index, fontsize=8)
        for i in range(m_calm.shape[0]):
            for j in range(m_calm.shape[1]):
                v = m_calm.values[i, j]
                if np.isfinite(v):
                    axes[0].text(j, i, f"{v:.1f}", ha="center", va="center",
                                  fontsize=7, color="black")
        axes[0].set_title(f"5 calm folds — RMSE 15-90  ({h}h horizon)")

        # RIGHT: storm fold 1 separately, on its own scale
        m_storm = m[[1]]
        norm_storm = mcolors.Normalize(vmin=600, vmax=950)
        im_right = axes[1].imshow(m_storm.values, aspect="auto", cmap=cmap, norm=norm_storm)
        axes[1].set_xticks([0])
        axes[1].set_xticklabels(["f1\n(storm)"], fontsize=8)
        for i in range(m_storm.shape[0]):
            v = m_storm.values[i, 0]
            if np.isfinite(v):
                axes[1].text(0, i, f"{v:.0f}", ha="center", va="center",
                              fontsize=7, color="white")
        axes[1].set_title("Storm fold 1\n(separate scale)")

        fig.suptitle(f"Top-15 models on {h}h: per-fold RMSE breakdown\n"
                      f"Storm fold (Jun 25-27, peak 86k) is a regime ALL models fail in. "
                      f"Calm folds (RMSE 15-90) are the meaningful comparison.",
                      y=1.04, fontsize=11)
        fig.tight_layout()
        out = fig_dir / f"cv_per_fold_h{h}.png"
        fig.savefig(out, dpi=130, bbox_inches="tight")
        plt.close(fig)
        print(f"saved {out}")

    # ====================================================================
    # 2. Best per family — TWO panels side by side: calm vs storm
    # ====================================================================
    for h in [24, 48]:
        sub = summary[(summary["horizon"] == h) & (summary["n"] >= 4)].copy()
        if sub.empty: continue
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
