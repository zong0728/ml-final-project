"""Generate the full set of EDA figures + tables that the report needs.

Run from project root:
    python -m src.eda_full

Outputs (saved into results/figures/ and report_assets/):
    - statewide_outage.png         (already produced by eda_storms; refreshed)
    - outage_distribution.png      log-histogram of (county,hour) outage values
    - county_heterogeneity.png     per-county zero-rate vs. tracked households
    - outage_autocorr.png          state-wide autocorrelation up to 7 days
    - weather_corr_top.png         top weather features by abs corr w/ statewide outage
    - tracked_county_table.csv     county_fips, name (if available), tracked, max_outage
"""
from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import xarray as xr

from . import config


def main() -> None:
    ds = xr.open_dataset(config.TRAIN_PATH)
    out = ds.out.transpose("timestamp", "location").values.astype(np.float32)
    weather = ds.weather.transpose("timestamp", "location", "feature").values.astype(np.float32)
    locations = [str(x) for x in ds.location.values]
    features = [str(x) for x in ds.feature.values]
    timestamps = pd.to_datetime(ds.timestamp.values)

    # tracked is per-county per-hour. For the report, take the per-county *max*
    # across time as the proxy for households-served upper bound.
    tracked = ds.tracked.transpose("timestamp", "location").values.astype(np.float32)
    tracked_max = np.nanmax(tracked, axis=0)  # (L,)

    fig_dir = config.FIGURES_DIR
    asset_dir = Path(config.PROJECT_ROOT / "report_assets")
    asset_dir.mkdir(exist_ok=True)

    # --- 1. Outage distribution ---
    flat = out.ravel()
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    axes[0].hist(flat, bins=80, log=True, color="steelblue", alpha=0.85)
    axes[0].set_title(f"All (county, hour) outage values\nN = {len(flat):,}, "
                      f"zero share = {(flat==0).mean():.1%}")
    axes[0].set_xlabel("outage count")
    axes[0].set_ylabel("frequency (log)")
    nz = flat[flat > 0]
    axes[1].hist(np.log10(nz), bins=60, color="firebrick", alpha=0.85)
    axes[1].set_title(f"Non-zero outage counts (log10)\nN_nonzero = {len(nz):,}")
    axes[1].set_xlabel("log10(outage)")
    fig.tight_layout()
    fig.savefig(fig_dir / "outage_distribution.png", dpi=120)
    print(f"  saved {fig_dir/'outage_distribution.png'}")

    # --- 2. Per-county heterogeneity ---
    zero_rate = (out == 0).mean(axis=0)
    max_out = out.max(axis=0)
    fig, ax = plt.subplots(figsize=(8, 5))
    sc = ax.scatter(tracked_max, max_out, c=zero_rate, cmap="viridis_r", s=30, alpha=0.85)
    ax.set_xscale("log")
    ax.set_yscale("symlog")
    ax.set_xlabel("max tracked households per county (log)")
    ax.set_ylabel("max hourly outage count (symlog)")
    ax.set_title("Per-county heterogeneity: tracked size × peak outage × idle rate")
    cb = plt.colorbar(sc, ax=ax)
    cb.set_label("fraction of hours with 0 outage")
    fig.tight_layout()
    fig.savefig(fig_dir / "county_heterogeneity.png", dpi=120)
    print(f"  saved {fig_dir/'county_heterogeneity.png'}")

    # --- 3. State-wide autocorrelation up to 7 days ---
    state = out.sum(axis=1)
    state = state - state.mean()
    max_lag = 24 * 7
    var = (state * state).mean()
    acf = []
    for k in range(0, max_lag + 1):
        acf.append(float((state[:len(state) - k] * state[k:]).mean() / var) if var > 0 else 0.0)
    acf = np.asarray(acf)
    fig, ax = plt.subplots(figsize=(11, 3.5))
    ax.bar(np.arange(len(acf)), acf, width=1.0, color="steelblue")
    for k in [24, 48, 168]:
        ax.axvline(k, color="red", lw=0.8, ls="--", alpha=0.7)
        ax.text(k + 1, ax.get_ylim()[1] * 0.9, f"{k}h", color="red", fontsize=8)
    ax.set_xlabel("lag (hours)")
    ax.set_ylabel("autocorrelation")
    ax.set_title("State-wide outage autocorrelation (7 days)")
    fig.tight_layout()
    fig.savefig(fig_dir / "outage_autocorr.png", dpi=120)
    print(f"  saved {fig_dir/'outage_autocorr.png'}")

    # --- 4. Top correlated weather features (state-aggregated) ---
    state_norm = out.sum(axis=1)
    weather_state = weather.mean(axis=1)  # (T, F): mean across counties
    corrs = np.zeros(weather_state.shape[1])
    valid = np.zeros(weather_state.shape[1], dtype=bool)
    for fi in range(weather_state.shape[1]):
        col = weather_state[:, fi]
        if np.nanstd(col) < 1e-8:
            continue
        c = np.corrcoef(state_norm, col)[0, 1]
        if np.isfinite(c):
            corrs[fi] = c
            valid[fi] = True
    abs_c = np.abs(corrs)
    top_idx = np.argsort(-abs_c)[:15]
    fig, ax = plt.subplots(figsize=(8, 5))
    names = [features[i] for i in top_idx]
    vals = [corrs[i] for i in top_idx]
    ax.barh(range(len(names)), vals, color=["steelblue" if v > 0 else "firebrick" for v in vals])
    ax.set_yticks(range(len(names))); ax.set_yticklabels(names)
    ax.invert_yaxis()
    ax.axvline(0, color="black", lw=0.5)
    ax.set_xlabel("Pearson correlation (state-aggregated)")
    ax.set_title("Top 15 weather features by |corr| with state-wide outage")
    fig.tight_layout()
    fig.savefig(fig_dir / "weather_corr_top.png", dpi=120)
    print(f"  saved {fig_dir/'weather_corr_top.png'}")

    # --- 5. County summary table ---
    county_df = pd.DataFrame({
        "fips": locations,
        "tracked_max": tracked_max.astype(int),
        "max_hourly_outage": max_out.astype(int),
        "mean_hourly_outage": out.mean(axis=0).round(2),
        "zero_rate": zero_rate.round(3),
    }).sort_values("max_hourly_outage", ascending=False)
    county_df.to_csv(asset_dir / "county_summary.csv", index=False)
    print(f"  saved {asset_dir/'county_summary.csv'}")
    print()
    print("Top 10 counties by peak outage:")
    print(county_df.head(10).to_string(index=False))

    # --- 6. Save state-wide ts as csv for report tables ---
    state_df = pd.DataFrame({
        "timestamp": timestamps, "state_outage_total": state_norm.astype(int),
    })
    state_df.to_csv(asset_dir / "statewide_outage.csv", index=False)


if __name__ == "__main__":
    main()
