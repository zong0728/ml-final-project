"""EDA + storm-event detection on the Michigan outage dataset.

Run this once at the project root:
    python -m src.eda_storms

Outputs:
    results/figures/statewide_outage.png  — full state-wide hourly time series
    results/figures/storm_zoom.png        — zoomed view of detected storms
    results/storm_events.csv              — start/end/peak of each detected event
    stdout: time-span, basic stats, storm count

The "storm" detector uses a simple threshold-based rule on the state-wide
sum of outages and merges consecutive events. We can refine if needed.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from . import config


# Storm detection knobs (we'll tune after seeing the data).
STATE_OUTAGE_THRESHOLD = 2_000      # state-wide total > this  -> "storm hour"
MIN_STORM_DURATION_HOURS = 6        # event must last at least this long
MERGE_GAP_HOURS = 12                # merge two events if gap is shorter than this
PRE_STORM_BUFFER_HOURS = 24         # include some lead-up hours when picking val windows
POST_STORM_BUFFER_HOURS = 12        # ditto for tail


def main() -> None:
    print(f"[EDA] Loading {config.TRAIN_PATH} ...")
    ds = xr.open_dataset(config.TRAIN_PATH)
    print(ds)
    print()

    out = ds.out.transpose("timestamp", "location").values.astype(np.float32)  # (T, L)
    weather = ds.weather.transpose("timestamp", "location", "feature").values.astype(np.float32)
    locations = [str(x) for x in ds.location.values]
    features = [str(x) for x in ds.feature.values]
    timestamps = pd.to_datetime(ds.timestamp.values)

    T, L = out.shape
    F = weather.shape[-1]
    print(f"[EDA] T={T} hours, L={L} counties, F={F} weather features")
    print(f"[EDA] Time range: {timestamps[0]}  →  {timestamps[-1]}")
    print(f"[EDA] Span: {(timestamps[-1] - timestamps[0]).total_seconds() / 3600:.0f} hours "
          f"({(timestamps[-1] - timestamps[0]).days} days)")
    print()

    # ---- State-wide outage time series ----
    statewide = out.sum(axis=1)  # (T,)
    print(f"[EDA] State-wide outage:")
    print(f"      min={statewide.min():.0f}  median={np.median(statewide):.0f}  "
          f"mean={statewide.mean():.1f}  max={statewide.max():.0f}")
    quantiles = [0.5, 0.75, 0.9, 0.95, 0.99, 0.995, 0.999]
    qvals = np.quantile(statewide, quantiles)
    for q, v in zip(quantiles, qvals):
        print(f"      q{q*100:5.1f}% = {v:.0f}")
    print()

    # ---- Per-county zero rate ----
    zero_rate = (out == 0).mean(axis=0)
    print(f"[EDA] Per-county zero-rate (fraction of hours with 0 outage):")
    print(f"      min={zero_rate.min():.3f}  median={np.median(zero_rate):.3f}  "
          f"max={zero_rate.max():.3f}  mean={zero_rate.mean():.3f}")
    print()

    # ---- Storm detection (threshold-based) ----
    above = statewide >= STATE_OUTAGE_THRESHOLD
    events = _find_runs(above, timestamps, statewide,
                        min_duration=MIN_STORM_DURATION_HOURS,
                        merge_gap=MERGE_GAP_HOURS)
    print(f"[EDA] Detected {len(events)} storm events with threshold>={STATE_OUTAGE_THRESHOLD}, "
          f"min_dur={MIN_STORM_DURATION_HOURS}h, merge_gap={MERGE_GAP_HOURS}h")
    storm_df = pd.DataFrame(events)
    print(storm_df.to_string(index=False))
    print()

    # Save event list
    out_csv = config.RESULTS_DIR / "storm_events.csv"
    storm_df.to_csv(out_csv, index=False)
    print(f"[EDA] Wrote {out_csv}")

    # ---- Figures ----
    fig_dir = config.FIGURES_DIR
    fig_dir.mkdir(exist_ok=True, parents=True)

    fig, ax = plt.subplots(figsize=(16, 4))
    ax.plot(timestamps, statewide, lw=0.6, color="steelblue")
    ax.axhline(STATE_OUTAGE_THRESHOLD, ls="--", color="red", lw=0.8, label=f"threshold={STATE_OUTAGE_THRESHOLD}")
    for i, ev in enumerate(events):
        ax.axvspan(ev["start"], ev["end"], alpha=0.2, color="orange")
    ax.set_yscale("symlog")
    ax.set_xlabel("time")
    ax.set_ylabel("state-wide hourly outage (symlog)")
    ax.set_title(f"Michigan state-wide hourly outage  ({timestamps[0].date()} → {timestamps[-1].date()})  "
                 f"  events={len(events)}")
    ax.legend()
    fig.tight_layout()
    fig.savefig(fig_dir / "statewide_outage.png", dpi=120)
    print(f"[EDA] Saved {fig_dir / 'statewide_outage.png'}")

    # Zoomed view: top-3 biggest events side by side
    if len(events) > 0:
        top = sorted(events, key=lambda e: -e["peak"])[:6]
        ncol = min(3, len(top))
        nrow = (len(top) + ncol - 1) // ncol
        fig, axes = plt.subplots(nrow, ncol, figsize=(5 * ncol, 3 * nrow), squeeze=False)
        for ax, ev in zip(axes.flat, top):
            t0 = ev["start"] - pd.Timedelta(hours=48)
            t1 = ev["end"] + pd.Timedelta(hours=48)
            mask = (timestamps >= t0) & (timestamps <= t1)
            ax.plot(timestamps[mask], statewide[mask], color="steelblue")
            ax.axvspan(ev["start"], ev["end"], alpha=0.2, color="orange")
            ax.set_title(f"event {ev['id']}: {ev['start'].date()}  peak={ev['peak']:.0f}")
            ax.set_yscale("symlog")
        for ax in axes.flat[len(top):]:
            ax.axis("off")
        fig.tight_layout()
        fig.savefig(fig_dir / "storm_zoom.png", dpi=120)
        print(f"[EDA] Saved {fig_dir / 'storm_zoom.png'}")

    # ---- Quick weather feature glance ----
    print()
    print(f"[EDA] First 10 weather feature names: {features[:10]}")
    print(f"[EDA] Last 10 weather feature names:  {features[-10:]}")

    # ---- Test set timestamps (for reference) ----
    print()
    for tag, p in [("test_24h", config.TEST_24H_PATH), ("test_48h", config.TEST_48H_PATH)]:
        ds_t = xr.open_dataset(p)
        ts_t = pd.to_datetime(ds_t.timestamp.values)
        print(f"[EDA] {tag}: {ts_t[0]} → {ts_t[-1]}  ({len(ts_t)} hours)")


def _find_runs(above: np.ndarray,
               timestamps: pd.DatetimeIndex,
               statewide: np.ndarray,
               min_duration: int,
               merge_gap: int) -> list[dict]:
    """Return list of {id, start, end, duration_h, peak, sum} dicts."""
    if not above.any():
        return []

    diff = np.diff(above.astype(np.int8))
    starts = list(np.where(diff == 1)[0] + 1)
    ends = list(np.where(diff == -1)[0] + 1)  # exclusive
    if above[0]:
        starts = [0] + starts
    if above[-1]:
        ends = ends + [len(above)]

    raw = list(zip(starts, ends))

    # Merge events whose gap is shorter than `merge_gap`
    merged: list[tuple[int, int]] = []
    for s, e in raw:
        if merged and s - merged[-1][1] < merge_gap:
            merged[-1] = (merged[-1][0], e)
        else:
            merged.append((s, e))

    # Filter by duration
    out_events = []
    for i, (s, e) in enumerate(merged):
        dur = e - s
        if dur < min_duration:
            continue
        out_events.append(dict(
            id=len(out_events),
            start=timestamps[s],
            end=timestamps[e - 1],
            start_idx=int(s),
            end_idx=int(e),
            duration_h=int(dur),
            peak=float(statewide[s:e].max()),
            sum=float(statewide[s:e].sum()),
        ))
    return out_events


if __name__ == "__main__":
    main()
