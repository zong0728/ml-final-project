"""Diagram of the rolling-origin CV scheme — for the report Methods section."""
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src import config
from src.data import drop_zero_variance_features, load_raw, make_rolling_folds


def main():
    ds_train, _, _ = load_raw()
    ds_train_f = drop_zero_variance_features(ds_train)
    folds24 = make_rolling_folds(ds_train_f, horizon=24, n_folds=6, stride_hours=72)
    folds48 = make_rolling_folds(ds_train_f, horizon=48, n_folds=6, stride_hours=72)

    timestamps = pd.to_datetime(ds_train_f.timestamp.values)
    T = len(timestamps)

    # Plot a Gantt-style diagram: each row is one fold, x-axis is timestamp.
    fig, axes = plt.subplots(2, 1, figsize=(12, 5), sharex=True)
    for ax, folds, h, label in [(axes[0], folds24, 24, "24-h horizon"),
                                  (axes[1], folds48, 48, "48-h horizon")]:
        for i, f in enumerate(folds):
            fit_lo = timestamps[0]
            fit_hi = timestamps[f["origin_idx"] - 1]
            val_lo = f["val_start_ts"]
            val_hi = f["val_end_ts"]
            ax.barh(i, (fit_hi - fit_lo).total_seconds() / 3600,
                    left=fit_lo, height=0.6, color="steelblue", alpha=0.6,
                    label="fit" if i == 0 else None)
            ax.barh(i, h, left=val_lo, height=0.6, color="firebrick",
                    label=f"val ({h}h)" if i == 0 else None)
        # Test as a thin orange bar at the right
        test_lo = timestamps[-1]
        ax.barh(len(folds) + 0.5, h, left=test_lo, height=0.4,
                color="orange", label="test (held by graders)")
        ax.set_yticks(list(range(len(folds))) + [len(folds) + 0.5])
        ax.set_yticklabels([f"fold {i}" for i in range(len(folds))] + ["TEST"])
        ax.set_xlabel("date (UTC)")
        ax.set_title(f"Rolling-origin CV — {label}")
        ax.legend(loc="upper left", fontsize=8)
        ax.grid(axis="x", alpha=0.3)

    fig.tight_layout()
    out = config.FIGURES_DIR / "cv_diagram.png"
    fig.savefig(out, dpi=130)
    print(f"saved {out}")


if __name__ == "__main__":
    main()
