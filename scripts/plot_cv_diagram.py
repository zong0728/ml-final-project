"""Diagram of the rolling-origin CV scheme — for the report Methods section."""
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.dates as mdates
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
    train_start = timestamps[0]
    train_end = timestamps[-1]

    # Use mdates.date2num to convert datetimes -> matplotlib day-floats explicitly,
    # so the barh `left` and `width` are in the same units (days).
    fig, axes = plt.subplots(2, 1, figsize=(11, 4.2), sharex=True)
    for ax, folds, h, label in [(axes[0], folds24, 24, "24-h horizon"),
                                  (axes[1], folds48, 48, "48-h horizon")]:
        for i, f in enumerate(folds):
            fit_lo_num = mdates.date2num(train_start)
            fit_hi_num = mdates.date2num(timestamps[f["origin_idx"] - 1])
            val_lo_num = mdates.date2num(f["val_start_ts"])
            val_hi_num = mdates.date2num(f["val_end_ts"])
            ax.barh(i, fit_hi_num - fit_lo_num, left=fit_lo_num,
                    height=0.7, color="steelblue", alpha=0.7,
                    label="fit window" if i == 0 else None)
            ax.barh(i, val_hi_num - val_lo_num, left=val_lo_num,
                    height=0.7, color="firebrick",
                    label=f"val ({h}h)" if i == 0 else None)

        # The actual test window held by the graders: starts immediately after
        # train_end, lasts h hours.
        test_lo_num = mdates.date2num(train_end)
        test_hi_num = mdates.date2num(train_end + pd.Timedelta(hours=h))
        ax.barh(len(folds) + 0.5, test_hi_num - test_lo_num, left=test_lo_num,
                height=0.5, color="orange",
                label="test (held by graders)")

        ax.set_yticks(list(range(len(folds))) + [len(folds) + 0.5])
        ax.set_yticklabels([f"fold {i}" for i in range(len(folds))] + ["TEST"],
                           fontsize=9)
        ax.invert_yaxis()
        ax.xaxis_date()
        ax.set_title(f"Rolling-origin CV — {label}", fontsize=10)
        ax.legend(loc="upper left", fontsize=8, ncol=3)
        ax.grid(axis="x", alpha=0.3)

    # Format the x-axis so dates show clearly (April → June 2023).
    axes[1].xaxis.set_major_locator(mdates.WeekdayLocator(byweekday=mdates.MO,
                                                            interval=2))
    axes[1].xaxis.set_major_formatter(mdates.DateFormatter("%b %d"))
    axes[1].set_xlabel("date (2023, UTC)")
    fig.autofmt_xdate(rotation=30, ha="right")

    fig.tight_layout()
    out = config.FIGURES_DIR / "cv_diagram.png"
    fig.savefig(out, dpi=130)
    print(f"saved {out}")


if __name__ == "__main__":
    main()
