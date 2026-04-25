"""Quick sanity check for rolling-origin CV folds. Run from project root:

    python -m scripts.check_folds
"""
from src.data import drop_zero_variance_features, load_raw, make_rolling_folds
from src import config


def main():
    ds_train, ds_test_24h, ds_test_48h = load_raw()
    ds_train_f = drop_zero_variance_features(ds_train)
    print(f"Train: {ds_train_f.dims}\n")

    for h in [24, 48]:
        folds = make_rolling_folds(ds_train_f, horizon=h,
                                   n_folds=config.N_FOLDS,
                                   stride_hours=config.FOLD_STRIDE_HOURS)
        print(f"horizon={h}h:  {len(folds)} folds")
        for f in folds:
            print(f"  fold {f['fold_id']}:  fit→{f['origin_idx']}h "
                  f"val={f['val_start_ts']}→{f['val_end_ts']}  "
                  f"sum={f['val_sum']:.0f}  peak={f['val_peak']:.0f}")
        print()


if __name__ == "__main__":
    main()
