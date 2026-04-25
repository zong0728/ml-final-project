"""Grid-search runner for hyperparameter sweeps.

Locally: python -m scripts.run_grid --grid lgb
On H200: sbatch scripts/h200/run_grid.sbatch lgb

Each grid entry registers a new model name (e.g. lightgbm__nl63_lr05_n800)
into the registry on the fly so that the standard runner machinery
(folds + caching + resume) just works.
"""
from __future__ import annotations

import argparse
import itertools
from typing import Iterable

import numpy as np

from src import config
from src.data import build_lag_table
from src.registry import register, MODEL_REGISTRY
from src.runner import run_all
from src.training import summarize_runs


# ----------------------------------------------------------------------------
# Helpers — lightweight closures that build a fit-predict from hyperparams.
# ----------------------------------------------------------------------------

_TABULAR_ORIGIN_STRIDE = 4


def _fit_predict_lgb(out_fit, weather_fit, ts_fit, locations, horizon, seed,
                     **lgb_kwargs):
    import lightgbm as lgb
    X_tr, y_tr, X_inf, _, _ = build_lag_table(
        out=out_fit, weather=weather_fit, timestamps=ts_fit,
        horizon=horizon, origin_stride=_TABULAR_ORIGIN_STRIDE,
    )
    use_log = lgb_kwargs.pop("log_target", True)
    if use_log:
        y = np.log1p(np.clip(y_tr, 0, None))
    else:
        y = y_tr
    model = lgb.LGBMRegressor(random_state=seed, verbosity=-1, **lgb_kwargs)
    model.fit(X_tr, y)
    pred = model.predict(X_inf)
    if use_log:
        cap = float(np.log1p(y_tr.max())) + 2.0
        pred = np.expm1(np.clip(pred, None, cap))
    cap_y = float(y_tr.max()) * 1.5 + 10.0
    pred = np.clip(pred, 0.0, cap_y)
    L = out_fit.shape[1]
    return pred.reshape(horizon, L).astype(np.float32)


def _fit_predict_xgb(out_fit, weather_fit, ts_fit, locations, horizon, seed,
                     **xgb_kwargs):
    import xgboost as xgb
    X_tr, y_tr, X_inf, _, _ = build_lag_table(
        out=out_fit, weather=weather_fit, timestamps=ts_fit,
        horizon=horizon, origin_stride=_TABULAR_ORIGIN_STRIDE,
    )
    use_log = xgb_kwargs.pop("log_target", True)
    y = np.log1p(np.clip(y_tr, 0, None)) if use_log else y_tr
    base_kwargs = dict(random_state=seed, tree_method="hist", verbosity=0)
    base_kwargs.update(xgb_kwargs)
    try:
        import torch
        if torch.cuda.is_available():
            base_kwargs["device"] = "cuda"
    except Exception:
        pass
    model = xgb.XGBRegressor(**base_kwargs)
    model.fit(X_tr, y)
    pred = model.predict(X_inf)
    if use_log:
        cap = float(np.log1p(y_tr.max())) + 2.0
        pred = np.expm1(np.clip(pred, None, cap))
    cap_y = float(y_tr.max()) * 1.5 + 10.0
    pred = np.clip(pred, 0.0, cap_y)
    L = out_fit.shape[1]
    return pred.reshape(horizon, L).astype(np.float32)


def _fit_predict_cat(out_fit, weather_fit, ts_fit, locations, horizon, seed,
                     **cat_kwargs):
    from catboost import CatBoostRegressor
    X_tr, y_tr, X_inf, _, feature_names = build_lag_table(
        out=out_fit, weather=weather_fit, timestamps=ts_fit,
        horizon=horizon, origin_stride=_TABULAR_ORIGIN_STRIDE,
    )
    cat_idx = [feature_names.index("location_idx")]
    y = np.log1p(np.clip(y_tr, 0, None))
    base = dict(random_seed=seed, verbose=False, cat_features=cat_idx,
                bootstrap_type="Bernoulli")
    base.update(cat_kwargs)
    X_tr_mix = X_tr.astype(object); X_tr_mix[:, cat_idx[0]] = X_tr[:, cat_idx[0]].astype(int)
    X_inf_mix = X_inf.astype(object); X_inf_mix[:, cat_idx[0]] = X_inf[:, cat_idx[0]].astype(int)
    model = CatBoostRegressor(**base)
    model.fit(X_tr_mix, y)
    pred = model.predict(X_inf_mix)
    cap = float(np.log1p(y_tr.max())) + 2.0
    pred = np.expm1(np.clip(pred, None, cap))
    cap_y = float(y_tr.max()) * 1.5 + 10.0
    pred = np.clip(pred, 0.0, cap_y)
    L = out_fit.shape[1]
    return pred.reshape(horizon, L).astype(np.float32)


# ----------------------------------------------------------------------------
# Grid definitions
# ----------------------------------------------------------------------------

LGB_GRID = list(itertools.product(
    [31, 63, 127],            # num_leaves
    [0.03, 0.05, 0.1],        # learning_rate
    [400, 800, 1500],         # n_estimators
    [0.7, 0.9],               # subsample
))

XGB_GRID = list(itertools.product(
    [4, 6, 8],
    [0.03, 0.05, 0.1],
    [400, 800],
))

CAT_GRID = list(itertools.product(
    [4, 6, 8],
    [0.03, 0.05, 0.1],
    [400, 800],
))

LGB_TWEEDIE_GRID = list(itertools.product(
    [31, 63],
    [0.05, 0.1],
    [400, 800],
    [1.2, 1.5, 1.8],          # tweedie variance power
))


def register_lgb_grid() -> list[str]:
    names = []
    for nl, lr, ne, ss in LGB_GRID:
        nm = f"lgb__nl{nl}_lr{int(lr*100):02d}_n{ne}_ss{int(ss*10)}"
        if nm in MODEL_REGISTRY:
            names.append(nm); continue
        kwargs = dict(num_leaves=nl, learning_rate=lr, n_estimators=ne,
                      subsample=ss, colsample_bytree=0.8, log_target=True)

        def _make(_kwargs=kwargs):
            def _fp(out_fit, weather_fit, timestamps_fit, locations, horizon, seed):
                preds = _fit_predict_lgb(out_fit, weather_fit, timestamps_fit,
                                         locations, horizon, seed, **_kwargs)
                return preds, _kwargs
            return _fp
        register(nm, tier="grid_lgb", stochastic=True, description=str(kwargs))(_make())
        names.append(nm)
    return names


def register_xgb_grid() -> list[str]:
    names = []
    for d, lr, ne in XGB_GRID:
        nm = f"xgb__d{d}_lr{int(lr*100):02d}_n{ne}"
        if nm in MODEL_REGISTRY:
            names.append(nm); continue
        kwargs = dict(max_depth=d, learning_rate=lr, n_estimators=ne,
                      subsample=0.8, colsample_bytree=0.8, log_target=True)

        def _make(_kwargs=kwargs):
            def _fp(out_fit, weather_fit, timestamps_fit, locations, horizon, seed):
                return (_fit_predict_xgb(out_fit, weather_fit, timestamps_fit,
                                         locations, horizon, seed, **_kwargs), _kwargs)
            return _fp
        register(nm, tier="grid_xgb", stochastic=True, description=str(kwargs))(_make())
        names.append(nm)
    return names


def register_cat_grid() -> list[str]:
    names = []
    for d, lr, ne in CAT_GRID:
        nm = f"cat__d{d}_lr{int(lr*100):02d}_n{ne}"
        if nm in MODEL_REGISTRY:
            names.append(nm); continue
        kwargs = dict(depth=d, learning_rate=lr, iterations=ne,
                      l2_leaf_reg=3.0, subsample=0.8)

        def _make(_kwargs=kwargs):
            def _fp(out_fit, weather_fit, timestamps_fit, locations, horizon, seed):
                return (_fit_predict_cat(out_fit, weather_fit, timestamps_fit,
                                         locations, horizon, seed, **_kwargs), _kwargs)
            return _fp
        register(nm, tier="grid_cat", stochastic=True, description=str(kwargs))(_make())
        names.append(nm)
    return names


def register_lgb_tweedie_grid() -> list[str]:
    names = []
    for nl, lr, ne, vp in LGB_TWEEDIE_GRID:
        nm = f"lgbtw__nl{nl}_lr{int(lr*100):02d}_n{ne}_vp{int(vp*10)}"
        if nm in MODEL_REGISTRY:
            names.append(nm); continue
        kwargs = dict(objective="tweedie", tweedie_variance_power=vp,
                      num_leaves=nl, learning_rate=lr, n_estimators=ne,
                      subsample=0.8, colsample_bytree=0.8, log_target=False)

        def _make(_kwargs=kwargs):
            def _fp(out_fit, weather_fit, timestamps_fit, locations, horizon, seed):
                return (_fit_predict_lgb(out_fit, weather_fit, timestamps_fit,
                                         locations, horizon, seed, **_kwargs), _kwargs)
            return _fp
        register(nm, tier="grid_lgbtw", stochastic=True, description=str(kwargs))(_make())
        names.append(nm)
    return names


# ----------------------------------------------------------------------------
# Entry point
# ----------------------------------------------------------------------------

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--grid", choices=["lgb", "xgb", "cat", "lgbtw", "all", "neural"],
                   default="all")
    p.add_argument("--seeds", nargs="+", type=int, default=[42])
    args = p.parse_args()

    names: list[str] = []
    if args.grid in ("lgb", "all"):
        names += register_lgb_grid()
    if args.grid in ("xgb", "all"):
        names += register_xgb_grid()
    if args.grid in ("cat", "all"):
        names += register_cat_grid()
    if args.grid in ("lgbtw", "all"):
        names += register_lgb_tweedie_grid()
    if args.grid == "neural":
        # Ship to the existing neural-net wrappers — H200 only.
        names += ["gru", "bilstm", "dlinear", "tcn", "transformer", "patchtst"]

    print(f"[Grid] running {len(names)} models, seeds={args.seeds}")
    for n in names:
        print(f"  - {n}")

    run_all(model_names=names, seeds=args.seeds, skip_on_error=True)
    print("\n========= Top 20 by mean RMSE =========")
    s = summarize_runs()
    print(s.head(20).to_string(index=False))


if __name__ == "__main__":
    main()
