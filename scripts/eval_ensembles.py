"""Evaluate multi-layer ensembles using cached fold-level val_preds.

This is purely diagnostic — no retraining. Uses
results/val_preds/<model>__seed{S}__h{H}__fold{F}.npy files.

Reports CV RMSE for:
  L1 (within-family bagging)  : multi-seed average of best config per family
  L2 (across-family ensemble) : equal-weight or inv-rmse-weighted mean of L1
                                 outputs across families
  L3 (stacking via leave-fold-out, optional)

    python -m scripts.eval_ensembles
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from src import config
from src.evaluation import avg_county_rmse
from src.data import drop_zero_variance_features, fold_arrays, load_raw, make_rolling_folds


VAL_PRED_DIR = config.RESULTS_DIR / "val_preds"


def _load_pred(model: str, seed: int, h: int, fold: int):
    p = VAL_PRED_DIR / f"{model}__seed{seed}__h{h}__fold{fold}.npy"
    return np.load(p) if p.exists() else None


def _load_all_seeds(model: str, h: int, fold: int) -> list[np.ndarray]:
    """Load every cached prediction for this (model, h, fold) across seeds."""
    pat = VAL_PRED_DIR.glob(f"{model}__seed*__h{h}__fold{fold}.npy")
    return [np.load(p) for p in sorted(pat)]


def main():
    ds_train, _, _ = load_raw()
    ds_train_f = drop_zero_variance_features(ds_train)

    # Champion configs by family (calm-period leaders from current data)
    family_champs = {
        "lgb_quantile":   "lgb_quantile_p50",
        "lgb_grid":       "lgb__nl63_lr03_n800_ss7",
        "cat":            "cat__d6_lr03_n400",
        "xgb":            "xgb__d8_lr03_n400",
        "two_stage":      "two_stage_lgb",
        "lightgbm_def":   "lightgbm",
        "lgb_v2_corr":    "lgb_v2_log_corrected",
    }

    print("="*100)
    print("Multi-layer ensemble evaluation on rolling-origin CV folds")
    print("="*100)

    for h in [24, 48]:
        folds = make_rolling_folds(ds_train_f, horizon=h,
                                    n_folds=config.N_FOLDS,
                                    stride_hours=config.FOLD_STRIDE_HOURS)
        print(f"\n--- horizon = {h}h ---\n")

        # Per-fold ground truth
        fold_truth = {}
        for fld in folds:
            _, _, out_val, _, _, _ = fold_arrays(ds_train_f, fld)
            fold_truth[fld["fold_id"]] = out_val

        # ========== L0: Single-seed champion per family ==========
        print(f"L0. Best single-seed model per family:")
        l0_per_fam = {}      # fam -> { fold: pred }
        for fam, model in family_champs.items():
            per_fold = {}
            for fld in folds:
                fid = fld["fold_id"]
                preds = _load_all_seeds(model, h, fid)
                if preds:
                    per_fold[fid] = preds[0]   # first seed
            l0_per_fam[fam] = per_fold
            if per_fold:
                rmses = [avg_county_rmse(fold_truth[f], p) for f, p in per_fold.items()]
                # calm = exclude fold 1
                calm = [r for f, r in zip(per_fold.keys(), rmses) if f != 1]
                storm = [r for f, r in zip(per_fold.keys(), rmses) if f == 1]
                calm_mean = np.mean(calm) if calm else np.nan
                storm_mean = np.mean(storm) if storm else np.nan
                print(f"  {fam:<18s} ({model:<35s}): calm={calm_mean:6.2f}  storm={storm_mean:6.0f}  folds={list(per_fold.keys())}")

        # ========== L1: Within-family multi-seed bagging ==========
        print(f"\nL1. Within-family bagging (mean across seeds):")
        l1_per_fam = {}
        for fam, model in family_champs.items():
            per_fold = {}
            for fld in folds:
                fid = fld["fold_id"]
                preds = _load_all_seeds(model, h, fid)
                if preds:
                    per_fold[fid] = np.mean(preds, axis=0)
            l1_per_fam[fam] = per_fold
            if per_fold:
                rmses = [avg_county_rmse(fold_truth[f], p) for f, p in per_fold.items()]
                calm = [r for f, r in zip(per_fold.keys(), rmses) if f != 1]
                storm = [r for f, r in zip(per_fold.keys(), rmses) if f == 1]
                if calm:
                    print(f"  {fam:<18s}: calm={np.mean(calm):6.2f}  storm={np.mean(storm) if storm else np.nan:6.0f}  "
                          f"(n_seeds={len(_load_all_seeds(model, h, list(per_fold.keys())[0]))})")

        # ========== L2: Across-family ensemble (equal weight) ==========
        print(f"\nL2. Across-family ensembles:")
        for k in [3, 5, 7]:
            # Pick top-k families by their L1 calm-period RMSE
            ranked = []
            for fam, per_fold in l1_per_fam.items():
                if not per_fold: continue
                calm_rmses = [avg_county_rmse(fold_truth[f], p)
                               for f, p in per_fold.items() if f != 1]
                if calm_rmses:
                    ranked.append((np.mean(calm_rmses), fam))
            ranked.sort()
            chosen_fams = [f for _, f in ranked[:k]]
            if not chosen_fams: continue

            for strat_name in ["mean", "median"]:
                folds_present = sorted(set(
                    fid for fam in chosen_fams for fid in l1_per_fam[fam].keys()
                ))
                fold_rmses = {}
                for fid in folds_present:
                    stack = []
                    for fam in chosen_fams:
                        if fid in l1_per_fam[fam]:
                            stack.append(l1_per_fam[fam][fid])
                    if not stack: continue
                    arr = np.stack(stack, axis=0)
                    ens = np.mean(arr, axis=0) if strat_name == "mean" else np.median(arr, axis=0)
                    fold_rmses[fid] = avg_county_rmse(fold_truth[fid], ens)

                calm_rmses = [r for f, r in fold_rmses.items() if f != 1]
                storm_rmses = [r for f, r in fold_rmses.items() if f == 1]
                if calm_rmses:
                    print(f"  Top-{k} families × {strat_name:6s}: calm={np.mean(calm_rmses):6.2f}  "
                          f"storm={np.mean(storm_rmses) if storm_rmses else np.nan:6.0f}  "
                          f"({chosen_fams})")


if __name__ == "__main__":
    main()
