"""Run AutoARIMA + Quantile LGB on the rolling-origin folds.

These are 'advanced' models that don't have a hparam grid — single config each,
fast to evaluate, useful as additional ensemble members.

    python -m scripts.run_extras
"""
from src.runner import run_all
from src.training import summarize_runs


EXTRA_MODELS = [
    # AutoARIMA: per-county SARIMA with order grid (replaces fixed (1,0,1))
    "auto_arima",

    # Quantile LGB: median + P90 for uncertainty
    "lgb_quantile_p50",
    "lgb_quantile_p90",

    # Hurdle / zero-inflated decomposition (Cragg 1971, Lambert 1992)
    "two_stage_lgb",
    "two_stage_xgb",

    # Per-county specialized regressors — diversity for ensemble
    "per_county_lgb",
    "per_county_xgb",
    "per_county_cat",
]


def main():
    run_all(model_names=EXTRA_MODELS, seeds=[42, 43], skip_on_error=True)
    print("\n========= Extras summary =========")
    s = summarize_runs()
    sub = s[s["model"].isin(EXTRA_MODELS)]
    print(sub.to_string(index=False))


if __name__ == "__main__":
    main()
