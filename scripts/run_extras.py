"""Run AutoARIMA + Quantile LGB on the rolling-origin folds.

These are 'advanced' models that don't have a hparam grid — single config each,
fast to evaluate, useful as additional ensemble members.

    python -m scripts.run_extras
"""
from src.runner import run_all
from src.training import summarize_runs


EXTRA_MODELS = [
    "auto_arima",
    "lgb_quantile_p50",
    "lgb_quantile_p90",
]


def main():
    run_all(model_names=EXTRA_MODELS, seeds=[42, 43], skip_on_error=True)
    print("\n========= Extras summary =========")
    s = summarize_runs()
    sub = s[s["model"].isin(EXTRA_MODELS)]
    print(sub.to_string(index=False))


if __name__ == "__main__":
    main()
