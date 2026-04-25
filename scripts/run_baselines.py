"""Run all baseline + SARIMAX models on the rolling-origin CV folds.

    python -m scripts.run_baselines
"""
from src.runner import run_all
from src.training import summarize_runs


BASELINE_MODELS = [
    "zero",
    "persistence",
    "seasonal_naive_24",
    "seasonal_naive_168",
    "historical_mean",
    "sarimax",
]


def main():
    run_all(model_names=BASELINE_MODELS, skip_on_error=True)
    print("\n========= Summary =========")
    print(summarize_runs().to_string(index=False))


if __name__ == "__main__":
    main()
