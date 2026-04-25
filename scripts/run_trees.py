"""Run tree-based models on rolling-origin CV folds.

    python -m scripts.run_trees
"""
from src.runner import run_all
from src.training import summarize_runs


TREE_MODELS = [
    "linreg_lag",
    "ridge_lag",
    "lightgbm",
    "lightgbm_tweedie",
    "xgboost",
    "catboost",
    "per_county_lgb",
]


def main():
    # Single seed first; we'll add more seeds on H200 later.
    run_all(model_names=TREE_MODELS, seeds=[42], skip_on_error=True)
    print("\n========= Summary =========")
    print(summarize_runs().to_string(index=False))


if __name__ == "__main__":
    main()
