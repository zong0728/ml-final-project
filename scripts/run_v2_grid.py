"""Run all v2-feature-set models on rolling-origin folds.

    python -m scripts.run_v2_grid
"""
from src.runner import run_all
from src.training import summarize_runs


V2_MODELS = [
    "lgb_v2",
    "lgb_v2_deep",
    "lgb_v2_mse",
    "lgb_v2_huber",
    "lgb_v2_weighted",
    "lgb_v2_log_corrected",
    "xgb_v2",
    "cat_v2",
    "lgb_tweedie_v2",
    "two_stage_lgb_v2",
]


def main():
    run_all(model_names=V2_MODELS, seeds=[42, 43], skip_on_error=True)
    print("\n========= v2 model summary =========")
    s = summarize_runs()
    sub = s[s["model"].isin(V2_MODELS)]
    print(sub.to_string(index=False))


if __name__ == "__main__":
    main()
