"""Run PCA + storm-expert grid on rolling-origin folds.

    python -m scripts.run_pca_storm
"""
from src.runner import run_all
from src.training import summarize_runs


PCA_STORM_MODELS = [
    # PCA on weather features (educator slide 13 hint)
    "lgb_pca20",
    "lgb_pca30",
    "xgb_pca20",
    "cat_pca20",

    # Storm-expert mixture-of-experts
    "lgb_storm_expert",
    "lgb_storm_blend",
]


def main():
    run_all(model_names=PCA_STORM_MODELS, seeds=[42, 43], skip_on_error=True)
    print("\n========= PCA + Storm-expert summary =========")
    s = summarize_runs()
    sub = s[s["model"].isin(PCA_STORM_MODELS)]
    print(sub.to_string(index=False))


if __name__ == "__main__":
    main()
