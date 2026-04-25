"""Generate LaTeX tables + policy section from current run results.

Outputs to report/:
    cv_table_h24.tex        — per-model CV RMSE table for 24h
    cv_table_h48.tex        — per-model CV RMSE table for 48h
    policy_section.tex      — full §5 text incl. selection rationale
"""
from pathlib import Path

import numpy as np
import pandas as pd

from src import config
from src.training import summarize_runs


REPORT_DIR = config.PROJECT_ROOT / "report"
REPORT_DIR.mkdir(exist_ok=True)

# A short pretty-name dict — keeps the table readable.
PRETTY = {
    "zero": "All-zero",
    "persistence": "Persistence ($y_t$ repeated)",
    "seasonal_naive_24": "Seasonal naive (24h)",
    "seasonal_naive_168": "Seasonal naive (168h)",
    "historical_mean": "Hist.\\ mean (county$\\times$hour$\\times$dow)",
    "sarimax": "SARIMAX(1,0,1) per county",
    "linreg_lag": "Linear regression on lag features",
    "ridge_lag": "Ridge ($\\alpha{=}1$)",
    "lightgbm": "LightGBM (log target)",
    "lightgbm_tweedie": "LightGBM (Tweedie loss)",
    "xgboost": "XGBoost",
    "catboost": "CatBoost",
    "per_county_lgb": "Per-county LightGBM",
    "ensemble_5way": "Ensemble (5 trees, equal weight)",
    "ensemble_6way": "Ensemble (6 trees, equal weight)",
}


def render_table(summary: pd.DataFrame, horizon: int) -> str:
    sub = summary[summary["horizon"] == horizon].copy()
    sub = sub.sort_values("val_rmse_mean")
    rows = []
    for _, r in sub.iterrows():
        nm = PRETTY.get(r["model"], r["model"].replace("_", "\\_"))
        rows.append(
            f"  {nm} & {r['val_rmse_mean']:.1f} & "
            f"{r['val_rmse_std']:.1f} & {r['val_rmse_min']:.1f} & "
            f"{r['val_rmse_max']:.1f} & {int(r['n'])} \\\\"
        )
    body = "\n".join(rows)
    return rf"""\begin{{table}}[t]
\centering
\caption{{{horizon}-hour rolling-origin CV: per-model RMSE (mean $\pm$ std across 6 folds, one seed). Lower is better. Fold count $n$ may be < 6 if a configuration was still in progress at writing time.}}
\label{{tab:cv{horizon}}}
\small
\begin{{tabular}}{{lrrrrr}}
\toprule
Model & mean & std & min & max & $n$ \\
\midrule
{body}
\bottomrule
\end{{tabular}}
\end{{table}}
"""


def write_policy_section(selection_path: Path, rationale_path: Path) -> str:
    if selection_path.exists():
        sel = selection_path.read_text().strip()
    else:
        sel = "[ pending — run scripts.finalize ]"
    if rationale_path.exists():
        r = pd.read_csv(rationale_path)
        rows = []
        for _, x in r.iterrows():
            rows.append(
                f"  {int(x['step'])} & {x['county_fips']} & "
                f"{x['marginal_served_household_hours']:.0f} & "
                f"{int(x['tracked_max']):,} & "
                f"{x['remaining_peak_at_pick_time']:.0f} \\\\"
            )
        header = (
            r"\begin{tabular}{rlrrr}" + "\n"
            r"\toprule" + "\n"
            r"step & county & served-hh-hours & tracked\_max & remaining peak \\" + "\n"
            r"\midrule" + "\n"
        )
        footer = "\n" + r"\bottomrule" + "\n" + r"\end{tabular}"
        rationale_tab = header + "\n".join(rows) + footer
    else:
        rationale_tab = "[ rationale table pending ]"

    eq_block = r"""
\[
M_s(c) = \sum_{h=1}^{48} \min\!\bigl( \hat y_{c,t+h} - {\rm consumed}_{s-1}(c,h),\ 1000,\ {\rm tracked\_max}(c) \bigr).
\]
"""

    body = (
        "We use the equal-weight ensemble's 48-hour test forecast to drive a "
        "greedy allocation of the 5 backup generators. The marginal benefit of "
        "placing one generator in county $c$ at decision step $s$ is the "
        "integral, across the 48-hour horizon, of the un-served outage that one "
        "generator (capacity 1{,}000 households) can absorb:\n"
        + eq_block +
        r"The per-hour cap at 1{,}000 is what gives the algorithm its "
        r"\emph{diminishing returns}: once a generator is placed in county $c$, "
        "subsequent generators contribute strictly less because the previously "
        "consumed capacity is subtracted. This makes $M_s$ a monotone "
        "submodular function over identical action sets, so the greedy "
        "algorithm is optimal (not merely a $1{-}1/e$ approximation).\n\n"
        r"A secondary cap at $\text{tracked\_max}(c)$ is included as a "
        "defensive guard for two future scenarios: (i) deployment to "
        "jurisdictions smaller than 1{,}000 households, and (ii) prediction "
        "pathologies where $\\hat y_c$ exceeds the county's actual household "
        "count. Neither binds in Michigan: the smallest county (Baraga, FIPS "
        "26013) tracks 2{,}327 households, and our predictions are clipped to "
        r"$1.5\times$ the historical max in any case. We retain the cap for "
        "code generality.\n\n"
        r"We greedily pick $c^\star_s = \arg\max_c M_s(c)$, deduct the served "
        r"capacity from $\text{remaining}$, and repeat for $s{=}1,\dots,5$. "
        r"Multiple generators \emph{may} go to the same county if its remaining "
        "served-household-hours after deduction still exceed every other "
        "county's marginal --- a feature, not a bug, since stacking generators "
        "in a high-impact county is allowed by the project description and is "
        "the optimal action when the predicted outage is many times larger "
        "than 1{,}000.\n\n"
        "Our recommendation is:"
        f"\n\n\\begin{{quote}}\n\\verb|{sel}|\n\\end{{quote}}\n\n"
        r"\paragraph{Rationale (per-step marginal benefit).}" + "\n"
        + rationale_tab + "\n\n"
        r"\paragraph{Limitations.}" + "\n"
        "(i) The training window is only 90 days, all in the same season, so "
        "any seasonality not represented (winter ice storms, fall hurricanes) "
        "is invisible to the model. (ii) No future-weather information is "
        "available at test time, which fundamentally limits 48-hour "
        "predictability of fast-moving convective events. (iii) The 5-county "
        "recommendation is derived from a point forecast; a probabilistic "
        "forecast would let us bound the regret of mis-allocation."
    )
    return body


def main():
    s = summarize_runs()
    if not s.empty:
        (REPORT_DIR / "cv_table_h24.tex").write_text(render_table(s, 24))
        (REPORT_DIR / "cv_table_h48.tex").write_text(render_table(s, 48))
        print(f"wrote cv_table_h24.tex ({len(s[s.horizon==24])} rows)")
        print(f"wrote cv_table_h48.tex ({len(s[s.horizon==48])} rows)")

    sel_path = config.RESULTS_DIR / "policy_selection.txt"
    ra_path = config.PROJECT_ROOT / "report_assets" / "policy_rationale.csv"
    (REPORT_DIR / "policy_section.tex").write_text(write_policy_section(sel_path, ra_path))
    print(f"wrote policy_section.tex (selection={sel_path.exists()}, rationale={ra_path.exists()})")


if __name__ == "__main__":
    main()
