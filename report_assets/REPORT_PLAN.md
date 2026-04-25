# 8-page Report Plan — Power Outage Forecasting

Hard limit: **8 pages** excluding refs/appendix. Rubric (PDF):

| Section weight | Item |
|---:|---|
| 30% | Prediction performance (24h RMSE 15% + 48h RMSE 15%) |
| 10% | Writing quality / clarity / organization |
| 15% | Modeling methodology and justification |
| 15% | Evaluation and discussion of prediction results |
| 30% | Policy recommendation and justification |

Net: **report quality is 70%**, prediction is 30%. Don't over-spend on RMSE.

---

## §1 Introduction (~0.75 page)
- Decision-maker view: Michigan utility planning under thunderstorm/wind season.
- Two tasks: 24h/48h forecast → 5-county generator pre-positioning.
- Why both forecast horizons matter for ops.
- Data summary in one sentence: 90 days × 83 counties × 109 weather features.

## §2 Data Analysis (~1.5 page)
**Key figures to produce → must save to `results/figures/`:**
- `statewide_outage.png` — full 90-day state-wide hourly time series with 26 storms shaded ✓ (done)
- `storm_zoom.png` — 6 largest storms detailed ✓ (done)
- `outage_distribution.png` — log-histogram showing zero-inflation
- `county_heterogeneity.png` — per-county zero-rate vs tracked households
- `weather_corr.png` — top-K weather features correlated with state-wide outage
- `outage_autocorr.png` — autocorrelation of state-wide outage (motivates lag features)

**Key claims to defend:**
- "Train ends mid-storm; test starts the very next hour" → motivates everything below.
- 70.5% of (county, hour) cells are zero → zero-inflation.
- Outage is highly autocorrelated → lag features dominate.
- Peak storm = 86k vs. 2nd-largest = 27k → heavy-tailed.

## §3 Methods (~1.75 page)
**Subsections:**
1. **Validation design** — why we replaced single-window split with rolling-origin CV.
   Show diagram: 6 folds × stride 72h × val=horizon. Mirrors test setup.
2. **Feature engineering** — outage lags, rolling stats, storm indicators (peak-24h,
   nonzero-count-24h, day-over-day delta), weather-at-t and weather lags (3/12/24h),
   calendar (sin/cos of hour/dow/month), location idx.
3. **Model families** (in order of complexity):
   - Baselines: all-zero, persistence, seasonal-naive, historical mean
   - Classical TS: SARIMAX(1,0,1)
   - Tree models: LightGBM (log target), LightGBM-Tweedie, XGBoost, CatBoost
   - Per-county LightGBM (heterogeneity)
   - Neural: GRU / BiLSTM / DLinear
   - Two-stage hurdle: classify zero/nonzero → predict positive
   - Ensemble: equal-weight mean (no learned weights → avoids val overfit)

## §4 Results (~1.5 page)
**Key tables:**
- Table 1: per-model 24h CV mean ± std RMSE (sorted)
- Table 2: per-model 48h CV mean ± std RMSE
- Table 3: ensemble vs best single model

**Key figures:**
- `cv_results_bar.png` — error bars per model
- `pred_vs_truth_storm.png` — predicted vs actual on a held-out storm (best model)
- `error_by_horizon.png` — RMSE at each forecast step h=1..48 for top model
- `feature_importance.png` — top-15 LightGBM gain features

## §5 Analysis & Policy Recommendation (~2 page → 30% of grade!)
**Reasoning chain:**
1. Use full-train ensemble to predict test 48h for every county.
2. For each county c, compute risk_c = max-over-h of pred[h, c] (peak risk)
   alternatively integrated risk = sum(pred[:,c]).
3. Constraint: each generator covers up to 1000 households.
4. Pick top-5 counties by `min(predicted_peak, tracked) / 1000` → priority score.
5. Discuss alternatives: top-5 by raw predicted total vs. by tracked-weighted vs.
   by uncertainty (where prediction is largest but also most uncertain).
6. Sensitivity analysis: what changes if we use 24h vs 48h horizon, mean vs peak.
7. Limitations: only 90 days of training, no real test ground truth, weather
   forecasts unavailable at test time.

**Output deliverable:**
- 5 FIPS codes in `[26XXX, 26XXX, 26XXX, 26XXX, 26XXX]` form

## §6 References (~0.25 page)
SARIMAX (Hyndman), LightGBM (Ke 2017), CatBoost (Prokhorenkova 2018),
PatchTST or DLinear if used (Zeng 2023), Tweedie (Jorgensen 1987).

---

## Per-task evidence checklist

Every experiment we run, we save **at least one of these** to `report_assets/`:
- A row in `cv_results.csv` (per-model per-fold per-horizon RMSE)
- A figure if it visualizes something non-obvious
- A markdown note in `report_assets/notes_*.md` if we made a design decision

This is non-negotiable: if I run an experiment and don't save evidence,
I cannot defend the choice in the report.
