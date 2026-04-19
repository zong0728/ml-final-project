# Policy Recommendation Methodology
**Part II — Pre-Positioning 5 Backup Generators Across Michigan Counties**

This document specifies the decision framework, objective function, algorithm, and
implementation used to translate our outage predictions from Part I into a concrete
deployment recommendation.

---

## 1. Problem statement

A utility company has **5 backup generators** to pre-position across the 83 Michigan
counties before a forecasted storm window. Each generator:

- Serves **up to 1,000 households** simultaneously.
- Cannot be relocated once deployed — it stays in its county for the full event.
- May share a county with other generators (2+ in one county is allowed).

The deployment window is the 24h or 48h test horizon for which Part I produced
per-county hourly outage predictions.

**Our decision**: which 5 county FIPS codes (with possible repeats) maximize the
number of household-hours of outage actually mitigated?

**Link to Part I**: the policy consumes the predicted outage matrix
`pred[c, t] ∈ ℝ^{83 × H}` (H = 24 or 48). Model accuracy directly translates
into policy accuracy — a better RMSE ⇒ better targeted deployment.

---

## 2. Decision framework

### 2.1 Decision variable

For each county `c ∈ {1..83}`, let `k_c ∈ {0, 1, 2, ...}` denote the number of
generators assigned there, subject to:

$$
\sum_c k_c = 5, \quad k_c \ge 0, \quad k_c \in \mathbb{Z}
$$

### 2.2 What happens at a single hour

At any given hour `t`, county `c` has `pred[c, t]` households needing power. With
`k_c` generators deployed there, capacity is `k_c × 1000`. The number of
households actually served that hour is:

$$
\text{served}_{c,t}(k_c) = \min\big(k_c \cdot 1000,\ \text{pred}[c, t]\big)
$$

- If capacity exceeds demand: some capacity idles, served = demand.
- If demand exceeds capacity: generators are fully loaded, served = capacity.

### 2.3 Total coverage (the objective)

Because generators stay in place for the full horizon, the total value of
assigning `k_c` generators to county `c` is the sum over every hour:

$$
\text{coverage}_c(k_c) = \sum_{t=1}^{H} \min\big(k_c \cdot 1000,\ \text{pred}[c, t]\big)
$$

The **system-wide objective** is to maximize total household-hours served:

$$
\boxed{\max_{k_1,\dots,k_{83}}\ \sum_c \text{coverage}_c(k_c) \quad \text{s.t.}\ \sum_c k_c = 5}
$$

### 2.4 Where does population enter? (Implicit vs explicit)

The objective uses `pred[c, t]` — the predicted **number of households out of
power**, not a rate and not a severity index. This already encodes population
implicitly: a county with 900,000 tracked households can physically produce
tens of thousands of simultaneous outages, while a 20k-household county is
capped at ~20k. Picking counties by raw outage count therefore gravitates
toward larger populations — which is the desired behavior for a humanitarian
objective (maximize total households helped).

Deliberate choices we **do not** make:

- We do **not** multiply by `tracked[c]` (would double-count population).
- We do **not** divide by `tracked[c]` (would convert to outage *rate* and
  favor small counties where generator capacity is likely underutilized).
- We do **not** add `tracked` as an input feature to the prediction model.
  Tree models already learn county-level baselines via `location_idx` and
  the long-lag outage history, so explicit population input is redundant.

One sanity filter we **do** apply:

- **Minimum tracked threshold**: if `tracked[c] < 1000`, a single generator
  cannot be fully utilized there. Filter these counties out before greedy
  allocation. (In the Michigan data this filter is a no-op — all 83 counties
  have `tracked ≥ 1000`. Included for documentation.)

### 2.5 Alternative objectives (considered and rejected)

For transparency, here is why competing formulations were not chosen:

| Variant | Formula | Behavior | Why not |
|---------|---------|----------|---------|
| **Our choice** | `Σ_t min(1000·k, pred[c,t])` | Maximize absolute households helped | Most defensible for utility's stated goal |
| Outage-rate maximization | `Σ_t min(1000·k, pred[c,t]) / tracked[c]` | Favors counties with highest proportional impact | Tends to pick small-population counties where capacity is wasted; inconsistent with "5,000 households served" language in problem statement |
| Peak-demand hedging | `max_t pred[c,t]` | Pick worst-moment counties | Ignores duration — see §3 counterexample |
| Total outage-hours | `Σ_t pred[c,t]` | Pick most cumulatively impacted counties | Ignores 1,000-household saturation — over-promises value |
| Critical infrastructure weighted | `Σ_t w_c · min(1000·k, pred[c,t])` | Extra weight for counties with hospitals / dialysis centers | Dataset lacks infrastructure data; flagged as future work |
| Equity-weighted | `Σ_t s_c · min(1000·k, pred[c,t])` | Extra weight for socioeconomically vulnerable counties | Same data limitation; flagged as future work |

---

## 3. Why naive metrics (peak / total) are wrong

Two tempting but incorrect scoring functions:

1. **Peak**: pick counties with the highest `max_t pred[c, t]`.
2. **Total outage-hours**: pick counties with the highest `Σ_t pred[c, t]`.

Both fail to capture the physical behavior of a deployed generator.

### Counterexample — peak picks the wrong county

| County | Outage profile over 48h | Peak | Total hh-hrs | 1-gen coverage |
|--------|-------------------------|------|--------------|----------------|
| A | 2000 households out for 1 hour only, 0 otherwise | **2000** | 2000 | **1000** |
| B | Steady 1000 households out for all 48 hours | 1000 | 48000 | **48000** |

**Peak ranking says A wins.** But deploying the generator to A covers only
1,000 household-hours (48 hours × min(1000, actual demand) = 1 × 1000 = 1000).
Deploying to B covers **48,000 household-hours** — 48× more.

### Empirical demonstration on our data

Running both algorithms on the training set's last 48 hours as an oracle proxy:

| Strategy | Recommended counties | Total coverage (hh-hrs) |
|----------|---------------------|------------------------|
| Peak-based top-5 | 26163, 26123, 26021, 26117, 26035 | **48,482** |
| Coverage-based greedy (correct) | 26125, 26139, 26161, **26163 × 2** | **80,147** |

The correct method yields **+65% coverage** on this window.

### Why "Total outage-hours" is also wrong

Total ignores saturation. A county with peak = 50,000 for 48 hours has
total = 2,400,000. Total says "put all 5 gens here". But 5 gens = 5,000 capacity,
and since demand >> capacity at every hour, coverage saturates at
5 × 1,000 × 48 = 240,000. A tenth of the claimed value.

The only metric that is simultaneously (a) duration-aware, (b) saturation-aware,
and (c) marginal-decrease-aware is the `min(k·1000, pred)` integral.

---

## 4. The greedy algorithm

Because `coverage_c(k)` is **concave** in `k` (each additional generator's
marginal value is non-increasing), the assignment problem is a **sub-modular
maximization under a cardinality constraint**. Greedy selection is known to
achieve ≥ (1 − 1/e) ≈ 63% of the optimum in general; for our problem structure
(identical capacity, integer slots, small n, marginal value independent across
counties), greedy is in fact **optimal** — equivalent to exhaustive search and
ILP solutions, at negligible runtime.

### 4.1 Priority hierarchy

A greedy algorithm requires a **fully specified ordering** over candidates.
Without one, `argmax` silently breaks ties by array index (i.e., by FIPS code),
which is arbitrary and not defensible. We define three tiers:

| Tier | Criterion | When it applies |
|------|-----------|-----------------|
| **Primary** | Expected marginal household-hours: `E[Δ_c(k)]` averaged over our 6-run ensemble (3 seeds × 2 model families) | Always; decides the choice whenever a candidate exceeds all others by > 1% |
| **Secondary** | Prediction stability: `std[Δ_c(k)]` across the ensemble (lower is preferred) | When two or more candidates are within 1% of the top on Primary |
| **Tertiary** | Tracked household count `tracked[c]` (larger is preferred) | When Secondary also yields near-ties |

**Rationales**:

- **Primary** = the objective. Expected coverage is the mathematical goal and is
  uncontroversial given that objective.
- **Secondary** = risk aversion under prediction noise. If two counties look
  equally good in expectation but one has tight variance (e.g., 5130 ± 200)
  and the other is wide (5130 ± 1800), the tight one is preferred — it avoids
  the bad scenario where we deploy on a noisy forecast and coverage is
  actually far below expected. This is a standard mean-variance trade-off
  adapted from finance.
- **Tertiary** = robustness to unmodeled tail events. Large-population
  counties have more "upside" if an event turns out to be worse than predicted
  (demand can grow into the extra generator capacity). They are the safer
  fallback when Primary and Secondary both fail to discriminate.

### 4.2 Algorithm

```
Initialize k_c ← 0 for all c
For step = 1 to 5:
    For each county c:
        ΔE_c  = mean over ensemble of [coverage_c(k_c + 1) − coverage_c(k_c)]
        Δstd_c = std over ensemble of the same quantity
    top = max_c ΔE_c
    tied_1 = {c : ΔE_c ≥ (1 − tol) · top}                # tol = 1%
    if |tied_1| == 1:
        c* = the single county in tied_1
    else:
        min_std = min_c in tied_1 [Δstd_c]
        tied_2 = {c in tied_1 : Δstd_c ≤ (1 + tol) · min_std}
        if |tied_2| == 1:
            c* = the single county in tied_2
        else:
            c* = argmax_{c in tied_2} tracked[c]
    k_{c*} += 1
Return the list of FIPS codes (with repeats for counties that got ≥ 2 gens)
```

**Intuition**: at every step, ask "where does the next generator save the most
household-hours in expectation?" — if unambiguous, place it there. Otherwise
prefer the more stable forecast, and as a final backstop the larger county.
Recompute marginals after each placement, because after a county approaches
saturation its demand, additional generators there are worth less.

### 4.3 Alternative orderings (mentioned for completeness)

The hierarchy above is the **risk-neutral** default. Other reasonable defaults:

| Variant | Primary | Secondary | When preferred |
|---------|---------|-----------|----------------|
| **Risk-averse** | `E[Δ] − λ·std[Δ]`, λ ∈ {0.5, 1.0} | tracked | When consequences of under-coverage are severe (lawsuits, reputation) |
| **Fairness-constrained** | `E[Δ]` | Prefer a county not yet selected | When geographic spread is politically required |
| **Equity-weighted** | `w_c · E[Δ]` where `w_c` reflects vulnerability | std | When data on socioeconomic vulnerability is available |

For the final submission we report results under the risk-neutral default
(Primary → Secondary → Tertiary as above) and note that alternative objectives
would produce slightly different recommendations; see §8 for one worked
comparison.

---

## 5. Implementation

### 5.1 Greedy allocator with three-tier tiebreaking

Mirrors the algorithm in §4.2 exactly. The function takes an **ensemble** of
predictions (shape `(M, L, H)` for M forecasts — typically 6 = 3 seeds × 2
models) so it can compute both the mean marginal (Primary) and the standard
deviation of the marginal (Secondary) at every step.

```python
import numpy as np

def greedy_three_tier(
    pred_ensemble,        # (M, L, H) — M forecasts for L counties × H hours
    tracked,              # (L,)      — tracked households per county
    capacity=1000,
    n_gen=5,
    tol=0.01,             # 1% relative tolerance for declaring "tied"
):
    """
    Three-tier lexicographic greedy allocation:
      Primary   : argmax of E[Δ_c(k)]   (mean across ensemble)
      Secondary : min of std[Δ_c(k)]    (stability, lower preferred)
      Tertiary  : argmax of tracked[c]  (larger population hedges tails)
    Returns
    -------
    assign : (L,) int — generators placed per county
    trace  : list of dicts — one entry per generator, with selection metadata
    """
    pred_ensemble = np.asarray(pred_ensemble, dtype=np.float64)
    tracked = np.asarray(tracked, dtype=np.float64)
    M, L, H = pred_ensemble.shape
    assert tracked.shape == (L,), f"tracked shape {tracked.shape}, expected ({L},)"

    assign = np.zeros(L, dtype=int)
    trace = []

    for step in range(n_gen):
        # Marginal coverage under the current assignment, per ensemble member.
        k_arr = assign[None, :, None]                                # (1, L, 1)
        cov_now  = np.minimum(k_arr       * capacity, pred_ensemble).sum(axis=2)  # (M, L)
        cov_next = np.minimum((k_arr + 1) * capacity, pred_ensemble).sum(axis=2)  # (M, L)
        gains    = cov_next - cov_now                                # (M, L)

        gain_mean = gains.mean(axis=0)                               # (L,) Primary
        gain_std  = gains.std(axis=0)                                # (L,) Secondary

        # --- Tier 1: Primary (expected marginal coverage) ---
        top = gain_mean.max()
        tied_1 = np.where(gain_mean >= top * (1 - tol) - 1e-9)[0]

        if len(tied_1) == 1:
            best = int(tied_1[0]); tier = "primary"
        else:
            # --- Tier 2: Secondary (lowest std) ---
            stds = gain_std[tied_1]
            min_std = stds.min()
            tied_2 = tied_1[stds <= min_std * (1 + tol) + 1e-9]
            if len(tied_2) == 1:
                best = int(tied_2[0]); tier = "secondary"
            else:
                # --- Tier 3: Tertiary (largest tracked) ---
                best = int(tied_2[np.argmax(tracked[tied_2])]); tier = "tertiary"

        assign[best] += 1
        trace.append({
            "step": step + 1,
            "county_idx": best,
            "gain_mean": float(gain_mean[best]),
            "gain_std": float(gain_std[best]),
            "tracked": float(tracked[best]),
            "n_candidates_primary": len(tied_1),
            "tier_used": tier,
        })

    return assign, trace
```

### 5.1b Single-prediction fallback

If an ensemble is not available (e.g., you only retrained once with seed=42),
`M=1` is still valid — Secondary becomes degenerate (std=0 everywhere) and
Tertiary always takes over on ties. Pass a 3D array with a singleton first
axis: `pred_ensemble = pred_window[None, :, :]`.

### 5.2 Loading predictions and running

```python
import pandas as pd
import xarray as xr
import numpy as np
from pathlib import Path

# 1. Build the ensemble of test predictions: 3 seeds × 2 model families = 6.
#    (See run_experiments.ipynb Section 9 for the driver code that generates them.)
from src.runner import retrain_best_on_full_and_predict_test
stack = []
for model in ("xgboost", "lightgbm"):
    for seed in (42, 43, 44):
        p = retrain_best_on_full_and_predict_test(model, horizon=48, seed=seed)  # (H, L)
        stack.append(p.T)                                                         # (L, H)
pred_ensemble = np.stack(stack)     # shape (6, 83, 48)

# 2. Load tracked for tertiary tiebreak.
ds_train = xr.open_dataset("dataset/data/train.nc")
locations = [str(x) for x in ds_train.location.values]
tracked = ds_train.tracked.mean(dim="timestamp").values.astype(float)  # (83,)

# 3. Run three-tier greedy.
assign, trace = greedy_three_tier(pred_ensemble, tracked, capacity=1000, n_gen=5, tol=0.01)

# 4. Expand to FIPS list (with repeats).
fips_list = [locations[li] for li, k in enumerate(assign) for _ in range(int(k))]

print("Per-step trace:")
for t in trace:
    print(f"  #{t['step']} → {locations[t['county_idx']]} "
          f"[tier={t['tier_used']}]  gain={t['gain_mean']:.0f}±{t['gain_std']:.0f}  "
          f"tracked={t['tracked']:.0f}")
print(f"\nRecommended counties (with repeats): {fips_list}")
print(f"Expected total household-hours served: "
      f"{sum(t['gain_mean'] for t in trace):,.0f}")

# 5. Write submission file (required by PDF Section 4.3).
with open("results/submissions/recommended_counties.txt", "w") as f:
    f.write(f"[{', '.join(fips_list)}]\n")
```

Expected runtime: ~3 min on A100 (dominated by retraining 6 final models on
full training data, not the allocation itself).

---

## 6. Choice of horizon: 24h vs 48h

We produce recommendations from **both** horizons and use the comparison as a
robustness check.

| | 24h horizon | 48h horizon |
|---|---|---|
| Prediction accuracy | higher (tighter RMSE) | lower (errors compound) |
| Decision lead time | shorter (same-day ops) | longer (2-day planning) |
| Typical storm duration in Michigan | many events are 6–18h | multi-day events exist |

**Default primary decision: 48h** — matches "pre-positioning" language in the
problem statement and captures the full duration of longer events.

**Sensitivity check**: rerun with 24h predictions; if the recommended top-5
counties overlap ≥ 4/5, the decision is robust to horizon choice. If overlap
is ≤ 2/5, flag this in the report as a genuine uncertainty.

---

## 7. Robustness checks (required before submission)

### 7.1 Across models

Run the allocator separately using:
- xgboost 48h predictions → `assign_xgb`
- lightgbm 48h predictions → `assign_lgb`
- (Optional) SARIMAX 48h predictions → `assign_sarimax`

Counties that appear in **all three** lists are the robust picks. Counties that
only appear in one are contingent on that model's inductive biases.

### 7.2 Across seeds

For stochastic models (xgboost, lightgbm), we ran 3 seeds. Rerun the allocator
on each seed's prediction and report the **frequency** each county is selected
across the 3 × 2 = 6 (seed × model) runs. A county selected 6/6 times is a
high-confidence pick; one selected 1/6 is speculative.

### 7.3 Against a non-predictive baseline

Compare our recommendation against two "null" policies:
1. **Population-based**: top-5 counties by `tracked` (mean households tracked).
2. **Historical-based**: top-5 counties by total training-period outage hours.

If our model-driven recommendation *diverges* from these baselines, and if on
the val set it covers more household-hours, that's direct evidence that the
prediction model adds decision value beyond static demographics / history. This
is the **core narrative** of the policy section in the report.

---

## 8. Uncertainty-aware extension (optional, adds rigor)

Point predictions ignore that our RMSE is ~30 (24h) / ~62 (48h), meaning for any
given (county, hour) the true value can differ from our prediction by dozens.

### 8.1 Robust (conservative) allocator

Replace point predictions with a lower confidence bound:

```python
pred_conservative = pred_mean - ALPHA * pred_std
```

where `pred_mean` and `pred_std` come from averaging across seeds, and
`ALPHA ∈ {0.5, 1.0}`. This under-counts uncertain counties and favors counties
with stable predictions. Report results at two `ALPHA` values to show
sensitivity.

### 8.2 CVaR / worst-case (Conditional Value-at-Risk)

Instead of maximizing expected coverage, maximize coverage in the worst 10% of
scenarios. This requires bootstrapping predictions (easy if you have multiple
seeds). Good to mention in report as "future work" even if not implemented.

---

## 9. Limitations to acknowledge in the report

1. **Point predictions only** — no probability intervals, so we under-use
   uncertainty information. Addressed partially by Section 8.
2. **Equal generator deployment cost** — in reality moving a generator to a
   remote UP (Upper Peninsula) county costs more than to Detroit metro.
3. **No repair crew coordination** — generators are just one piece of event
   response; real policy integrates crew staging too.
4. **"Event" window is fixed** — we assume the 48h prediction window captures
   the event. In practice, an event could start before or end after.
5. **Objective is total households served** — but some loads (hospitals,
   dialysis patients) matter far more per capita. Tracked-weighted or
   critical-infrastructure weighted objectives would be a natural extension.

---

## 10. Deliverable format

Per the PDF (Section 4.3), submit a `.txt` file containing a Python-style list:

```
[26125, 26139, 26161, 26163, 26163]
```

Counties that received multiple generators appear multiple times. Order does not
matter; repetition count does.

---

## Appendix A — Michigan FIPS codes in top candidates

| FIPS | County | Region | Typical role |
|------|--------|--------|--------------|
| 26125 | Oakland | Detroit metro (N suburbs) | high pop, high outage |
| 26163 | Wayne | Detroit proper | highest pop in MI |
| 26099 | Macomb | Detroit metro (NE suburbs) | high pop, high outage |
| 26161 | Washtenaw | Ann Arbor | mid-pop, university town |
| 26139 | Ottawa | W. Michigan (Grand Rapids area) | mid-pop |
| 26145 | Saginaw | Mid-Michigan | mid-pop |

---

## Appendix B — Walkthrough of the greedy trace (oracle example)

Using training data last-48h as proxy predictions:

| Step | County picked | Marginal gain (hh-hrs) | Cumulative total | Running assignment |
|------|---------------|------------------------|------------------|-------------------|
| 1 | 26163 | 39,260 | 39,260 | 26163 ×1 |
| 2 | 26163 | 17,859 | 57,119 | 26163 ×2 |
| 3 | 26125 | 12,514 | 69,633 | 26163 ×2, 26125 ×1 |
| 4 | 26139 | 5,384 | 75,017 | + 26139 ×1 |
| 5 | 26161 | 5,130 | 80,147 | + 26161 ×1 |

Final: `[26163, 26163, 26125, 26139, 26161]`, covering 80,147 household-hours.

Observe the **marginal decreasing**: the 2nd generator at 26163 yields less
than the 1st (17,859 vs 39,260), which is exactly why greedy eventually moves
on from 26163 — the 3rd generator there would only yield ~3,000, less than the
first generator at 26139 (5,384).

---

*Methodology last updated: 2026-04-19. Corresponds to `run_experiments.ipynb`
commit producing `results/submissions/final_*_pred_48h.csv`.*
