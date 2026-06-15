# Monte Carlo + Correlation Summary
## Quant50 Survivor Sleeve — Decision Document

Generated: 2026-06-14  
Methods: block-bootstrap (N=10,000, L=50 days), Politis-Romano-style fixed-block circular resampling.  
Block bootstrap samples WHOLE ROWS of the aligned multi-strategy + SPY matrix, preserving
all cross-strategy and strategy-vs-benchmark correlation structure. Strategies are never
resampled independently.

**Critical caveat — printed on every result:** These simulations resample the observed
2000-2024 history. They cannot simulate factor crowding (AUM-driven decay), a prolonged
sideways regime absent from the sample, structural change in the equity risk premium, or
a correlation regime not seen in 25 years. Block bootstrap widens error bars honestly but
the central-tendency is still anchored to the observed past. Treat all numbers as conditional
on past-resembles-future.

---

## Task 1 Result: Do the Survivors Diversify?

**Short answer: barely. You have ~2 effective independent bets, not 7.**

Full-sample N_eff = **2.29 / 7**. PC1 explains **64.1%** of all strategy variance. The first
two PCs together explain 75.1%. There is one dominant factor — equity market direction —
and roughly one other factor.

Crash-conditional N_eff falls further to **1.93 / 7**. Diversification partially collapses
precisely when it is needed most, consistent with fair-weather diversification theory.

### Correlation highlights

| Pair (crash regime) | Full-sample | Crash | Delta |
|---------------------|-------------|-------|-------|
| S30 LowVol × S49 DollarR | +0.71 | +0.90 | **+0.19** |
| S46 RiskPar × S49 DollarR | +0.51 | +0.76 | **+0.25** |
| S46 RiskPar × S30 LowVol | +0.58 | +0.77 | **+0.19** |
| S35 SellMay × S49 DollarR | +0.56 | +0.78 | **+0.22** |

7 of 21 strategy pairs show crash correlation rising > 0.10. The worst offender is S49 (Dollar
Regime): it correlates at 0.87-0.90 with S30 and S31 in crashes because EEM (its risk-on leg)
sells off in USD-safe-haven events, perfectly timing a simultaneous position reduction in
the equity-beta cluster.

### Cluster structure (used for correlation-aware weighting)

- **Equity-beta cluster** (S08, S30, S31, S35, S49): 5 strategies that all converge in bear
  markets. Average within-cluster full-sample correlation: ~0.63. Lock up in crashes.
  Recommended total budget: 50% split equally → 10% each.
- **Multi-asset cluster** (S46, S02): genuinely lower correlation to the equity cluster (0.45-0.59).
  S46 holds bonds + real assets (TLT, GLD, DBC, VNQ) providing the only meaningful hedge.
  S02 trades a mix of futures + ETFs across asset classes. Budget: 50% → 25% each.

---

## Task 2 Result: Monte Carlo Distributions

### 25-Year Horizon (bootstrapped over full 2000-2024 sample)

| Portfolio | CAGR p5/med/p95 | Sharpe med | MDD worst-5% | MDD median | P(DD>30%) | P(DD>40%) | P(beat SPY) |
|-----------|-----------------|------------|--------------|------------|-----------|-----------|-------------|
| Core 3: Equal-wt | 4.2% / 8.3% / 12.3% | 0.68 | -44.7% | -29.8% | 48.7% | 12.1% | **68.6%** |
| Core 3: Inv-vol | 4.3% / 8.2% / 11.9% | 0.69 | -44.6% | -29.4% | 46.2% | 11.0% | **65.5%** |
| Core 3: Cluster-parity | 3.9% / 7.7% / 11.6% | 0.68 | -44.3% | -29.1% | 44.6% | 10.9% | **57.7%** |
| Full 7: Equal-wt | 4.7% / 8.7% / 12.7% | 0.73 | -42.2% | -28.5% | 40.6% | 7.7% | **77.2%** |
| Full 7: Inv-vol | 4.4% / 8.1% / 11.8% | 0.71 | **-40.8%** | **-27.3%** | **33.3%** | **5.9%** | 66.1% |
| Full 7: Cluster-parity | **4.9% / 9.2% / 13.4%** | **0.77** | -42.8% | -28.8% | 43.4% | 8.8% | **82.2%** |

### 5-Year Horizon

| Portfolio | CAGR p5/med/p95 | Sharpe med | MDD worst-5% | P(DD>30%) | P(beat SPY) |
|-----------|-----------------|------------|--------------|-----------|-------------|
| Core 3: Equal-wt | -0.6% / 8.5% / 16.9% | 0.71 | -34.0% | 10.4% | **58.1%** |
| Core 3: Inv-vol | -0.7% / 8.3% / 16.9% | 0.72 | -33.9% | 9.8% | **57.1%** |
| Core 3: Cluster-parity | -0.9% / 7.9% / 16.2% | 0.70 | -33.5% | 9.1% | **52.6%** |
| Full 7: Equal-wt | -0.2% / 8.8% / 17.7% | 0.74 | -32.8% | 8.2% | **62.4%** |
| Full 7: Inv-vol | -0.2% / 8.1% / 16.3% | 0.73 | **-31.2%** | **6.5%** | 56.8% |
| Full 7: Cluster-parity | 0.0% / 9.3% / 18.8% | **0.79** | -33.1% | 9.1% | **64.9%** |

### Block structure check

Near-zero probability of >40% drawdown would indicate the bootstrap failed to capture tail
events. **P(MDD>40%) is 6-12% for the 25-year horizon** — non-zero and realistic for a window
spanning GFC, COVID, and rate-hike cycles. The block structure correctly preserved the
correlated drawdown structure from sampled bear markets.

---

## Task 3 Result: Execution Fragility

All four tested strategies returned **STABLE** across all three perturbation dimensions.

| Strategy | Jitter max degradation | Cost 3x adj Sharpe | Param ±10% range | VERDICT |
|----------|----------------------|---------------------|------------------|---------|
| S02 TS-Mom | <1% across ±2d | 0.567 | 0.651 – 0.703 | **STABLE** |
| S31 VolTgt | <2% across ±2d | 0.569 | 0.829 – 0.839 | **STABLE** |
| S08 Sector (control) | <2% across ±2d | 0.657 | see robustness.py | **STABLE** |
| S46 RiskPar (control) | <0.4% across ±2d | 0.786 | n/a (single-param) | **STABLE** |

**Why jitter sensitivity is low for all four:** Monthly-rebalancing strategies (S08, S46) are
nearly immune to 1-2 day execution timing. Daily-signal strategies (S02, S31) use vol-scaled
positions and monthly rebalance execution, so a 1-2 day lag in a 63-day or 20-day lookback
window has negligible effect on the signal direction.

**Why parameter sensitivity is low for S31:** Vol-targeting at lookback 18-20-22 yields
Sharpe 0.829-0.839 — the mean-reversion in SPY vol is stable enough that the precise lookback
barely matters. The edge comes from being out of peak drawdown, not from lookback calibration.

**Why S46 is the most robust:** Turnover 0.54x/yr means 3x costs degrade Sharpe only from
0.806 to 0.786. Even at 10x costs it would remain viable. This is the cost-insensitive hedge.

---

## Plain-Language Answers

### 1. Do the survivors diversify?

Technically yes, meaningfully less than the count implies. Seven strategies yield 2.3 effective
independent bets at full-sample and 1.9 in crashes. You own one large equity-beta bet
(S08/S30/S31/S35/S49), one multi-asset bet (S46 bonds+real assets), and one cross-market
momentum bet (S02). Think of it as 3 buckets, not 7 strategies.

The diversification that exists — S46 vs the equity cluster — is real and captured in the
numbers: adding S46 to a pure S08/S30 portfolio reduces the worst-5% drawdown from ~-47%
to ~-44%. Not transformative, but measurable.

### 2. What is the realistic 25-year range and 95th-percentile drawdown?

**For the recommended portfolio (Full 7, Cluster-parity):**

- Median CAGR: **9.2%** (conditional on observed 2000-2024 regime)
- Central range (25th-75th pct): roughly 7-11% CAGR
- Wide range (5th-95th pct): **4.9% to 13.4%** CAGR
- Median max drawdown: **-28.8%** (you will likely see a -30% drawdown over 25 years)
- Worst 5% of paths max drawdown: **-42.8%** (this is what to size against)
- P(drawdown exceeds 40%): **8.8%** — roughly 1-in-11 chance over a 25-year horizon

Over any 5-year window: CAGR can be negative (5th pct = 0.0%), and you will likely see a
-18% median peak drawdown. The 5th-worst-5% path has a -33% drawdown inside 5 years.

**Size against -43%.** That is not a tail scenario — it is the 95th-percentile outcome for
a 25-year run of a strategy portfolio that was mostly long equities throughout GFC and COVID.

### 3. Do the maybe/narrow strategies earn their place?

**Yes.** Comparing Full 7 vs Core 3 under cluster-parity weighting:

| Metric | Core 3 | Full 7 | Improvement |
|--------|--------|--------|-------------|
| Median CAGR | 7.7% | 9.2% | +1.5 pp |
| Sharpe (median) | 0.68 | 0.77 | +0.09 |
| Worst-5% MDD | -44.3% | -42.8% | -1.5 pp better |
| P(beat SPY, 25yr) | 57.7% | 82.2% | +24.5 pp |

Adding S31, S35, S02, and S49 (with cluster-parity down-weighting of S49 and the equity-beta
cluster) meaningfully improves the central tendency and the odds of outperforming the index
over long horizons. The drawdown impact is small because you never over-weight any single
strategy.

S31 (Vol Targeting) deserves special mention: it contributes the highest OOS Sharpe (0.833)
and its parameter stability (vol_lookback 18-22 all yield 0.83+) is exceptional. Its IS
Sharpe of 0.297 was the concern — the OOS result now suggests the strategy's edge is
structural (smoothing compounded vol events) rather than overfitted to the IS period.

### 4. Is the active sleeve worth running over SPY?

**Over 25 years: yes, with high probability (82% for the best portfolio configuration).**
**Over 5 years: roughly a coin flip (58-65% depending on weighting).**

The 5-year number is the honest benchmark for any given performance review cycle. Over 5
years, the active sleeve underperforms in roughly 35-42% of scenarios. Anyone evaluating
this sleeve on a 1-3 year horizon will regularly see underperformance — this is expected
and should not trigger strategy changes.

Over the full 25-year window, the active sleeve adds roughly 4% annualized median CAGR vs SPY
(the bootstrapped SPY median is ~5-6% CAGR over the 2000-2024 window, which includes two
major bear markets). But this edge is entirely conditional on the historical equity risk
premium, factor premia, and regime composition repeating.

---

## Recommended Portfolio Configuration

**Full 7 strategies, Cluster-parity weighting:**

| Strategy | Weight | Cluster |
|----------|--------|---------|
| S46 Risk Parity | 25% | Multi-asset (bond + real assets) |
| S02 TS Momentum | 25% | Multi-asset (cross-market futures/ETF) |
| S08 Sector Rotation | 10% | Equity-beta |
| S30 Low Volatility (LF) | 10% | Equity-beta |
| S31 Vol Targeting | 10% | Equity-beta |
| S35 Sell in May | 10% | Equity-beta |
| S49 Dollar Regime | 10% | Equity-beta |

**Why this allocation:**
1. S46 and S02 together (50%) provide the only genuine cross-asset diversification. S46 is the
   defensive hedge; S02 has the lowest crash-regime correlation with the equity cluster.
2. The equity-beta cluster (50%) is spread across 5 strategies, none above 10%, preventing
   any single strategy from dominating drawdown.
3. Cluster-parity beats equal-weight on every return metric and beats inverse-vol on CAGR
   and P(outperform SPY) — making it the dominant choice for long-horizon investors who can
   tolerate the slightly higher MDD vs inverse-vol.

**Alternative for drawdown-sensitive investors:** Full 7, Inverse-vol weighting.
Reduces P(MDD>40%) from 8.8% to 5.9%, worst-5% MDD from -42.8% to -40.8%. Costs: lower
median CAGR (8.1% vs 9.2%) and lower P(beat SPY) (66% vs 82%).

---

## What This Analysis Cannot Tell You

1. **Factor crowding.** As assets flow into momentum, low-vol, and risk-parity strategies,
   the edge compresses. The bootstrap resamples the pre-crowd era. The OOS performance is
   already partially post-crowd (2017-2024), which is why OOS improvements in S08 and S46
   are encouraging — but there is no guarantee this continues.

2. **Structural breaks.** A prolonged low-vol, trending-up market (like 1995-1999) absent
   from the bootstrap sample would produce median outcomes not represented in the fan chart.
   Similarly, a Japan-style 30-year stagnation would collapse the equity-beta cluster entirely.

3. **Correlation regime shifts.** The bond-equity correlation went positive in 2022 for the
   first time in two decades. S46's TLT holding stopped hedging during the rate-hike cycle.
   If the positive bond-equity regime persists, N_eff could fall further to near 1.0.

4. **Implementation costs at scale.** The cost model assumes 10 bps one-way. Above ~$5-10M
   AUM for S08 (monthly turnover in 3 ETFs) and ~$2-5M for S02 (monthly multi-instrument),
   market impact will exceed this assumption.

5. **Rebalancing drag not captured.** The parquet-cached returns are gross of rebalancing
   friction within multi-asset portfolios. At the portfolio-of-strategies level, monthly
   rebalancing back to target weights adds additional costs not modeled here.

---

## Related Files

| File | Content |
|------|---------|
| `results/is_oos_table.csv` | 50-strategy IS/OOS table with ROBUST/DECAY/MIRAGE/FRAGILE/WEAK/STUB flags |
| `results/quarantine.md` | S07/S26/S45 exclusion documentation |
| `results/robustness_report.md` | Cost stress, 12-pt parameter grids, regime breakdown |
| `results/DECISION_SUMMARY.md` | Final per-strategy deploy verdicts |
| `results/corr_full.png` | Full-sample 7x7 correlation heatmap |
| `results/corr_crash.png` | Crash-conditional correlation heatmap |
| `results/corr_rolling.png` | Rolling 252-day avg pairwise correlation (2000-2024) |
| `results/corr_eigenvalues.png` | Eigenvalue decomposition + N_eff visualization |
| `results/mc_fan_*.png` | Fan charts (12 total: 6 portfolio specs × 25yr + 5yr) |
| `results/mc_tables.md` | Full percentile tables for all 6 portfolio specs |
| `results/perturbation_report.md` | Jitter/cost/parameter fragility detail |
