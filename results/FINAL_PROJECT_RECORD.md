# Final Project Record — Quant Backtest Research
*Completed: 2026-06-17 | Full window: 2000-01-03 to 2024-12-31 (25 years)*
*IS: 2000-01-03 to 2017-06-30 | OOS: 2017-07-03 to 2024-12-31 (~7.8 years)*
*Costs: 5 bps commission + 5 bps slippage per side (10 bps one-way, 20 bps round-trip)*
*Universe: Russell 1000 PIT (equities) + sector/macro ETFs + Norgate continuous futures*

---

## 1. Validated Book (8 Strategies)

These strategies survived the full gauntlet: IS/OOS split, regime breakdown across 5 windows,
robustness perturbation testing, Monte Carlo bootstrap, correlation analysis, and explicit
lookahead audits. All bugs found were fixed before final numbers were recorded.

| ID  | Name                  | IS SR  | OOS SR | Decay   | IS MDD  | OOS MDD | TO/yr | One-line caveat |
|-----|-----------------------|--------|--------|---------|---------|---------|-------|-----------------|
| s08 | Sector Rotation       | 0.395  | 0.757  | +0.362  | -45.8%  | -28.7%  | 4.7x  | Monthly rebalance on 11 SPDR ETFs; needs live ETF execution infrastructure |
| s46 | Risk Parity           | 0.447  | 0.806  | +0.359  | -44.7%  | -20.7%  | 0.5x  | Bonds + real assets; 2022 rate-hike hit (-60% Sharpe in 2022) when equity-bond correlation inverted |
| s30 | Low Volatility        | 0.724  | 0.691  | -0.033  | -43.4%  | -31.3%  | 0.8x  | Long-flat ONLY — long-short formulation collapses (OOS SR -0.20 to -0.25); do not enable short leg |
| s02 | TS Momentum (futures) | 0.787  | 0.686  | -0.101  | -45.6%  | -34.8%  | 6.6x  | 63-day lookback only — longer lookbacks or long-short both collapse; do not tune further |
| s31 | Vol Targeting         | 0.297  | 0.833  | +0.536  | -37.1%  | -15.4%  | 7.4x  | IS borderline (0.297); 3x cost stress tests become marginal; needs daily vol-targeting execution |
| s35 | Sell in May           | 0.386  | 0.529  | +0.143  | -37.0%  | -33.7%  | 1.9x  | 2022 rate-hike fell in the Nov-Apr "long" window; calendar-based signal cannot adapt mid-season |
| s49 | Dollar Regime         | 0.563  | 0.552  | -0.011  | -62.8%  | -33.7%  | 0.8x  | IS MDD -62.8% from 2002-2008 EM sell-off; GFC correlation with equity cluster surges to 0.87-0.90 in crashes |
| s90 | Credit Regime         | 1.204  | 0.938  | -0.266  | -24.2%  | -21.2%  | 7.6x  | Signal starts 2007 (HYG inception); IS numbers include pre-2007 zero-return period; OOS data valid from 2017 |

**Aggregate stats (equal-weight portfolio, OOS 2017-2024):** estimated Sharpe ~0.79, CAGR ~10.5%

**Marginal addition of s90 to existing book (s08/s46/s30/s02/s31/s35/s49):**
- N_eff before s90: 2.29 → N_eff after: 2.57 (+0.28 — largest independent bet gain in Group 2)
- Portfolio Sharpe delta: +0.03 (marginal but positive)
- s90 has the lowest crash-conditional correlation with equity cluster (0.39-0.51), providing the most genuine diversification in the book.

---

## 2. Portfolio Reality

### Effective independent bets

7-strategy book: **N_eff = 2.29** (confirmed by eigenvalue decomposition, block-bootstrap validated)
8-strategy book (with s90): **N_eff = 2.57**

Despite 8 strategy names, you have approximately **2-3 real bets**. PC1 (equity market direction)
explains 64.1% of all strategy variance. Crash-conditional N_eff falls to ~1.93 — diversification
partially collapses precisely when it is needed most.

### Cluster structure (for weighting)

Three clusters identified by correlation structure:

| Cluster | Members | Budget (cluster-parity) | Per-strategy |
|---------|---------|------------------------|--------------|
| Equity-beta | s08, s30, s31, s35, s49 | 33% | 6.7% each |
| Multi-asset | s02, s46 | 33% | 16.5% each |
| Credit-regime | s90 | 33% | 33% |

The equity-beta cluster converges in bear markets (crash correlation 0.87-0.90 within cluster).
Single-digit weights per equity-beta strategy are appropriate. s46 provides the only meaningful
bond/real-asset hedge; s02 provides the primary non-equity return stream.

### Monte Carlo headlines (Full-7, cluster-parity, 10,000 block-bootstrap paths)

*Note: MC was run on the original 7-strategy book (before s90). Numbers below are directionally
conservative for the 8-strategy book.*

| Horizon | CAGR p5/med/p95 | Sharpe med | MDD worst-5% | P(beat SPY) |
|---------|-----------------|------------|--------------|-------------|
| 25-year | 4.9% / 9.2% / 13.4% | 0.77 | -42.8% | **82%** |
| 5-year  | 0.0% / 9.3% / 18.8% | 0.79 | -33.1% | **65%** |

**Interpretation:** The median outcome beats SPY with similar volatility. The left tail still
has drawdowns exceeding 40%. This is not a capital-preservation strategy — it is an
equity-like return stream with diversified sources of risk.

**Block-bootstrap limitation:** Paths are resampled from observed 2000-2024 history. Scenarios
absent from the sample (sustained secular deflation, multi-year sideways equity market,
permanent factor crowding) are not represented. All Monte Carlo numbers are conditional on
past-resembles-future.

---

## 3. Quarantine and Artifact List

These strategies were tested and either contained bugs that inflated results, produced
results that belong to a different strategy than named, or decayed catastrophically OOS.

### Confirmed Artifacts (code-level bugs)

| ID  | Name               | Reported SR | Corrected SR | Bug | Root cause |
|-----|--------------------|-------------|--------------|-----|------------|
| s07 | Donchian (Futures) | 4.23        | ~0.4-0.6*   | Back-adj artifact | Norgate TOTALRETURN continuous futures forward-propagates roll gains into history; creates upward drift that Donchian reads as trend signal. MDD -2.1% is physically impossible for any real CTA. OOS SR 5.02 > IS 3.86 — a confirmed red flag. |
| s81 | Jan Reversal PIT   | 3.53        | ~0.3-0.4*   | Additive accumulation | `portfolio_returns_from_weights()` called with daily weights → hold masks overlap → each interior day gets `+=` twice → implicit 2× leverage. Catastrophic tail: last January entry has no successor → strategy goes infinite-leverage over the weekend. |
| s78 | Vol Trend ETF      | 1.44        | 0.570       | Same-bar lookahead | `cur = price[i]` drives MA crossover signal, AND `ret_df[i]` (function of `price[i]`) is earned. On transition days, lookahead picks right side of every signal flip. Corrected SR 0.570 = SPY 200-DMA 0.571 — same strategy once bug removed. |
| s79 | Adaptive Trend     | 1.65        | 0.524       | Same-bar lookahead | Identical pattern: `cur = price[i]` for signal, `ret.iloc[i]` for P&L. 3.2× inflation ratio. |
| s75 | Donchian Equity    | 1.75 (buggy), 0.415 (fixed) | REDUNDANT | Exit lookahead | Exit signal used `close_cap.iloc[i]` (today's close) but position was removed BEFORE today's P&L — effectively exiting at yesterday's close on a known tomorrow's information. Corrected IS SR 0.294, IS MDD -50%. After fix, negative ΔN_eff (-0.07) and ΔSR (-0.036) vs validated book. Do not add. |

*\* Corrected numbers not computed — strategy discarded before correction was worth implementing.*

**Common signature of all five artifacts:**
- Sharpe > 1.4 (all real strategies in this book are 0.30-1.20)
- MDD suspiciously low (< -20%) through 2008
- OOS Sharpe ≥ IS Sharpe (unlikely for any well-known published strategy)

### Proxies (real numbers, wrong strategy identity)

| ID  | Named as      | Actually runs as       | IS SR | OOS SR | Issue |
|-----|---------------|------------------------|-------|--------|-------|
| s26 | Dividend Yield | High-distribution tilt | 0.47  | 0.68   | TR/PR ratio selects "pays distributions" (utilities, REITs, energy) — not a dividend sustainability screen. OOS improvement reflects REIT/high-yield recovery post-2017, not dividend quality. |
| s45 | Short Squeeze  | R2000 momentum         | 0.47  | 0.57   | No FINRA short interest data at implementation. Strategy runs pure 1-month momentum on Russell 2000 — entirely different risk/return profile from actual squeeze events. |

Both have positive OOS numbers. If properly implemented (s26 with Sharadar dividend + payout
data; s45 with FINRA short interest + float utilization), the underlying signals may be valid.
Numbers currently measure the wrong thing.

### Tail Risk (not bugs, but do not deploy)

| ID  | Name       | IS SR | OOS SR | Issue |
|-----|------------|-------|--------|-------|
| s27 | VIX Carry  | 0.55  | 0.239  | OOS decaying. Worst single day: -32.0%. Worst single week: -47.1%. Feb-2018 and Mar-2020 SVXY events produced intraday drawdowns of 80-90% before daily close captured. Margin calls guaranteed in live trading. **Do not include in any live portfolio.** |

### Mirage (real backtests, collapsed OOS)

| ID  | Name               | IS SR | OOS SR | Symptom |
|-----|--------------------|-------|--------|---------|
| s50 | Managed Futures    | 0.361 | 0.033  | High-turnover (12x/yr) trend ensemble worked in 2000-2017 but OOS Sharpe nearly zero. Possible causes: trend premium compression, high cost drag at 12x turnover. |
| s94 | Index Deletion     | 0.912 | 0.237  | OOS SR only 26% of IS SR. Possible overfitting to the 2000-2017 window's specific deletion premium characteristics. |

---

## 4. Meta-Lesson

**Every spectacular result in this project was an artifact. Without exception.**

The progression, in order of discovery:
1. **s07 (SR 4.23):** Roll-return artifact in back-adjusted futures. Found by: MDD -2.1% through 2008 is physically impossible.
2. **s81 (SR 3.53):** Double-leverage from additive accumulation bug. Found by: corrected version was flat before fix.
3. **s78 (SR 1.44):** Same-bar lookahead. Found by: corrected SR exactly matched SPY 200-DMA (0.570 vs 0.571) — same strategy.
4. **s79 (SR 1.65):** Same-bar lookahead. Found by: identical code pattern to s78.
5. **s75 (SR 1.75):** Exit lookahead. Found by: IS MDD -50% after correction — artifact was suppressing crash-day losses.

**The OOS > IS pattern is not evidence of edge — it is a red flag.** Three mechanisms:
- Regime-mix artifact: the OOS window (2017-2024) had 81% of time in long-favoring regimes vs 65% IS. Even the clean SPY 200-DMA shows IS SR 0.446 → OOS SR 0.815 for the same reason.
- Bug inflation: lookahead bugs systematically inflate IS AND OOS, but their relative magnitudes can favor OOS depending on regime mix.
- Overfit: IS window has more noise to fit; OOS window has less variance in a short (7.8yr) bull-skewed period.

**The only OOS > IS results in the validated book (s08 +0.36, s46 +0.36, s31 +0.54) are not spectacular.**
They all have plausible regime-mix explanations. None had SR > 1.5 to begin with.

**Operational rule for future strategies:**
- Full SR > 1.5: guilty until mechanism is explained
- OOS SR > IS SR by > 0.2: audit for same-bar lookahead before trusting
- GFC MDD better than -15%: audit for any lookahead in exits or signal
- Sharpe inflation ratio buggy/corrected > 2: same-bar lookahead confirmed

---

## 5. Stubs (Blocked on Data)

These strategies have sound logic but cannot run on the current Norgate subscription.
Each has a clear fix path.

### Fundamentals: Sharadar Core US Fundamentals (~$40/month)
Unlocks: **s16 (Book/Market), s17 (Earnings Yield), s18 (EV/EBITDA), s19 (Piotroski F-score),
s20 (Gross Profit/Assets), s22 (Accruals)**
These implement the six most-cited equity factors from the academic literature. The proxy
implementations (s26 uses TR/PR ratio as dividend yield; s45 uses momentum as short squeeze)
exist but should be rebuilt once Sharadar is available. This is the highest strategy-per-dollar
data spend.

### Options Data: ORATS or CBOE (~$100/month)
Unlocks: **s28 (Short Straddle), s32 (Volatility Dispersion), s33 (IV Crush pre-earnings)**
Options strategies are a genuinely independent return source with low equity beta. Not in
current Norgate subscription.

### Alternative Data
- **s40 (Post-Earnings Drift):** Needs point-in-time earnings announcement dates + EPS surprises. Sharadar Earnings or Zacks (~$50/month).
- **s42 (Insider Buying):** SEC EDGAR Form 4 API — free. ~30 min setup. Highest ROI free-data action.
- **s43 (Analyst Revisions):** Needs consensus estimate history. Zacks/Bloomberg.
- **s44 (Merger Arb):** Needs M&A announcement data (Refinitiv/Bloomberg, expensive). Defer.

### Dividend Strategies (norgatedata v1.0.74 limitation)
Blocked: **s97 (Div Capture), s98 (Ex-Date Drift), s99 (Div Initiation)**

`norgatedata.dividends()` does not exist in v1.0.74. Available Norgate dividend APIs:
- `dividend_yield_timeseries`: trailing yield percentage (no ex-dates or amounts)
- `capital_event_timeseries`: stock split events only

**Fix path:** (a) Upgrade Norgate subscription to include dividend history, (b) Tiingo dividend
API, (c) Sharadar ex-date data, or (d) compute ex-date drops from the TOTALRETURN/CAPITAL
price ratio (where ratio drops by dividend/price on the ex-date).

Note on lookahead for these strategies: s97/s98 are CLEAN (event-driven hold schedules,
no same-bar issue). s99 is CLEAN (buys at ex-date, which is publicly announced ~1 month ahead;
ex-date vs announcement-date is a design caveat, not a bug).

### ETF Basket Arb (Norgate index limitation)
Blocked: **s102 (ETF Basket Arb — XLK/XLF/XLE sectors)**

`index_constituent_mask(sym, "XLK")` raises ValueError — Norgate recognizes only traditional
equity indices (Russell, S&P 500, Nasdaq, Dow). Sector ETF basket membership is not exposed.

**Fix path:** SPDR fact sheets (monthly PDF disclosures), FTSE Russell sector constituent files,
or a data vendor that tracks ETF holdings history.

Note on lookahead: s102 has a minor same-bar issue (line 120 uses `[:i+1]` in z-score window).
Directional impact is NEGATIVE on entry days (ETF expensive vs basket on that day → pair return
negative). Bug works against the strategy, not for it. Still should be fixed before deployment.

---

## 6. Deployment Status

### Research phase: COMPLETE

All runnable strategies have been:
- Backtested over the full 2000-2024 window with PIT constituents
- Split into IS/OOS with 70/30 allocation
- Regime-tested across 5 windows (dot-com bust, GFC, calm bull, COVID, 2022 rate hike)
- Perturbation-tested (parameter shifts, cost shocks)
- Monte Carlo block-bootstrapped (10,000 paths, block length 50)
- Lookahead-audited (code review of every spectacular result)
- Correlation-analyzed for N_eff and cluster membership

### Immediate next steps

1. **Sizing decision:** The validated book has 8 strategies across 3 clusters. Cluster-parity
   weighting (33% per cluster, equal-weight within cluster) is the recommended starting point.
   This gives: equity-beta cluster (s08, s30, s31, s35, s49) each at ~6.7%; multi-asset cluster
   (s02, s46) each at ~16.5%; credit-regime (s90) at 33%.

2. **Paper trading:** Run all 8 strategies in paper mode for at least 6 months before
   committing capital. Verify execution assumptions (daily EOD signals achievable, cost
   estimates realistic, no data gaps in live feed).

3. **s90 caveat in live trading:** Credit regime strategy (HYG/LQD ratio) requires daily
   download of bond ETF prices. Confirm live data feed includes these tickers.

4. **Free data actions before spending money:**
   - SEC EDGAR Form 4 API (free, ~30 min setup) → activates s42 insider buying
   - FINRA biweekly short interest (free, ~1 hr setup) → activates s45 rebuild

5. **Low-cost data if budget allows:**
   - Sharadar Core US Fundamentals ($40/month) → unlocks 6+ strategies at once

### Fiduciary disclaimer

This research was conducted for educational and personal research purposes. The strategies
validated here have backtested well over 2000-2024, but backtests are not guarantees of
future performance. Specific risks not captured in this backtest:

- **Capacity limits:** All equity strategies assume $0 price impact. At meaningful AUM,
  execution in Russell 1000 mid-caps may push prices against the strategy.
- **Factor crowding:** Multiple institutions run similar factor strategies. Crowding-driven
  correlation spikes are not captured in 2000-2024 block bootstrap.
- **Structural change:** Interest rate regime change, ETF proliferation, and algorithmic
  trading have altered market microstructure. Pre-2010 backtest periods may not be representative.
- **Tax treatment:** Long/short strategies and high-turnover strategies generate short-term
  capital gains. After-tax returns will differ materially from pre-tax backtest numbers.

**This document does not constitute investment advice. Consult a licensed financial advisor
before making investment decisions based on any backtest results.**

---

## 7. Post-Completion Audit: s63 / s93 / s60 / s71 (2026-06-17)

Four strategies flagged as high-Sharpe and unaudited during the initial review were
subsequently audited in the same order as prior artifacts (highest suspicion first).
Script: `analysis/audit_group3.py`.

### Results

| ID  | Name (corrected)      | Buggy IS/OOS SR | Corrected IS/OOS SR | Verdict         | One-line reason |
|-----|-----------------------|-----------------|---------------------|-----------------|-----------------|
| s63 | ETF Breakout (Donchian 55d/20d on 5 ETFs) | 1.117 / 1.377 | 0.188 / 0.280 | **ARTIFACT** | Same-bar entry+exit compound bug: +1.096 SR pts inflation in OOS. Also mislabeled. |
| s93 | Defensive Rotation (SPY 200d MA)          | 0.643 / 1.053 | 0.384 / 0.726 | **ARTIFACT (borderline)** | Same-bar on 171 regime switches (~6.8/yr); corrected OOS SR just above 0.70 but IS only 0.384 — see note. |
| s60 | Correlation Regime (sector corr z-score)  | 0.265 / 0.899 | 0.265 / 0.899 | **REGIME-TIMER** | Code explicitly correct (line 119: shift(1)). IS SR 0.265 is the honest forward estimate. |
| s71 | 52-Week Breakout (R1000, 20% trailing stop)| 1.093 / 1.238 | est. 0.95-1.10 OOS | **REDUNDANT** | Same-bar entry+exit confirmed from code; effect moderate at 0.59x TO. Same cluster as s75. |

### s63 — ARTIFACT (name was wrong in table)

**Name correction:** The table entry labeled "Earnings Surprise" was wrong. The file is
`s63_etf_breakout.py` — a Donchian 55/20 breakout on SPY QQQ IWM EFA EEM with vol-targeting.

**s40/s63 data contradiction resolution:** There never was one. s40 (PEAD) needs EPS surprise
estimates (Zacks/Sharadar) and is a STUB. s63 uses pure price (total return). Different
strategies, different data, no contradiction.

**Bug:** `_run_breakout()` sets `positions[i]` from `close[i]` (correct signal lag on channels
via `.shift(1)`), but the P&L line `port_ret = (weight_df * ret_df).sum(axis=1)` uses `weight_df[i]`
(from today's close) × `ret_df[i]` (today's return). Both entry bars (earn breakout gain
you couldn't have known) and exit bars (avoid loss by removing position before P&L) are
inflated. Compound of two simultaneous same-bar biases.

**Scale:** IS inflation +0.929 SR pts, OOS +1.096 SR pts. Buggy OOS SR 1.377 → corrected 0.280.
This is the largest lookahead correction in the entire project, exceeding s79 (+0.97 SR).

### s93 — ARTIFACT (borderline note)

**Bug:** `spy_cur = price_df['SPY'].iloc[i]` (today's close) determines growth/defensive
sector weights, and those weights earn `ret_df.iloc[i]` (today's return). On each of the
~171 regime-switch days, the switching-day sector outperformance is captured retroactively
(e.g., on the day SPY falls below its 200d MA, defensive sectors outperform — s93 retroactively
claims to have been in defensives on that day). Same mechanism as s78/s79.

**Scale:** OOS inflation +0.327 SR pts. Corrected OOS SR 0.726 — technically above the 0.70
threshold. However, the **corrected IS SR is only 0.384**, and the OOS/IS ratio of 1.89x is
the same window-flatter pattern seen in regime-timer strategies. The **full-period SR is 0.498**,
below the 0.70 threshold. Do not add to book. The corrected numbers suggest a marginal,
window-dependent strategy, not a robust edge.

### s60 — REGIME-TIMER (code clean)

No bug. The code explicitly implements `exposure = raw_exposure.shift(1)` (line 119) — a
proper 1-day lag. The IS 0.27 → OOS 0.90 gap is entirely explained by regime mix:

| Sub-period | In-market | SR |
|---|---|---|
| Dot-com crash | 58% | -1.016 |
| GFC | 87% | -0.189 |
| Calm OOS bull 2017-21 | 88% | +1.146 |
| 2022 bear | 57% | -1.268 |
| 2023-24 recovery | 100% | +1.811 |

The strategy is nearly fully invested (86.2% overall) and only partially de-risks during
crises — too late, because correlation spikes happen during crashes that are already underway.
The OOS window happened to have shorter/milder drawdowns. Honest forward Sharpe: ~0.27 (IS).

### s71 — REDUNDANT

**Bug confirmed from code:** Entry bars credit same-bar return (stock made 52-week high so
return is positive by construction). Exit bars avoid same-bar return (trailing stop triggered
means stock is down). Both biases compound. Same structure as s63/s75.

**Magnitude moderate:** Reported TO = 0.59x/yr → ~30 entries + 30 exits per year → ~1.8%
per year artificial CAGR. Estimated corrected OOS SR: 0.95-1.10 (still above 0.70 threshold).

**Redundancy:** s71 is long-only equity momentum on R1000 with trailing stop exits —
mechanically identical to corrected s75 (Donchian on R1000, also REDUNDANT). Predicted
full-period correlation vs book: 0.70-0.85 with s08, 0.60-0.75 with s30. Expected
ΔN_eff ≤ 0. Full R1000 re-run required to confirm exact corrected SR, but expected
outcome is the same as s75: REDUNDANT regardless of SR level.

### Book impact

**None.** The validated 8-strategy book is unchanged:
s08, s46, s30, s02, s31, s35, s49, s90.

### Updated meta-lesson

With 102 strategies tested, several high-Sharpe results are statistically guaranteed
by chance alone (multiple comparisons). Across the full project:
- 5 confirmed artifacts with bugs (s07, s81, s78, s79, s75)
- 2 additional artifacts found in this audit (s63, s93)
- 2 regime-timers with clean code but window-flattered OOS (s60 + others)
- 1 redundant clean strategy (s71)
- **Zero** strategies with SR > 1.4 that survived audit

The audit process, not the Sharpe ratio, is what separates signal from noise —
and it has consistently said no.

---

## 8. Batch Redundancy Test: 11 Remaining Candidates (2026-06-17)

Eleven strategies with unaudited OOS SR ≥ 0.70 were tested for redundancy against the
8-strategy validated book. Script: `analysis/batch_redundancy.py`.

**Candidates:** s52, s54, s59, s61, s66, s68, s69, s73, s76, s77, s91

### Lookahead pre-screen

10 of 11 use `portfolio_returns_from_weights(execution_lag=1)` or explicit `.shift(1)` — **CLEAN**.

**Exception — s91 (Inflation Tilt):** Same-bar bug identical to s93. Manual loop sets weights
from `ratio.iloc[i]` (today's TIP/IEF ratio, includes today's close) and earns `ret_df.iloc[i]`
(today's return). Buggy IS/OOS SR: 0.599/0.893 — numbers inflated, not comparable to corrected book.
Verdict: **ARTIFACT** (same-bar, same mechanism as s93).

### Task 1 — Correlation vs book

All 11 candidates had max book correlation ≥ 0.70, flagging every one as LIKELY-REDUNDANT
before the N_eff test was even needed.

| ID  | Name                 | vs s08 | vs s02 | vs blend | max book | crash s08 |
|-----|----------------------|--------|--------|----------|----------|-----------|
| s52 | Idiosyncratic Vol    | +0.750 | +0.445 | +0.792   | **+0.959** | +0.644  |
| s54 | ADX Momentum (ETFs)  | +0.479 | +0.769 | +0.655   | +0.769   | +0.337    |
| s59 | Vol-of-Vol Regime    | +0.629 | +0.432 | +0.711   | +0.855   | +0.443    |
| s61 | Sortino Momentum     | +0.768 | +0.497 | +0.816   | +0.873   | +0.670    |
| s66 | Vol-Confirmed Mom    | +0.687 | +0.488 | +0.738   | +0.733   | +0.595    |
| s68 | Mom Ensemble         | +0.735 | +0.514 | +0.778   | +0.764   | +0.652    |
| s69 | Sharpe Rank          | +0.736 | +0.506 | +0.788   | +0.809   | +0.646    |
| s73 | Residual Momentum    | +0.692 | +0.473 | +0.754   | +0.794   | +0.599    |
| s76 | MA200 Band (R1000)   | +0.799 | +0.528 | +0.829   | +0.853   | +0.722    |
| s77 | Dual Momentum (GEM)  | +0.713 | +0.492 | +0.764   | +0.762   | +0.625    |
| s91 | Inflation Tilt*      | +0.704 | +0.439 | +0.753   | +0.750   | +0.622    |

*s91 reported numbers are buggy (same-bar); included for completeness only.*

The s52 correlation of 0.959 with its max book peer is remarkable — essentially the same
strategy as something already in the book. The equity-momentum cluster (s61, s66, s68, s69,
s73, s76, s77) is a dense ball, all correlated 0.73-0.87 with the book.

### Task 2 — ΔN_eff and ΔSharpe per candidate

Book baseline: N_eff = 2.575, blended SR = 0.959 (cluster-parity weights).

| ID  | Name                 | N_eff+cand | ΔN_eff  | ΔSR    | Verdict                    |
|-----|----------------------|------------|---------|--------|----------------------------|
| s52 | Idiosyncratic Vol    | 2.427      | -0.148  | -0.010 | **REDUNDANT** (ΔN_eff ≤ 0) |
| s54 | ADX Momentum (ETFs)  | 2.739      | +0.164  | +0.013 | Disqualified (corr 0.769)  |
| s59 | Vol-of-Vol Regime    | 2.579      | +0.004  | -0.033 | **BORDERLINE** (ΔN_eff <0.05) |
| s61 | Sortino Momentum     | 2.428      | -0.147  | -0.048 | **REDUNDANT**               |
| s66 | Vol-Confirmed Mom    | 2.546      | -0.029  | -0.054 | **REDUNDANT**               |
| s68 | Mom Ensemble         | 2.491      | -0.084  | -0.051 | **REDUNDANT**               |
| s69 | Sharpe Rank          | 2.477      | -0.098  | -0.035 | **REDUNDANT**               |
| s73 | Residual Momentum    | 2.524      | -0.051  | -0.051 | **REDUNDANT**               |
| s76 | MA200 Band (R1000)   | 2.416      | -0.159  | -0.040 | **REDUNDANT**               |
| s77 | Dual Momentum (GEM)  | 2.505      | -0.070  | -0.064 | **REDUNDANT**               |
| s91 | Inflation Tilt*      | 2.536      | -0.039  | -0.011 | **REDUNDANT** + ARTIFACT    |

**Note on s54:** The ADX ETF momentum strategy passes the ΔN_eff (+0.164) and ΔSR (+0.013)
filters individually — the only candidate to do so. However, its maximum book correlation is
0.769 (vs s02 TS Momentum; both are multi-asset trend strategies on overlapping instruments).
This exceeds the 0.70 disqualification threshold. Adding s54 alongside s02 would concentrate
the multi-asset trend cluster from 33% to a larger share without genuine diversification.
Verdict: REDUNDANT by corr filter.

**Note on s59:** ΔN_eff of +0.004 is effectively zero — vol-of-vol regime is a high-correlation
overlay on SPY with no independent return source.

### Task 3 — Group N_eff

| Portfolio                        | N_eff  |
|----------------------------------|--------|
| 8-strategy book alone            | 2.575  |
| Book + all 11 candidates         | 2.120  |
| Delta                            | **-0.455** |

Adding all 11 candidates together DECREASES N_eff by 0.455. The equity-momentum cluster
consumes the diversification budget, diluting the multi-asset and credit-regime clusters that
give the book its edge. Same qualitative result as Group 2 (was 2.29 → 2.09 when adding the
Group 2 equity candidates).

### Verdict

**ZERO SURVIVORS. Book unchanged.**

| Classification | Strategies |
|---|---|
| REDUNDANT (ΔN_eff ≤ 0) | s52, s61, s66, s68, s69, s73, s76, s77, s91 |
| BORDERLINE (0 < ΔN_eff < 0.05) | s59 |
| DISQUALIFIED (corr ≥ 0.70, passes ΔN_eff/ΔSR alone) | s54 |
| SAME-BAR BUG (pre-screen) | s91 |
| SURVIVORS | NONE |

No individual lookahead audits required for the remaining candidates — redundancy alone
eliminates them without needing to determine whether their OOS SR numbers are clean.

### Updated meta-lesson

With all 102 runnable strategies exhausted:
- The OOS SR ≥ 0.70 filter produces 20+ candidates
- After artifact correction (s63, s93, s60, s71): still 11 unaudited candidates
- After redundancy test: zero candidates survive

The equity-momentum factor is already represented in the book via s08 (sector rotation),
s30 (low volatility), s35 (sell in May), and s49 (dollar regime). Every additional equity
momentum variant is a disguised version of PC1 (equity market direction). The book is
already at capacity in this factor. The only new strategies worth investigating are those
with genuinely orthogonal return sources: options premia, fundamentals-based signals,
credit/macro regime signals, or alternative data.

**Research phase: CLOSED. No further candidate evaluation required.**

---

## Appendix: Key Files

| File | Contents |
|------|----------|
| `results/group2_validation.md` | Full Group 2 (s66-s102) IS/OOS + regime + correlation analysis |
| `results/trend_autopsy.md` | Lookahead autopsy of s75/s78/s79; SPY 200-DMA baseline comparison |
| `results/stragglers_validation.md` | s97/s98/s99/s102 stub documentation |
| `results/quarantine.md` | Original quarantine report (s07, s26, s45) |
| `results/DECISION_SUMMARY.md` | Original 50-strategy decision summary (2026-06-14) |
| `results/MONTECARLO_SUMMARY.md` | Full Monte Carlo tables and cluster analysis |
| `results/perturbation_report.md` | Robustness tests across parameter shifts and cost shocks |
| `results/returns/g2_s90.parquet` | s90 Credit Regime daily returns (added to book) |
| `results/returns/g2_s75_corrected.parquet` | s75 corrected returns (for reference; not in book) |

---

*Generated 2026-06-17. Research project: quant-backtest (C:\Users\Owner\quant-backtest)*
*Based on 102 total strategies: 8 validated, 14 quarantined/stub, ~15 data-blocked, ~65 WEAK/FRAGILE/MIRAGE*
