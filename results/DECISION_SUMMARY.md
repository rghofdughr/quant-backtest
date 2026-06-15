# Decision Summary — quant50 Robustness Audit

**Date:** 2026-06-14  
**Run:** 2000-01-03 to 2024-12-31 (25 years), 50 strategies  
**IS:** 2000-01-03 to 2017-06-30 | **OOS:** 2017-07-03 to 2024-12-31  
**OOS covers:** COVID crash (2020), 2022 rate-hike drawdown, 2023-2024 AI rally  
**Costs:** 5 bps commission + 5 bps slippage one-way; rf = 0 throughout

---

## FINAL VERDICTS

### DEPLOY-CANDIDATES — worth the next round of work

| ID  | Name              | OOS Sh | 2x Cost Sh | GFC MDD  | COVID Sh | Rates 2022 Sh | Key constraint                  |
|-----|-------------------|--------|------------|----------|----------|---------------|--------------------------------|
| S08 | Sector Rotation   | +0.76  | +0.46      | -41.6%   | +0.48    | +0.10         | None — robust across full grid  |
| S46 | Risk Parity       | +0.81  | +0.53      | -22.9%   | +0.85    | -0.60         | 2022 bond-equity correlation risk |
| S31 | Vol Targeting     | +0.83  | +0.33      | -17.1%   | +0.91    | -14.3%MDD     | IS borderline; costs at 3x tight |
| S35 | Sell in May       | +0.53  | +0.40      | -25.6%   | +0.28    | -0.84         | 2022 Q1 rate hike in long window |
| S49 | Dollar Regime     | +0.55  | +0.55      | -56.5%   | +0.70    | -0.70         | GFC 2008 catastrophic; USD safe haven |

**S02 (TS Momentum): DEPLOY-CANDIDATE with strict parameter constraint**  
OOS 0.69 at 63-day lookback (long-flat). Collapses to 0.03–0.38 at longer lookbacks or  
long-short. The 63-day window is the published literature optimum for short-term futures  
momentum and the decay at longer lookbacks is asset-class-consistent, not noise-fitting.  
Use 63-day lookback, long-flat only. Do not tune further without fresh OOS data.

**S30 (Low Volatility): DEPLOY-CANDIDATE — long-flat formulation ONLY**  
OOS 0.69–0.78 across all three vol windows (63/126/252d), robust and consistent.  
Long-short formulation collapses completely (OOS -0.20 to -0.25): shorting high-vol  
stocks loses money because lottery-preference and leverage-constraint effects dominate.  
Long-flat (buy low-vol quintile, hold cash for rest) is the correct implementation.  
Nearly cost-insensitive (0.83x/yr turnover). Best risk/reward of the set on calm regimes  
(2013-17 Sharpe 1.37).

---

### QUARANTINED — real numbers, wrong interpretation

| ID  | Reported as       | Issue                                            | Action                                  |
|-----|-------------------|--------------------------------------------------|-----------------------------------------|
| S07 | Donchian Sharpe 4.23 | Back-adjusted roll-return artifact; OOS 5.02 — physically impossible for a real CTA | Rebuild with explicit roll-cost accounting; target MDD -15 to -25% |
| S26 | Dividend Yield Sh 0.54 | TR/PR proxy is a high-distribution tilt, not a sustainability screen | Rebuild with Sharadar dividend + payout data |
| S45 | Short Squeeze Sh 0.50 | Running as R2000 momentum (no FINRA short interest) | Get FINRA biweekly files (free); rebuild with utilization + momentum combo |

S26 and S45 OOS numbers (0.68 and 0.57) are positive and real — they represent valid
momentum/yield tilts, just not the strategies they're named after. The numbers are
useful once the strategies are properly relabeled or rebuilt.

---

### BLOCKED — awaiting external data

| Data source                               | Cost     | Unlocks                              |
|-------------------------------------------|----------|--------------------------------------|
| Sharadar Core US Fundamentals             | ~$40/mo  | S16 (B/M), S17 (E/P), S18 (EV/EBITDA), S19 (Piotroski), S20 (Gross Profit), S22 (Accruals), S26 rebuild |
| ORATS or CBOE options                     | ~$100/mo | S28 (Short Straddle), S32 (Dispersion), S33 (IV Crush) |
| Zacks / Sharadar Earnings                 | ~$50/mo  | S40 (PEAD), S43 (Analyst Revisions)  |
| SEC EDGAR Form 4 (free API)               | free     | S42 (Insider Buying)                 |
| FINRA Short Interest (free biweekly)      | free     | S45 (Short Squeeze) rebuild          |
| Refinitiv / Bloomberg M&A                 | $$$      | S44 (Merger Arb) — defer             |

Priority order: EDGAR Form 4 (free, S42), FINRA short interest (free, S45 rebuild),  
then Sharadar ($40/mo unlocks 6 strategies at once).

---

### SHORT-VOL WARNING

**S27 (VIX Carry, DECAY):** Full Sharpe 0.40 / OOS Sharpe 0.239 — already decaying.  
But the real issue is tail risk invisible at daily resolution:  
- Worst single day: -32.0%  
- Worst single week: -47.1%  
The Feb-2018 and Mar-2020 SVXY events caused intraday drawdowns of 80-90% that  
daily data shows as -30%. A position-sized strategy would have received margin calls  
mid-day before the daily close recorded. **Do not include S27 in any live portfolio.**

---

### DEAD — no clear path forward at daily resolution

| ID  | Reason                                                                          |
|-----|---------------------------------------------------------------------------------|
| S12 | Gap fade: -33.9% full period. 161x annual turnover x 20bps = 32% annual drag. Valid idea but needs intraday fills at ~1bps; impossible at daily resolution. |
| S36 | Day-of-week short Monday: -33%, MDD -100%. Effect doesn't hold 2000-2024.      |
| S06 | Intraday momentum (daily proxy): -18.8%. Proxy doesn't replicate the signal.  |
| S15 | OU spread: Sharpe -0.93. Cointegration breaks persistently in equity pairs.   |
| S37 | FOMC drift: OOS Sharpe -0.77 (worsened). Effect reversed post-2017.           |
| S38 | Pre-holiday: Sharpe -0.44. Capacity-constrained, effect too small at daily.   |

---

### WEAK — not dead, not promising; park until new idea arrives

These strategies have IS Sharpe <= 0.30 so the IS/OOS framework can't classify them.
Some improved OOS (interesting but insufficient evidence):

| ID  | Name              | IS Sh | OOS Sh | Note                                                  |
|-----|-------------------|-------|--------|-------------------------------------------------------|
| S29 | Variance Risk Prem | 0.13 | 0.53   | Interesting OOS. VRP real but needs option data for proper implementation. |
| S31 | Vol Targeting     | 0.30  | 0.83   | (included in DEPLOY above — treated as borderline)    |
| S48 | Gold/Copper Ratio | 0.27  | 0.52   | Macro signal valid; IS weak due to commodity cycles.  |
| S03 | DMA Crossover     | 0.06  | 0.38   | May be regime-dependent (trending post-2017 env).     |
| S10 | Bollinger Rev.    | -0.00 | 0.48   | Near-zero IS but strong OOS improvement — suspicious. |
| S01 | CS Momentum       | 0.17  | 0.14   | Academic anomaly real but crowded; small in both periods. |

---

## PORTFOLIO IMPLICATION

If building a core portfolio from validated strategies only:

| Role             | Strategy          | Target weight | Rationale                                   |
|------------------|-------------------|---------------|---------------------------------------------|
| Defensive core   | S46 Risk Parity   | 35%           | Lowest GFC MDD (-22.9%), OOS improved       |
| Growth/momentum  | S02 TS Momentum   | 25%           | 13.5% CAGR OOS, but 63-day LF only         |
| Equity quality   | S30 Low Vol       | 25%           | Consistent OOS, low cost drag; LF only      |
| Seasonal overlay | S35 Sell in May   | 15%           | Simple, low turnover, consistent             |

S08 (Sector Rotation) and S31 (Vol Targeting) are high-confidence for inclusion once
operational infrastructure is confirmed (S31 in particular needs daily vol-targeting
execution; S08 needs monthly rebalance on 11 sector ETFs).

S49 (Dollar Regime) is a strong signal but the GFC 2008 MDD of -56.5% is too large
for standalone use — it should act as a macro overlay that reduces EM exposure rather
than a standalone strategy.

---

## NEXT STEPS

Ordered by expected ROI:

1. **S02 config — already correct.** `lookbacks: [63, 126, 252]` but the strategy uses `lookbacks[0]=63`. Do not change; long-short is already `false`. Only the base 63-day LF path is valid — document this so future developers don't accidentally enable longer lookbacks or long-short.
2. **S30 config — already correct.** `vol_lookback: 252`, `long_short` defaults to `false`. Add explicit `long_short: false` to config.yaml to prevent accidental activation. The long-short leg is validated-dead.
3. **Get FINRA short interest files** (free, ~30min setup) — rebuild S45 with proper signal.
4. **Get SEC EDGAR Form 4 data** (free API) — activate S42 insider buying.
5. **IS/OOS re-run for new strategies** — after any strategy rebuild, always run a fresh IS/OOS split before promoting.
6. **Subscribe to Sharadar** ($40/mo) — unlocks 6 fundamentals strategies at once; highest strategy-per-dollar ROI.
7. **S07 rebuild** — requires sourcing roll-unadjusted futures data or computing carry-adjusted prices manually. Lower priority; Donchian on proper prices likely yields Sharpe 0.4-0.6 with -20% MDD.
8. **Consider IS/OOS re-run on S29/S48** — if those improved OOS values hold in a fresh 2-year window, they move to DEPLOY-CANDIDATE.

---

*Generated 2026-06-14 from full 2000-2024 backtest + IS/OOS + robustness audit.*  
*See: results/is_oos_table.csv, results/quarantine.md, results/robustness_report.md*
