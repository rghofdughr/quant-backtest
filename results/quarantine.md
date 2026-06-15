# Quarantine Report

**Purpose:** Strategies excluded from the primary ranking table because their reported
numbers do not reflect tradeable alpha. Numbers are real; the *interpretation* is wrong.
All three have separate `results/s??_metrics.json` files and are still importable —
they are quarantined from rankings, not deleted.

---

## S07 — Donchian Channel Breakout (Turtle)

**Status:** INVALID — back-adjusted futures artifact  
**Reported:** Full Sharpe 4.23 / MDD -2.1% / CAGR 15.4%  
**IS:** Sharpe 3.86 / CAGR 13.2% / MDD -2.0%  
**OOS:** Sharpe 5.02 / CAGR 20.9% / MDD -1.7%

**Why this is wrong:**

The strategy runs on 8 Norgate continuous futures (ES, CL, NG, DX, EMD, HG, ZS, SB) using
`StockPriceAdjustmentType.TOTALRETURN`. Norgate back-adjusts continuous contracts by forward-
propagating roll gains/losses into the historical price series. For commodity futures with
positive carry (backwardation during 2000-2014), this creates a systematic *upward* drift in
the back-adjusted price series that has nothing to do with spot price trends.

The strategy interprets this drift as Donchian breakout entries and sits long for months during
what is essentially a "pay me the roll yield" artifact. The 2008-2013 commodity supercycle
exaggerates this further.

Evidence that MDD -2.1% is an artifact:
- The real Turtle Trading program had 40-60% MDD drawdowns through 2011-2013 when commodity
  trends broke down. No daily-OHLC Donchian system on 8 markets produces -2.1% MDD over 25 years.
- OOS Sharpe 5.02 is higher than IS Sharpe 3.86. Genuine trend-following strategies decay OOS
  (trend premium has compressed); a Sharpe that *improves* OOS on a well-known strategy is
  a red flag for artifact harvesting.

**Rebuild target:** Use roll-adjusted returns where the continuous price is stitched at roll
dates without carry-forward adjustment. Carry should be modeled explicitly as a separate P&L
component. Target realistic metrics: Sharpe 0.4-0.7, MDD -15% to -25%, CAGR 5-12%.

---

## S26 — Dividend Yield (Proxy)

**Status:** PROXY — strategy identity mismatch  
**Reported:** Full Sharpe 0.54 / MDD -67.6% / CAGR 9.8%  
**IS:** Sharpe 0.47 / CAGR 8.0% / MDD -67.6%  
**OOS:** Sharpe 0.68 / CAGR 14.0% / MDD -48.1%

**Why this is in quarantine:**

The implementation uses the ratio of total-return price to price-return price (TR/PR) as a
dividend yield proxy. Ranking stocks by TR/PR selects "names that pay distributions" — which
is predominantly a *size and sector* tilt (small caps and utilities dominate).

The published dividend yield anomaly (Blume 1980, Litzenberger & Ramaswamy 1982) requires:
1. Actual dividend yield data (ex-dividend dates, declared amounts)
2. Payout sustainability screening (exclude dividend traps — stocks cutting dividends)
3. Growth filter (sustained dividend growth over 2-3 years, not just TR/PR ratio)

The proxy captures *some* of the right signal, which is why numbers are positive. But without
sustainability screening, it also loads into energy/commodity high-yielders and REITs that are
vulnerable to regime changes. The OOS improvement likely reflects REIT and high-yield recovery
post-2017, not validated dividend anomaly exposure.

**Numbers are still meaningful** as a "high-distribution stock" factor — just not as a dividend
quality screen. Reported separately from the primary ranking.

**Rebuild target:** Plug in Sharadar Core US Fundamentals (dividends_per_share,
eps, book_value tables) to compute TTM yield, payout ratio, and 3-year consecutive growth.

---

## S45 — Short Squeeze (Proxy)

**Status:** PROXY — core signal missing  
**Reported:** Full Sharpe 0.50 / MDD -72.7% / CAGR 10.8%  
**IS:** Sharpe 0.47 / CAGR 9.8% / MDD -72.7%  
**OOS:** Sharpe 0.57 / CAGR 13.0% / MDD -48.2%

**Why this is in quarantine:**

S45 is labeled "short squeeze" but runs as a **pure Russell 2000 momentum strategy** because
FINRA biweekly short interest data was not available at implementation time. The strategy ranks
R2000 names by 1-month momentum (no short-interest filter, no days-to-cover, no float
utilization, no cost-to-borrow).

What the numbers actually represent: small-cap momentum alpha over 2000-2024, which is a
well-documented but separate anomaly from short squeeze. Short squeeze events (GME Jan 2021,
AMC, BBBY, etc.) are event-driven with completely different timing, holding periods, and
risk characteristics.

The positive numbers are *real* — they just describe momentum, not squeeze dynamics. Keeping
them in the primary ranking would create confusion about what risk is being taken.

**Numbers are reported separately** in the proxy bucket.

**Rebuild target:** Download FINRA short interest biweekly files (free from finra.org/finra-data).
Build short-interest utilization = short_interest / float_shares. Signal = high utilization +
recent positive price momentum (potential squeeze candidates). This is materially different from
the current strategy and needs a fresh backtest once the data is obtained.

---

## Proxy Bucket (not invalidated, just relabeled)

These strategies produce real numbers but for a different strategy than named:

| ID  | Reported as    | Actually runs as          | IS Sh | OOS Sh | Notes                          |
|-----|----------------|---------------------------|-------|--------|--------------------------------|
| S26 | Dividend Yield | High-distribution tilt    | 0.47  | 0.68   | REIT/utility/energy heavy      |
| S45 | Short Squeeze  | R2000 momentum            | 0.47  | 0.57   | No short-interest data         |

Verdict for S26/S45: **positive OOS numbers**; worth revisiting once the missing data is in.
The underlying signal may be valid once properly implemented.

---

*Generated: 2026-06-14. Quarantine decisions are permanent until a corrected rebuild is completed.*
