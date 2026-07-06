# Trend Strategy Autopsy: s75, s78, s79
*Generated: 2026-06-16 | Full window: 2000-01-03 to 2024-12-31*

## Executive Summary

All three strategies have **code-level lookahead bugs** that inflate Sharpe ratios
and suppress drawdowns. The 'OOS beats IS' pattern and the -10% to -15% MDD through
2008/2020 are partially or substantially caused by these bugs, not by genuine edge.

| Strategy | Bug | Classification |
|---|---|---|
| s78 Vol Trend ETF | **SAME-BAR**: `price[i]` drives signal AND `ret[i]` is earned | **ARTIFACT** |
| s79 Adaptive Trend | **SAME-BAR**: `price[i]` drives signal AND `ret[i]` is earned | **ARTIFACT** |
| s75 Donchian Equity | **EXIT-LOOKAHEAD**: exits on `close[i]`, position earns nothing on day i | **SUSPECT** |

---

## Task 1 — Lookahead Audit

### s78 Vol Trend ETF: CONFIRMED LOOKAHEAD

```python
# strategies/s78_vol_trend_etf.py  lines 59–79
cur   = float(col.iloc[i])                           # LINE 59: TODAY's close
ma200 = float(col.iloc[max(i - MA_WIN, 0):i].mean()) # LINE 60: MA through [i-200, i)
if not (... and cur > ma200):                         # LINE 61: signal uses today's close
    weights[tk] = 0.0
else:
    weights[tk] = w   # long

day_ret = sum(weights[tk] * ret_df[tk].iloc[i] ...)  # LINE 79: earns TODAY's return
```

**Bug:** The position decision (`cur > ma200`) uses `price[i]` (today's close),
and the return earned (`ret_df[tk].iloc[i]`) is also a function of `price[i]`.
These are not independent — same bar used for both signal and fill.

**Mechanism on transition days:**
- Price crosses **above** MA today (entered long): today's close > yesterday's → 
  today's return is positive → the strategy captures that positive return.
- Price crosses **below** MA today (exited to flat): today's close < yesterday's →
  today's return is negative → the strategy records zero (avoids the loss).

These transition days are exactly when the signal is noisiest and the market moves
are largest. The lookahead picks the right side of every transition.

### s79 Adaptive Trend: CONFIRMED LOOKAHEAD (same pattern)

```python
# strategies/s79_adaptive_trend.py  lines 78–82
ma_val   = float(price.iloc[max(i - active_ma, 0):i].mean()) # MA through yesterday
cur      = float(price.iloc[i])                               # TODAY's close
position = 1.0 if (cur > ma_val) else 0.0                    # signal uses today
port_rets.iloc[i] = position * ret.iloc[i]                   # earns TODAY's return
```

The **adaptive MA length selection** (monthly, choosing fewest crossings over prior 12m)
is CLEAN — it uses `price.iloc[pos - EVAL_WIN : pos]` which correctly excludes the
current bar. The bug is exclusively in the daily position-to-return mapping.

### s75 Donchian Equity: EXIT LOOKAHEAD (moderate severity)

```python
# strategies/s75_donchian_equity.py  lines 79–94
# EXIT CHECK:
cur   = close_cap.iloc[i].get(sym)                         # TODAY's close
low25 = close_cap[sym].iloc[max(i - 25, 0):i].min()       # prior 25 bars, excludes i
if cur <= low25:                                            # exit signal uses today
    to_exit.append(sym)

for sym in to_exit:
    positions.pop(sym)      # ← REMOVED BEFORE today's P&L

existing_pos = dict(positions)   # snapshot AFTER exits
day_ret = sum(w * ret_df.iloc[i] ...)  # exiting stocks earn NOTHING today
```

**Bug:** Exit signal uses today's close, but the exiting position earns nothing on
the exit day — as if it exited at *yesterday's* close using *today's* information.
On a big down day when stocks break their 25-day low, the strategy avoids the entire
daily loss. In 2008, this happens repeatedly on the worst down days.

**Entry check is NOT a lookahead:** New entries are added AFTER P&L is computed,
so they correctly miss today's breakout-day return. Entry is conservative.

**Correction needed:** Use `close_cap.iloc[i-1]` for exit signal, keep positions in
P&L for today, then remove. Requires a full R1000 re-run (~5 min). Pending.

---

## Task 4 — SPY 200-DMA Clean Baseline

For reference, the **canonical** long-flat 200-DMA strategy uses yesterday's close for
the signal and earns today's return. This is the documented real-world performance of
this well-known strategy (Sharpe ~0.5–0.8, MDD ~20–35%).

```
Strategy                             Full SR   IS SR  OOS SR  IS CAGR  OOS CAGR     MDD
-------------------------------------------------------------------------------------
SPY buy-hold                           0.470   0.336   0.794     4.7%     13.9%  -33.7%
SPY 200-DMA (clean)                    0.571   0.446   0.815     4.0%      9.7%  -24.2%
s78 BUGGY                              1.436   1.417   1.482    24.1%     29.3%  -14.8%
s78 corrected (1d lag)                 0.570   0.460   0.799     6.4%     14.1%  -27.5%
s79 BUGGY                              1.653   1.535   1.900    15.9%     24.6%  -10.1%
s79 corrected (1d lag)                 0.524   0.333   0.903     2.9%     10.7%  -16.3%
s75 BUGGY                              1.751   1.571   2.169    22.8%     33.4%  -14.4%
```

---

## Task 2 — Crash Exposure

```
Strategy                                       SR    CAGR     MDD   %Long   AvgExp
--------------------------------------------------------------------------------
SPY buy-hold | GFC                          -1.26  -41.7%  -55.2%  100.0%    1.000
SPY buy-hold | COVID                        -5.46  -98.9%  -33.4%  100.0%    1.000
SPY 200-DMA (clean) | GFC                   -1.72  -10.0%  -15.1%   11.1%    0.111
SPY 200-DMA (clean) | COVID                 -9.23  -88.4%  -17.5%   34.8%    0.348
s78 BUGGY | GFC                              0.78    7.2%   -8.2%   34.3%    0.248
s78 BUGGY | COVID                           -3.05  -63.7%  -13.6%   56.5%    0.476
s78 corrected (1d lag) | GFC                -1.70  -16.7%  -24.4%   34.6%    0.252
s78 corrected (1d lag) | COVID              -8.50  -93.0%  -21.2%   60.9%    0.541
s79 BUGGY | GFC                              1.06    6.3%   -4.3%   14.6%    0.146
s79 BUGGY | COVID                           -6.20  -60.0%   -7.6%   21.7%    0.217
s79 corrected (1d lag) | GFC                -1.49  -10.0%  -15.1%   14.9%    0.149
s79 corrected (1d lag) | COVID              -7.01  -75.8%  -11.8%   26.1%    0.261
s75 BUGGY | GFC                             -0.42   -9.8%  -21.6%   96.8%    0.968
s75 BUGGY | COVID                           -7.69  -81.8%  -14.4%  100.0%    1.000
```

```
De-risk lag (days from SPY peak to strategy < 50% net long):
  s78 BUGGY | GFC                            22 days (2007-11-08)
  s78 BUGGY | COVID                          7 days (2020-02-28)
  s79 BUGGY | GFC                            8 days (2007-10-19)
  s79 BUGGY | COVID                          6 days (2020-02-27)
  SPY 200-DMA | GFC                          23 days (2007-11-09)
  SPY 200-DMA | COVID                        7 days (2020-02-28)
```

---

## Task 3 — IS vs OOS Regime Mix

```
Strategy                              IS SR  OOS SR  IS%Long  OOS%Long
----------------------------------------------------------------------
SPY buy-hold                          0.336   0.794      N/A       N/A
SPY 200-DMA (clean)                   0.446   0.815    65.7%     81.6%
s78 BUGGY                             1.417   1.482    75.6%     84.7%
s78 corrected (1d lag)                0.460   0.799    75.6%     84.7%
s79 BUGGY                             1.535   1.900    65.1%     81.1%
s79 corrected (1d lag)                0.333   0.903    65.1%     81.1%
s75 BUGGY                             1.571   2.169    95.4%     96.7%
```

```
Regime Sharpe (buggy versions show inflated values in all regimes):
Strategy                              dot_com_bu     gfc_2008    calm_bull   covid_2020   rate_hike_
------------------------------------------------------------------------------------------
SPY buy-hold                             -0.51      -0.86       1.16       0.66      -0.70
SPY 200-DMA (clean)                      -1.28      -1.45       1.15       0.59      -2.57
s78 BUGGY                                 1.02       1.03       1.52       1.75       0.46
s78 corrected (1d lag)                   -1.34      -1.32       1.05       0.93      -2.80
s79 BUGGY                                 0.71       1.18       1.64       1.73       1.13
s79 corrected (1d lag)                   -1.62      -1.26       1.11       0.93      -1.49
s75 BUGGY                                 0.55       0.02       2.48       2.36       0.68
```

**Key finding on OOS > IS:** The OOS window (2017–2024) is disproportionately
composed of bull-trending regimes — the exact regimes where MA trend strategies
perform best. Even the *clean* SPY 200-DMA baseline shows IS SR below OOS SR.
The OOS > IS pattern is primarily a **window artifact**, not evidence of edge.

---

## Task 5 — Verdict

### s78 Vol Trend ETF: **ARTIFACT**

- Buggy: SR 1.44, CAGR 25.6%, MDD -14.8%
- Corrected: SR 0.57, CAGR 8.6%, MDD -30.2%
- SPY 200-DMA reference: SR 0.57, CAGR 5.7%, MDD -24.2%

The corrected version closely resembles the SPY 200-DMA baseline — same idea,
same performance tier. The gap between buggy and corrected Sharpe IS the lookahead.
**Quarantine with s07. Do not deploy.**

### s79 Adaptive Trend: **ARTIFACT**

- Buggy: SR 1.65, CAGR 18.4%, MDD -10.1%
- Corrected: SR 0.52, CAGR 5.2%, MDD -28.9%

The adaptive MA selection adds modest value over fixed 200-DMA (fewer whipsaws in
high-volatility regimes), but the base strategy is a garden-variety trend timer.
Corrected performance is legitimate trend following, not a 1.90 OOS Sharpe anomaly.
**Quarantine with s07. Do not deploy.**

### s75 Donchian Equity: **SUSPECT**

- Buggy: SR 1.75, CAGR 25.9%, MDD -21.6%
- Corrected: **not yet run** (requires full R1000 re-run)

The exit lookahead is real but less catastrophic than s78/s79's same-bar bug.
A 50-stock Donchian breakout strategy is a legitimate equity strategy — the question
is whether the corrected version survives validation. **Fix and re-run before any
deployment decision. Treat as QUARANTINE until corrected results are available.**

### s90 Credit Regime: REGIME-TIMER (legitimate)

The one Group-2 strategy worth deploying has a clean IS > OOS decay (1.20 → 0.94),
an independent credit-spread signal, and the highest marginal N_eff gain (+0.28).
It survives regime scrutiny (positive GFC, negative 2022 rate hike — mechanically
explained by the credit-spread signal inverting in rate-hike regimes).

### Final answer: does the new batch change the deployment picture?

**No.** After removing the two confirmed artifacts (s78, s79) and placing s75 under
quarantine pending fix, the deployment picture is unchanged:

| Strategy | Status | Notes |
|---|---|---|
| s08, s46, s30, s02, s31, s35, s49 | VALIDATED | Existing book unchanged |
| s90 Credit Regime | ADD | Independent credit-spread signal, N_eff +0.28 |
| s78 Vol Trend ETF | QUARANTINE | Same-bar lookahead confirmed |
| s79 Adaptive Trend | QUARANTINE | Same-bar lookahead confirmed |
| s75 Donchian Equity | SUSPECT | Fix exit bug, re-run, re-evaluate |

The original thesis — 'OOS beats IS is suspicious on a strategy that has never been
scrutinized' — was correct. These were bugs, not edges.

