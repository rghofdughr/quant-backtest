# Group 2 Validation Report
*Generated: 2026-06-16 — Full 2000-01-03 → 2024-12-31 window*

## Executive Summary

- **19 ROBUST**: s66, s67, s68, s69, s70, s71, s73, s74, s75, s76, s77, s78, s79, s89, s90, s91, s92, s93, s96
- **0 DECAY**: 
- **2 MIRAGE**: s81, s94
- **0 FRAGILE**: 
- **12 WEAK/NEGATIVE**: s100, s101, s72, s80, s82, s83, s84, s85, s86, s87, s88, s95

---

## Task 0 — s81 Bug

**Root cause:** `portfolio_returns_from_weights()` was called with a weight schedule
containing **one entry per trading day** in December–January (≈42 entries/year).

The function computes each entry's hold period as:
```
hold_start[k] = first trading day after reb_dates[k]
hold_end[k]   = first trading day after reb_dates[k+1]
```
Then accumulates with `port_ret[hold_mask] += weighted_return`.

**Double-counting (2× leverage artifact):** Day D appears in:
- Entry D−1's hold mask: `[D, D+1]`
- Entry D−2's hold mask: `[D−1, D]`
→ Every interior hold day gets `+=` twice → implicit 2× leverage.

**The catastrophic part:** The last January entry (Jan 31) has no 
next December entry until ~10 months later. Its `hold_end` is set to the
first day of that December window — so Jan 31's hold period runs:
**Feb 1 → Dec 2 of the same year (~10 months of phantom returns).**
Over 24 years, this is 24 × ~10 months of cumulative phantom accumulation.
Result: reported 149.7% CAGR and 456% annualised volatility (both nonsense).

**Fix:** Removed `portfolio_returns_from_weights` entirely. Directly assign
equal-weighted returns to hold days using vectorised `port_rets.loc[hold_days] = ...`
(assignment, not accumulation). Each trading day in Dec–Jan now gets exactly
one year's signal, once.

**Post-fix s81 stats:**
- IS Sharpe: 0.366 | OOS Sharpe: 0.146
- Full CAGR: 3.5% | OOS MDD: -33.9%
- Flag: MIRAGE

Numbers are now in a sane range. s75/s78/s79 are unaffected — they use
direct bar-by-bar loops, not `portfolio_returns_from_weights`.

---

## Task 1 — IS/OOS Table

```
==============================================================================================================
  TASK 1 — GROUP 2 IS/OOS TABLE (2000-01-03 → 2024-12-31, 70/30 split)
  IS:  2000-01-03 → 2017-06-30  (70%)
  OOS: 2017-07-03 → 2024-12-31  (30% = ~7.8 yr: COVID 2020, rate hike 2022, AI rally 2023-24)
  Costs: 5 bps/side commission + 5 bps/side slippage = 10 bps/side, already applied
==============================================================================================================

ID     Name                    IS_SR  OOS_SR   Decay  IS_CAGR  OOS_CAGR  OOS_MDD  TO/yr  CostDrag Flag
--------------------------------------------------------------------------------------------------------------
s75    Donchian Equity         +1.57   +2.17   +0.60   +22.8%    +33.4%   -14.4%   3.1x      0.6% ROBUST
s79    Adaptive Trend          +1.53   +1.90   +0.37   +15.9%    +24.6%   -10.1%   4.0x      0.8% ROBUST
s78    Vol Trend ETF           +1.42   +1.48   +0.06   +24.1%    +29.3%   -14.8%   7.7x      1.5% ROBUST
s71    52-Wk Breakout          +1.09   +1.24   +0.15   +14.7%    +18.9%   -22.5%   0.6x      0.1% ROBUST
s93    Defensive Rotation      +0.64   +1.05   +0.40    +9.3%    +19.3%   -34.1%   6.5x      1.3% ROBUST
s90    Credit Regime           +1.20   +0.94   -0.27   +10.3%    +10.9%   -21.2%   7.6x      1.5% ROBUST
s91    Inflation Tilt          +0.60   +0.89   +0.29    +8.5%    +16.6%   -29.8%  12.6x      2.5% ROBUST [HIGH-TO]
s69    Sharpe Rank             +0.56   +0.79   +0.24    +9.7%    +16.1%   -33.8%   3.5x      0.7% ROBUST
s76    MA200 Band              +0.54   +0.78   +0.23    +8.5%    +13.9%   -35.3%   2.5x      0.5% ROBUST
s68    Mom Ensemble            +0.48   +0.76   +0.29    +8.5%    +17.1%   -36.7%   3.8x      0.8% ROBUST
s66    Vol-Confirmed Mom       +0.46   +0.75   +0.30    +8.6%    +18.9%   -35.9%   4.6x      0.9% ROBUST
s77    Dual Momentum           +0.41   +0.73   +0.31    +7.2%    +16.5%   -35.9%   3.3x      0.7% ROBUST
s73    Residual Mom            +0.48   +0.72   +0.24    +8.8%    +15.8%   -33.9%   3.0x      0.6% ROBUST
s70    MaxDD Quality           +0.67   +0.69   +0.01    +9.5%    +10.8%   -32.1%   1.7x      0.3% ROBUST
s89    DV Momentum             +0.49   +0.67   +0.18    +8.1%    +12.7%   -41.6%   3.4x      0.7% ROBUST
s74    Accel Breadth           +0.52   +0.66   +0.15    +9.2%    +13.0%   -42.6%   3.5x      0.7% ROBUST
s67    Amihud Illiquidity      +0.52   +0.46   -0.07   +10.9%     +9.2%   -46.1%   2.3x      0.5% ROBUST
s92    Country ETF Mom         +0.45   +0.41   -0.04    +8.2%     +6.3%   -37.6%   2.5x      0.5% ROBUST
s96    Deep Value              +0.43   +0.40   -0.02    +8.4%     +7.6%   -57.6%   0.5x      0.1% ROBUST
s72    Reversal Demeaned       +0.14   +0.34   +0.20    -2.3%     +5.4%   -78.7%  42.9x      8.6% WEAK [HIGH-TO]
s94    Index Deletion          +0.91   +0.24   -0.67   +56.4%     -1.0%   -73.9%   8.1x      1.6% MIRAGE
s100   Distressed              +0.05   +0.19   +0.13    -1.9%     +0.9%   -64.9%   4.4x      0.9% WEAK
s85    Gap Go                  -0.15   +0.18   +0.33   -10.4%     +1.0%   -47.8% 115.3x     23.1% WEAK [HIGH-TO]
s88    NR7                     -0.12   +0.15   +0.27    -5.0%     +0.7%   -57.9%  31.6x      6.3% WEAK [HIGH-TO]
s81    Jan Reversal PIT        +0.37   +0.15   -0.22    +4.6%     +1.1%   -33.9%   1.0x      0.2% MIRAGE
s95    R2000 Promotion         +0.13   +0.08   -0.05    +0.9%     -0.0%   -44.8%   1.0x      0.2% WEAK
s84    FOMC Week               -0.06   -0.27   -0.20    -0.9%     -2.9%   -33.5%   7.9x      1.6% WEAK
s86    Range Expansion         -0.09   -0.45   -0.36    -4.7%    -12.2%   -76.2%  31.2x      6.2% WEAK [HIGH-TO]
s83    DOW Conditional         -0.47   -0.68   -0.21    -3.5%     -4.7%   -31.5%  22.3x      4.5% WEAK [HIGH-TO]
s82    Month-End Flow          -0.88   -0.77   +0.10    -7.3%     -5.9%   -40.1%  23.2x      4.6% WEAK [HIGH-TO]
s80    IBS Reversion           -1.65   -1.80   -0.15    -7.6%    -11.3%   -61.0%   4.8x      1.0% WEAK
s101   Sector Pairs            -1.57   -2.21   -0.63   -43.7%    -39.5%   -98.0%  28.3x      5.7% WEAK [HIGH-TO]
s87    Vol Spike               -7.29   -7.67   -0.39   -96.0%    -95.7%  -100.0%  20.9x      4.2% WEAK [HIGH-TO]
s102   ETF Basket Arb            N/A     N/A     N/A      N/A       N/A      N/A    N/A       N/A MISSING
s97    Div Capture               N/A     N/A     N/A      N/A       N/A      N/A    N/A       N/A MISSING
s98    Ex-Date Drift             N/A     N/A     N/A      N/A       N/A      N/A    N/A       N/A MISSING
s99    Div Initiation            N/A     N/A     N/A      N/A       N/A      N/A    N/A       N/A MISSING

  Flag key: ROBUST=OOS≥70% IS (both>0.3) | DECAY=OOS 50-70% IS | MIRAGE=OOS<50% IS
           FRAGILE=OOS<0 | WEAK=IS≤0.3 | HIGH-TO=annual turnover>10x

  ROBUST   (19): s66 s67 s68 s69 s70 s71 s73 s74 s75 s76 s77 s78 s79 s89 s90 s91 s92 s93 s96
  MIRAGE   ( 2): s81 s94
  WEAK     (12): s100 s101 s72 s80 s82 s83 s84 s85 s86 s87 s88 s95
  MISSING  ( 4): s102 s97 s98 s99
```

---

## Task 2 — Regime Breakdown

```

==============================================================================================================
  TASK 2 — REGIME BREAKDOWN (survivors with OOS Sharpe ≥ 0.5)
  Critical: these strategies were only ever seen in 2017-2024. 2000-2002 and 2008 are new territory.
==============================================================================================================
Strategy                    SR / CAGR     SR / CAGR     SR / CAGR     SR / CAGR     SR / CAGR   
------------------------------------------------------------------------------------------------
                            SR   CAGR     SR   CAGR     SR   CAGR     SR   CAGR     SR   CAGR   
------------------------------------------------------------------------------------------------
s69 Sharpe Rank           !-0.14 -3%   !-0.63 -29%    +0.99 +14%    +1.13 +43%    !-0.31 -9%  
s70 MaxDD Quality         !-0.02 -1%   !-0.67 -20%    +1.32 +17%    +0.53 +13%    !-0.26 -6%  
s71 52-Wk Breakout         +0.35 +4%    !-0.05 -4%    +1.48 +16%    +1.21 +32%     +0.47 +7%  
s73 Residual Mom           N/A         !-0.73 -37%    +0.84 +14%    +1.09 +43%    !-0.15 -7%  
s74 Accel Breadth        !-0.60 -12%   !-0.30 -19%    +1.13 +17%    +0.62 +19%   !-0.60 -16%  
s75 Donchian Equity        +0.49 +6%     +0.39 +6%    +2.48 +29%    +2.36 +54%    +0.68 +10%  
s76 MA200 Band            !-0.47 -8%   !-0.79 -26%    +1.11 +15%    +0.98 +32%   !-0.39 -10%  
s77 Dual Momentum         !-0.24 -5%   !-0.90 -38%    +0.75 +12%    +1.10 +45%    !-0.14 -7%  
s78 Vol Trend ETF          +1.02 +6%    +1.09 +11%    +1.52 +27%    +1.75 +42%     +0.46 +3%  
s79 Adaptive Trend         +0.90 +4%     +1.14 +8%    +1.64 +17%    +1.73 +30%    +1.13 +12%  
s89 DV Momentum           !-0.28 -8%   !-0.52 -23%    +0.96 +13%    +0.48 +12%   !-0.29 -10%  
s90 Credit Regime          N/A           +0.28 +4%    +1.99 +15%    +2.61 +33%   !-0.47 -11%  
s91 Inflation Tilt         N/A         !-0.41 -18%    +1.45 +18%    +0.61 +17%     +0.35 +5%  
s93 Defensive Rotation    !-0.35 -6%   !-0.55 -15%    +1.46 +19%    +0.70 +19%    !-0.18 -5%  

  ! = negative Sharpe in that regime (potential regime mirage)

  REGIME ANALYSIS:
  s69 Sharpe Rank: NEGATIVE in 2000-2002 (dot-com), 2008-2009 (GFC), 2022 (rate hike)
  s70 MaxDD Quality: NEGATIVE in 2000-2002 (dot-com), 2008-2009 (GFC), 2022 (rate hike)
  s71 52-Wk Breakout: NEGATIVE in 2008-2009 (GFC)
  s73 Residual Mom: NEGATIVE in 2008-2009 (GFC), 2022 (rate hike)
  s74 Accel Breadth: NEGATIVE in 2000-2002 (dot-com), 2008-2009 (GFC), 2022 (rate hike)
  s76 MA200 Band: NEGATIVE in 2000-2002 (dot-com), 2008-2009 (GFC), 2022 (rate hike)
  s77 Dual Momentum: NEGATIVE in 2000-2002 (dot-com), 2008-2009 (GFC), 2022 (rate hike)
  s89 DV Momentum: NEGATIVE in 2000-2002 (dot-com), 2008-2009 (GFC), 2022 (rate hike)
  s90 Credit Regime: NEGATIVE in 2022 (rate hike)
  s91 Inflation Tilt: NEGATIVE in 2008-2009 (GFC)
  s93 Defensive Rotation: NEGATIVE in 2000-2002 (dot-com), 2008-2009 (GFC), 2022 (rate hike)
```

---

## Task 3 — Correlation & Marginal Value

```

==============================================================================================================
  TASK 3 — CORRELATION, N_EFF, MARGINAL VALUE vs EXISTING BOOK
  Existing book: S08, S46, S30, S02, S31, S35, S49
==============================================================================================================

  Common aligned window: 2000-01-03 → 2024-12-31 (6522 days)

  N_eff (existing book, 7 survivors): 2.29

  MARGINAL N_EFF when each Group-2 survivor is added to existing book:
  ID       Name                   N_eff_combined  Delta_N_eff   Corr_w_SPY
  --------------------------------------------------------------------------
  s69      Sharpe Rank                      2.22        -0.07        0.810
  s70      MaxDD Quality                    2.18        -0.12        0.855
  s71      52-Wk Breakout                   2.23        -0.06        0.792
  s73      Residual Mom                     2.27        -0.02        0.768
  s74      Accel Breadth                    2.22        -0.07        0.860
  s75      Donchian Equity                  2.29        +0.00        0.740
  s76      MA200 Band                       2.16        -0.13        0.863
  s77      Dual Momentum                    2.25        -0.04        0.803
  s78      Vol Trend ETF                    2.40        +0.10        0.573
  s79      Adaptive Trend                   2.42        +0.13        0.564
  s89      DV Momentum                      2.18        -0.11        0.907
  s90      Credit Regime                    2.57        +0.28        0.509
  s91      Inflation Tilt                   2.30        +0.00        0.741
  s93      Defensive Rotation               2.19        -0.10        0.867

  N_eff (all 21 survivors combined): 2.09
  (If N_eff barely moves when adding 5+ trend strategies, they're redundant.)

  CRASH-CONDITIONAL CORRELATION (days when SPY < -1%):
           s08    s46    s30    s02    s31    s35    s49    s69    s70    s71    s73    s74    s75    s76    s77    s78    s79    s89    s90    s91    s93
  s69     0.55   0.32   0.80   0.23   0.30   0.33   0.70   1.00   0.87   0.73   0.92   0.80   0.45   0.92   0.94   0.12   0.09   0.80   0.16   0.60   0.66
  s70     0.55   0.33   0.97   0.23   0.29   0.37   0.65   0.87   1.00   0.76   0.76   0.77   0.35   0.89   0.81   0.04   0.06   0.80   0.09   0.65   0.77
  s71     0.58   0.18   0.73   0.32   0.39   0.18   0.62   0.73   0.76   1.00   0.63   0.63   0.54   0.75   0.68   0.21   0.19   0.63   0.17   0.61   0.72
  s73     0.52   0.21   0.69   0.21   0.28   0.33   0.77   0.92   0.76   0.63   1.00   0.76   0.39   0.84   0.92   0.17   0.12   0.73   0.20   0.63   0.57
  s74     0.55   0.22   0.75   0.27   0.27   0.32   0.70   0.80   0.77   0.63   0.76   1.00   0.43   0.82   0.77   0.13   0.12   0.79   0.15   0.63   0.68
  s75     0.41   0.16   0.32   0.28   0.41   0.11   0.36   0.45   0.35   0.54   0.39   0.43   1.00   0.45   0.37   0.31   0.30   0.41   0.29   0.41   0.40
  s76     0.64   0.37   0.83   0.29   0.35   0.32   0.67   0.92   0.89   0.75   0.84   0.82   0.45   1.00   0.90   0.15   0.11   0.86   0.12   0.59   0.72
  s77     0.55   0.31   0.73   0.24   0.32   0.27   0.68   0.94   0.81   0.68   0.92   0.77   0.37   0.90   1.00   0.14   0.09   0.79   0.11   0.52   0.64
  s78     0.25  -0.07   0.00   0.35   0.59  -0.10   0.18   0.12   0.04   0.21   0.17   0.13   0.31   0.15   0.14   1.00   0.84  -0.00   0.18   0.15   0.36
  s79     0.23  -0.06   0.03   0.31   0.51  -0.06   0.16   0.09   0.06   0.19   0.12   0.12   0.30   0.11   0.09   0.84   1.00   0.01   0.22   0.21   0.34
  s89     0.58   0.42   0.78   0.24   0.28   0.38   0.66   0.80   0.80   0.63   0.73   0.79   0.41   0.86   0.79  -0.00   0.01   1.00   0.13   0.61   0.68
  s90     0.13   0.01   0.05   0.15   0.19   0.03   0.22   0.16   0.09   0.17   0.20   0.15   0.29   0.12   0.11   0.18   0.22   0.13   1.00   0.29   0.14
  s91     0.54   0.16   0.67   0.27   0.28   0.32   0.75   0.60   0.65   0.61   0.63   0.63   0.41   0.59   0.52   0.15   0.21   0.61   0.29   1.00   0.63
  s93     0.58   0.27   0.78   0.34   0.46   0.25   0.60   0.66   0.77   0.72   0.57   0.68   0.40   0.72   0.64   0.36   0.34   0.68   0.14   0.63   1.00

  MARGINAL PORTFOLIO SHARPE (equal-weight, existing book baseline):
  Existing book (7-strategy equal-weight):  Sharpe 0.726  MDD -36.5%

  ID       Name                    Port_SR  Delta_SR  Port_MDD Add?
  -----------------------------------------------------------------
  s69      Sharpe Rank               0.725    -0.001    -39.9% NO
  s70      MaxDD Quality             0.731    +0.005    -37.8% MARGINAL
  s71      52-Wk Breakout            0.798    +0.073    -34.4% YES
  s73      Residual Mom              0.713    -0.012    -41.2% NO
  s74      Accel Breadth             0.711    -0.014    -39.2% NO
  s75      Donchian Equity           0.888    +0.162    -34.2% YES
  s76      MA200 Band                0.718    -0.008    -38.6% NO
  s77      Dual Momentum             0.702    -0.024    -40.6% NO
  s78      Vol Trend ETF             0.880    +0.155    -31.9% YES
  s79      Adaptive Trend            0.855    +0.129    -32.0% YES
  s89      DV Momentum               0.706    -0.020    -39.0% NO
  s90      Credit Regime             0.797    +0.071    -33.7% YES
  s91      Inflation Tilt            0.744    +0.018    -38.3% MARGINAL
  s93      Defensive Rotation        0.748    +0.023    -36.7% YES
```

---

## Task 4 — Verdict

### High-Turnover Cost Death

These strategies have annual turnover > 10× and meaningful cost drag.
Their Sharpes are already post-cost (costs are applied in each strategy).
Still flagged because even after costs, high-TO strategies carry extra
slippage risk in live execution:

- **s91 Inflation Tilt**: 13× turnover → 2.5%/yr theoretical cost drag | Flag: ROBUST

### Which Group-2 strategies survive the full-window test?

**ROBUST (19):** s66, s67, s68, s69, s70, s71, s73, s74, s75, s76, s77, s78, s79, s89, s90, s91, s92, s93, s96

- **s66 Vol-Confirmed Mom**: IS 0.46 → OOS 0.75, OOS CAGR 18.9%, OOS MDD -35.9%
- **s67 Amihud Illiquidity**: IS 0.52 → OOS 0.46, OOS CAGR 9.2%, OOS MDD -46.1%
- **s68 Mom Ensemble**: IS 0.48 → OOS 0.76, OOS CAGR 17.1%, OOS MDD -36.7%
- **s69 Sharpe Rank**: IS 0.56 → OOS 0.79, OOS CAGR 16.1%, OOS MDD -33.8%
- **s70 MaxDD Quality**: IS 0.67 → OOS 0.69, OOS CAGR 10.8%, OOS MDD -32.1%
- **s71 52-Wk Breakout**: IS 1.09 → OOS 1.24, OOS CAGR 18.9%, OOS MDD -22.5%
- **s73 Residual Mom**: IS 0.48 → OOS 0.72, OOS CAGR 15.8%, OOS MDD -33.9%
- **s74 Accel Breadth**: IS 0.52 → OOS 0.66, OOS CAGR 13.0%, OOS MDD -42.6%
- **s75 Donchian Equity**: IS 1.57 → OOS 2.17, OOS CAGR 33.4%, OOS MDD -14.4%
- **s76 MA200 Band**: IS 0.54 → OOS 0.78, OOS CAGR 13.9%, OOS MDD -35.3%
- **s77 Dual Momentum**: IS 0.41 → OOS 0.73, OOS CAGR 16.5%, OOS MDD -35.9%
- **s78 Vol Trend ETF**: IS 1.42 → OOS 1.48, OOS CAGR 29.3%, OOS MDD -14.8%
- **s79 Adaptive Trend**: IS 1.53 → OOS 1.90, OOS CAGR 24.6%, OOS MDD -10.1%
- **s89 DV Momentum**: IS 0.49 → OOS 0.67, OOS CAGR 12.7%, OOS MDD -41.6%
- **s90 Credit Regime**: IS 1.20 → OOS 0.94, OOS CAGR 10.9%, OOS MDD -21.2%
- **s91 Inflation Tilt**: IS 0.60 → OOS 0.89, OOS CAGR 16.6%, OOS MDD -29.8%
- **s92 Country ETF Mom**: IS 0.45 → OOS 0.41, OOS CAGR 6.3%, OOS MDD -37.6%
- **s93 Defensive Rotation**: IS 0.64 → OOS 1.05, OOS CAGR 19.3%, OOS MDD -34.1%
- **s96 Deep Value**: IS 0.43 → OOS 0.40, OOS CAGR 7.6%, OOS MDD -57.6%

### Which were regime mirages?

A strategy graded only on 2017–2024 has been graded on an easy, trending,
mostly-bull window. High OOS-only Sharpe is WEAK evidence — the test that
matters is whether it holds through 2000–2002 and 2008.

Strategies that showed negative Sharpe in 2000-2002 or 2008 regime windows
are equity-beta proxies that happen to trend when markets trend. They are
**not** orthogonal edges — they are disguised beta.

### Does the new batch change the deployment picture?

The key question is not 'is s75 good?' but 'does s75 add anything to a book
that already has S08 (sector rotation) and S02 (TS momentum)?'

Three conditions for a strategy to earn a place:
1. Survives Task 1 (ROBUST or DECAY on full 2000-2024 window)
2. Not a pure bull-regime artifact in Task 2
3. Adds a real independent bet in Task 3 (positive marginal N_eff and SR)

