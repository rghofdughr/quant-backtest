# Robustness Report

ROBUST strategies: S02, S08, S30, S35, S46, S49  (OOS Sharpe >= 0.7x IS, IS > 0.3, both positive)  
Edge case included: S31 (IS 0.30, OOS 0.83)  
Base costs: 5 bps commission + 5 bps slippage one-way (10 bps total one-way).  
IS: 2000-01-03 to 2017-06-30 | OOS: 2017-07-03 to 2024-12-31


---

## 1. Cost Stress

Additional annual cost drag at nx multiplier vs 1x base = (n-1) x turnover x 0.20% (round-trip 10bps x 2).  
Sharpe approximation: delta = -extra_drag / annualized_vol.  
Flag: strategy becomes non-viable (Sharpe < 0.20) at 2x costs.


| ID  | Name              | TO/yr |  1x Sh |  2x Sh |  3x Sh |  2x CAGR | Non-viable@2x? |
|-----|-------------------|-------|--------|--------|--------|----------|----------------|

| S02 | TS Momentum       |  6.6x | +0.75 | +0.68 | +0.61 |   +12.2% | no             |

| S08 | Sector Rotation   |  4.7x | +0.51 | +0.46 | +0.40 |    +6.8% | no             |

| S30 | Low Volatility    |  0.8x | +0.71 | +0.70 | +0.69 |    +9.9% | no             |

| S35 | Sell in May       |  1.9x | +0.43 | +0.40 | +0.38 |    +4.7% | no             |

| S46 | Risk Parity       |  0.5x | +0.54 | +0.53 | +0.52 |    +5.9% | no             |

| S49 | Dollar Regime     |  0.8x | +0.56 | +0.55 | +0.54 |    +9.9% | no             |

| S31 | Vol Targeting     |  7.4x | +0.46 | +0.33 | +0.19 |    +3.1% | no             |


---

## 2. Parameter Sensitivity

A robust strategy should work across most of the grid.  
OVERFIT flag: OOS Sharpe collapses (< 0.20) outside one specific combo.


### S02 — TS Momentum: lookback (3m/6m/12m) x long-flat/long-short


| Config          | IS Sh | OOS Sh | Full Sh | OOS CAGR |
|-----------------|-------|--------|---------|----------|

| lb=63d LF       | +0.79 | +0.69 | +0.75 |   +13.5% |

| lb=63d LS       | +0.33 | +0.09 | +0.27 |    +0.3% |

| lb=126d LF      | +0.73 | +0.38 | +0.61 |    +6.1% |

| lb=126d LS      | +0.24 | +0.03 | +0.19 |    -0.6% |

| lb=252d LF      | +0.83 | +0.37 | +0.66 |    +5.9% |

| lb=252d LS      | +0.43 | +0.23 | +0.38 |    +2.2% |


OOS Sharpe range: 0.03 to 0.69  (COLLAPSES at some settings)


### S08 — Sector Rotation: top_n (1/2/3/5) x lookback months (1/3/6)


| Config          | IS Sh | OOS Sh | Full Sh | OOS CAGR |
|-----------------|-------|--------|---------|----------|

| top1 lb1m       | +0.21 | +0.46 | +0.28 |    +7.7% |

| top1 lb3m       | +0.16 | +0.52 | +0.28 |   +10.2% |

| top1 lb6m       | +0.45 | +0.74 | +0.55 |   +16.6% |

| top2 lb1m       | +0.21 | +0.72 | +0.34 |   +10.9% |

| top2 lb3m       | +0.29 | +0.82 | +0.47 |   +15.9% |

| top2 lb6m       | +0.50 | +0.69 | +0.56 |   +12.9% |

| top3 lb1m       | +0.39 | +0.82 | +0.51 |   +12.1% |

| top3 lb3m       | +0.40 | +0.76 | +0.51 |   +13.3% |

| top3 lb6m       | +0.51 | +0.82 | +0.61 |   +14.7% |

| top5 lb1m       | +0.59 | +0.94 | +0.68 |   +13.2% |

| top5 lb3m       | +0.54 | +0.71 | +0.59 |   +11.6% |

| top5 lb6m       | +0.59 | +0.86 | +0.68 |   +15.1% |


OOS Sharpe range: 0.46 to 0.94  (ROBUST across grid)


### S30 — Low Volatility: vol_lookback (63/126/252d) x long-flat/long-short


| Config          | IS Sh | OOS Sh | Full Sh | OOS CAGR |
|-----------------|-------|--------|---------|----------|

| vol=63d LF      | +0.77 | +0.78 | +0.77 |   +11.8% |

| vol=63d LS      | +0.03 | -0.20 | -0.03 |    -7.6% |

| vol=126d LF     | +0.78 | +0.72 | +0.76 |   +10.8% |

| vol=126d LS     | +0.09 | -0.24 | -0.00 |    -8.8% |

| vol=252d LF     | +0.72 | +0.69 | +0.71 |   +10.6% |

| vol=252d LS     | +0.05 | -0.25 | -0.03 |    -8.9% |


OOS Sharpe range: -0.25 to 0.78  (COLLAPSES at some settings)


### S35 / S46 / S49 — Parameter notes

- **S35 (Sell in May):** Only parameter is the calendar window (Nov-Apr), which is the published Bouman & Jacobsen specification. No meaningful grid to test.
- **S46 (Risk Parity):** Assets list is configurable (SPY/TLT/GLD/DBC/VNQ). Monthly rebalance is hardcoded. Low turnover (0.54x/yr) makes it very cost-insensitive.
- **S49 (Dollar Regime):** DMA windows (50/200) are hardcoded. Variations (20/50, 100/200) would test signal sensitivity but require code modification.



---

## 3. Regime Breakdown


Four distinct macro regimes + worst-day/worst-week for short-vol strategies.


**Sharpe by regime** (rf=0, annualized)


| ID  | Name              | GFC 2008         | Calm 2013-17     | COVID 2020       | Rates 2022       |
|-----|-------------------|------------------|------------------|------------------|------------------|

| S02 | TS Momentum       |            -0.21 |            +0.23 |            +1.33 |            -0.42 |

| S08 | Sector Rotation   |            -1.15 |            +0.70 |            +0.48 |            +0.10 |

| S30 | Low Volatility    |            -0.71 |            +1.37 |            +0.49 |            +0.01 |

| S35 | Sell in May       |            -0.31 |            +0.93 |            +0.28 |            -0.84 |

| S46 | Risk Parity       |            -0.15 |            +0.29 |            +0.85 |            -0.60 |

| S49 | Dollar Regime     |            -1.08 |            +0.67 |            +0.70 |            -0.70 |

| S31 | Vol Targeting     |            -1.32 |            +0.74 |            +0.91 |            -1.08 |


**Max Drawdown by regime**


| ID  | Name              | GFC 2008         | Calm 2013-17     | COVID 2020       | Rates 2022       |
|-----|-------------------|------------------|------------------|------------------|------------------|

| S02 | TS Momentum       |           -45.6% |           -42.8% |           -32.1% |           -28.1% |

| S08 | Sector Rotation   |           -41.6% |           -14.5% |           -28.7% |           -23.9% |

| S30 | Low Volatility    |           -32.4% |           -11.2% |           -31.3% |           -17.0% |

| S35 | Sell in May       |           -25.6% |           -13.2% |           -33.7% |           -17.2% |

| S46 | Risk Parity       |           -22.9% |           -14.0% |           -18.2% |           -20.7% |

| S49 | Dollar Regime     |           -56.5% |           -17.6% |           -33.7% |           -24.5% |

| S31 | Vol Targeting     |           -17.1% |           -12.6% |           -11.5% |           -14.3% |


**CAGR by regime**


| ID  | Name              | GFC 2008         | Calm 2013-17     | COVID 2020       | Rates 2022       |
|-----|-------------------|------------------|------------------|------------------|------------------|

| S02 | TS Momentum       |            -8.2% |            +2.4% |           +47.7% |           -13.1% |

| S08 | Sector Rotation   |           -30.1% |            +8.5% |           +11.0% |            -0.1% |

| S30 | Low Volatility    |           -23.6% |           +15.9% |           +11.2% |            -1.4% |

| S35 | Sell in May       |           -10.7% |            +7.8% |            +3.8% |           -13.4% |

| S46 | Risk Parity       |            -3.9% |            +2.0% |           +13.1% |            -9.7% |

| S49 | Dollar Regime     |           -44.7% |            +9.4% |           +19.1% |           -17.7% |

| S31 | Vol Targeting     |           -14.5% |            +7.9% |           +10.7% |           -11.5% |


### Short-Vol Tail Risk: S27 (VIX Carry)


S27 was flagged DECAY (IS Sharpe 0.471, OOS Sharpe 0.239 — partial breakdown).  
Daily daily resolution understates tail risk for short-vol strategies:  


- Worst single day: **-31.99%**

- Worst single week: **-47.09%**

- Full Sharpe 0.40 masks these tail events. Feb-2018 VIX spike and  
  Mar-2020 VIX spike caused intraday moves of 80-90% for SVXY  
  that daily returns do not fully capture.  
- **Do not classify S27 as DEPLOY-CANDIDATE regardless of Sharpe.**



---

## 4. Per-Strategy Verdicts


### S02 — TS Momentum: **NEEDS-WORK**


- IS Sharpe: 0.79 | OOS Sharpe: 0.69  

- Annual turnover: 6.57x | Sharpe at 2x costs: +0.68  

- 2008 GFC: Sharpe -0.21, MDD -45.6%, CAGR  -8.2%  

- COVID 2020: Sharpe +1.33, MDD -32.1%, CAGR +47.7%  

- Rates 2022: Sharpe -0.42, MDD -28.1%, CAGR -13.1%  

- Calm 2013-17: Sharpe +0.23, MDD -42.8%, CAGR  +2.4%  


### S08 — Sector Rotation: **DEPLOY-CANDIDATE**


- IS Sharpe: 0.40 | OOS Sharpe: 0.76  

- Annual turnover: 4.71x | Sharpe at 2x costs: +0.46  

- 2008 GFC: Sharpe -1.15, MDD -41.6%, CAGR -30.1%  

- COVID 2020: Sharpe +0.48, MDD -28.7%, CAGR +11.0%  

- Rates 2022: Sharpe +0.10, MDD -23.9%, CAGR  -0.1%  

- Calm 2013-17: Sharpe +0.70, MDD -14.5%, CAGR  +8.5%  


### S30 — Low Volatility: **NEEDS-WORK**


- IS Sharpe: 0.72 | OOS Sharpe: 0.69  

- Annual turnover: 0.83x | Sharpe at 2x costs: +0.70  

- 2008 GFC: Sharpe -0.71, MDD -32.4%, CAGR -23.6%  

- COVID 2020: Sharpe +0.49, MDD -31.3%, CAGR +11.2%  

- Rates 2022: Sharpe +0.01, MDD -17.0%, CAGR  -1.4%  

- Calm 2013-17: Sharpe +1.37, MDD -11.2%, CAGR +15.9%  


### S35 — Sell in May: **DEPLOY-CANDIDATE**


- IS Sharpe: 0.39 | OOS Sharpe: 0.53  

- Annual turnover: 1.93x | Sharpe at 2x costs: +0.40  

- 2008 GFC: Sharpe -0.31, MDD -25.6%, CAGR -10.7%  

- COVID 2020: Sharpe +0.28, MDD -33.7%, CAGR  +3.8%  

- Rates 2022: Sharpe -0.84, MDD -17.2%, CAGR -13.4%  

- Calm 2013-17: Sharpe +0.93, MDD -13.2%, CAGR  +7.8%  


### S46 — Risk Parity: **DEPLOY-CANDIDATE**


- IS Sharpe: 0.45 | OOS Sharpe: 0.81  

- Annual turnover: 0.54x | Sharpe at 2x costs: +0.53  

- 2008 GFC: Sharpe -0.15, MDD -22.9%, CAGR  -3.9%  

- COVID 2020: Sharpe +0.85, MDD -18.2%, CAGR +13.1%  

- Rates 2022: Sharpe -0.60, MDD -20.7%, CAGR  -9.7%  

- Calm 2013-17: Sharpe +0.29, MDD -14.0%, CAGR  +2.0%  


### S49 — Dollar Regime: **DEPLOY-CANDIDATE**


- IS Sharpe: 0.56 | OOS Sharpe: 0.55  

- Annual turnover: 0.81x | Sharpe at 2x costs: +0.55  

- 2008 GFC: Sharpe -1.08, MDD -56.5%, CAGR -44.7%  

- COVID 2020: Sharpe +0.70, MDD -33.7%, CAGR +19.1%  

- Rates 2022: Sharpe -0.70, MDD -24.5%, CAGR -17.7%  

- Calm 2013-17: Sharpe +0.67, MDD -17.6%, CAGR  +9.4%  


### S31 — Vol Targeting: **DEPLOY-CANDIDATE (borderline IS)**


- IS Sharpe: 0.30 | OOS Sharpe: 0.83  

- Annual turnover: 7.37x | Sharpe at 2x costs: +0.33  

- 2008 GFC: Sharpe -1.32, MDD -17.1%, CAGR -14.5%  

- COVID 2020: Sharpe +0.91, MDD -11.5%, CAGR +10.7%  

- Rates 2022: Sharpe -1.08, MDD -14.3%, CAGR -11.5%  

- Calm 2013-17: Sharpe +0.74, MDD -12.6%, CAGR  +7.9%  


### Summary


| ID  | Name              | IS Sh | OOS Sh | 2x Cost Sh | Verdict               |
|-----|-------------------|-------|--------|------------|-----------------------|

| S02 | TS Momentum       | +0.79 | +0.69 |      +0.68 | NEEDS-WORK            |

| S08 | Sector Rotation   | +0.40 | +0.76 |      +0.46 | DEPLOY-CANDIDATE      |

| S30 | Low Volatility    | +0.72 | +0.69 |      +0.70 | NEEDS-WORK            |

| S35 | Sell in May       | +0.39 | +0.53 |      +0.40 | DEPLOY-CANDIDATE      |

| S46 | Risk Parity       | +0.45 | +0.81 |      +0.53 | DEPLOY-CANDIDATE      |

| S49 | Dollar Regime     | +0.56 | +0.55 |      +0.55 | DEPLOY-CANDIDATE      |

| S31 | Vol Targeting     | +0.30 | +0.83 |      +0.33 | DEPLOY-CANDIDATE (borderline IS) |


*S27 (VIX Carry, DECAY): excluded from deploy candidates — short-vol tail risk not adequately modeled at daily resolution. See quarantine-adjacent note above.*
