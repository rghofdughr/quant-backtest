# Monte Carlo Summary

Block-bootstrap Monte Carlo: N=10,000 paths, block L=50 days (0.2 months)  
Common window bootstrapped: 2000-01-03 to 2024-12-31  (6,522 trading days)  
Strategies and SPY resampled together (same block draws) — SPY comparison is apples-to-apples.  


---

## Core (S08+S46+S30) | Equal-weight


### Core (S08+S46+S30) | Equal-weight  |  25-year horizon
| Metric         |  5th  | 25th  | median | 75th  | 95th  |
|----------------|-------|-------|--------|-------|-------|
| CAGR           |  4.2% |  6.6% |   8.3% |  9.9% | 12.3% |
| Ann Vol        | 11.6% | 12.2% |  12.9% | 13.5% | 14.5% |
| Sharpe         |  0.37 |  0.55 |   0.68 |  0.81 |  1.02 |
| Max DD         |-44.7% |-35.3% | -29.8% |-26.0% |-20.4% |

**Risk events:**
- P(underperform SPY): **31.4%**
- P(MDD > 30%):        **48.7%**
- P(MDD > 40%):        **12.1%**
- P(MDD > 50%):        **1.6%**
- 95th-pct MDD (size against): **-20.4%**

### Core (S08+S46+S30) | Equal-weight  |  5-year horizon
| Metric         |  5th  | 25th  | median | 75th  | 95th  |
|----------------|-------|-------|--------|-------|-------|
| CAGR           | -0.6% |  4.7% |   8.5% | 12.0% | 16.9% |
| Ann Vol        | 10.4% | 11.4% |  12.3% | 13.9% | 16.4% |
| Sharpe         |  0.03 |  0.41 |   0.71 |  1.01 |  1.43 |
| Max DD         |-34.0% |-25.4% | -18.5% |-13.4% | -9.8% |

**Risk events:**
- P(underperform SPY): **41.9%**
- P(MDD > 30%):        **10.4%**
- P(MDD > 40%):        **1.5%**
- P(MDD > 50%):        **0.1%**
- 95th-pct MDD (size against): **-9.8%**

---

## Core (S08+S46+S30) | Inverse-vol


### Core (S08+S46+S30) | Inverse-vol  |  25-year horizon
| Metric         |  5th  | 25th  | median | 75th  | 95th  |
|----------------|-------|-------|--------|-------|-------|
| CAGR           |  4.3% |  6.6% |   8.2% |  9.7% | 11.9% |
| Ann Vol        | 11.3% | 12.0% |  12.5% | 13.2% | 14.2% |
| Sharpe         |  0.38 |  0.56 |   0.69 |  0.82 |  1.01 |
| Max DD         |-44.6% |-34.7% | -29.4% |-25.5% |-20.2% |

**Risk events:**
- P(underperform SPY): **34.5%**
- P(MDD > 30%):        **46.2%**
- P(MDD > 40%):        **11.0%**
- P(MDD > 50%):        **1.7%**
- 95th-pct MDD (size against): **-20.2%**

### Core (S08+S46+S30) | Inverse-vol  |  5-year horizon
| Metric         |  5th  | 25th  | median | 75th  | 95th  |
|----------------|-------|-------|--------|-------|-------|
| CAGR           | -0.7% |  4.7% |   8.3% | 11.8% | 16.9% |
| Ann Vol        | 10.2% | 11.1% |  12.0% | 13.7% | 16.1% |
| Sharpe         |  0.02 |  0.42 |   0.72 |  1.02 |  1.47 |
| Max DD         |-33.9% |-25.0% | -18.3% |-13.3% | -9.5% |

**Risk events:**
- P(underperform SPY): **42.9%**
- P(MDD > 30%):        **9.8%**
- P(MDD > 40%):        **1.5%**
- P(MDD > 50%):        **0.1%**
- 95th-pct MDD (size against): **-9.5%**

---

## Core (S08+S46+S30) | Cluster-parity


### Core (S08+S46+S30) | Cluster-parity  |  25-year horizon
| Metric         |  5th  | 25th  | median | 75th  | 95th  |
|----------------|-------|-------|--------|-------|-------|
| CAGR           |  3.9% |  6.2% |   7.7% |  9.3% | 11.6% |
| Ann Vol        | 11.0% | 11.6% |  12.1% | 12.7% | 13.6% |
| Sharpe         |  0.36 |  0.55 |   0.68 |  0.81 |  1.01 |
| Max DD         |-44.3% |-34.5% | -29.1% |-25.0% |-20.3% |

**Risk events:**
- P(underperform SPY): **42.3%**
- P(MDD > 30%):        **44.6%**
- P(MDD > 40%):        **10.9%**
- P(MDD > 50%):        **1.7%**
- 95th-pct MDD (size against): **-20.3%**

### Core (S08+S46+S30) | Cluster-parity  |  5-year horizon
| Metric         |  5th  | 25th  | median | 75th  | 95th  |
|----------------|-------|-------|--------|-------|-------|
| CAGR           | -0.9% |  4.4% |   7.9% | 11.4% | 16.2% |
| Ann Vol        |  9.9% | 10.9% |  11.7% | 13.2% | 15.3% |
| Sharpe         | -0.00 |  0.41 |   0.70 |  1.01 |  1.45 |
| Max DD         |-33.5% |-24.4% | -18.6% |-13.5% | -9.4% |

**Risk events:**
- P(underperform SPY): **47.4%**
- P(MDD > 30%):        **9.1%**
- P(MDD > 40%):        **1.5%**
- P(MDD > 50%):        **0.2%**
- 95th-pct MDD (size against): **-9.4%**

---

## Full (all 7) | Equal-weight


### Full (all 7) | Equal-weight  |  25-year horizon
| Metric         |  5th  | 25th  | median | 75th  | 95th  |
|----------------|-------|-------|--------|-------|-------|
| CAGR           |  4.7% |  7.0% |   8.7% | 10.3% | 12.7% |
| Ann Vol        | 11.3% | 11.9% |  12.4% | 13.0% | 13.9% |
| Sharpe         |  0.42 |  0.60 |   0.73 |  0.87 |  1.06 |
| Max DD         |-42.2% |-33.3% | -28.5% |-25.4% |-19.3% |

**Risk events:**
- P(underperform SPY): **22.8%**
- P(MDD > 30%):        **40.6%**
- P(MDD > 40%):        **7.7%**
- P(MDD > 50%):        **0.9%**
- 95th-pct MDD (size against): **-19.3%**

### Full (all 7) | Equal-weight  |  5-year horizon
| Metric         |  5th  | 25th  | median | 75th  | 95th  |
|----------------|-------|-------|--------|-------|-------|
| CAGR           | -0.2% |  5.1% |   8.8% | 12.4% | 17.7% |
| Ann Vol        | 10.4% | 11.2% |  11.9% | 13.3% | 15.7% |
| Sharpe         |  0.06 |  0.45 |   0.74 |  1.05 |  1.49 |
| Max DD         |-32.8% |-24.6% | -17.7% |-13.1% | -9.9% |

**Risk events:**
- P(underperform SPY): **37.6%**
- P(MDD > 30%):        **8.2%**
- P(MDD > 40%):        **1.0%**
- P(MDD > 50%):        **0.1%**
- 95th-pct MDD (size against): **-9.9%**

---

## Full (all 7) | Inverse-vol


### Full (all 7) | Inverse-vol  |  25-year horizon
| Metric         |  5th  | 25th  | median | 75th  | 95th  |
|----------------|-------|-------|--------|-------|-------|
| CAGR           |  4.4% |  6.5% |   8.1% |  9.6% | 11.8% |
| Ann Vol        | 10.8% | 11.3% |  11.8% | 12.3% | 13.2% |
| Sharpe         |  0.41 |  0.59 |   0.71 |  0.85 |  1.04 |
| Max DD         |-40.8% |-31.8% | -27.3% |-24.3% |-18.5% |

**Risk events:**
- P(underperform SPY): **33.9%**
- P(MDD > 30%):        **33.3%**
- P(MDD > 40%):        **5.9%**
- P(MDD > 50%):        **0.6%**
- 95th-pct MDD (size against): **-18.5%**

### Full (all 7) | Inverse-vol  |  5-year horizon
| Metric         |  5th  | 25th  | median | 75th  | 95th  |
|----------------|-------|-------|--------|-------|-------|
| CAGR           | -0.2% |  4.7% |   8.1% | 11.4% | 16.3% |
| Ann Vol        | 10.0% | 10.7% |  11.4% | 12.6% | 14.8% |
| Sharpe         |  0.05 |  0.44 |   0.73 |  1.02 |  1.46 |
| Max DD         |-31.2% |-23.1% | -17.0% |-12.8% | -9.5% |

**Risk events:**
- P(underperform SPY): **43.2%**
- P(MDD > 30%):        **6.5%**
- P(MDD > 40%):        **0.8%**
- P(MDD > 50%):        **0.1%**
- 95th-pct MDD (size against): **-9.5%**

---

## Full (all 7) | Cluster-parity


### Full (all 7) | Cluster-parity  |  25-year horizon
| Metric         |  5th  | 25th  | median | 75th  | 95th  |
|----------------|-------|-------|--------|-------|-------|
| CAGR           |  4.9% |  7.4% |   9.2% | 10.9% | 13.4% |
| Ann Vol        | 11.4% | 11.9% |  12.4% | 12.9% | 13.7% |
| Sharpe         |  0.44 |  0.63 |   0.77 |  0.91 |  1.11 |
| Max DD         |-42.8% |-34.0% | -28.8% |-25.1% |-20.1% |

**Risk events:**
- P(underperform SPY): **17.8%**
- P(MDD > 30%):        **43.4%**
- P(MDD > 40%):        **8.8%**
- P(MDD > 50%):        **1.2%**
- 95th-pct MDD (size against): **-20.1%**

### Full (all 7) | Cluster-parity  |  5-year horizon
| Metric         |  5th  | 25th  | median | 75th  | 95th  |
|----------------|-------|-------|--------|-------|-------|
| CAGR           |  0.0% |  5.5% |   9.3% | 13.2% | 18.8% |
| Ann Vol        | 10.5% | 11.3% |  11.9% | 13.1% | 15.3% |
| Sharpe         |  0.07 |  0.49 |   0.79 |  1.11 |  1.55 |
| Max DD         |-33.1% |-24.9% | -18.2% |-13.7% |-10.1% |

**Risk events:**
- P(underperform SPY): **35.1%**
- P(MDD > 30%):        **9.1%**
- P(MDD > 40%):        **1.1%**
- P(MDD > 50%):        **0.1%**
- 95th-pct MDD (size against): **-10.1%**
