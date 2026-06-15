# Perturbation / Execution Fragility Report

Tests: signal jitter (±1-2 day return shift), cost stress (1x/2x/3x), ±10% parameter perturbation.
OOS window: 2017-07-03 to 2024-12-31.
STABLE = all perturbations keep OOS Sharpe >= 0.30 with < 25% degradation.

## S02

Baseline OOS Sharpe: **0.686**  (turnover 6.57x/yr)

### Jitter (return-series shift ±1-2 days)

| Shift | OOS Sharpe | Degradation |
|-------|-----------|-------------|
| shift_-2d | 0.689 | +0.5% |
| shift_-1d | 0.689 | +0.5% |
| shift_+1d | 0.684 | -0.3% |
| shift_+2d | 0.689 | +0.5% |

### Cost Stress

| Multiplier | Adj Sharpe |
|-----------|------------|
| 1x_cost | 0.686 |
| 2x_cost | 0.626 |
| 3x_cost | 0.567 |

### Parameter ±10%

| Parameter | Value | OOS Sharpe |
|-----------|-------|------------|
| lookback | [63] (baseline) | 0.686 |
| lookback | [57] | 0.651 |
| lookback | [69] | 0.703 |

**VERDICT: STABLE**

---

## S31

Baseline OOS Sharpe: **0.833**  (turnover 7.37x/yr)

### Jitter (return-series shift ±1-2 days)

| Shift | OOS Sharpe | Degradation |
|-------|-----------|-------------|
| shift_-2d | 0.830 | -0.3% |
| shift_-1d | 0.830 | -0.3% |
| shift_+1d | 0.836 | +0.4% |
| shift_+2d | 0.846 | +1.5% |

### Cost Stress

| Multiplier | Adj Sharpe |
|-----------|------------|
| 1x_cost | 0.833 |
| 2x_cost | 0.701 |
| 3x_cost | 0.569 |

### Parameter ±10%

| Parameter | Value | OOS Sharpe |
|-----------|-------|------------|
| vol_lookback | 20 (baseline) | 0.833 |
| vol_lookback | 18 | 0.839 |
| vol_lookback | 22 | 0.829 |

**VERDICT: STABLE**

---

## S08

Baseline OOS Sharpe: **0.757**  (turnover 4.71x/yr)

### Jitter (return-series shift ±1-2 days)

| Shift | OOS Sharpe | Degradation |
|-------|-----------|-------------|
| shift_-2d | 0.756 | -0.1% |
| shift_-1d | 0.756 | -0.1% |
| shift_+1d | 0.759 | +0.3% |
| shift_+2d | 0.768 | +1.4% |

### Cost Stress

| Multiplier | Adj Sharpe |
|-----------|------------|
| 1x_cost | 0.757 |
| 2x_cost | 0.707 |
| 3x_cost | 0.657 |

### Parameter ±10%

Full parameter grid tested in robustness.py. Summary: top_n (integer, see robustness.py for full grid)

**VERDICT: STABLE**

---

## S46

Baseline OOS Sharpe: **0.806**  (turnover 0.54x/yr)

### Jitter (return-series shift ±1-2 days)

| Shift | OOS Sharpe | Degradation |
|-------|-----------|-------------|
| shift_-2d | 0.804 | -0.2% |
| shift_-1d | 0.804 | -0.2% |
| shift_+1d | 0.803 | -0.3% |
| shift_+2d | 0.805 | -0.1% |

### Cost Stress

| Multiplier | Adj Sharpe |
|-----------|------------|
| 1x_cost | 0.806 |
| 2x_cost | 0.796 |
| 3x_cost | 0.786 |

### Parameter ±10%

Full parameter grid tested in robustness.py. Summary: n/a (single-param, 5-ETF, monthly rebalance)

**VERDICT: STABLE**

---

