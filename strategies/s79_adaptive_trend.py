"""
S79 — Adaptive lookback trend on SPY.
Each month, select the MA length (from a fixed candidate set) that had the lowest
whipsaw cost (number of crossings) over the prior 12m. Trade that length next month.
The selection rule is locked — no in-sample fitting of the candidate set itself.
"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import logging
import numpy as np
import pandas as pd
from data import load_price_series, ADJ_TOTALRETURN
from engine import apply_costs

log = logging.getLogger(__name__)
TRADING_DAYS = 252
DESCRIPTION = "Adaptive lookback trend on SPY: monthly select MA length (from fixed set) with fewest prior-12m crossings."
# Fixed candidate set — never optimised, just tested
MA_CANDIDATES = [50, 100, 150, 200]
EVAL_WIN = 252  # 12m evaluation window

def _count_crossings(price: pd.Series, ma_len: int) -> int:
    ma = price.rolling(ma_len, min_periods=ma_len // 2).mean()
    above = (price > ma).astype(int)
    return int(above.diff().abs().sum())

def run(config):
    cfg = config["backtest"]
    start, end = cfg["start_date"], cfg["end_date"]
    cache = config["paths"]["cache_dir"]
    cost_bps = config["costs"]["equity_cost_bps"]
    slip_bps = config["costs"]["equity_slippage_bps"]
    start_load = "1997-01-01"

    spy_df = load_price_series("SPY", start=start_load, end=end, adjustment=ADJ_TOTALRETURN, cache_dir=cache)
    if spy_df.empty:
        raise RuntimeError("S79: no SPY data")

    trading_idx = pd.bdate_range(start, end)
    price = spy_df["Close"].reindex(trading_idx, method="ffill")
    ret = price.pct_change(fill_method=None).fillna(0.0)

    reb_dates = pd.date_range(start, end, freq="BME")
    WARMUP = EVAL_WIN + max(MA_CANDIDATES) + 5

    # Current MA length state
    current_ma = MA_CANDIDATES[-1]  # default to longest (most stable)
    ma_schedule = {}  # date -> ma_length

    for rd in reb_dates:
        rd = min(rd, trading_idx[-1])
        pos = trading_idx.searchsorted(rd)
        if pos < WARMUP:
            continue
        window = price.iloc[max(pos - EVAL_WIN, 0):pos]
        crossings = {m: _count_crossings(window, m) for m in MA_CANDIDATES}
        current_ma = min(crossings, key=crossings.get)
        ma_schedule[rd] = current_ma

    port_rets = pd.Series(0.0, index=trading_idx)
    to_series = pd.Series(0.0, index=trading_idx)
    active_ma = MA_CANDIDATES[-1]
    prev_pos = 0.0

    reb_sorted = sorted(ma_schedule.keys())

    for i, dt in enumerate(trading_idx):
        if dt < pd.Timestamp(start):
            continue
        if i < max(MA_CANDIDATES) + 5:
            continue

        # Update MA on rebalance dates
        reb_before = [r for r in reb_sorted if r <= dt]
        if reb_before:
            active_ma = ma_schedule.get(reb_before[-1], active_ma)

        ma_val = float(price.iloc[max(i - active_ma, 0):i].mean())
        cur = float(price.iloc[i])
        position = 1.0 if (np.isfinite(cur) and np.isfinite(ma_val) and cur > ma_val) else 0.0

        port_rets.iloc[i] = position * ret.iloc[i]
        to_series.iloc[i] = abs(position - prev_pos) / 2.0
        prev_pos = position

    net_ret = apply_costs(port_rets, to_series, cost_bps, slip_bps)
    spy = load_price_series("SPY", start, end, ADJ_TOTALRETURN, cache)
    bm = spy["Close"].pct_change(fill_method=None).reindex(net_ret.index)
    ann_to = float(to_series.sum() / max(len(to_series) / TRADING_DAYS, 1))
    return {"returns": net_ret, "benchmark": bm, "description": DESCRIPTION, "turnover_annual": ann_to}
