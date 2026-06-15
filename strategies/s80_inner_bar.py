"""
S80 — Internal-bar-strength reversion on SPY/QQQ.
Entry: close in bottom 10% of 10-day high-low range AND price above 200d MA.
Exit: close above the 5-day midpoint.
Uses ADJ_CAPITAL for price-level (range) computation; ADJ_TOTALRETURN for returns.
Distinct from RSI(2): the 200d trend filter is strict and the exit is range-based.
"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import logging
import numpy as np
import pandas as pd
from data import load_price_series, ADJ_TOTALRETURN, ADJ_CAPITAL
from engine import apply_costs

log = logging.getLogger(__name__)
TRADING_DAYS = 252
DESCRIPTION = "IBS reversion on SPY: close in bottom 10% of 10d range AND above 200d MA; exit above 5d midpoint."
RANGE_WIN = 10
EXIT_WIN = 5
MA_WIN = 200

def run(config):
    cfg = config["backtest"]
    start, end = cfg["start_date"], cfg["end_date"]
    cache = config["paths"]["cache_dir"]
    cost_bps = config["costs"]["equity_cost_bps"]
    slip_bps = config["costs"]["equity_slippage_bps"]
    start_load = "1997-01-01"

    spy_c = load_price_series("SPY", start=start_load, end=end, adjustment=ADJ_CAPITAL, cache_dir=cache)
    spy_r = load_price_series("SPY", start=start_load, end=end, adjustment=ADJ_TOTALRETURN, cache_dir=cache)
    if spy_c.empty or spy_r.empty:
        raise RuntimeError("S80: no SPY data")

    trading_idx = pd.bdate_range(start, end)
    close_c = spy_c["Close"].reindex(trading_idx, method="ffill")
    high_c = spy_c["High"].reindex(trading_idx, method="ffill") if "High" in spy_c.columns else close_c
    low_c = spy_c["Low"].reindex(trading_idx, method="ffill") if "Low" in spy_c.columns else close_c
    ret = spy_r["Close"].reindex(trading_idx, method="ffill").pct_change(fill_method=None).fillna(0.0)

    port_rets = pd.Series(0.0, index=trading_idx)
    to_series = pd.Series(0.0, index=trading_idx)

    in_position = False
    prev_pos = 0.0
    WARMUP = MA_WIN + RANGE_WIN + 5

    for i, dt in enumerate(trading_idx):
        if dt < pd.Timestamp(start):
            continue
        if i < WARMUP:
            continue

        cur = float(close_c.iloc[i])
        ma200 = float(close_c.iloc[max(i - MA_WIN, 0):i].mean())
        hi10 = float(high_c.iloc[max(i - RANGE_WIN, 0):i + 1].max())
        lo10 = float(low_c.iloc[max(i - RANGE_WIN, 0):i + 1].min())
        rng = hi10 - lo10

        if not all(np.isfinite(v) for v in [cur, ma200, hi10, lo10]):
            position = 0.0
        elif in_position:
            # Exit: close above 5-day midpoint
            hi5 = float(high_c.iloc[max(i - EXIT_WIN, 0):i + 1].max())
            lo5 = float(low_c.iloc[max(i - EXIT_WIN, 0):i + 1].min())
            mid5 = (hi5 + lo5) / 2.0
            if cur > mid5:
                in_position = False
            position = 1.0 if in_position else 0.0
        else:
            # Entry: close in bottom 10% of 10d range AND above 200d MA
            if rng > 1e-6:
                ibs = (cur - lo10) / rng  # 0 = at low, 1 = at high
                if ibs <= 0.10 and cur > ma200:
                    in_position = True
            position = 1.0 if in_position else 0.0

        port_rets.iloc[i] = position * ret.iloc[i]
        to_series.iloc[i] = abs(position - prev_pos) / 2.0
        prev_pos = position

    net_ret = apply_costs(port_rets, to_series, cost_bps, slip_bps)
    spy = load_price_series("SPY", start, end, ADJ_TOTALRETURN, cache)
    bm = spy["Close"].pct_change(fill_method=None).reindex(net_ret.index)
    ann_to = float(to_series.sum() / max(len(to_series) / TRADING_DAYS, 1))
    return {"returns": net_ret, "benchmark": bm, "description": DESCRIPTION, "turnover_annual": ann_to}
