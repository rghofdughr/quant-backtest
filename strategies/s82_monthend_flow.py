"""
S82 — Month-end rebalancing flow.
Long SPY last 2 trading days of month; fade (short SPY) first 2 trading days of next month.
Distinct from S34 (turn-of-month): that was long last 1 day + 3 forward; this captures
the fade-at-start separately as a distinct signal component.
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
DESCRIPTION = "Month-end rebalancing flow: long SPY last 2 days of month, short SPY first 2 days of next month."

def run(config):
    cfg = config["backtest"]
    start, end = cfg["start_date"], cfg["end_date"]
    cache = config["paths"]["cache_dir"]
    cost_bps = config["costs"]["equity_cost_bps"]
    slip_bps = config["costs"]["equity_slippage_bps"]
    start_load = "1997-01-01"

    spy_df = load_price_series("SPY", start=start_load, end=end, adjustment=ADJ_TOTALRETURN, cache_dir=cache)
    if spy_df.empty:
        raise RuntimeError("S82: no SPY data")

    trading_idx = pd.bdate_range(start, end)
    ret = spy_df["Close"].reindex(trading_idx, method="ffill").pct_change(fill_method=None).fillna(0.0)

    # Build set of last-2 and first-2 trading days of each month
    by_month = {}
    for dt in trading_idx:
        key = (dt.year, dt.month)
        by_month.setdefault(key, []).append(dt)

    long_days = set()
    short_days = set()
    for (yr, mo), days in by_month.items():
        days_sorted = sorted(days)
        for d in days_sorted[-2:]:
            long_days.add(d)
        for d in days_sorted[:2]:
            short_days.add(d)

    port_rets = pd.Series(0.0, index=trading_idx)
    to_series = pd.Series(0.0, index=trading_idx)
    prev_pos = 0.0

    for i, dt in enumerate(trading_idx):
        if dt < pd.Timestamp(start):
            continue
        if dt in long_days and dt not in short_days:
            position = 1.0
        elif dt in short_days and dt not in long_days:
            position = -1.0
        else:
            position = 0.0

        port_rets.iloc[i] = position * ret.iloc[i]
        to_series.iloc[i] = abs(position - prev_pos) / 2.0
        prev_pos = position

    net_ret = apply_costs(port_rets, to_series, cost_bps, slip_bps)
    spy = load_price_series("SPY", start, end, ADJ_TOTALRETURN, cache)
    bm = spy["Close"].pct_change(fill_method=None).reindex(net_ret.index)
    ann_to = float(to_series.sum() / max(len(to_series) / TRADING_DAYS, 1))
    return {"returns": net_ret, "benchmark": bm, "description": DESCRIPTION, "turnover_annual": ann_to}
