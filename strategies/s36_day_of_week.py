"""
S36 — Day-of-week effects (cross-sectional and time-series)
Universe:  SPY; optionally ES futures (same for daily holding)
Signal:    Historically, Monday has weak returns; Friday has strong returns.
           Go long SPY on Thursday–Friday (buy Wed close, sell Fri close).
           Go flat (or short via inverse) on Monday.
Execution: Hold specific days; daily position change.
"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import logging
import numpy as np
import pandas as pd

from data import load_price_series, ADJ_TOTALRETURN
from engine import apply_costs

log = logging.getLogger(__name__)
DESCRIPTION = "Day-of-week seasonality, SPY long Thu-Fri short Mon, daily"
TRADING_DAYS = 252

LONG_DAYS  = {3, 4}   # Thursday=3, Friday=4
SHORT_DAYS = {0}       # Monday=0


def run(config: dict) -> dict:
    cfg      = config["backtest"]
    start    = cfg["start_date"]
    end      = cfg["end_date"]
    cache    = config["paths"]["cache_dir"]
    cost_bps = config["costs"]["equity_cost_bps"]
    slip_bps = config["costs"]["equity_slippage_bps"]

    spy = load_price_series("SPY", start=start, end=end, adjustment=ADJ_TOTALRETURN, cache_dir=cache)
    if spy.empty:
        return {"returns": pd.Series(dtype=float), "benchmark": pd.Series(dtype=float),
                "description": DESCRIPTION, "turnover_annual": 0.0}

    trading_idx = pd.bdate_range(start, end)
    ret = spy["Close"].reindex(trading_idx, method="ffill").pct_change(fill_method=None).fillna(0.0)
    dow = pd.Series(trading_idx.dayofweek, index=trading_idx)

    # Signal: +1 on long days, -1 on short days, 0 otherwise
    # Shift 1: use yesterday's day-of-week to determine today's signal
    # Actually: DOW is known before open, so signal for day T = DOW of T (no shift needed
    # for end-of-day execution entering at prior close).
    signal = pd.Series(0.0, index=trading_idx)
    signal[dow.isin(LONG_DAYS)]  =  1.0
    signal[dow.isin(SHORT_DAYS)] = -1.0

    port_ret = signal * ret
    to = signal.diff().abs().fillna(0.0)
    net_ret = apply_costs(port_ret, to, cost_bps, slip_bps)

    # Event study: mean return by day of week
    for d, name in enumerate(["Mon","Tue","Wed","Thu","Fri"]):
        mask = (dow == d)
        if mask.any():
            log.info("S36: %s avg = %.3f%%", name, float(ret[mask].mean()) * 100)

    bm = ret.copy()
    ann_to = float(to.sum() / max(len(to) / TRADING_DAYS, 1))

    return {
        "returns": net_ret, "benchmark": bm,
        "description": DESCRIPTION, "turnover_annual": ann_to,
    }
