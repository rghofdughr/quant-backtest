"""
S34 — Turn-of-month effect (Lakonishok & Smidt 1988)
Universe:  SPY (and breadth-weighted S&P 500 variant)
Signal:    Long from last trading day of month through first N trading days of next month.
           Flat otherwise.
Execution: Market on close of entry day; exit at close of last holding day.
Sweep:     Window from [-1,+3] to [-1,+5]; SPY only vs S&P 500 breadth.
"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import logging
import numpy as np
import pandas as pd

from data import load_price_series, ADJ_TOTALRETURN
from engine import apply_costs

log = logging.getLogger(__name__)
DESCRIPTION = "Turn-of-month effect, SPY, long last day of month through first 3 trading days"
TRADING_DAYS = 252


def run(config: dict) -> dict:
    cfg      = config["backtest"]
    start    = cfg["start_date"]
    end      = cfg["end_date"]
    cache    = config["paths"]["cache_dir"]

    # Window: enter on last_day_of_month - pre_days + 1 through first_days of next month
    pre_days   = 1   # include last N days of ending month (1 = last day only)
    post_days  = 3   # first N days of new month

    cost_bps = config["costs"]["equity_cost_bps"]
    slip_bps = config["costs"]["equity_slippage_bps"]

    spy = load_price_series("SPY", start=start, end=end,
                            adjustment=ADJ_TOTALRETURN, cache_dir=cache)
    if spy.empty:
        return {"returns": pd.Series(dtype=float), "benchmark": pd.Series(dtype=float),
                "description": DESCRIPTION, "turnover_annual": 0.0}

    trading_idx = pd.bdate_range(start, end)
    close = spy["Close"].reindex(trading_idx, method="ffill")
    ret   = close.pct_change(fill_method=None).fillna(0.0)

    # Build set of "turn-of-month" trading days
    in_window = pd.Series(False, index=trading_idx)

    months = pd.date_range(start, end, freq="MS")  # month starts
    for ms in months:
        # Last trading day(s) of previous month
        prev_month_days = trading_idx[trading_idx < ms]
        if len(prev_month_days) >= pre_days:
            entry_days = prev_month_days[-pre_days:]
            in_window[entry_days] = True

        # First N trading days of this month
        this_month_days = trading_idx[
            (trading_idx >= ms) &
            (trading_idx < ms + pd.offsets.MonthEnd(1))
        ]
        exit_days = this_month_days[:post_days]
        in_window[exit_days] = True

    # Shift signal 1 day: act on next close after signal
    signal = in_window.shift(1).fillna(False).astype(bool)

    port_ret = signal.astype(float) * ret
    to       = signal.astype(float).diff().abs().fillna(0.0)
    net_ret  = apply_costs(port_ret, to, cost_bps, slip_bps)

    bm = ret.copy()
    ann_to = float(to.sum() / max(len(to) / TRADING_DAYS, 1))
    in_pct = float(signal.mean()) * 100

    log.info("S34 done. In-window: %.1f%% of days, ann turnover: %.1fx", in_pct, ann_to)

    return {
        "returns": net_ret, "benchmark": bm,
        "description": DESCRIPTION, "turnover_annual": ann_to,
    }
