"""
s112 — Turn-of-Month Effect
Signal: Long SPY on last 3 trading days of month + first 3 trading days of next month.
Cash otherwise. (Ariel 1987; Lakonishok & Smidt 1988)
"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import logging
import numpy as np
import pandas as pd
from data import load_price_series, ADJ_TOTALRETURN
from engine import apply_costs

log = logging.getLogger(__name__)
DESCRIPTION = "Turn-of-month: long SPY last 3 + first 3 trading days of each month, cash otherwise"

N_DAYS = 3  # days before and after month boundary

def run(config):
    cfg   = config["backtest"]
    start, end = cfg["start_date"], cfg["end_date"]
    cache = config["paths"]["cache_dir"]
    cost_bps = config["costs"]["equity_cost_bps"]
    slip_bps = config["costs"]["equity_slippage_bps"]

    spy = load_price_series("SPY", start=start, end=end, adjustment=ADJ_TOTALRETURN, cache_dir=cache)
    idx = pd.bdate_range(start, end)
    spy_r = spy["Close"].reindex(idx, method="ffill").pct_change(fill_method=None).fillna(0.0)

    # For each month, mark the last N and first N trading days
    in_market = pd.Series(False, index=idx)

    for (yr, mo), group in spy_r.groupby([spy_r.index.year, spy_r.index.month]):
        days = group.index
        if len(days) >= N_DAYS:
            in_market.loc[days[-N_DAYS:]] = True   # last 3 trading days
        else:
            in_market.loc[days] = True
        if len(days) >= N_DAYS:
            in_market.loc[days[:N_DAYS]] = True    # first 3 trading days
        else:
            in_market.loc[days] = True

    # Signal is purely calendar-based (no lookahead) — no shift needed
    port_ret = pd.Series(0.0, index=idx)
    port_ret[in_market] = spy_r[in_market]

    # Turnover: entry/exit on market boundaries
    to = in_market.astype(float).diff().abs().fillna(0.0)
    net_ret = apply_costs(port_ret, to, cost_bps, slip_bps)
    ann_to = float(to.sum() / max(len(to) / 252, 1))
    return {"returns": net_ret, "turnover_annual": ann_to, "description": DESCRIPTION}
