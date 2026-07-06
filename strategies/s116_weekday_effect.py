"""
s116 — Weekday Effect
Monday historically has the lowest (often negative) average return.
Signal: Long SPY Tue-Fri, cash on Monday.
"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import logging
import numpy as np
import pandas as pd
from data import load_price_series, ADJ_TOTALRETURN
from engine import apply_costs

log = logging.getLogger(__name__)
DESCRIPTION = "Weekday effect: long SPY Tue-Fri (avoid Monday which has lowest avg return)"

def run(config):
    cfg   = config["backtest"]
    start, end = cfg["start_date"], cfg["end_date"]
    cache = config["paths"]["cache_dir"]
    cost_bps = config["costs"]["equity_cost_bps"]
    slip_bps = config["costs"]["equity_slippage_bps"]

    spy = load_price_series("SPY", start=start, end=end, adjustment=ADJ_TOTALRETURN, cache_dir=cache)
    idx = pd.bdate_range(start, end)
    spy_r = spy["Close"].reindex(idx, method="ffill").pct_change(fill_method=None).fillna(0.0)

    # dayofweek: 0=Monday, 1=Tue, ..., 4=Friday
    not_monday = pd.Series(idx.dayofweek != 0, index=idx)

    port_ret = pd.Series(0.0, index=idx)
    port_ret[not_monday] = spy_r[not_monday]

    to = not_monday.astype(float).diff().abs().fillna(0.0)
    net_ret = apply_costs(port_ret, to, cost_bps, slip_bps)
    ann_to = float(to.sum() / max(len(to) / 252, 1))
    return {"returns": net_ret, "turnover_annual": ann_to, "description": DESCRIPTION}
