"""
s104 — Gold Regime
Signal: GLD above 200d MA -> 50% GLD + 50% SPY (gold trending, hedge equity).
        GLD below 200d MA -> 100% SPY.
Monthly rebalance.
"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import logging
import numpy as np
import pandas as pd
from data import load_price_series, ADJ_TOTALRETURN
from engine import portfolio_returns_from_weights, apply_costs

log = logging.getLogger(__name__)
DESCRIPTION = "Gold regime: tilt to GLD when above 200d MA, else 100% SPY"

def run(config):
    cfg   = config["backtest"]
    start, end = cfg["start_date"], cfg["end_date"]
    cache = config["paths"]["cache_dir"]
    cost_bps = config["costs"]["equity_cost_bps"]
    slip_bps = config["costs"]["equity_slippage_bps"]

    spy = load_price_series("SPY", start=start, end=end, adjustment=ADJ_TOTALRETURN, cache_dir=cache)
    gld = load_price_series("GLD", start=start, end=end, adjustment=ADJ_TOTALRETURN, cache_dir=cache)

    idx = pd.bdate_range(start, end)
    spy_c = spy["Close"].reindex(idx, method="ffill")
    gld_c = gld["Close"].reindex(idx, method="ffill")

    spy_r = spy_c.pct_change(fill_method=None).fillna(0.0)
    gld_r = gld_c.pct_change(fill_method=None).fillna(0.0)

    gld_ma200 = gld_c.rolling(200).mean()
    gld_above = (gld_c > gld_ma200).shift(1).fillna(False)

    # Monthly rebalance: use month-end signal, hold all month
    monthly_above = gld_above.resample("ME").last().reindex(idx, method="ffill").fillna(False)
    gld_available = gld_c.notna().resample("ME").last().reindex(idx, method="ffill").fillna(False)

    # When GLD available and above MA: 50/50. Else: 100% SPY.
    gld_wt = pd.Series(0.0, index=idx)
    spy_wt = pd.Series(1.0, index=idx)
    in_gold = monthly_above & gld_available
    gld_wt[in_gold] = 0.5
    spy_wt[in_gold] = 0.5

    port_ret = spy_wt * spy_r + gld_wt * gld_r
    prev_gld = gld_wt.shift(1).fillna(0.0)
    prev_spy = spy_wt.shift(1).fillna(1.0)
    to = (gld_wt - prev_gld).abs() + (spy_wt - prev_spy).abs()
    to = to / 2  # one-way

    net_ret = apply_costs(port_ret, to, cost_bps, slip_bps)
    ann_to = float(to.sum() / max(len(to) / 252, 1))
    return {"returns": net_ret, "turnover_annual": ann_to, "description": DESCRIPTION}
