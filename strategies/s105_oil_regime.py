"""
s105 — Oil / Energy Regime
Signal: XLE 12-month return > SPY 12-month return -> energy outperforming -> tilt XLE.
        XLE in downtrend (below 200d MA) -> hold SPY only.
        Else: 50% XLE + 50% SPY. Monthly rebalance.
"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import logging
import numpy as np
import pandas as pd
from data import load_price_series, ADJ_TOTALRETURN
from engine import apply_costs

log = logging.getLogger(__name__)
DESCRIPTION = "Oil/energy regime: tilt XLE when outperforming and in uptrend, else SPY"

def run(config):
    cfg   = config["backtest"]
    start, end = cfg["start_date"], cfg["end_date"]
    cache = config["paths"]["cache_dir"]
    cost_bps = config["costs"]["equity_cost_bps"]
    slip_bps = config["costs"]["equity_slippage_bps"]

    spy = load_price_series("SPY", start=start, end=end, adjustment=ADJ_TOTALRETURN, cache_dir=cache)
    xle = load_price_series("XLE", start=start, end=end, adjustment=ADJ_TOTALRETURN, cache_dir=cache)

    idx = pd.bdate_range(start, end)
    spy_c = spy["Close"].reindex(idx, method="ffill")
    xle_c = xle["Close"].reindex(idx, method="ffill")

    spy_r = spy_c.pct_change(fill_method=None).fillna(0.0)
    xle_r = xle_c.pct_change(fill_method=None).fillna(0.0)

    xle_mom12 = xle_c.pct_change(252)
    spy_mom12 = spy_c.pct_change(252)
    xle_ma200 = xle_c.rolling(200).mean()

    xle_outperform = (xle_mom12 > spy_mom12)
    xle_uptrend    = (xle_c > xle_ma200)

    # Signal: both outperform and uptrend -> 50/50; else -> 100% SPY
    signal = (xle_outperform & xle_uptrend).shift(1).fillna(False)
    monthly_signal = signal.resample("ME").last().reindex(idx, method="ffill").fillna(False)

    port_ret = pd.Series(0.0, index=idx)
    port_ret[~monthly_signal] = spy_r[~monthly_signal]
    port_ret[monthly_signal]  = 0.5 * spy_r[monthly_signal] + 0.5 * xle_r[monthly_signal]

    to = monthly_signal.astype(float).diff().abs().fillna(0.0) * 0.5
    net_ret = apply_costs(port_ret, to, cost_bps, slip_bps)
    ann_to = float(to.sum() / max(len(to) / 252, 1))
    return {"returns": net_ret, "turnover_annual": ann_to, "description": DESCRIPTION}
