"""
s103 — Yield Curve Regime
Signal: TLT 63d return > SHY 63d return -> bonds outperforming short end -> risk-off -> long TLT
        Else risk-on -> long SPY. Monthly rebalance.
"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import logging
import numpy as np
import pandas as pd
from data import load_price_series, ADJ_TOTALRETURN
from engine import apply_costs

log = logging.getLogger(__name__)
DESCRIPTION = "Yield curve regime: long TLT when bonds outrun short end (risk-off), else long SPY"

def run(config):
    cfg   = config["backtest"]
    start, end = cfg["start_date"], cfg["end_date"]
    cache = config["paths"]["cache_dir"]
    cost_bps = config["costs"]["equity_cost_bps"]
    slip_bps = config["costs"]["equity_slippage_bps"]

    spy = load_price_series("SPY", start=start, end=end, adjustment=ADJ_TOTALRETURN, cache_dir=cache)
    tlt = load_price_series("TLT", start=start, end=end, adjustment=ADJ_TOTALRETURN, cache_dir=cache)
    shy = load_price_series("SHY", start=start, end=end, adjustment=ADJ_TOTALRETURN, cache_dir=cache)

    idx = pd.bdate_range(start, end)
    spy_c = spy["Close"].reindex(idx, method="ffill")
    tlt_c = tlt["Close"].reindex(idx, method="ffill")
    shy_c = shy["Close"].reindex(idx, method="ffill")

    spy_r = spy_c.pct_change(fill_method=None).fillna(0.0)
    tlt_r = tlt_c.pct_change(fill_method=None).fillna(0.0)

    # Monthly signal: is TLT 63d return > SHY 63d return (bonds outrunning short end)?
    tlt_mom = tlt_c.pct_change(63)
    shy_mom = shy_c.pct_change(63)
    risk_off = (tlt_mom > shy_mom).shift(1).fillna(False)  # 1-day execution lag

    # Resample to monthly: hold position all month based on last month-end signal
    monthly_signal = risk_off.resample("ME").last().reindex(idx, method="ffill").fillna(False)

    port_ret = pd.Series(0.0, index=idx)
    port_ret[~monthly_signal] = spy_r[~monthly_signal]
    port_ret[monthly_signal]  = tlt_r[monthly_signal]

    to = monthly_signal.astype(float).diff().abs().fillna(0.0)
    net_ret = apply_costs(port_ret, to, cost_bps, slip_bps)
    ann_to = float(to.sum() / max(len(to) / 252, 1))
    return {"returns": net_ret, "turnover_annual": ann_to, "description": DESCRIPTION}
