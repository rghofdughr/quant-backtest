"""
s107 — Real Estate Regime (VNQ trend)
Signal: VNQ above 200d MA -> long VNQ; below -> cash. Monthly rebalance.
"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import logging
import numpy as np
import pandas as pd
from data import load_price_series, ADJ_TOTALRETURN
from engine import apply_costs

log = logging.getLogger(__name__)
DESCRIPTION = "Real estate regime: long VNQ when above 200d MA, else cash"

def run(config):
    cfg   = config["backtest"]
    start, end = cfg["start_date"], cfg["end_date"]
    cache = config["paths"]["cache_dir"]
    cost_bps = config["costs"]["equity_cost_bps"]
    slip_bps = config["costs"]["equity_slippage_bps"]

    vnq = load_price_series("VNQ", start=start, end=end, adjustment=ADJ_TOTALRETURN, cache_dir=cache)
    if vnq.empty:
        return {"returns": pd.Series(dtype=float), "turnover_annual": 0.0, "description": DESCRIPTION}

    idx = pd.bdate_range(start, end)
    vnq_c = vnq["Close"].reindex(idx, method="ffill")
    vnq_r = vnq_c.pct_change(fill_method=None).fillna(0.0)

    ma200 = vnq_c.rolling(200).mean()
    above = (vnq_c > ma200).shift(1).fillna(False)

    monthly_above = above.resample("ME").last().reindex(idx, method="ffill").fillna(False)
    vnq_available = vnq_c.notna().resample("ME").last().reindex(idx, method="ffill").fillna(False)

    in_market = monthly_above & vnq_available
    port_ret = pd.Series(0.0, index=idx)
    port_ret[in_market] = vnq_r[in_market]

    to = in_market.astype(float).diff().abs().fillna(0.0)
    net_ret = apply_costs(port_ret, to, cost_bps, slip_bps)
    ann_to = float(to.sum() / max(len(to) / 252, 1))
    return {"returns": net_ret, "turnover_annual": ann_to, "description": DESCRIPTION}
