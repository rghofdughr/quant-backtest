"""
s113 — January Barometer
"As goes January, so goes the year." (Yale Hirsch 1972)
Signal: If SPY return in January > 0 -> long SPY Feb-Dec. Else -> cash.
Annual signal; position held constant within the year.
"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import logging
import numpy as np
import pandas as pd
from data import load_price_series, ADJ_TOTALRETURN
from engine import apply_costs

log = logging.getLogger(__name__)
DESCRIPTION = "January barometer: long SPY if January positive, cash if negative, annual signal"

def run(config):
    cfg   = config["backtest"]
    start, end = cfg["start_date"], cfg["end_date"]
    cache = config["paths"]["cache_dir"]
    cost_bps = config["costs"]["equity_cost_bps"]
    slip_bps = config["costs"]["equity_slippage_bps"]

    spy = load_price_series("SPY", start=start, end=end, adjustment=ADJ_TOTALRETURN, cache_dir=cache)
    idx = pd.bdate_range(start, end)
    spy_c = spy["Close"].reindex(idx, method="ffill")
    spy_r = spy_c.pct_change(fill_method=None).fillna(0.0)

    # For each year, compute January total return, then signal Feb 1 onward
    in_market = pd.Series(False, index=idx)

    for yr in range(idx.year.min(), idx.year.max() + 1):
        jan_days = idx[(idx.year == yr) & (idx.month == 1)]
        if len(jan_days) < 10:
            continue
        jan_start_price = spy_c.loc[jan_days[0]]
        jan_end_price   = spy_c.loc[jan_days[-1]]
        if pd.isna(jan_start_price) or pd.isna(jan_end_price):
            continue
        jan_ret = (jan_end_price / jan_start_price) - 1.0
        # Signal applies Feb 1 through Dec 31 of same year (not Jan of next year)
        feb_to_dec = idx[(idx.year == yr) & (idx.month >= 2)]
        if jan_ret > 0:
            in_market.loc[feb_to_dec] = True

    port_ret = pd.Series(0.0, index=idx)
    port_ret[in_market] = spy_r[in_market]

    to = in_market.astype(float).diff().abs().fillna(0.0)
    net_ret = apply_costs(port_ret, to, cost_bps, slip_bps)
    ann_to = float(to.sum() / max(len(to) / 252, 1))
    return {"returns": net_ret, "turnover_annual": ann_to, "description": DESCRIPTION}
