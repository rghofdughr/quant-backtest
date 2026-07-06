"""
s114 — Pre-Holiday Effect
Long SPY on the trading day immediately before each US market holiday.
(Ariel 1990: pre-holiday returns average 8x normal daily returns)
"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import logging
from datetime import date
import numpy as np
import pandas as pd
from data import load_price_series, ADJ_TOTALRETURN
from engine import apply_costs

log = logging.getLogger(__name__)
DESCRIPTION = "Pre-holiday effect: long SPY on T-1 before each US market holiday"

def _us_holidays(start_yr, end_yr):
    """Generate approximate US market holiday dates (observed)."""
    from pandas.tseries.holiday import USFederalHolidayCalendar
    cal = USFederalHolidayCalendar()
    return cal.holidays(start=f"{start_yr}-01-01", end=f"{end_yr}-12-31")

def run(config):
    cfg   = config["backtest"]
    start, end = cfg["start_date"], cfg["end_date"]
    cache = config["paths"]["cache_dir"]
    cost_bps = config["costs"]["equity_cost_bps"]
    slip_bps = config["costs"]["equity_slippage_bps"]

    spy = load_price_series("SPY", start=start, end=end, adjustment=ADJ_TOTALRETURN, cache_dir=cache)
    idx = pd.bdate_range(start, end)
    spy_r = spy["Close"].reindex(idx, method="ffill").pct_change(fill_method=None).fillna(0.0)

    # Get all US federal holidays in range
    try:
        holidays = _us_holidays(idx.year.min(), idx.year.max())
    except Exception:
        # Fallback: hardcode approximate Christmas, Thanksgiving, July 4, Labor Day
        holidays = pd.DatetimeIndex([])

    # Mark T-1 before each holiday: find last trading day before each holiday
    pre_holiday_days = set()
    for h in holidays:
        # Find last trading day strictly before the holiday
        preceding = idx[idx < h]
        if len(preceding) > 0:
            pre_holiday_days.add(preceding[-1])

    in_market = pd.Series(False, index=idx)
    for d in pre_holiday_days:
        if d in idx:
            in_market.loc[d] = True

    port_ret = pd.Series(0.0, index=idx)
    port_ret[in_market] = spy_r[in_market]

    to = in_market.astype(float).diff().abs().fillna(0.0)
    net_ret = apply_costs(port_ret, to, cost_bps, slip_bps)
    ann_to = float(to.sum() / max(len(to) / 252, 1))
    pct_days = float(in_market.mean()) * 100
    log.info("s114: pre-holiday days %.1f%% of all trading days", pct_days)
    return {"returns": net_ret, "turnover_annual": ann_to, "description": DESCRIPTION}
