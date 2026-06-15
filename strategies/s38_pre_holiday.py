"""
S38 — Pre-holiday drift (Ariel 1990)
Universe:  SPY
Signal:    Long SPY on the last trading day before each US market holiday.
Execution: Enter at close of pre-holiday day; exit at next open (approx next close).
"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import logging
import numpy as np
import pandas as pd

from data import load_price_series, ADJ_TOTALRETURN
from engine import apply_costs
from calendars import market_holidays

log = logging.getLogger(__name__)
DESCRIPTION = "Pre-holiday drift, SPY, long day before each US market holiday"
TRADING_DAYS = 252


def run(config: dict) -> dict:
    cfg      = config["backtest"]
    start    = cfg["start_date"]
    end      = cfg["end_date"]
    cache    = config["paths"]["cache_dir"]

    cost_bps = config["costs"]["equity_cost_bps"]
    slip_bps = config["costs"]["equity_slippage_bps"]

    spy = load_price_series("SPY", start=start, end=end,
                            adjustment=ADJ_TOTALRETURN, cache_dir=cache)
    if spy.empty:
        return {"returns": pd.Series(dtype=float), "benchmark": pd.Series(dtype=float),
                "description": DESCRIPTION, "turnover_annual": 0.0}

    trading_idx = pd.bdate_range(start, end)
    ret = spy["Close"].reindex(trading_idx, method="ffill").pct_change(fill_method=None).fillna(0.0)

    # Get holidays and find the last trading day before each
    holidays = market_holidays(start, end)
    pre_holiday = pd.Series(False, index=trading_idx)

    for hol in holidays:
        hol_ts = pd.Timestamp(hol)
        prior  = trading_idx[trading_idx < hol_ts]
        if prior.empty:
            continue
        # The return OF the day after the pre-holiday day = the next day's return
        # We want to capture the close-to-close return on the day before the holiday
        # i.e., hold on the pre-holiday day itself (enter at prior-day close, exit at pre-holiday close)
        pre_hol_day = prior[-1]
        pre_holiday[pre_hol_day] = True

    # Use shifted signal: act on next day (returns of the pre-holiday day)
    # shift(1) means: signal from yesterday's close → enter at that day's open ≈ close
    # Actually, for pre-holiday we want the RETURN OF THE PRE-HOLIDAY DAY itself
    # No shift: signal at close of (pre_hol - 1), capture return on pre_hol_day
    port_ret = pre_holiday.astype(float) * ret

    hol_rets = ret[pre_holiday]
    log.info("S38: %d pre-holiday events | avg return: %.3f%%",
             int(pre_holiday.sum()), float(hol_rets.mean()) * 100)

    to = pre_holiday.astype(float).diff().abs().fillna(0.0)
    net_ret = apply_costs(port_ret, to, cost_bps, slip_bps)

    bm = ret.copy()
    ann_to = float(to.sum() / max(len(to) / TRADING_DAYS, 1))

    return {
        "returns": net_ret, "benchmark": bm,
        "description": DESCRIPTION, "turnover_annual": ann_to,
    }
