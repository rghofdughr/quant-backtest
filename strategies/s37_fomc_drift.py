"""
S37 — Pre-FOMC announcement drift (Lucca & Moench 2015)
Universe:  SPY
Signal:    Long SPY in the 24 hours before each scheduled FOMC announcement.
           Enter at close of day-before announcement; exit at close of announcement day.
Execution: Close-to-close holding over the FOMC day.
Analysis:  Event-study average return + compounded tradeable strategy.
"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import logging
import numpy as np
import pandas as pd

from data import load_price_series, ADJ_TOTALRETURN
from engine import apply_costs
from calendars import FOMC_DATES

log = logging.getLogger(__name__)
DESCRIPTION = "Pre-FOMC announcement drift, SPY 24h window, close-to-close"
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

    # FOMC announcement dates within our period
    fomc_set = {pd.Timestamp(d) for d in FOMC_DATES
                if pd.Timestamp(start) <= pd.Timestamp(d) <= pd.Timestamp(end)}

    # Signal: hold SPY on FOMC announcement day (the 24h window ending at FOMC close)
    # In practice: enter at prior day's close, exit at FOMC-day close.
    fomc_signal = pd.Series(False, index=trading_idx)
    for fomc_d in fomc_set:
        # Find the closest trading day on or after the FOMC date
        candidates = trading_idx[trading_idx >= fomc_d]
        if candidates.empty:
            continue
        hold_day = candidates[0]
        fomc_signal[hold_day] = True

    # Enter at prior close (shift 1 backward): this is the FOMC day return
    # No shift needed here: we're buying at prior-day close, capturing FOMC-day return
    port_ret = fomc_signal.astype(float) * ret

    # Event study: average FOMC-day return vs average non-FOMC day
    fomc_rets     = ret[fomc_signal]
    non_fomc_rets = ret[~fomc_signal & (ret != 0)]
    log.info("S37: %d FOMC events | avg FOMC-day ret: %.3f%% | avg non-FOMC: %.3f%%",
             fomc_signal.sum(),
             float(fomc_rets.mean()) * 100,
             float(non_fomc_rets.mean()) * 100)

    to = fomc_signal.astype(float).diff().abs().fillna(0.0)
    net_ret = apply_costs(port_ret, to, cost_bps, slip_bps)

    bm = ret.copy()
    ann_to = float(to.sum() / max(len(to) / TRADING_DAYS, 1))

    return {
        "returns": net_ret, "benchmark": bm,
        "description": DESCRIPTION, "turnover_annual": ann_to,
    }
