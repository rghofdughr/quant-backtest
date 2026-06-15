"""
S33 — Earnings implied-volatility crush
Reference: Goyal & Saretto (2009); extensive practitioner literature
Universe:  S&P 500 stocks with upcoming earnings announcements
Signal:    Sell straddle 1-2 days before earnings announcement; close on announcement day.
           IV inflated before earnings, collapses after announcement regardless of direction.
Data:      REQUIRES options data (ORATS) + earnings calendar (Zacks or Compustat).
           FundamentalsStub.earnings_surprise contains the earnings date framework.
           Stub: implements trade timing and P&L structure; awaiting options + earnings dates.
"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import logging
import pandas as pd

from data import FundamentalsStub, load_price_series, ADJ_TOTALRETURN

log = logging.getLogger(__name__)
DESCRIPTION = "Earnings IV crush (pre-announcement vol sell) [STUB — needs ORATS + Zacks earnings]"
TRADING_DAYS = 252


def run(config: dict) -> dict:
    cfg   = config["backtest"]
    start = cfg["start_date"]
    end   = cfg["end_date"]
    cache = config["paths"]["cache_dir"]

    trading_idx = pd.bdate_range(start, end)
    spy = load_price_series("SPY", start=start, end=end, adjustment=ADJ_TOTALRETURN, cache_dir=cache)
    bm  = spy["Close"].reindex(trading_idx, method="ffill").pct_change(fill_method=None).fillna(0.0) if not spy.empty else pd.Series(0.0, index=trading_idx)

    fund = FundamentalsStub()
    try:
        fund.earnings_surprise("SPY", pd.Timestamp(start).date())
    except NotImplementedError:
        log.warning("S33: requires ORATS options data + earnings dates (Zacks/Compustat)")

    return {"returns": pd.Series(0.0, index=trading_idx), "benchmark": bm,
            "description": DESCRIPTION, "turnover_annual": 0.0}
