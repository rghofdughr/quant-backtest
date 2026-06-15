"""
S17 — Earnings yield (E/P ratio value factor)
Reference: Basu (1977); Fama & French (1992)
Universe:  S&P 500 C&P PIT
Signal:    Trailing 12-month EPS / price. Long top quintile (cheapest on earnings).
           4-month lag for reporting delay.
Data:      REQUIRES Sharadar Core US Fundamentals.
           FundamentalsStub.earnings_yield(symbols, date) must be implemented.
"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import logging
import pandas as pd

from data import FundamentalsStub, load_price_series, watchlist_symbols, ADJ_TOTALRETURN
from engine import portfolio_returns_from_weights, apply_costs

log = logging.getLogger(__name__)
DESCRIPTION = "Earnings yield (E/P) quintile, S&P 500 PIT [STUB — needs Sharadar]"
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
        fund.earnings_yield("SPY", pd.Timestamp(start).date())
    except NotImplementedError:
        log.warning("S17: earnings_yield not implemented — needs Sharadar fundamentals data")

    return {"returns": pd.Series(0.0, index=trading_idx), "benchmark": bm,
            "description": DESCRIPTION, "turnover_annual": 0.0}
