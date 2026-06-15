"""
S20 — Gross profitability anomaly
Reference: Novy-Marx (2013)
Universe:  Russell 1000 C&P PIT
Signal:    Gross profit / total assets. Long top quintile (most profitable). Annual rebalance.
           Gross profit = revenue − COGS.
Data:      REQUIRES Sharadar Core US Fundamentals.
           FundamentalsStub.gross_profitability(symbols, date) must be implemented.
"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import logging
import pandas as pd

from data import FundamentalsStub, load_price_series, ADJ_TOTALRETURN
from engine import apply_costs

log = logging.getLogger(__name__)
DESCRIPTION = "Gross profitability factor, Russell 1000 PIT [STUB — needs Sharadar]"
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
        fund.gross_profitability("SPY", pd.Timestamp(start).date())
    except NotImplementedError:
        log.warning("S20: gross_profitability not implemented — needs Sharadar fundamentals data")

    return {"returns": pd.Series(0.0, index=trading_idx), "benchmark": bm,
            "description": DESCRIPTION, "turnover_annual": 0.0}
