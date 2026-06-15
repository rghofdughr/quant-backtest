"""
S43 — Analyst estimate revision momentum
Reference: Chan, Jegadeesh & Lakonishok (1996)
Universe:  S&P 500 C&P PIT
Signal:    3-month earnings estimate revision momentum (ERev = (new_est - old_est) / price).
           Long stocks with highest upward EPS revisions; short lowest.
Data:      REQUIRES I/B/E/S, FactSet, or Zacks estimate history.
           FundamentalsStub.analyst_revisions(symbols, date) → Series of revision scores.
"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import logging
import pandas as pd

from data import FundamentalsStub, load_price_series, ADJ_TOTALRETURN

log = logging.getLogger(__name__)
DESCRIPTION = "Analyst revision momentum [STUB — needs I/B/E/S or Zacks estimate history]"
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
        fund.analyst_revisions("SPY", pd.Timestamp(start).date())
    except NotImplementedError:
        log.warning("S43: analyst_revisions not implemented. Needs I/B/E/S, FactSet, or Zacks estimates.")

    return {"returns": pd.Series(0.0, index=trading_idx), "benchmark": bm,
            "description": DESCRIPTION, "turnover_annual": 0.0}
