"""
S44 — Merger arbitrage (risk arbitrage)
Reference: Mitchell & Pulvino (2001); Baker & Savasoglu (2002)
Universe:  All announced M&A deals with public US target companies
Signal:    Long target at announcement; size by deal probability × spread / downside.
           If deal breaks: target loses ~20-30% of prior premium → risk management critical.
Data:      REQUIRES M&A deal database: Refinitiv SDC Platinum, Bloomberg, or specialty vendor.
           Deal database includes: announcement date, target, acquirer, deal price,
           deal structure (cash/stock), deal status, completion probability.
           FundamentalsStub.ma_deal(symbols, date) is the stub interface.
Note:      SEC 8-K filings (free via EDGAR) capture public announcements but require
           NLP parsing to extract deal terms. No affordable automated source.
"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import logging
import pandas as pd

from data import FundamentalsStub, load_price_series, ADJ_TOTALRETURN

log = logging.getLogger(__name__)
DESCRIPTION = "Merger arbitrage [STUB — needs Refinitiv/Bloomberg M&A deal database]"
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
        fund.ma_deal("SPY", pd.Timestamp(start).date())
    except NotImplementedError:
        log.warning("S44: ma_deal not implemented. Needs Refinitiv SDC Platinum or Bloomberg M&A (expensive).")

    return {"returns": pd.Series(0.0, index=trading_idx), "benchmark": bm,
            "description": DESCRIPTION, "turnover_annual": 0.0}
