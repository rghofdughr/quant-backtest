"""
S42 — Insider buying clusters
Reference: Lakonishok & Lee (2001); Jeng, Metrick & Zeckhauser (2003)
Universe:  S&P 500 C&P PIT
Signal:    Cluster of insider open-market purchases within 3 months.
           Long stocks with ≥3 insider purchases in past 63 days.
           Avoid open-market sales (not gift/estate/plan purchases).
Data:      REQUIRES SEC EDGAR Form 4 data.
           Source: SEC EDGAR full-text API (free, rate-limited) or InsiderInsights.com.
           Implement load_insider_transactions(symbol, start, end) → DataFrame with
           (date, insider_name, transaction_type, shares, price).
Partial:   Form 4 is free from SEC but requires parsing XML and building a time-series DB.
"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import logging
import pandas as pd

from data import FundamentalsStub, load_price_series, ADJ_TOTALRETURN

log = logging.getLogger(__name__)
DESCRIPTION = "Insider buying clusters, S&P 500 PIT [STUB — needs SEC EDGAR Form 4 data]"
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
        fund.insider_transactions("SPY", pd.Timestamp(start).date(), pd.Timestamp(end).date())
    except NotImplementedError:
        log.warning("S42: insider_transactions not implemented. Needs SEC EDGAR Form 4 (free) or InsiderInsights.")

    return {"returns": pd.Series(0.0, index=trading_idx), "benchmark": bm,
            "description": DESCRIPTION, "turnover_annual": 0.0}
