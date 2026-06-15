"""
S22 — Accruals anomaly
Reference: Sloan (1996)
Universe:  S&P 500 C&P PIT
Signal:    Total accruals = (ΔNet operating assets) / avg total assets.
           High accruals → earnings quality low → short; low accruals → long.
Data:      REQUIRES Sharadar: net_income, CFO (cash from operations), total_assets.
           Accruals = (Net income − CFO) / avg total assets.
           FundamentalsStub.accruals(symbols, date) must be implemented.
"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import logging
import pandas as pd

from data import FundamentalsStub, load_price_series, ADJ_TOTALRETURN
from engine import apply_costs

log = logging.getLogger(__name__)
DESCRIPTION = "Accruals anomaly (earnings quality), S&P 500 PIT [STUB — needs Sharadar]"
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
        fund.accruals("SPY", 2023)
    except NotImplementedError:
        log.warning("S22: accruals not implemented — needs Sharadar fundamentals data")

    return {"returns": pd.Series(0.0, index=trading_idx), "benchmark": bm,
            "description": DESCRIPTION, "turnover_annual": 0.0}
