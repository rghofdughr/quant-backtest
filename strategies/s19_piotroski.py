"""
S19 — Piotroski F-score (fundamental quality screen)
Reference: Piotroski (2000)
Universe:  Russell 1000 C&P PIT
Signal:    9 binary criteria from annual financial statements → F-score 0-9.
           Long F-score >= 7 (strong), Short F-score <= 2 (weak), monthly rebalance.
           Criteria: ROA, CFO, delta-ROA, accruals, delta-leverage, delta-liquidity,
                     equity issuance, delta-gross-margin, delta-asset-turnover.
Data:      REQUIRES Sharadar Core US Fundamentals.
           FundamentalsStub.piotroski_fscore(symbols, date) must be implemented.
"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import logging
import pandas as pd

from data import FundamentalsStub, load_price_series, ADJ_TOTALRETURN
from engine import apply_costs

log = logging.getLogger(__name__)
DESCRIPTION = "Piotroski F-score quality screen, Russell 1000 PIT [STUB — needs Sharadar]"
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
        fund.piotroski_fscore("SPY", 2023)
    except NotImplementedError:
        log.warning("S19: piotroski_fscore not implemented — needs Sharadar fundamentals data")

    return {"returns": pd.Series(0.0, index=trading_idx), "benchmark": bm,
            "description": DESCRIPTION, "turnover_annual": 0.0}
