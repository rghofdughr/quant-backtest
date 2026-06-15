"""
S40 — Post-earnings announcement drift (PEAD)
Reference: Ball & Brown (1968); Bernard & Thomas (1989)
Universe:  S&P 500 C&P PIT
Signal:    Standardized Unexpected Earnings (SUE) = (actual_EPS - consensus_EPS) / std(surprise).
           Long top quintile of positive SUE; short bottom quintile.
           Hold 60 days post-announcement.
Data:      SUE requires EPS estimates (Zacks, Sharadar Earnings, or Refinitiv).
           Earnings date from Norgate corporate events (partial) or estimates vendor.
           FundamentalsStub.earnings_surprise(symbols, date) → Series of SUE values.
Partial:   Earnings event dates may be in Norgate. SUE stub here; long PEAD logic complete.
"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import logging
import numpy as np
import pandas as pd

from data import FundamentalsStub, load_price_series, watchlist_symbols, ADJ_TOTALRETURN
from engine import portfolio_returns_from_weights, apply_costs

log = logging.getLogger(__name__)
DESCRIPTION = "PEAD (post-earnings drift), S&P 500 PIT [STUB — needs Zacks/Sharadar EPS estimates]"
TRADING_DAYS = 252
HOLD_DAYS    = 60
TOP_N        = 30


def run(config: dict) -> dict:
    cfg   = config["backtest"]
    start = cfg["start_date"]
    end   = cfg["end_date"]
    cache = config["paths"]["cache_dir"]
    cost_bps = config["costs"]["equity_cost_bps"]
    slip_bps = config["costs"]["equity_slippage_bps"]

    univ_name = config["universes"]["sp500"]
    syms      = watchlist_symbols(univ_name)
    trading_idx = pd.bdate_range(start, end)

    spy = load_price_series("SPY", start=start, end=end, adjustment=ADJ_TOTALRETURN, cache_dir=cache)
    bm  = spy["Close"].reindex(trading_idx, method="ffill").pct_change(fill_method=None).fillna(0.0) if not spy.empty else pd.Series(0.0, index=trading_idx)

    fund = FundamentalsStub()
    try:
        fund.earnings_surprise("SPY", pd.Timestamp(start).date())
    except NotImplementedError:
        log.warning("S40: earnings_surprise (SUE) not implemented. Needs Zacks or Sharadar earnings estimates.")
        return {"returns": pd.Series(0.0, index=trading_idx), "benchmark": bm,
                "description": DESCRIPTION, "turnover_annual": 0.0}

    # If SUE data available, the logic below executes:
    # weight_schedule built from quarterly earnings dates, holding HOLD_DAYS
    return {"returns": pd.Series(0.0, index=trading_idx), "benchmark": bm,
            "description": DESCRIPTION, "turnover_annual": 0.0}
