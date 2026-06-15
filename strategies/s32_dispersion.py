"""
S32 — Dispersion trade (short index vol, long stock vol)
Reference: Driessen, Maenhout & Vilkov (2009)
Universe:  SPX + large cap stock options (top 50 S&P 500 constituents)
Signal:    Short SPX straddle, long straddles on components (weighted by index weight).
           Profit when realized correlation < implied correlation.
Data:      REQUIRES options data: ORATS, CBOE LiveVol (for both index and stock options).
           Stub: implements correlation premium framework; awaiting options loader.
"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import logging
import pandas as pd

from data import load_price_series, ADJ_TOTALRETURN

log = logging.getLogger(__name__)
DESCRIPTION = "Dispersion trade: short index vol, long stock vol [STUB — needs options data]"
TRADING_DAYS = 252


def run(config: dict) -> dict:
    cfg   = config["backtest"]
    start = cfg["start_date"]
    end   = cfg["end_date"]
    cache = config["paths"]["cache_dir"]

    trading_idx = pd.bdate_range(start, end)
    spy = load_price_series("SPY", start=start, end=end, adjustment=ADJ_TOTALRETURN, cache_dir=cache)
    bm  = spy["Close"].reindex(trading_idx, method="ffill").pct_change(fill_method=None).fillna(0.0) if not spy.empty else pd.Series(0.0, index=trading_idx)

    log.warning("S32: dispersion trade requires ORATS/CBOE options data for both index and constituents")

    return {"returns": pd.Series(0.0, index=trading_idx), "benchmark": bm,
            "description": DESCRIPTION, "turnover_annual": 0.0}
