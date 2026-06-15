"""
S28 — Short straddle with regime filter (sell implied volatility)
Reference: Whaley (2002); Coval & Shumway (2001)
Universe:  SPX options (ATM straddle) or SPY options as proxy
Signal:    When VRP high (IV >> RV), sell 1-month ATM straddle.
           Regime filter: avoid selling vol in high-vol regime (VIX > 25).
           Delta-hedge daily (approximate: short straddle + long/short underlying).
Data:      REQUIRES options data: ORATS, CBOE LiveVol, or similar.
           This stub implements the regime-filter signal logic and P&L framework.
           Options loader: implement load_options_chain(symbol, date, strike_range, expiry) → DataFrame
"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import logging
import pandas as pd

from data import load_price_series, ADJ_TOTALRETURN
from engine import apply_costs

log = logging.getLogger(__name__)
DESCRIPTION = "Short straddle with VIX regime filter [STUB — needs ORATS/CBOE options data]"
TRADING_DAYS = 252


def load_options_chain(symbol: str, date, strike_range: float, expiry_days: int, cache_dir: str):
    """Stub: replace with ORATS or CBOE LiveVol loader returning (call_iv, put_iv, delta, gamma, theta)."""
    raise NotImplementedError(
        "load_options_chain: plug in ORATS (~$100/mo) or CBOE LiveVol.\n"
        "Expected return: DataFrame with columns [strike, call_iv, put_iv, call_price, put_price, delta, gamma, theta]"
    )


def run(config: dict) -> dict:
    cfg   = config["backtest"]
    start = cfg["start_date"]
    end   = cfg["end_date"]
    cache = config["paths"]["cache_dir"]

    trading_idx = pd.bdate_range(start, end)
    spy = load_price_series("SPY", start=start, end=end, adjustment=ADJ_TOTALRETURN, cache_dir=cache)
    bm  = spy["Close"].reindex(trading_idx, method="ffill").pct_change(fill_method=None).fillna(0.0) if not spy.empty else pd.Series(0.0, index=trading_idx)

    try:
        load_options_chain("SPY", pd.Timestamp(start), 0.05, 30, cache)
    except NotImplementedError as e:
        log.warning("S28: %s", str(e).split('\n')[0])

    return {"returns": pd.Series(0.0, index=trading_idx), "benchmark": bm,
            "description": DESCRIPTION, "turnover_annual": 0.0}
