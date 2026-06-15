"""
S29 — Variance risk premium (VRP)
Universe:  SPY short volatility; VIX as implied vol proxy
Signal:    VRP = VIX − realized_vol(21-day) of SPY.
           When VRP is high (implied >> realized), premium is rich → short vol (short SPY put-spread proxy).
           Traded here via SPY short/flat (proxy): when VRP high, expect VIX mean-reversion → SPY long.
           When VRP negative (implied < realized, stress regime) → move to cash.
Note:      Full VRP trade requires options (straddle/variance swap).
           This implements the equity-bias side: long SPY when VRP high, flat/cash when VRP negative.
           VIX loaded from Norgate as index UVXY/VXX/^VIX proxy.
"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import logging
import numpy as np
import pandas as pd

from data import load_price_series, load_futures_series, ADJ_TOTALRETURN
from engine import apply_costs

log = logging.getLogger(__name__)
DESCRIPTION = "Variance risk premium, SPY long/flat, VIX minus realized vol signal"
TRADING_DAYS = 252
RV_LOOKBACK  = 21      # realized vol window (days)
VRP_THRESH   = 0.02    # VRP > 2 vol points → long SPY; < 0 → flat


def _load_vix(start: str, end: str, cache: str) -> pd.Series:
    """Try to load VIX from Norgate; fall back to UVXY/VXX proxy."""
    for sym in ["$VIX", "VIX", "^VIX", "VIXY"]:
        df = load_price_series(sym, start=start, end=end, adjustment=ADJ_TOTALRETURN, cache_dir=cache)
        if not df.empty and len(df) > 100:
            log.info("S29: VIX proxy loaded via %s", sym)
            return df["Close"]
    # Try VX futures as VIX proxy
    df = load_futures_series("VX", start=start, end=end, cache_dir=cache)
    if not df.empty:
        log.info("S29: VIX proxy loaded via VX futures")
        return df["Close"]
    log.warning("S29: VIX not available; VRP signal will be flat")
    return pd.Series(dtype=float)


def run(config: dict) -> dict:
    cfg      = config["backtest"]
    start    = cfg["start_date"]
    end      = cfg["end_date"]
    cache    = config["paths"]["cache_dir"]
    cost_bps = config["costs"]["equity_cost_bps"]
    slip_bps = config["costs"]["equity_slippage_bps"]

    spy = load_price_series("SPY", start=start, end=end, adjustment=ADJ_TOTALRETURN, cache_dir=cache)
    if spy.empty:
        return {"returns": pd.Series(dtype=float), "benchmark": pd.Series(dtype=float),
                "description": DESCRIPTION, "turnover_annual": 0.0}

    vix_close = _load_vix(start, end, cache)

    trading_idx = pd.bdate_range(start, end)
    spy_r = spy["Close"].reindex(trading_idx, method="ffill").pct_change(fill_method=None).fillna(0.0)

    # Realized vol (annualized)
    rv = spy_r.rolling(RV_LOOKBACK).std() * np.sqrt(TRADING_DAYS)

    if not vix_close.empty:
        # VIX is in percentage (e.g., 20 = 20% annualized vol)
        vix_s = vix_close.reindex(trading_idx, method="ffill") / 100.0
        vrp   = vix_s - rv
        signal = ((vrp > VRP_THRESH).astype(float).shift(1).fillna(0.0) > 0)
    else:
        # Without VIX: rough signal using RV only — go long when RV is low (VRP tends to be high)
        rv_median = rv.expanding().median()
        signal = (rv < rv_median).shift(1).fillna(False)
        log.info("S29: Using RV-only proxy signal (VIX unavailable)")

    port_ret = signal.astype(float) * spy_r
    to = signal.astype(float).diff().abs().fillna(0.0)
    net_ret = apply_costs(port_ret, to, cost_bps, slip_bps)

    bm = spy_r.copy()
    long_pct = float(signal.mean()) * 100
    ann_to = float(to.sum() / max(len(trading_idx) / TRADING_DAYS, 1))
    log.info("S29 done. Long SPY: %.1f%% of days, ann turnover: %.2fx", long_pct, ann_to)

    return {
        "returns": net_ret, "benchmark": bm,
        "description": DESCRIPTION, "turnover_annual": ann_to,
    }
