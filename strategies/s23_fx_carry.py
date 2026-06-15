"""
S23 — FX carry (interest rate differential)
Universe:  G10 currency futures available in Norgate (6E, 6J, 6B, 6A, 6C, 6S, 6N)
           Each traded against USD.
Signal:    True carry = foreign short-rate − USD short-rate.
           Proxy (no FRED): rank currencies by 12-month roll-adjusted return of continuous
           futures vs USD index (DX). High return ≈ high carry.
           Long top 3 high-carry currencies, short bottom 3, vol-targeted.
Note:      True interest-rate differential requires FRED/central bank rate data.
           This implementation uses momentum on futures as carry proxy (Moskowitz et al.)
"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import logging
import numpy as np
import pandas as pd

from data import load_futures_series, load_price_series, ADJ_TOTALRETURN
from engine import apply_costs

log = logging.getLogger(__name__)
DESCRIPTION = "FX carry via G10 currency futures, roll-yield proxy, long-short monthly"
TRADING_DAYS = 252
LOOKBACK     = 252   # 12m momentum as carry proxy
VOL_LOOKBACK = 63
VOL_TARGET   = 0.10
MAX_LEV      = 1.5

FX_FUTURES = ["6E", "6J", "6B", "6A", "6C", "6S", "6N"]   # EUR, JPY, GBP, AUD, CAD, CHF, NZD
FX_ETFS    = ["FXE", "FXY", "FXB", "FXA", "FXC", "FXF"]   # ETF fallbacks for missing futures


def run(config: dict) -> dict:
    cfg      = config["backtest"]
    start    = cfg["start_date"]
    end      = cfg["end_date"]
    cache    = config["paths"]["cache_dir"]
    cost_bps = config["costs"]["equity_cost_bps"]
    slip_bps = config["costs"]["equity_slippage_bps"]

    trading_idx = pd.bdate_range(start, end)

    close_map = {}
    for sym in FX_FUTURES:
        df = load_futures_series(sym, start=start, end=end, cache_dir=cache)
        if not df.empty:
            close_map[sym] = df["Close"]

    # Supplement with ETF proxies for any unavailable futures
    if len(close_map) < 3:
        for sym in FX_ETFS:
            if sym not in close_map:
                df = load_price_series(sym, start=start, end=end, adjustment=ADJ_TOTALRETURN, cache_dir=cache)
                if not df.empty:
                    close_map[sym] = df["Close"]

    if len(close_map) < 3:
        log.error("S23: only %d FX instruments available", len(close_map))
        return {"returns": pd.Series(dtype=float), "benchmark": pd.Series(dtype=float),
                "description": DESCRIPTION, "turnover_annual": 0.0}

    close_df = pd.DataFrame(close_map).reindex(trading_idx, method="ffill")
    ret_df   = close_df.pct_change(fill_method=None).fillna(0.0)

    mom_df   = close_df.pct_change(LOOKBACK, fill_method=None)   # 12m carry proxy
    rvol_df  = ret_df.rolling(VOL_LOOKBACK).std() * np.sqrt(TRADING_DAYS)
    rvol_df  = rvol_df.replace(0, np.nan).ffill().fillna(0.05)

    n_mkts   = close_df.shape[1]
    n_long   = max(1, n_mkts // 3)
    n_short  = max(1, n_mkts // 3)

    port_ret = pd.Series(0.0, index=trading_idx)
    to_series = pd.Series(0.0, index=trading_idx)

    prev_wts = pd.Series(0.0, index=close_df.columns)
    month_starts = trading_idx[
        (trading_idx.month != pd.Series(trading_idx.month).shift(1).fillna(-1).values)
    ]

    cur_wts = pd.Series(0.0, index=close_df.columns)
    for t in trading_idx:
        if t in month_starts:
            ranks = mom_df.loc[t].dropna().rank()
            if ranks.empty:
                cur_wts[:] = 0.0
            else:
                longs  = ranks.nlargest(n_long).index
                shorts = ranks.nsmallest(n_short).index
                wts    = pd.Series(0.0, index=close_df.columns)
                for s in longs:
                    wts[s] = (VOL_TARGET / rvol_df.loc[t, s]) / n_long
                for s in shorts:
                    wts[s] = -(VOL_TARGET / rvol_df.loc[t, s]) / n_short
                wts = wts.clip(-MAX_LEV, MAX_LEV)
                cur_wts = wts

        port_ret[t] = float((cur_wts * ret_df.loc[t]).sum())
        to_series[t] = float((cur_wts - prev_wts).abs().sum())
        prev_wts = cur_wts.copy()

    net_ret = apply_costs(port_ret, to_series, cost_bps, slip_bps)

    bm = ret_df.mean(axis=1)
    ann_to = float(to_series.sum() / max(len(trading_idx) / TRADING_DAYS, 1))
    log.info("S23 done. %d FX instruments | ann turnover: %.2fx", len(close_map), ann_to)

    return {
        "returns": net_ret, "benchmark": bm,
        "description": DESCRIPTION, "turnover_annual": ann_to,
    }
