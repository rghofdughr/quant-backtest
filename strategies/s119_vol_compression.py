"""
s119 — Volatility Compression (Bollinger Squeeze)
Signal: When SPY 20d Bollinger Band width is at a 252-day low (most compressed in a year),
go long SPY expecting a breakout. Hold until BB width expands above its 252d median. Else cash.
"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import logging
import numpy as np
import pandas as pd
from data import load_price_series, ADJ_TOTALRETURN
from engine import apply_costs

log = logging.getLogger(__name__)
DESCRIPTION = "Vol compression: long SPY when Bollinger Band width at 252d low (squeeze signal)"

BB_WIN      = 20
BB_STD      = 2.0
LOOKBACK    = 252  # window for historical min comparison
EXPAND_PCT  = 0.50  # exit when BB width >= 50th percentile of 252d window

def run(config):
    cfg   = config["backtest"]
    start, end = cfg["start_date"], cfg["end_date"]
    cache = config["paths"]["cache_dir"]
    cost_bps = config["costs"]["equity_cost_bps"]
    slip_bps = config["costs"]["equity_slippage_bps"]

    spy = load_price_series("SPY", start=start, end=end, adjustment=ADJ_TOTALRETURN, cache_dir=cache)
    idx = pd.bdate_range(start, end)
    spy_c = spy["Close"].reindex(idx, method="ffill")
    spy_r = spy_c.pct_change(fill_method=None).fillna(0.0)

    sma = spy_c.rolling(BB_WIN).mean()
    std = spy_c.rolling(BB_WIN).std()
    # BB width normalized: (upper - lower) / middle = 2*BB_STD*std / sma
    bb_width = (2 * BB_STD * std) / sma.replace(0, np.nan)

    # Rolling min and median over past 252 days (no forward look: use [t-252:t])
    bb_min  = bb_width.rolling(LOOKBACK).min()
    bb_med  = bb_width.rolling(LOOKBACK).quantile(EXPAND_PCT)

    # Squeeze: today's BB width <= its 252d rolling min (compressed)
    # Use .shift(1) to avoid using today's signal to earn today's return
    squeeze = (bb_width.shift(1) <= bb_min.shift(1)).fillna(False)
    expanded = (bb_width.shift(1) >= bb_med.shift(1)).fillna(True)

    # State machine: enter on squeeze, exit on expansion
    in_market = pd.Series(False, index=idx)
    active = False
    for i in range(len(idx)):
        if not active and squeeze.iloc[i]:
            active = True
        elif active and expanded.iloc[i]:
            active = False
        in_market.iloc[i] = active

    port_ret = pd.Series(0.0, index=idx)
    port_ret[in_market] = spy_r[in_market]

    to = in_market.astype(float).diff().abs().fillna(0.0)
    net_ret = apply_costs(port_ret, to, cost_bps, slip_bps)
    ann_to = float(to.sum() / max(len(to) / 252, 1))
    pct = float(in_market.mean()) * 100
    log.info("s119: in market %.1f%% of days", pct)
    return {"returns": net_ret, "turnover_annual": ann_to, "description": DESCRIPTION}
