"""
S10 — Bollinger Band mean reversion on liquid ETFs
Universe:  SPY, QQQ, IWM, EEM, TLT, GLD, GDX + 11 SPDR sector ETFs
Signal:    Close below lower Bollinger Band → long; exit at middle band (SMA).
           Optional short: close above upper band → short; exit at middle band.
Execution: Signal at close of day t; execute at open of t+1 (approx next close).
Sizing:    Equal allocation per active position.
Sweep:     window [10,20], sigma [1.5,2.0,2.5], long-only vs long-short
"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import logging
import numpy as np
import pandas as pd

from data import load_price_series, ADJ_TOTALRETURN
from engine import apply_costs

log = logging.getLogger(__name__)
DESCRIPTION = "Bollinger Band reversion, liquid ETFs, long on lower-band touch"
TRADING_DAYS = 252

ETFS = [
    "SPY","QQQ","IWM","EEM","TLT","GLD","GDX",
    "XLK","XLF","XLE","XLV","XLI","XLY","XLP","XLB","XLU",
]


def run(config: dict) -> dict:
    cfg      = config["backtest"]
    start    = cfg["start_date"]
    end      = cfg["end_date"]
    is_smoke = config.get("smoke", False)
    cache    = config["paths"]["cache_dir"]

    s10_cfg   = config.get("strategies", {}).get("s10", {})
    bb_window = s10_cfg.get("bb_window", 20)
    bb_sigma  = s10_cfg.get("bb_sigma",  2.0)
    long_short = s10_cfg.get("long_short", False)

    cost_bps = config["costs"]["equity_cost_bps"]
    slip_bps = config["costs"]["equity_slippage_bps"]

    syms = ETFS[:6] if is_smoke else ETFS

    prices = {}
    for sym in syms:
        df = load_price_series(sym, start=start, end=end,
                               adjustment=ADJ_TOTALRETURN, cache_dir=cache)
        if not df.empty and "Close" in df.columns:
            prices[sym] = df["Close"]

    log.info("S10: %d ETFs loaded", len(prices))

    trading_idx = pd.bdate_range(start, end)
    close_df  = pd.DataFrame(prices).reindex(trading_idx, method="ffill")

    # Bollinger Bands for each ETF
    sma   = close_df.rolling(bb_window).mean()
    std   = close_df.rolling(bb_window).std()
    upper = sma + bb_sigma * std
    lower = sma - bb_sigma * std

    # Signals (shifted 1 day for no look-ahead — we act on yesterday's close signal)
    below_lower = (close_df < lower).shift(1).fillna(False)  # entry long
    above_upper = (close_df > upper).shift(1).fillna(False)  # entry short (if enabled)
    at_mid_long = (close_df >= sma).shift(1).fillna(False)   # exit long
    at_mid_short= (close_df <= sma).shift(1).fillna(False)   # exit short

    # Build per-asset position via state machine (vectorized with forward-fill)
    n = len(trading_idx)
    position_df = pd.DataFrame(0.0, index=trading_idx, columns=list(prices.keys()))

    for sym in prices:
        pos = 0.0
        positions = []
        for i in range(n):
            if pos == 1.0:
                if at_mid_long.iloc[i][sym]:
                    pos = 0.0
            elif pos == -1.0 and long_short:
                if at_mid_short.iloc[i][sym]:
                    pos = 0.0
            # Entry
            if pos == 0.0:
                if below_lower.iloc[i][sym]:
                    pos = 1.0
                elif long_short and above_upper.iloc[i][sym]:
                    pos = -1.0
            positions.append(pos)
        position_df[sym] = positions

    # Equal-weight active positions
    n_active = position_df.abs().sum(axis=1).clip(lower=1e-9)
    # Normalise each row by number of active positions
    weight_df = position_df.div(n_active, axis=0)
    # If nothing active, weight_df is all-zero → cash position

    # Daily portfolio return
    ret_df   = close_df.pct_change(fill_method=None).fillna(0.0)
    port_ret = (weight_df * ret_df).sum(axis=1)

    # Turnover: weight changes day-over-day / 2
    to = weight_df.diff().abs().sum(axis=1) / 2.0

    net_ret = apply_costs(port_ret, to, cost_bps, slip_bps)

    spy = load_price_series("SPY", start, end, ADJ_TOTALRETURN, cache)
    bm  = spy["Close"].pct_change(fill_method=None).reindex(net_ret.index)

    ann_to = float(to.sum() / max(len(to) / TRADING_DAYS, 1))
    log.info("S10 done (window=%d, sigma=%.1f, ls=%s). Ann turnover: %.2fx",
             bb_window, bb_sigma, long_short, ann_to)

    return {
        "returns": net_ret, "benchmark": bm,
        "description": DESCRIPTION, "turnover_annual": ann_to,
    }
