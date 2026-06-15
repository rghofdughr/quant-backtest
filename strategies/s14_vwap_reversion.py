"""
S14 — VWAP reversion (intraday → daily analog)
Reference: Several intraday studies show price > VWAP → sell; price < VWAP → buy.
Universe:  S&P 500 PIT or SPY
Signal:    True: intraday price vs. cumulative VWAP → mean-revert to VWAP.
           DAILY PROXY: daily close vs. 20-day VWAP proxy (sum(Close*Vol) / sum(Vol) over 20 days).
           When close >> 20-day VWAP → short; when close << 20-day VWAP → long.
Note:      Requires intraday bars for proper VWAP implementation.
           Vendors: Polygon.io or Norgate intraday add-on.
"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import logging
import numpy as np
import pandas as pd

from data import load_price_series, ADJ_TOTALRETURN
from engine import apply_costs

log = logging.getLogger(__name__)
DESCRIPTION = "VWAP reversion daily proxy (true version needs intraday), 20-day VWAP z-score"
TRADING_DAYS = 252
VWAP_WINDOW  = 20
Z_ENTER      = 1.5
Z_EXIT       = 0.2


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

    trading_idx = pd.bdate_range(start, end)
    close_s = spy["Close"].reindex(trading_idx, method="ffill")
    vol_s   = spy["Volume"].reindex(trading_idx, method="ffill").fillna(1.0)
    ret     = close_s.pct_change(fill_method=None).fillna(0.0)

    # 20-day VWAP proxy
    vwap = (close_s * vol_s).rolling(VWAP_WINDOW).sum() / vol_s.rolling(VWAP_WINDOW).sum()
    std  = close_s.rolling(VWAP_WINDOW).std()
    z    = ((close_s - vwap) / std.replace(0, np.nan)).fillna(0.0)

    # State machine: entry/exit based on z-score
    signal  = pd.Series(0.0, index=trading_idx)
    pos     = 0.0
    z_prev  = z.shift(1).fillna(0.0)
    for t in trading_idx:
        z_t = z_prev[t]
        if pos == 0.0:
            if z_t >  Z_ENTER:
                pos = -1.0   # short (overextended above VWAP)
            elif z_t < -Z_ENTER:
                pos =  1.0   # long (overextended below VWAP)
        else:
            if abs(z_t) < Z_EXIT:
                pos = 0.0
        signal[t] = pos

    port_ret = signal * ret
    to = signal.diff().abs().fillna(0.0)
    net_ret = apply_costs(port_ret, to, cost_bps, slip_bps)

    bm = ret.copy()
    ann_to = float(to.sum() / max(len(trading_idx) / TRADING_DAYS, 1))
    log.info("S14 done (DAILY PROXY). Ann turnover: %.2fx", ann_to)
    log.warning("S14: true VWAP reversion requires intraday bars (Polygon/NDU intraday add-on)")

    return {
        "returns": net_ret, "benchmark": bm,
        "description": DESCRIPTION + " [PROXY ONLY — needs intraday bars]",
        "turnover_annual": ann_to,
    }
