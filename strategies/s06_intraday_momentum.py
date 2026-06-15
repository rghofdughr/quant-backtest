"""
S06 — Intraday momentum (first 30-min return predicts last 30-min return)
Reference: Gao, Han, Li & Zhou (2018)
Universe:  S&P 500 futures (ES) or SPY
Signal:    True implementation: first-30-min return (open to 10am) predicts last-30-min
           return (3:30pm to close). Requires intraday bars.
           DAILY PROXY (used here): Open-to-Open momentum from prior day.
           open_ret = Open[t] / Close[t-1] - 1 (overnight component)
           close_ret = Close[t] / Open[t] - 1 (intraday component)
           Signal: if open_ret[t-1] > 0, go long at open[t]; exit at close[t].
           Approximated via close-to-close with Open-derived signal.
Note:      Requires intraday bars for proper implementation. See data_availability_report.md.
           Vendors: Polygon.io or Norgate intraday add-on.
"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import logging
import numpy as np
import pandas as pd

from data import load_price_series, load_futures_series, ADJ_TOTALRETURN
from engine import apply_costs

log = logging.getLogger(__name__)
DESCRIPTION = "Intraday momentum daily proxy, SPY open/close signal (true version needs intraday)"
TRADING_DAYS = 252


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
    open_s  = spy["Open"].reindex(trading_idx, method="ffill") if "Open" in spy.columns else close_s

    ret_cc = close_s.pct_change(fill_method=None).fillna(0.0)    # close-to-close
    # Overnight gap (proxy for "first 30-min" directional signal)
    overnight = (open_s / close_s.shift(1) - 1).fillna(0.0)
    # Intraday (proxy for "last 30-min" component)
    intraday  = (close_s / open_s - 1).fillna(0.0)

    # Signal: if yesterday's overnight gap was positive, expect momentum continuation today
    signal = (overnight.shift(1) > 0).astype(float).shift(1).fillna(0.0)
    # Long when prior overnight was positive
    port_ret = signal * ret_cc

    to = signal.diff().abs().fillna(0.0)
    net_ret = apply_costs(port_ret, to, cost_bps, slip_bps)

    bm = ret_cc.copy()
    long_pct = float((signal > 0).mean()) * 100
    ann_to = float(to.sum() / max(len(trading_idx) / TRADING_DAYS, 1))
    log.info("S06 done (DAILY PROXY). Active: %.1f%% of days | ann turnover: %.2fx", long_pct, ann_to)
    log.warning("S06: true intraday implementation requires 30-min bars (Polygon/NDU intraday add-on)")

    return {
        "returns": net_ret, "benchmark": bm,
        "description": DESCRIPTION + " [PROXY ONLY — needs intraday bars]",
        "turnover_annual": ann_to,
    }
