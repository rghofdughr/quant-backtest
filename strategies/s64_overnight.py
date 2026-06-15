"""
S64 — Overnight vs intraday decomposition (SPY)
Strategy:  Long SPY overnight only (hold from close to next open, collect overnight return).
           Flat during intraday session.
Return:    Overnight return on day t = today_Open / prev_Close - 1.
Costs:     Every trading day requires a buy at close and a sell at next open.
           2 legs per day → daily cost = 2 × (cost_bps + slip_bps) / 10000.
           With 10bps each-way cost + 10bps slip: daily cost = 40bps = 50.4% annual drag.
Benchmark: SPY (full day, close-to-close).
Note:      This strategy is primarily illustrative; net returns are likely negative
           after realistic costs. Gross performance vs intraday decomposition is the key insight.
"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import logging
import numpy as np
import pandas as pd
from data import load_price_series, ADJ_TOTALRETURN
from engine import apply_costs

log = logging.getLogger(__name__)
TRADING_DAYS = 252
DESCRIPTION = "Overnight SPY return (close-to-open) with realistic round-trip costs every day"


def run(config: dict) -> dict:
    cfg = config["backtest"]
    start = cfg["start_date"]
    end = cfg["end_date"]
    cache = config["paths"]["cache_dir"]

    cost_bps = config["costs"]["equity_cost_bps"]
    slip_bps = config["costs"]["equity_slippage_bps"]

    spy = load_price_series("SPY", start=start, end=end,
                            adjustment=ADJ_TOTALRETURN, cache_dir=cache)
    if spy.empty:
        empty = pd.Series(dtype=float)
        return {"returns": empty, "benchmark": empty,
                "description": DESCRIPTION, "turnover_annual": 0.0}

    trading_idx = pd.bdate_range(start, end)

    # Close and Open series aligned to trading calendar
    close = spy["Close"].reindex(trading_idx, method="ffill")
    bm_ret = close.pct_change(fill_method=None).fillna(0.0)  # full-day SPY return

    # Overnight return: Open[t] / Close[t-1] - 1
    if "Open" in spy.columns:
        open_ = spy["Open"].reindex(trading_idx, method="ffill")
        overnight_ret = (open_ / close.shift(1) - 1).fillna(0.0)
        intraday_ret = (close / open_ - 1).fillna(0.0)
    else:
        # No Open data: approximate using 50/50 split or just use bm_ret
        log.warning("S64: No 'Open' column in SPY data; falling back to full-day return as overnight proxy")
        overnight_ret = bm_ret.copy()
        intraday_ret = pd.Series(0.0, index=trading_idx)

    # Position is always +1 (we hold overnight every day)
    # Gross portfolio return = overnight return
    gross_ret = overnight_ret.copy()

    # Cost model: every day we buy at close and sell at open → 2 legs per day
    # Each leg costs (cost_bps + slip_bps) bps one-way
    # Turnover = 1.0 every day (full portfolio round-trip)
    # apply_costs deducts: turnover * (cost_bps + slip_bps) / 10000 * 2  per day
    # So with turnover = 1.0 daily: daily cost = 2 * (cost_bps + slip_bps) / 10000
    turnover = pd.Series(1.0, index=trading_idx)

    net_ret = apply_costs(gross_ret, turnover, cost_bps, slip_bps)

    # Summary statistics for logging
    gross_ann = float((1 + gross_ret).prod() ** (TRADING_DAYS / len(gross_ret)) - 1)
    net_ann = float((1 + net_ret).prod() ** (TRADING_DAYS / len(net_ret)) - 1)
    intraday_ann = float((1 + intraday_ret).prod() ** (TRADING_DAYS / len(intraday_ret)) - 1)
    daily_cost_bps = 2 * (cost_bps + slip_bps)

    log.info(
        "S64 done. Gross overnight CAGR=%.2f%%, Net CAGR=%.2f%%, Intraday CAGR=%.2f%%, "
        "daily cost=%.1fbps (%.1f%% annual drag)",
        gross_ann * 100, net_ann * 100, intraday_ann * 100,
        daily_cost_bps, daily_cost_bps / 100 * TRADING_DAYS,
    )

    ann_to = float(turnover.sum() / max(len(trading_idx) / TRADING_DAYS, 1))

    return {
        "returns": net_ret,
        "benchmark": bm_ret,
        "description": DESCRIPTION,
        "turnover_annual": ann_to,
    }
