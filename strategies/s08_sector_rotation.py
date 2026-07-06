"""
S08 — Sector rotation on relative strength (SPDR ETFs)
Universe:  11 SPDR sector ETFs (XLK, XLF, XLE, XLV, XLI, XLY, XLP, XLRE, XLB, XLU, XLC)
Signal:    Trailing N-month total return; hold top-K sectors equal-weight.
           Optional absolute-momentum cash filter: if top sector's own N-month return < 0,
           go to cash (short-term T-bills = 0% for simplicity).
Rebalance: Monthly (month-end business day)
Execution: Next trading day close
Sweep:     top_n [2,3,4], lookback_months [1,3,6], with/without abs-mom cash filter
"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import logging
import numpy as np
import pandas as pd

from data import load_price_series, ADJ_TOTALRETURN
from engine import portfolio_returns_from_weights, apply_costs

log = logging.getLogger(__name__)
DESCRIPTION = "Sector rotation by 3-month relative strength, SPDR ETFs, top-3 equal-weight"

# XLC launched June 2018; include it but accept shorter history
SECTORS = ["XLK", "XLF", "XLE", "XLV", "XLI", "XLY", "XLP", "XLRE", "XLB", "XLU", "XLC"]
TRADING_DAYS = 252


def _load_sectors(config) -> dict[str, pd.Series]:
    cache = config["paths"]["cache_dir"]
    start = "1998-01-01"
    end   = config["backtest"]["end_date"]
    prices = {}
    for sym in SECTORS:
        df = load_price_series(sym, start=start, end=end,
                               adjustment=ADJ_TOTALRETURN, cache_dir=cache)
        if not df.empty and "Close" in df.columns:
            prices[sym] = df["Close"]
    log.info("S08: loaded %d sector ETFs", len(prices))
    return prices


def run(config: dict) -> dict:
    cfg      = config["backtest"]
    start    = cfg["start_date"]
    end      = cfg["end_date"]
    is_smoke = config.get("smoke", False)

    s08_cfg          = config.get("strategies", {}).get("s08", {})
    top_n            = s08_cfg.get("top_n", 3)
    lb_months        = s08_cfg.get("lookback_months", 3)
    cash_filt        = s08_cfg.get("abs_momentum_cash_filter", True)
    graded_cash_filt = s08_cfg.get("graded_cash_filter", False)

    cost_bps = config["costs"]["equity_cost_bps"]
    slip_bps = config["costs"]["equity_slippage_bps"]

    prices   = _load_sectors(config)
    close_df = pd.DataFrame(prices).sort_index()
    # Lookback in business days (approx)
    lb_days = lb_months * 21

    reb_dates = pd.date_range(start, end, freq="BME")
    weight_schedule: dict = {}

    for d in reb_dates:
        avail_d = close_df.index[close_df.index <= d]
        if len(avail_d) < lb_days + 2:
            continue
        snap_row = len(avail_d) - 1

        p_now  = close_df.iloc[snap_row]
        p_past = close_df.iloc[max(0, snap_row - lb_days)]

        returns = ((p_now / p_past) - 1.0).dropna().sort_values(ascending=False)
        if returns.empty:
            continue

        if graded_cash_filt:
            # V6: drop each top-N holding with negative momentum individually;
            # equal-weight survivors; go to cash only if none are positive.
            selected = [s for s in returns.index[:top_n] if returns[s] > 0]
            if not selected:
                weight_schedule[d] = {}
                continue
        else:
            # Original all-or-nothing cash filter
            if cash_filt and returns.iloc[0] <= 0:
                weight_schedule[d] = {}
                continue
            selected = returns.iloc[:top_n].index.tolist()

        w = {s: 1.0 / len(selected) for s in selected}
        weight_schedule[d] = w

    if not weight_schedule:
        return {"returns": pd.Series(dtype=float), "benchmark": pd.Series(dtype=float),
                "description": DESCRIPTION, "turnover_annual": 0.0}

    gross_ret, to = portfolio_returns_from_weights(weight_schedule, close_df, start, end)
    net_ret = apply_costs(gross_ret, to, cost_bps, slip_bps)

    spy = load_price_series("SPY", start, end, ADJ_TOTALRETURN, config["paths"]["cache_dir"])
    bm  = spy["Close"].pct_change().reindex(net_ret.index)

    n_with_pos = sum(1 for w in weight_schedule.values() if w)
    cash_pct   = 1 - n_with_pos / max(len(weight_schedule), 1)
    ann_to     = float(to.sum() / max(len(to) / TRADING_DAYS, 1))

    log.info("S08 done. Top-%d sectors, %d-month lookback, in-cash %.0f%% of time, ann turnover %.2fx",
             top_n, lb_months, cash_pct * 100, ann_to)

    return {
        "returns":         net_ret,
        "benchmark":       bm,
        "description":     DESCRIPTION,
        "turnover_annual": ann_to,
    }
