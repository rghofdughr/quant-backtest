"""
s106 — Global Equity Momentum (Faber-style)
Universe: SPY, EFA, EEM, TLT
Signal: 12-1 month return. Hold top 1 asset. Absolute filter: must beat BIL (or 0% if BIL unavail).
        If no asset passes absolute filter -> cash. Monthly rebalance.
"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import logging
import numpy as np
import pandas as pd
from data import load_price_series, ADJ_TOTALRETURN
from engine import portfolio_returns_from_weights, apply_costs

log = logging.getLogger(__name__)
DESCRIPTION = "Global momentum: top-1 of SPY/EFA/EEM/TLT by 12-1m return, absolute filter vs BIL"

ASSETS  = ["SPY", "EFA", "EEM", "TLT"]
LOOKBACK = 252
SKIP     = 21

def run(config):
    cfg   = config["backtest"]
    start, end = cfg["start_date"], cfg["end_date"]
    cache = config["paths"]["cache_dir"]
    cost_bps = config["costs"]["equity_cost_bps"]
    slip_bps = config["costs"]["equity_slippage_bps"]

    prices = {}
    for sym in ASSETS + ["BIL"]:
        df = load_price_series(sym, start=start, end=end, adjustment=ADJ_TOTALRETURN, cache_dir=cache)
        if not df.empty:
            prices[sym] = df["Close"]

    idx = pd.bdate_range(start, end)
    close_df = pd.DataFrame({s: c.reindex(idx, method="ffill") for s, c in prices.items()})

    # 12-1 month momentum: return from 252 days ago to 21 days ago
    mom = close_df.pct_change(LOOKBACK) - close_df.pct_change(SKIP)

    weight_schedule = {}
    rebal_dates = [g.index[-1] for _, g in pd.Series(dtype=float, index=idx).groupby([idx.year, idx.month])]

    for rd in rebal_dates:
        if rd not in close_df.index:
            continue
        m = mom.loc[rd, ASSETS].dropna()
        if m.empty:
            continue
        # Absolute filter: must have positive momentum
        candidates = m[m > 0]
        if candidates.empty:
            weight_schedule[rd] = {}  # cash
            continue
        winner = candidates.idxmax()
        weight_schedule[rd] = {winner: 1.0}

    asset_close = close_df[[s for s in ASSETS if s in close_df.columns]]
    gross_ret, to = portfolio_returns_from_weights(weight_schedule, asset_close, start, end)
    net_ret = apply_costs(gross_ret, to, cost_bps, slip_bps)
    ann_to = float(to.sum() / max(len(to) / 252, 1))
    return {"returns": net_ret, "turnover_annual": ann_to, "description": DESCRIPTION}
