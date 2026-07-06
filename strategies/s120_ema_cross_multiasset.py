"""
s120 — Multi-Asset EMA Cross (Trend Following)
For each of SPY, TLT, GLD, EFA, EEM: long when price > EMA(200), else flat.
Equal-weight the "on" assets. If none are above EMA(200) -> cash. Monthly rebalance.
"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import logging
import numpy as np
import pandas as pd
from data import load_price_series, ADJ_TOTALRETURN
from engine import portfolio_returns_from_weights, apply_costs

log = logging.getLogger(__name__)
DESCRIPTION = "Multi-asset EMA(200) cross: equal-weight assets above their 200d EMA, monthly"

ASSETS  = ["SPY", "TLT", "GLD", "EFA", "EEM"]
EMA_WIN = 200

def run(config):
    cfg   = config["backtest"]
    start, end = cfg["start_date"], cfg["end_date"]
    cache = config["paths"]["cache_dir"]
    cost_bps = config["costs"]["equity_cost_bps"]
    slip_bps = config["costs"]["equity_slippage_bps"]

    prices = {}
    for sym in ASSETS:
        df = load_price_series(sym, start=start, end=end, adjustment=ADJ_TOTALRETURN, cache_dir=cache)
        if not df.empty:
            prices[sym] = df["Close"]

    idx = pd.bdate_range(start, end)
    close_df = pd.DataFrame({s: c.reindex(idx, method="ffill") for s, c in prices.items()})

    ema_df = close_df.ewm(span=EMA_WIN, adjust=False).mean()
    above  = close_df > ema_df  # True/False for each asset each day

    weight_schedule = {}
    rebal_dates = [g.index[-1] for _, g in pd.Series(dtype=float, index=idx).groupby([idx.year, idx.month])]

    for rd in rebal_dates:
        if rd not in above.index:
            continue
        # Use previous bar's signal (shift(1) equivalent: use signal AT rd, apply from rd+1 via engine lag)
        row = above.loc[rd]
        on_assets = row[row].index.tolist()
        if not on_assets:
            weight_schedule[rd] = {}  # cash
        else:
            n = len(on_assets)
            weight_schedule[rd] = {s: 1.0 / n for s in on_assets}

    gross_ret, to = portfolio_returns_from_weights(weight_schedule, close_df, start, end)
    net_ret = apply_costs(gross_ret, to, cost_bps, slip_bps)
    ann_to = float(to.sum() / max(len(to) / 252, 1))
    return {"returns": net_ret, "turnover_annual": ann_to, "description": DESCRIPTION}
