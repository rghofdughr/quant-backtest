"""
s111 — ETF Z-score Mean Reversion
Signal: 20d z-score of returns for 8 macro ETFs. Long 2 most-negative-z-score ETFs.
Weekly rebalance (every 5 trading days).
"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import logging
import numpy as np
import pandas as pd
from data import load_price_series, ADJ_TOTALRETURN
from engine import portfolio_returns_from_weights, apply_costs

log = logging.getLogger(__name__)
DESCRIPTION = "ETF z-score reversion: long 2 most-oversold macro ETFs by 20d z-score, weekly"

UNIVERSE = ["SPY", "TLT", "GLD", "EFA", "EEM", "XLE", "VNQ", "HYG"]
Z_WIN    = 20
N_HOLD   = 2
REBAL_FREQ = 5  # trading days

def run(config):
    cfg   = config["backtest"]
    start, end = cfg["start_date"], cfg["end_date"]
    cache = config["paths"]["cache_dir"]
    cost_bps = config["costs"]["equity_cost_bps"]
    slip_bps = config["costs"]["equity_slippage_bps"]

    prices = {}
    for sym in UNIVERSE:
        df = load_price_series(sym, start=start, end=end, adjustment=ADJ_TOTALRETURN, cache_dir=cache)
        if not df.empty:
            prices[sym] = df["Close"]

    idx = pd.bdate_range(start, end)
    close_df = pd.DataFrame({s: c.reindex(idx, method="ffill") for s, c in prices.items()})

    # 20d z-score of price: (price - SMA20) / std20
    sma = close_df.rolling(Z_WIN).mean()
    std = close_df.rolling(Z_WIN).std()
    zscore_df = (close_df - sma) / std.replace(0, np.nan)

    weight_schedule = {}
    rebal_idx = [idx[i] for i in range(Z_WIN, len(idx), REBAL_FREQ)]

    for rd in rebal_idx:
        # Use signal from previous day (shift already embedded by using rd as key with lag=1)
        z = zscore_df.loc[rd].dropna()
        if len(z) < N_HOLD:
            continue
        picks = z.nsmallest(N_HOLD).index.tolist()
        weight_schedule[rd] = {s: 1.0 / N_HOLD for s in picks}

    gross_ret, to = portfolio_returns_from_weights(weight_schedule, close_df, start, end)
    net_ret = apply_costs(gross_ret, to, cost_bps, slip_bps)
    ann_to = float(to.sum() / max(len(to) / 252, 1))
    return {"returns": net_ret, "turnover_annual": ann_to, "description": DESCRIPTION}
