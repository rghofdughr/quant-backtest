"""
s109 — Sector RSI Mean Reversion
Signal: Buy 2 most-oversold sector SPDRs (lowest 14d RSI). Monthly rebalance.
Universe: 9 classic SPDR sector ETFs (available from 1998).
"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import logging
import numpy as np
import pandas as pd
from data import load_price_series, ADJ_TOTALRETURN
from engine import portfolio_returns_from_weights, apply_costs

log = logging.getLogger(__name__)
DESCRIPTION = "Sector RSI reversion: long 2 most-oversold SPDR sectors by 14d RSI monthly"

SECTORS = ["XLE", "XLF", "XLI", "XLK", "XLB", "XLV", "XLY", "XLP", "XLU"]
RSI_WIN  = 14
N_HOLD   = 2

def _rsi(series, window=14):
    delta = series.diff()
    gain = delta.clip(lower=0).rolling(window).mean()
    loss = (-delta.clip(upper=0)).rolling(window).mean()
    rs = gain / loss.replace(0, np.nan)
    return 100 - 100 / (1 + rs)

def run(config):
    cfg   = config["backtest"]
    start, end = cfg["start_date"], cfg["end_date"]
    cache = config["paths"]["cache_dir"]
    cost_bps = config["costs"]["equity_cost_bps"]
    slip_bps = config["costs"]["equity_slippage_bps"]

    prices = {}
    for sym in SECTORS:
        df = load_price_series(sym, start=start, end=end, adjustment=ADJ_TOTALRETURN, cache_dir=cache)
        if not df.empty:
            prices[sym] = df["Close"]

    idx = pd.bdate_range(start, end)
    close_df = pd.DataFrame({s: c.reindex(idx, method="ffill") for s, c in prices.items()})

    rsi_df = close_df.apply(lambda col: _rsi(col, RSI_WIN))

    weight_schedule = {}
    rebal_dates = [g.index[-1] for _, g in pd.Series(dtype=float, index=idx).groupby([idx.year, idx.month])]

    for rd in rebal_dates:
        if rd not in rsi_df.index:
            continue
        r = rsi_df.loc[rd].dropna()
        if len(r) < N_HOLD:
            continue
        # Buy N_HOLD most oversold (lowest RSI), equal weight
        picks = r.nsmallest(N_HOLD).index.tolist()
        weight_schedule[rd] = {s: 1.0 / N_HOLD for s in picks}

    gross_ret, to = portfolio_returns_from_weights(weight_schedule, close_df, start, end)
    net_ret = apply_costs(gross_ret, to, cost_bps, slip_bps)
    ann_to = float(to.sum() / max(len(to) / 252, 1))
    return {"returns": net_ret, "turnover_annual": ann_to, "description": DESCRIPTION}
