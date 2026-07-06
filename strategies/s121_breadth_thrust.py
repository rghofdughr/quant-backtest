"""
s121 — Market Breadth Thrust (Sector ETF Breadth)
Breadth proxy: % of 9 sector ETFs above their 200d MA.
Signal: >= 7/9 sectors above MA (broad bull) -> 100% SPY.
        5-6/9 -> 50% SPY. <= 4/9 -> cash. Monthly rebalance.
"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import logging
import numpy as np
import pandas as pd
from data import load_price_series, ADJ_TOTALRETURN
from engine import apply_costs

log = logging.getLogger(__name__)
DESCRIPTION = "Sector breadth thrust: SPY allocation scales with % of sectors above 200d MA"

SECTORS = ["XLE", "XLF", "XLI", "XLK", "XLB", "XLV", "XLY", "XLP", "XLU"]
MA_WIN  = 200

def run(config):
    cfg   = config["backtest"]
    start, end = cfg["start_date"], cfg["end_date"]
    cache = config["paths"]["cache_dir"]
    cost_bps = config["costs"]["equity_cost_bps"]
    slip_bps = config["costs"]["equity_slippage_bps"]

    prices = {}
    for sym in SECTORS + ["SPY"]:
        df = load_price_series(sym, start=start, end=end, adjustment=ADJ_TOTALRETURN, cache_dir=cache)
        if not df.empty:
            prices[sym] = df["Close"]

    idx = pd.bdate_range(start, end)
    close_df = pd.DataFrame({s: c.reindex(idx, method="ffill") for s, c in prices.items()})

    sector_cols = [s for s in SECTORS if s in close_df.columns]
    spy_r = close_df["SPY"].pct_change(fill_method=None).fillna(0.0) if "SPY" in close_df.columns else pd.Series(0.0, index=idx)

    ma_df = close_df[sector_cols].rolling(MA_WIN).mean()
    above_df = (close_df[sector_cols] > ma_df)

    # Fraction of sectors above MA, shifted 1 day for execution lag
    pct_above = above_df.mean(axis=1).shift(1).fillna(0.0)

    # Monthly: use month-end pct_above, hold all month
    monthly_pct = pct_above.resample("ME").last().reindex(idx, method="ffill").fillna(0.0)

    # Allocation: 0%, 50%, or 100% SPY based on breadth
    spy_alloc = pd.Series(0.0, index=idx)
    spy_alloc[monthly_pct >= 7/9]                                     = 1.0
    spy_alloc[(monthly_pct >= 5/9) & (monthly_pct < 7/9)]             = 0.5

    port_ret = spy_alloc * spy_r

    prev_alloc = spy_alloc.shift(1).fillna(0.0)
    to = (spy_alloc - prev_alloc).abs()
    net_ret = apply_costs(port_ret, to, cost_bps, slip_bps)
    ann_to = float(to.sum() / max(len(to) / 252, 1))
    return {"returns": net_ret, "turnover_annual": ann_to, "description": DESCRIPTION}
