"""
s108 — Short-Term Weekly Reversal (R1000)
Signal: Buy bottom quintile of R1000 by 5-day return. Long-only, equal weight.
Rebalance every 5 trading days. (Jegadeesh 1990 short-term reversal)
"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import logging
import numpy as np
import pandas as pd
from data import (load_price_series, watchlist_symbols, index_constituent_mask,
                  ADJ_TOTALRETURN, compute_dollar_volume)
from engine import portfolio_returns_from_weights, apply_costs

log = logging.getLogger(__name__)
DESCRIPTION = "Weekly reversal: long bottom quintile of R1000 by 5d return, rebalance every 5 days"

LOOKBACK  = 5   # 1-week
QUINTILE  = 0.20
REBAL_N   = 5   # trading days between rebalances
MIN_STOCKS = 20

def run(config):
    cfg   = config["backtest"]
    start, end = cfg["start_date"], cfg["end_date"]
    cache = config["paths"]["cache_dir"]
    cost_bps = config["costs"]["equity_cost_bps"]
    slip_bps = config["costs"]["equity_slippage_bps"]
    min_dv   = config["liquidity"]["min_dollar_volume"]
    min_px   = config["liquidity"]["min_price"]
    wl_name  = config["universes"]["russell1000"]
    idx_name = config["universes"]["russell1000_index"]

    symbols = watchlist_symbols(wl_name)
    close_map, dv_map, mask_map = {}, {}, {}
    for sym in symbols:
        df = load_price_series(sym, start="1997-01-01", end=end, adjustment=ADJ_TOTALRETURN, cache_dir=cache)
        if df.empty or df["Close"].max() < min_px:
            continue
        close_map[sym] = df["Close"]
        dv_map[sym] = compute_dollar_volume(df)
        m = index_constituent_mask(sym, idx_name, start="1997-01-01", end=end, cache_dir=cache)
        if not m.empty:
            mask_map[sym] = m

    idx = pd.bdate_range(start, end)
    close_df = pd.DataFrame(close_map).reindex(idx, method="ffill")
    dv_df    = pd.DataFrame(dv_map).reindex(idx, method="ffill")
    member_df = (pd.DataFrame(mask_map).reindex(idx, method="ffill")
                 .infer_objects(copy=False).fillna(False).astype(bool))
    for col in close_df.columns:
        if col not in member_df.columns:
            member_df[col] = False

    ret_df = close_df.pct_change(fill_method=None)
    mom5_df = close_df.pct_change(LOOKBACK)

    weight_schedule = {}
    rebal_positions = range(LOOKBACK + 1, len(idx), REBAL_N)

    for pos in rebal_positions:
        rd = idx[pos]
        dv_now  = dv_df.iloc[pos - 1]
        members = member_df.iloc[pos] if pos < len(member_df) else member_df.iloc[-1]

        valid = [c for c in close_df.columns
                 if members.get(c, False) and dv_now.get(c, 0) >= min_dv]
        if len(valid) < MIN_STOCKS:
            continue

        mom = mom5_df.loc[rd, valid].dropna()
        if len(mom) < MIN_STOCKS:
            continue

        cutoff = mom.quantile(QUINTILE)
        picks  = mom[mom <= cutoff].index.tolist()
        if not picks:
            continue
        weight_schedule[rd] = {s: 1.0 / len(picks) for s in picks}

    gross_ret, to = portfolio_returns_from_weights(weight_schedule, close_df, start, end)
    net_ret = apply_costs(gross_ret, to, cost_bps, slip_bps)
    ann_to = float(to.sum() / max(len(to) / 252, 1))
    return {"returns": net_ret, "turnover_annual": ann_to, "description": DESCRIPTION}
