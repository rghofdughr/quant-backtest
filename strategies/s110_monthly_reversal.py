"""
s110 — Monthly Reversal (R1000)
Signal: Buy bottom decile of R1000 by 21-day return. Long-only, equal weight.
Monthly rebalance. (Short-term reversal at monthly frequency)
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
DESCRIPTION = "Monthly reversal: long bottom decile of R1000 by 21d return, monthly rebalance"

LOOKBACK  = 21
DECILE    = 0.10
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

    mom_df = close_df.pct_change(LOOKBACK)

    weight_schedule = {}
    rebal_dates = pd.date_range(start, end, freq="BME")

    for rd in rebal_dates:
        rd = min(rd, idx[-1])
        pos = idx.searchsorted(rd)
        if pos < LOOKBACK + 1:
            continue

        dv_now  = dv_df.iloc[pos - 1]
        members = member_df.iloc[pos] if pos < len(member_df) else member_df.iloc[-1]

        valid = [c for c in close_df.columns
                 if members.get(c, False) and dv_now.get(c, 0) >= min_dv]
        if len(valid) < MIN_STOCKS:
            continue

        mom = mom_df.loc[rd, valid].dropna()
        if len(mom) < MIN_STOCKS:
            continue

        cutoff = mom.quantile(DECILE)
        picks  = mom[mom <= cutoff].index.tolist()
        if not picks:
            continue
        weight_schedule[rd] = {s: 1.0 / len(picks) for s in picks}

    gross_ret, to = portfolio_returns_from_weights(weight_schedule, close_df, start, end)
    net_ret = apply_costs(gross_ret, to, cost_bps, slip_bps)
    ann_to = float(to.sum() / max(len(to) / 252, 1))
    return {"returns": net_ret, "turnover_annual": ann_to, "description": DESCRIPTION}
