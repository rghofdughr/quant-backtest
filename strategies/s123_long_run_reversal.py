"""
s123 — Long-Run Reversal (DeBondt & Thaler 1985)
Signal:    48-month cumulative return, skipping last 12 months
           (i.e. return from t-60m to t-12m)
Long:      bottom decile (multi-year losers)
Short:     top decile (multi-year winners)
Rebalance: Annual (first trading day of each year)
Universe:  Russell 1000 PIT
"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import logging
import numpy as np
import pandas as pd
from data import load_price_series, watchlist_symbols, index_constituent_mask, ADJ_TOTALRETURN, compute_dollar_volume
from engine import portfolio_returns_from_weights, apply_costs

log = logging.getLogger(__name__)
DESCRIPTION = "Long-Run Reversal: buy 4yr losers, short 4yr winners, annual rebalance (DeBondt-Thaler)"

SIGNAL_WINDOW = 1008   # ~48 months (t-60m to t-12m)
SKIP_WINDOW   = 252    # skip last 12 months
DECILE        = 0.10
MIN_STOCKS    = 30
LOAD_START    = "1994-01-01"   # need 6yr pre-backtest for full lookback


def run(config):
    cfg      = config["backtest"]
    start, end = cfg["start_date"], cfg["end_date"]
    cache    = config["paths"]["cache_dir"]
    cost_bps = config["costs"]["equity_cost_bps"]
    slip_bps = config["costs"]["equity_slippage_bps"]
    min_dv   = config["liquidity"]["min_dollar_volume"]
    min_px   = config["liquidity"]["min_price"]
    wl_name  = config["universes"]["russell1000"]
    idx_name = config["universes"]["russell1000_index"]

    symbols = watchlist_symbols(wl_name)
    close_map, dv_map, mask_map = {}, {}, {}
    for sym in symbols:
        df = load_price_series(sym, start=LOAD_START, end=end,
                               adjustment=ADJ_TOTALRETURN, cache_dir=cache)
        if df.empty or df["Close"].max() < min_px:
            continue
        close_map[sym] = df["Close"]
        dv_map[sym]    = compute_dollar_volume(df)
        m = index_constituent_mask(sym, idx_name, start=LOAD_START, end=end, cache_dir=cache)
        if not m.empty:
            mask_map[sym] = m

    # Build on extended index so lookback reaches pre-backtest prices
    full_idx  = pd.bdate_range(LOAD_START, end)
    close_df  = pd.DataFrame(close_map).reindex(full_idx, method="ffill")
    dv_df     = pd.DataFrame(dv_map).reindex(full_idx, method="ffill")
    member_df = (pd.DataFrame(mask_map).reindex(full_idx, method="ffill")
                 .infer_objects(copy=False).fillna(False).astype(bool))
    for col in close_df.columns:
        if col not in member_df.columns:
            member_df[col] = False

    weight_schedule = {}

    # Annual rebalance: first trading day of each calendar year within backtest window
    backtest_idx = pd.bdate_range(start, end)
    for yr in range(pd.Timestamp(start).year, pd.Timestamp(end).year + 1):
        yr_days = backtest_idx[backtest_idx.year == yr]
        if len(yr_days) == 0:
            continue
        date = yr_days[0]
        pos  = full_idx.searchsorted(date)

        min_pos = SIGNAL_WINDOW + SKIP_WINDOW + 10
        if pos < min_pos:
            continue

        # PIT filters
        members  = member_df.iloc[pos]
        dv_now   = dv_df.iloc[pos]
        px_now   = close_df.iloc[pos]
        valid = [c for c in close_df.columns
                 if members.get(c, False)
                 and dv_now.get(c, 0.0) >= min_dv
                 and px_now.get(c, 0.0) >= min_px]
        if len(valid) < MIN_STOCKS:
            continue

        # Signal: return from (pos - SIGNAL_WINDOW - SKIP_WINDOW) to (pos - SKIP_WINDOW)
        p_old = close_df.iloc[pos - SIGNAL_WINDOW - SKIP_WINDOW][valid]
        p_mid = close_df.iloc[pos - SKIP_WINDOW][valid]
        ret   = (p_mid / p_old - 1).replace([np.inf, -np.inf], np.nan).dropna()
        if len(ret) < MIN_STOCKS:
            continue

        lo_cut = ret.quantile(DECILE)
        hi_cut = ret.quantile(1 - DECILE)
        longs  = ret[ret <= lo_cut].index.tolist()   # worst performers → BUY
        shorts = ret[ret >= hi_cut].index.tolist()   # best performers  → SHORT

        if not longs or not shorts:
            continue

        w = {s:  1.0 / len(longs)  for s in longs}
        for s in shorts:
            w[s] = w.get(s, 0.0) - 1.0 / len(shorts)
        weight_schedule[date] = w

    gross_ret, to = portfolio_returns_from_weights(weight_schedule, close_df, start, end)
    net_ret = apply_costs(gross_ret, to, cost_bps, slip_bps)
    ann_to  = float(to.sum() / max(len(to) / 252, 1))
    return {"returns": net_ret, "turnover_annual": ann_to, "description": DESCRIPTION}
