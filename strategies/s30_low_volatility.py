"""
S30 — Low-volatility anomaly (Baker-Bradley-Wurgler 2011)
Universe:  S&P 500 point-in-time
Signal:    Trailing 252-day realized volatility. Long lowest-vol quintile; short
           highest-vol quintile (or long-only vs SPY benchmark).
Rebalance: Monthly
Execution: Next trading day close
Sizing:    Equal-weight within each quintile; dollar-neutral (L/S variant).
Sweep:     vol vs beta ranking, quintile vs decile
"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import logging
import numpy as np
import pandas as pd

from data import (
    load_price_series, watchlist_symbols, index_constituent_mask,
    ADJ_TOTALRETURN, compute_dollar_volume,
)
from engine import portfolio_returns_from_weights, apply_costs

log = logging.getLogger(__name__)
DESCRIPTION = "Low-volatility anomaly, S&P 500 PIT, long low-vol quintile vs SPY"
TRADING_DAYS = 252


def run(config: dict) -> dict:
    cfg      = config["backtest"]
    start    = cfg["start_date"]
    end      = cfg["end_date"]
    is_smoke = config.get("smoke", False)
    cache    = config["paths"]["cache_dir"]

    s30_cfg     = config.get("strategies", {}).get("s30", {})
    vol_lookback = s30_cfg.get("vol_lookback", 252)
    long_short   = s30_cfg.get("long_short", False)
    n_quantile   = 5

    cost_bps = config["costs"]["equity_cost_bps"]
    slip_bps = config["costs"]["equity_slippage_bps"]
    min_dv   = config["liquidity"]["min_dollar_volume"]
    min_px   = config["liquidity"]["min_price"]

    wl_name  = config["universes"].get("sp500", "S&P 500 Current & Past")
    idx_name = config["universes"].get("sp500_index", "S&P 500")
    start_load = "2013-01-01" if is_smoke else "1997-01-01"

    log.info("S30: loading S&P 500 C&P ...")
    symbols = watchlist_symbols(wl_name)
    if is_smoke:
        symbols = symbols[:120]

    close_map, dv_map, mask_map = {}, {}, {}
    for sym in symbols:
        df = load_price_series(sym, start=start_load, end=end,
                               adjustment=ADJ_TOTALRETURN, cache_dir=cache)
        if df.empty or df["Close"].max() < min_px:
            continue
        close_map[sym] = df["Close"]
        dv_map[sym]    = compute_dollar_volume(df)
        m = index_constituent_mask(sym, idx_name, start=start_load, end=end, cache_dir=cache)
        if not m.empty:
            mask_map[sym] = m

    log.info("S30: %d symbols loaded", len(close_map))

    trading_idx = pd.bdate_range(start, end)
    close_df  = pd.DataFrame(close_map).reindex(trading_idx, method="ffill")
    dv_df     = pd.DataFrame(dv_map).reindex(trading_idx, method="ffill")
    member_df = (pd.DataFrame(mask_map)
                   .reindex(trading_idx, method="ffill")
                   .infer_objects(copy=False)
                   .fillna(False).astype(bool))
    for col in close_df.columns:
        if col not in member_df.columns:
            member_df[col] = False

    # Compute rolling annualised vol for all symbols at once
    ret_df  = close_df.pct_change(fill_method=None)
    vol_df  = ret_df.rolling(vol_lookback).std() * np.sqrt(TRADING_DAYS)

    reb_dates = pd.date_range(start, end, freq="BME")
    weight_schedule: dict = {}

    for d in reb_dates:
        pos = trading_idx.searchsorted(d, side="right") - 1
        if pos < vol_lookback:
            continue

        vols    = vol_df.iloc[pos]
        prices  = close_df.iloc[pos]
        dv_row  = dv_df.iloc[pos]
        members = member_df.iloc[pos]

        valid = (
            members.reindex(vols.index, fill_value=False) &
            (prices.reindex(vols.index) >= min_px) &
            (dv_row.reindex(vols.index) >= min_dv)
        )
        vols_valid = vols[valid].dropna()
        if len(vols_valid) < n_quantile * 2:
            continue

        n_each = max(1, len(vols_valid) // n_quantile)
        ranked = vols_valid.sort_values()
        lows   = ranked.iloc[:n_each].index.tolist()    # lowest vol → long
        highs  = ranked.iloc[-n_each:].index.tolist()   # highest vol → short (if L/S)

        w = {s: 1.0 / len(lows) for s in lows}
        if long_short:
            for s in highs:
                w[s] = -1.0 / len(highs)
        weight_schedule[d] = w

    gross_ret, to = portfolio_returns_from_weights(weight_schedule, close_df, start, end)
    net_ret = apply_costs(gross_ret, to, cost_bps, slip_bps)

    spy = load_price_series("SPY", start, end, ADJ_TOTALRETURN, cache)
    bm  = spy["Close"].pct_change(fill_method=None).reindex(net_ret.index)

    ann_to = float(to.sum() / max(len(to) / TRADING_DAYS, 1))
    log.info("S30 done (long_short=%s). Ann turnover: %.2fx", long_short, ann_to)

    return {
        "returns": net_ret, "benchmark": bm,
        "description": DESCRIPTION, "turnover_annual": ann_to,
    }
