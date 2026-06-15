"""
S99 — Dividend initiation momentum.
Detect first-ever dividend payment by comparing dividend history.
Long stocks that initiated dividends; hold 63 trading days.
Caveat: Norgate gives payment/ex-date, not announcement date —
there may be a lag between announcement and detection here.
"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import logging
import numpy as np
import pandas as pd
from data import (load_price_series, watchlist_symbols, index_constituent_mask,
                  ADJ_TOTALRETURN, compute_dollar_volume, load_dividends)
from engine import apply_costs

log = logging.getLogger(__name__)
TRADING_DAYS = 252
DESCRIPTION = "Dividend initiation: long S&P 500 stocks paying first-ever dividend; 63d hold. Ex-date based (not announcement)."
HOLD_DAYS = 63
MIN_HISTORY_YEARS = 2  # require 2yr of data before initiation to confirm it's truly first

def run(config):
    cfg = config["backtest"]
    start, end = cfg["start_date"], cfg["end_date"]
    is_smoke = config.get("smoke", False)
    cache = config["paths"]["cache_dir"]
    cost_bps = config["costs"]["equity_cost_bps"]
    slip_bps = config["costs"]["equity_slippage_bps"]
    min_dv = config["liquidity"]["min_dollar_volume"]
    min_px = config["liquidity"]["min_price"]
    wl_name = config["universes"].get("sp500", "S&P 500 Current & Past")
    idx_name = config["universes"].get("sp500_index", "S&P 500")
    start_load = "2013-01-01" if is_smoke else "1995-01-01"

    symbols = watchlist_symbols(wl_name)
    if is_smoke:
        symbols = symbols[:200]

    close_map, dv_map, mask_map, div_map = {}, {}, {}, {}
    for sym in symbols:
        df = load_price_series(sym, start=start_load, end=end, adjustment=ADJ_TOTALRETURN, cache_dir=cache)
        if df.empty or df["Close"].max() < min_px:
            continue
        close_map[sym] = df["Close"]
        dv_map[sym] = compute_dollar_volume(df)
        m = index_constituent_mask(sym, idx_name, start=start_load, end=end, cache_dir=cache)
        if not m.empty:
            mask_map[sym] = m
        try:
            divs = load_dividends(sym, cache_dir=cache)
            if not divs.empty:
                div_map[sym] = divs
        except Exception:
            pass

    if not div_map:
        raise RuntimeError("S99: no dividend data loaded.")

    trading_idx = pd.bdate_range(start, end)
    close_df = pd.DataFrame(close_map).reindex(trading_idx, method="ffill")
    dv_df = pd.DataFrame(dv_map).reindex(trading_idx, method="ffill")
    member_df = (pd.DataFrame(mask_map).reindex(trading_idx, method="ffill")
                 .infer_objects(copy=False).fillna(False).astype(bool))
    for col in close_df.columns:
        if col not in member_df.columns:
            member_df[col] = False

    ret_df = close_df.pct_change(fill_method=None).fillna(0.0)
    start_ts = pd.Timestamp(start)
    grace_period = pd.Timedelta(days=MIN_HISTORY_YEARS * 365)

    # Find initiation events: first dividend that falls within [start, end] AND
    # stock had at least 2yr of price history before the ex-date with NO prior dividends
    initiations = []
    for sym, divs in div_map.items():
        if sym not in close_map:
            continue
        divs_sorted = divs.sort_index()
        first_div_date = divs_sorted.index[0]
        first_div_ts = pd.Timestamp(first_div_date)
        # Only count as initiation if first dividend is within our backtest window
        if first_div_ts < start_ts:
            continue
        # Must have enough price history before first dividend
        price_history = close_df[sym].dropna() if sym in close_df.columns else pd.Series()
        price_start = price_history.index[0] if len(price_history) > 0 else first_div_ts
        if (first_div_ts - price_start) < grace_period:
            continue
        initiations.append((first_div_ts, sym))

    initiations = sorted(initiations)
    log.info("S99: %d dividend initiation events", len(initiations))

    position_df = pd.DataFrame(0.0, index=trading_idx, columns=list(close_map.keys()))
    for init_date, sym in initiations:
        if sym not in ret_df.columns:
            continue
        start_idx = trading_idx.searchsorted(init_date)
        if start_idx >= len(trading_idx):
            continue
        members = member_df.iloc[min(start_idx, len(member_df) - 1)]
        dv_now = dv_df.iloc[max(start_idx - 1, 0)]
        if not members.get(sym, False) or dv_now.get(sym, 0) < min_dv:
            continue
        end_idx = min(start_idx + HOLD_DAYS, len(trading_idx))
        position_df.loc[trading_idx[start_idx:end_idx], sym] += 1.0

    n_active = position_df.abs().sum(axis=1).clip(lower=1e-9)
    weight_df = position_df.div(n_active, axis=0)
    weight_df[(n_active < 0.5).values] = 0.0

    port_ret = (weight_df * ret_df).sum(axis=1)
    to = weight_df.diff().abs().sum(axis=1) / 2.0
    net_ret = apply_costs(port_ret, to, cost_bps, slip_bps)

    spy = load_price_series("SPY", start, end, ADJ_TOTALRETURN, cache)
    bm = spy["Close"].pct_change(fill_method=None).reindex(net_ret.index)
    ann_to = float(to.sum() / max(len(to) / TRADING_DAYS, 1))
    return {"returns": net_ret, "benchmark": bm, "description": DESCRIPTION, "turnover_annual": ann_to}
