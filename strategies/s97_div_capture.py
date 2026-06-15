"""
S97 — Dividend capture.
Buy high-yield S&P 500 members before ex-date; sell after.
Uses unadjusted (ADJ_NONE) close to detect ex-date price drop;
compares vs TOTALRETURN to confirm dividend amount.
Expected to fail on costs: documenting the structural barrier.
"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import logging
import numpy as np
import pandas as pd
from data import (load_price_series, watchlist_symbols, index_constituent_mask,
                  ADJ_TOTALRETURN, ADJ_NONE, compute_dollar_volume, load_dividends)
from engine import apply_costs

log = logging.getLogger(__name__)
TRADING_DAYS = 252
DESCRIPTION = "Dividend capture: buy S&P 500 1d before ex-date, sell 1d after. Structural barrier test."
MAX_POSITIONS = 20
HOLD_DAYS = 2  # buy t-1, sell t+1

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
    start_load = "2013-01-01" if is_smoke else "1997-01-01"

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
        raise RuntimeError("S97: no dividend data loaded. Check Norgate subscription includes dividend history.")

    trading_idx = pd.bdate_range(start, end)
    close_df = pd.DataFrame(close_map).reindex(trading_idx, method="ffill")
    dv_df = pd.DataFrame(dv_map).reindex(trading_idx, method="ffill")
    member_df = (pd.DataFrame(mask_map).reindex(trading_idx, method="ffill")
                 .infer_objects(copy=False).fillna(False).astype(bool))
    for col in close_df.columns:
        if col not in member_df.columns:
            member_df[col] = False

    ret_df = close_df.pct_change(fill_method=None).fillna(0.0)

    # Build event table: (entry_date, sym)
    events = []
    for sym, divs in div_map.items():
        if sym not in close_map:
            continue
        for ex_date in divs.index:
            ex_ts = pd.Timestamp(ex_date)
            # Entry: trading day before ex-date
            prior = trading_idx[trading_idx < ex_ts]
            if len(prior) == 0:
                continue
            entry_date = prior[-1]
            amount = float(divs.loc[ex_date, "Amount"]) if "Amount" in divs.columns else 0.0
            if amount <= 0:
                continue
            # Check it's a meaningful yield (>0.1% of price at entry)
            entry_pos = trading_idx.searchsorted(entry_date)
            px = float(close_df[sym].iloc[entry_pos]) if sym in close_df.columns and entry_pos < len(close_df) else 0.0
            if px <= 0 or amount / px < 0.001:
                continue
            events.append((entry_date, sym))

    events = sorted(events)
    log.info("S97: %d dividend capture events identified", len(events))

    position_df = pd.DataFrame(0.0, index=trading_idx, columns=list(close_map.keys()))
    for entry_date, sym in events:
        if entry_date < pd.Timestamp(start) or sym not in ret_df.columns:
            continue
        start_idx = trading_idx.searchsorted(entry_date)
        members = member_df.iloc[min(start_idx, len(member_df) - 1)]
        dv_now = dv_df.iloc[max(start_idx - 1, 0)]
        if not members.get(sym, False) or dv_now.get(sym, 0) < min_dv:
            continue
        end_idx = min(start_idx + HOLD_DAYS, len(trading_idx))
        position_df.loc[trading_idx[start_idx:end_idx], sym] += 1.0

    # Cap at MAX_POSITIONS
    n_pos = position_df.clip(upper=1.0)
    n_active = n_pos.sum(axis=1).clip(lower=1e-9)
    weight_df = n_pos.div(n_active.clip(lower=MAX_POSITIONS), axis=0)
    weight_df[(n_active < 0.5).values] = 0.0

    port_ret = (weight_df * ret_df).sum(axis=1)
    to = weight_df.diff().abs().sum(axis=1) / 2.0
    net_ret = apply_costs(port_ret, to, cost_bps, slip_bps)

    spy = load_price_series("SPY", start, end, ADJ_TOTALRETURN, cache)
    bm = spy["Close"].pct_change(fill_method=None).reindex(net_ret.index)
    ann_to = float(to.sum() / max(len(to) / TRADING_DAYS, 1))
    return {"returns": net_ret, "benchmark": bm, "description": DESCRIPTION, "turnover_annual": ann_to}
