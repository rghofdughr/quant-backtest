"""
S95 — R2000 → R1000 promotion momentum.
Detect stocks moving from Russell 2000 to Russell 1000 membership
(must be in R2000 and then in R1000; typically June reconstitution).
Long promoted names for 60 days post-promotion.
"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import logging
import numpy as np
import pandas as pd
from data import load_price_series, watchlist_symbols, index_constituent_mask, ADJ_TOTALRETURN
from engine import apply_costs

log = logging.getLogger(__name__)
TRADING_DAYS = 252
DESCRIPTION = "R2000->R1000 promotion: long promoted stocks for 60d post-effective-date. June reconstitution clustering noted."
HOLD_DAYS = 60

def run(config):
    cfg = config["backtest"]
    start, end = cfg["start_date"], cfg["end_date"]
    is_smoke = config.get("smoke", False)
    cache = config["paths"]["cache_dir"]
    cost_bps = config["costs"]["equity_cost_bps"]
    slip_bps = config["costs"]["equity_slippage_bps"]
    min_px = config["liquidity"]["min_price"]
    # Need to check both R2000 and R1000 memberships
    wl_r2000 = config["universes"].get("russell2000", "Russell 2000 Current & Past")
    wl_r1000 = config["universes"].get("russell1000", "Russell 1000 Current & Past")
    start_load = "2013-01-01" if is_smoke else "1999-01-01"

    # Load all symbols from both universes
    syms_r2000 = set(watchlist_symbols(wl_r2000))
    syms_r1000 = set(watchlist_symbols(wl_r1000))
    # Candidates: symbols that appear in both (moved from R2000 to R1000)
    candidates = syms_r2000 & syms_r1000
    if is_smoke:
        candidates = list(candidates)[:300]
    else:
        candidates = list(candidates)

    log.info("S95: %d candidate symbols (in both R2000 and R1000 C&P)", len(candidates))

    close_map, mask_r2000, mask_r1000 = {}, {}, {}
    for sym in candidates:
        df = load_price_series(sym, start=start_load, end=end, adjustment=ADJ_TOTALRETURN, cache_dir=cache)
        if df.empty:
            continue
        close_map[sym] = df["Close"]
        m2 = index_constituent_mask(sym, "Russell 2000", start=start_load, end=end, cache_dir=cache)
        m1 = index_constituent_mask(sym, "Russell 1000", start=start_load, end=end, cache_dir=cache)
        if not m2.empty:
            mask_r2000[sym] = m2
        if not m1.empty:
            mask_r1000[sym] = m1

    trading_idx = pd.bdate_range(start, end)
    close_df = pd.DataFrame(close_map).reindex(trading_idx, method="ffill")

    # Detect promotion events: R2000 member becomes R1000 member (1->0 in R2000, 0->1 in R1000)
    promotions = []
    for sym in close_map:
        if sym not in mask_r1000:
            continue
        m1 = mask_r1000[sym].reindex(trading_idx, method="ffill").fillna(False).astype(bool)
        # R1000 addition date
        changes1 = m1.astype(int).diff()
        r1000_add_dates = trading_idx[changes1.values == 1]
        for d in r1000_add_dates:
            # Must have been in R2000 within last 90 days (before promotion)
            if sym in mask_r2000:
                m2 = mask_r2000[sym].reindex(trading_idx, method="ffill").fillna(False)
                d_pos = trading_idx.searchsorted(d)
                lookback_start = max(d_pos - 90, 0)
                was_r2000 = m2.iloc[lookback_start:d_pos].any()
                if was_r2000:
                    promotions.append((d, sym))

    promotions = sorted(promotions)
    log.info("S95: detected %d R2000->R1000 promotions", len(promotions))

    ret_df = close_df.pct_change(fill_method=None).fillna(0.0)
    position_df = pd.DataFrame(0.0, index=trading_idx, columns=list(close_map.keys()))

    for promo_date, sym in promotions:
        if promo_date < pd.Timestamp(start) or sym not in ret_df.columns:
            continue
        start_idx = trading_idx.searchsorted(promo_date)
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
