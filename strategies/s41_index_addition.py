"""
S41 — Index addition/deletion effect (Harris & Gurel 1986, Shleifer 1986)
Universe:  S&P 500 C&P membership changes detected via index_constituent_timeseries
Signal:    Long new additions from effective date; short (or avoid) deletions.
           Pure membership-change signal — no look-ahead into announcement date.
           (Announcement dates are ~1-2 weeks before effective; this captures the
           residual price impact after the announcement pop, which is often revertible.)
Execution: Enter at close of effective date; hold for 20 trading days.
Sizing:    Equal-weight across active positions; long-only or long-short.
Note:      Norgate gives effective-date membership changes; announcement dates
           would require a secondary source for the full pre-announcement drift.
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
from engine import apply_costs

log = logging.getLogger(__name__)
DESCRIPTION = "S&P 500 index addition/deletion, long additions 20-day hold, effective date"
TRADING_DAYS = 252
HOLD_DAYS    = 20   # hold 20 trading days after effective date


def _detect_changes(mask_map: dict[str, pd.Series],
                    trading_idx: pd.DatetimeIndex) -> dict[str, list[pd.Timestamp]]:
    """
    Detect dates when each symbol was added to or removed from the index.
    Returns {'additions': [(date, sym),...], 'deletions': [(date, sym),...]}
    """
    additions = []
    deletions = []

    for sym, mask in mask_map.items():
        m_reindexed = mask.reindex(trading_idx, method="ffill").fillna(False).astype(bool)
        changes = m_reindexed.astype(int).diff()
        add_dates = trading_idx[changes == 1]
        del_dates = trading_idx[changes == -1]
        for d in add_dates:
            additions.append((d, sym))
        for d in del_dates:
            deletions.append((d, sym))

    return {"additions": sorted(additions), "deletions": sorted(deletions)}


def run(config: dict) -> dict:
    cfg      = config["backtest"]
    start    = cfg["start_date"]
    end      = cfg["end_date"]
    is_smoke = config.get("smoke", False)
    cache    = config["paths"]["cache_dir"]

    cost_bps = config["costs"]["equity_cost_bps"]
    slip_bps = config["costs"]["equity_slippage_bps"]
    min_dv   = config["liquidity"]["min_dollar_volume"]
    min_px   = config["liquidity"]["min_price"]
    long_short = False   # long additions only; can extend to short deletions

    wl_name  = config["universes"].get("sp500", "S&P 500 Current & Past")
    idx_name = config["universes"].get("sp500_index", "S&P 500")
    start_load = "2013-01-01" if is_smoke else "1999-01-01"

    log.info("S41: loading S&P 500 C&P ...")
    symbols = watchlist_symbols(wl_name)
    if is_smoke:
        symbols = symbols[:200]  # need more symbols to catch membership changes

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

    log.info("S41: %d symbols loaded, %d with constituent masks", len(close_map), len(mask_map))

    trading_idx = pd.bdate_range(start, end)
    close_df = pd.DataFrame(close_map).reindex(trading_idx, method="ffill")
    dv_df    = pd.DataFrame(dv_map).reindex(trading_idx, method="ffill")
    ret_df   = close_df.pct_change(fill_method=None).fillna(0.0)

    changes = _detect_changes(mask_map, trading_idx)
    log.info("S41: detected %d additions, %d deletions",
             len(changes["additions"]), len(changes["deletions"]))

    # Build daily position matrix: each event opens a 20-day position
    position_df = pd.DataFrame(0.0, index=trading_idx, columns=list(close_map.keys()))

    for event_date, sym in changes["additions"]:
        if event_date < pd.Timestamp(start) or sym not in ret_df.columns:
            continue
        start_idx = trading_idx.searchsorted(event_date)
        end_idx   = min(start_idx + HOLD_DAYS, len(trading_idx))
        position_df.loc[trading_idx[start_idx:end_idx], sym] += 1.0

    if long_short:
        for event_date, sym in changes["deletions"]:
            if event_date < pd.Timestamp(start) or sym not in ret_df.columns:
                continue
            start_idx = trading_idx.searchsorted(event_date)
            end_idx   = min(start_idx + HOLD_DAYS, len(trading_idx))
            position_df.loc[trading_idx[start_idx:end_idx], sym] -= 1.0

    # Normalise: equal-weight across active positions each day
    n_active = position_df.abs().sum(axis=1).clip(lower=1e-9)
    weight_df = position_df.div(n_active, axis=0)
    weight_df[(n_active < 0.5).values] = 0.0  # all-cash when nothing active

    port_ret = (weight_df * ret_df.reindex(trading_idx).fillna(0.0)).sum(axis=1)
    to       = weight_df.diff().abs().sum(axis=1) / 2.0

    net_ret  = apply_costs(port_ret, to, cost_bps, slip_bps)

    spy = load_price_series("SPY", start, end, ADJ_TOTALRETURN, cache)
    bm  = spy["Close"].pct_change(fill_method=None).reindex(net_ret.index)

    ann_to = float(to.sum() / max(len(to) / TRADING_DAYS, 1))
    active_days = float((position_df.abs().sum(axis=1) > 0).mean()) * 100
    log.info("S41 done. Active: %.1f%% of days, ann turnover: %.2fx", active_days, ann_to)

    return {
        "returns": net_ret, "benchmark": bm,
        "description": DESCRIPTION, "turnover_annual": ann_to,
    }
