"""
S85 — Gap-and-go continuation.
Long S&P 500 PIT stocks that gap up >1.5% (vs prior close) on rising volume
(today's volume > 50d avg). Hold to close of that day only (intraday).
Uses ADJ_CAPITAL for price-level gaps; unadjusted volume from Turnover.
"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import logging
import numpy as np
import pandas as pd
from data import load_price_series, watchlist_symbols, index_constituent_mask, ADJ_CAPITAL, ADJ_TOTALRETURN, compute_dollar_volume
from engine import apply_costs

log = logging.getLogger(__name__)
TRADING_DAYS = 252
DESCRIPTION = "Gap-and-go continuation: long S&P 500 PIT stocks gapping up >1.5% on rising volume; daily in-day hold."
GAP_THRESH = 0.015
VOL_WIN = 50

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

    close_cap_map, open_cap_map, vol_map, dv_map, mask_map = {}, {}, {}, {}, {}
    for sym in symbols:
        df_c = load_price_series(sym, start=start_load, end=end, adjustment=ADJ_CAPITAL, cache_dir=cache)
        df_r = load_price_series(sym, start=start_load, end=end, adjustment=ADJ_TOTALRETURN, cache_dir=cache)
        if df_c.empty or df_c["Close"].max() < min_px:
            continue
        close_cap_map[sym] = df_c["Close"]
        open_cap_map[sym] = df_c["Open"] if "Open" in df_c.columns else df_c["Close"]
        vol_map[sym] = df_c["Volume"] if "Volume" in df_c.columns else pd.Series(dtype=float)
        dv_map[sym] = compute_dollar_volume(df_r)
        m = index_constituent_mask(sym, idx_name, start=start_load, end=end, cache_dir=cache)
        if not m.empty:
            mask_map[sym] = m

    trading_idx = pd.bdate_range(start, end)
    close_cap = pd.DataFrame(close_cap_map).reindex(trading_idx, method="ffill")
    open_cap = pd.DataFrame(open_cap_map).reindex(trading_idx, method="ffill")
    vol_df = pd.DataFrame(vol_map).reindex(trading_idx)
    dv_df = pd.DataFrame(dv_map).reindex(trading_idx, method="ffill")
    member_df = (pd.DataFrame(mask_map).reindex(trading_idx, method="ffill")
                 .infer_objects(copy=False).fillna(False).astype(bool))
    for col in close_cap.columns:
        if col not in member_df.columns:
            member_df[col] = False

    # Intraday return: open-to-close using TOTALRETURN prices
    close_ret_map = {}
    for sym in close_cap_map.keys():
        df_r = load_price_series(sym, start=start_load, end=end, adjustment=ADJ_TOTALRETURN, cache_dir=cache)
        if not df_r.empty and "Open" in df_r.columns:
            o = df_r["Open"].reindex(trading_idx, method="ffill")
            c = df_r["Close"].reindex(trading_idx, method="ffill")
            close_ret_map[sym] = (c / o - 1.0).fillna(0.0)
        else:
            close_ret_map[sym] = pd.Series(0.0, index=trading_idx)

    intra_ret = pd.DataFrame(close_ret_map)

    port_rets = pd.Series(0.0, index=trading_idx)
    to_series = pd.Series(0.0, index=trading_idx)
    WARMUP = VOL_WIN + 1

    for i, dt in enumerate(trading_idx):
        if dt < pd.Timestamp(start):
            continue
        if i < WARMUP:
            continue

        members = member_df.iloc[i] if i < len(member_df) else member_df.iloc[-1]
        dv_now = dv_df.iloc[i - 1]

        valid_cols = [
            c for c in close_cap.columns
            if members.get(c, False) and dv_now.get(c, 0) >= min_dv
        ]

        gap_names = []
        for c in valid_cols:
            open_today = float(open_cap.iloc[i].get(c, np.nan))
            close_prev = float(close_cap.iloc[i - 1].get(c, np.nan))
            if not (np.isfinite(open_today) and np.isfinite(close_prev) and close_prev > 0):
                continue
            gap = open_today / close_prev - 1.0
            if gap < GAP_THRESH:
                continue
            # Volume confirmation: today > 50d avg
            vol_today = float(vol_df.iloc[i].get(c, 0))
            vol_avg = float(vol_df[c].iloc[max(i - VOL_WIN, 0):i].mean()) if c in vol_df.columns else 0.0
            if vol_today > vol_avg and vol_avg > 0:
                gap_names.append(c)

        if not gap_names:
            continue

        n = len(gap_names)
        day_ret = sum(float(intra_ret.iloc[i].get(c, 0.0)) for c in gap_names) / n
        port_rets.iloc[i] = day_ret
        to_series.iloc[i] = 0.5  # full turnover each day (positions are 1-day only)

    net_ret = apply_costs(port_rets, to_series, cost_bps, slip_bps)
    spy = load_price_series("SPY", start, end, ADJ_TOTALRETURN, cache)
    bm = spy["Close"].pct_change(fill_method=None).reindex(net_ret.index)
    ann_to = float(to_series.sum() / max(len(to_series) / TRADING_DAYS, 1))
    return {"returns": net_ret, "benchmark": bm, "description": DESCRIPTION, "turnover_annual": ann_to}
