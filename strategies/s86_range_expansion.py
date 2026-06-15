"""
S86 — Range-expansion breakout.
Enter (long) when today's true range > 2x ATR(20) AND close is in top 25% of today's range.
Only for liquid S&P 500 PIT members. Hold 3 trading days then exit.
Uses ADJ_CAPITAL for range computation; ADJ_TOTALRETURN for returns.
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
DESCRIPTION = "Range-expansion breakout: TR > 2x ATR(20) and close in top 25% of range; hold 3 days. Liquid S&P 500 PIT."
ATR_WIN = 20
MULT = 2.0
CLOSE_THRESH = 0.75  # close in top 25% of range
HOLD_DAYS = 3

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

    close_cap, high_cap, low_cap, close_ret_map, dv_map, mask_map = {}, {}, {}, {}, {}, {}
    for sym in symbols:
        df_c = load_price_series(sym, start=start_load, end=end, adjustment=ADJ_CAPITAL, cache_dir=cache)
        df_r = load_price_series(sym, start=start_load, end=end, adjustment=ADJ_TOTALRETURN, cache_dir=cache)
        if df_c.empty or df_c["Close"].max() < min_px:
            continue
        close_cap[sym] = df_c["Close"]
        high_cap[sym] = df_c["High"] if "High" in df_c.columns else df_c["Close"]
        low_cap[sym] = df_c["Low"] if "Low" in df_c.columns else df_c["Close"]
        close_ret_map[sym] = df_r["Close"]
        dv_map[sym] = compute_dollar_volume(df_r)
        m = index_constituent_mask(sym, idx_name, start=start_load, end=end, cache_dir=cache)
        if not m.empty:
            mask_map[sym] = m

    trading_idx = pd.bdate_range(start, end)
    close_c = pd.DataFrame(close_cap).reindex(trading_idx, method="ffill")
    high_c = pd.DataFrame(high_cap).reindex(trading_idx, method="ffill")
    low_c = pd.DataFrame(low_cap).reindex(trading_idx, method="ffill")
    close_r = pd.DataFrame(close_ret_map).reindex(trading_idx, method="ffill")
    dv_df = pd.DataFrame(dv_map).reindex(trading_idx, method="ffill")
    member_df = (pd.DataFrame(mask_map).reindex(trading_idx, method="ffill")
                 .infer_objects(copy=False).fillna(False).astype(bool))
    for col in close_c.columns:
        if col not in member_df.columns:
            member_df[col] = False

    ret_df = close_r.pct_change(fill_method=None).fillna(0.0)

    # Track open positions: {sym: days_remaining}
    positions = {}  # {sym: days_left}
    port_rets = pd.Series(0.0, index=trading_idx)
    to_series = pd.Series(0.0, index=trading_idx)
    WARMUP = ATR_WIN + 2

    for i, dt in enumerate(trading_idx):
        if dt < pd.Timestamp(start):
            continue
        if i < WARMUP:
            continue

        members = member_df.iloc[i] if i < len(member_df) else member_df.iloc[-1]
        dv_now = dv_df.iloc[i - 1]

        # Decrement holding periods and remove expired
        to_exit = [s for s, d in positions.items() if d <= 0]
        for s in to_exit:
            del positions[s]
        positions = {s: d - 1 for s, d in positions.items()}

        # P&L for positions open BEFORE today's entries (no same-bar entry return)
        existing_pos = list(positions.keys())
        if existing_pos:
            day_ret = sum(float(ret_df.iloc[i].get(s, 0.0)) for s in existing_pos) / len(existing_pos)
            port_rets.iloc[i] = day_ret

        valid_cols = [
            c for c in close_c.columns
            if members.get(c, False) and dv_now.get(c, 0) >= min_dv and c not in positions
        ]

        new_entries = []
        for c in valid_cols:
            hi = float(high_c.iloc[i].get(c, np.nan))
            lo = float(low_c.iloc[i].get(c, np.nan))
            cl = float(close_c.iloc[i].get(c, np.nan))
            cl_prev = float(close_c.iloc[i - 1].get(c, np.nan))
            if not all(np.isfinite(v) for v in [hi, lo, cl, cl_prev]):
                continue
            tr = max(hi - lo, abs(hi - cl_prev), abs(lo - cl_prev))
            atr = float((high_c[c].iloc[max(i - ATR_WIN, 0):i] -
                         low_c[c].iloc[max(i - ATR_WIN, 0):i]).mean())
            if atr <= 0 or tr < MULT * atr:
                continue
            rng = hi - lo
            if rng <= 0:
                continue
            close_pct = (cl - lo) / rng
            if close_pct >= CLOSE_THRESH:
                new_entries.append(c)

        for s in new_entries:
            positions[s] = HOLD_DAYS

        to_series.iloc[i] = len(new_entries) / max(len(existing_pos) + len(new_entries), 1) / 2.0

    net_ret = apply_costs(port_rets, to_series, cost_bps, slip_bps)
    spy = load_price_series("SPY", start, end, ADJ_TOTALRETURN, cache)
    bm = spy["Close"].pct_change(fill_method=None).reindex(net_ret.index)
    ann_to = float(to_series.sum() / max(len(to_series) / TRADING_DAYS, 1))
    return {"returns": net_ret, "benchmark": bm, "description": DESCRIPTION, "turnover_annual": ann_to}
