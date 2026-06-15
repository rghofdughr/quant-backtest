"""
S87 — Volume-spike reversal.
Fade S&P 500 PIT stocks with volume > 3x 50d avg AND a large adverse move (down >2%).
Hold 5 trading days. Expects reversion after oversold-on-volume flush.
"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import logging
import numpy as np
import pandas as pd
from data import load_price_series, watchlist_symbols, index_constituent_mask, ADJ_TOTALRETURN, compute_dollar_volume
from engine import apply_costs

log = logging.getLogger(__name__)
TRADING_DAYS = 252
DESCRIPTION = "Volume-spike reversal: long S&P 500 PIT stocks with volume >3x avg and adverse day >2%; hold 5 days."
VOL_MULT = 3.0
VOL_WIN = 50
RET_THRESH = -0.02  # must be down at least 2%
HOLD_DAYS = 5
MAX_POSITIONS = 30

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

    close_map, vol_map, dv_map, mask_map = {}, {}, {}, {}
    for sym in symbols:
        df = load_price_series(sym, start=start_load, end=end, adjustment=ADJ_TOTALRETURN, cache_dir=cache)
        if df.empty or df["Close"].max() < min_px:
            continue
        close_map[sym] = df["Close"]
        vol_map[sym] = df["Volume"] if "Volume" in df.columns else pd.Series(dtype=float)
        dv_map[sym] = compute_dollar_volume(df)
        m = index_constituent_mask(sym, idx_name, start=start_load, end=end, cache_dir=cache)
        if not m.empty:
            mask_map[sym] = m

    trading_idx = pd.bdate_range(start, end)
    close_df = pd.DataFrame(close_map).reindex(trading_idx, method="ffill")
    vol_df = pd.DataFrame(vol_map).reindex(trading_idx)
    dv_df = pd.DataFrame(dv_map).reindex(trading_idx, method="ffill")
    member_df = (pd.DataFrame(mask_map).reindex(trading_idx, method="ffill")
                 .infer_objects(copy=False).fillna(False).astype(bool))
    for col in close_df.columns:
        if col not in member_df.columns:
            member_df[col] = False

    ret_df = close_df.pct_change(fill_method=None).fillna(0.0)

    positions = {}  # {sym: days_left}
    port_rets = pd.Series(0.0, index=trading_idx)
    to_series = pd.Series(0.0, index=trading_idx)
    WARMUP = VOL_WIN + 2

    for i, dt in enumerate(trading_idx):
        if dt < pd.Timestamp(start):
            continue
        if i < WARMUP:
            continue

        members = member_df.iloc[i] if i < len(member_df) else member_df.iloc[-1]
        dv_now = dv_df.iloc[i - 1]

        to_exit = [s for s, d in positions.items() if d <= 0]
        for s in to_exit:
            del positions[s]
        positions = {s: d - 1 for s, d in positions.items()}

        if len(positions) < MAX_POSITIONS:
            valid_cols = [
                c for c in close_df.columns
                if members.get(c, False) and dv_now.get(c, 0) >= min_dv and c not in positions
            ]
            for c in valid_cols:
                if len(positions) >= MAX_POSITIONS:
                    break
                day_ret_c = float(ret_df.iloc[i].get(c, 0.0))
                if day_ret_c > RET_THRESH:
                    continue
                vol_today = float(vol_df.iloc[i].get(c, 0))
                vol_avg = float(vol_df[c].iloc[max(i - VOL_WIN, 0):i].mean()) if c in vol_df.columns else 0.0
                if vol_avg > 0 and vol_today >= VOL_MULT * vol_avg:
                    positions[c] = HOLD_DAYS

        if positions:
            all_pos = list(positions.keys())
            day_ret = sum(float(ret_df.iloc[i].get(s, 0.0)) for s in all_pos) / len(all_pos)
            port_rets.iloc[i] = day_ret
            new_entries_count = sum(1 for d in positions.values() if d == HOLD_DAYS)
            to_series.iloc[i] = new_entries_count / max(len(all_pos), 1) / 2.0

    net_ret = apply_costs(port_rets, to_series, cost_bps, slip_bps)
    spy = load_price_series("SPY", start, end, ADJ_TOTALRETURN, cache)
    bm = spy["Close"].pct_change(fill_method=None).reindex(net_ret.index)
    ann_to = float(to_series.sum() / max(len(to_series) / TRADING_DAYS, 1))
    return {"returns": net_ret, "benchmark": bm, "description": DESCRIPTION, "turnover_annual": ann_to}
