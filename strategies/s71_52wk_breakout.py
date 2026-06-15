import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import logging
import numpy as np
import pandas as pd
from data import load_price_series, watchlist_symbols, index_constituent_mask, ADJ_CAPITAL, ADJ_TOTALRETURN, compute_dollar_volume
from engine import apply_costs

log = logging.getLogger(__name__)
TRADING_DAYS = 252
DESCRIPTION = "52-week-high breakout continuation on R1000 PIT; 20% trailing stop exit. Uses ADJ_CAPITAL to prevent dividend jumps."
MAX_POSITIONS = 50
TRAIL_PCT = 0.20

def run(config):
    cfg = config["backtest"]
    start, end = cfg["start_date"], cfg["end_date"]
    is_smoke = config.get("smoke", False)
    cache = config["paths"]["cache_dir"]
    cost_bps = config["costs"]["equity_cost_bps"]
    slip_bps = config["costs"]["equity_slippage_bps"]
    min_dv = config["liquidity"]["min_dollar_volume"]
    min_px = config["liquidity"]["min_price"]
    wl_name = config["universes"].get("russell1000", "Russell 1000 Current & Past")
    idx_name = config["universes"].get("russell1000_index", "Russell 1000")
    start_load = "2013-01-01" if is_smoke else "1997-01-01"

    symbols = watchlist_symbols(wl_name)
    if is_smoke:
        symbols = symbols[:150]

    close_cap_map, close_ret_map, dv_map, mask_map = {}, {}, {}, {}
    for sym in symbols:
        # ADJ_CAPITAL for price-level (52wk high); ADJ_TOTALRETURN for returns
        df_c = load_price_series(sym, start=start_load, end=end, adjustment=ADJ_CAPITAL, cache_dir=cache)
        df_r = load_price_series(sym, start=start_load, end=end, adjustment=ADJ_TOTALRETURN, cache_dir=cache)
        if df_c.empty or df_r.empty or df_c["Close"].max() < min_px:
            continue
        close_cap_map[sym] = df_c["Close"]
        close_ret_map[sym] = df_r["Close"]
        dv_map[sym] = compute_dollar_volume(df_r)
        m = index_constituent_mask(sym, idx_name, start=start_load, end=end, cache_dir=cache)
        if not m.empty:
            mask_map[sym] = m

    trading_idx = pd.bdate_range(start, end)
    close_cap = pd.DataFrame(close_cap_map).reindex(trading_idx, method="ffill")
    close_ret = pd.DataFrame(close_ret_map).reindex(trading_idx, method="ffill")
    dv_df = pd.DataFrame(dv_map).reindex(trading_idx, method="ffill")
    member_df = (pd.DataFrame(mask_map).reindex(trading_idx, method="ffill")
                 .infer_objects(copy=False).fillna(False).astype(bool))
    for col in close_cap.columns:
        if col not in member_df.columns:
            member_df[col] = False

    ret_df = close_ret.pct_change(fill_method=None).fillna(0.0)
    WIN = 252
    start_ts = pd.Timestamp(start)
    end_ts = pd.Timestamp(end)

    # Track open positions: {sym: (entry_price, peak_price)}
    positions = {}
    port_rets = pd.Series(0.0, index=trading_idx)
    to_series = pd.Series(0.0, index=trading_idx)

    for i, dt in enumerate(trading_idx):
        if dt < start_ts or dt > end_ts:
            continue
        if i < WIN:
            continue

        members = member_df.iloc[i] if i < len(member_df) else member_df.iloc[-1]
        dv_now = dv_df.iloc[i - 1]

        # Update peak prices and check trailing stops for open positions
        to_exit = []
        for sym, (entry_px, peak_px) in list(positions.items()):
            cur_px = close_cap.iloc[i].get(sym)
            if cur_px is None or not np.isfinite(cur_px):
                to_exit.append(sym)
                continue
            new_peak = max(peak_px, cur_px)
            positions[sym] = (entry_px, new_peak)
            if cur_px < new_peak * (1 - TRAIL_PCT):
                to_exit.append(sym)

        for sym in to_exit:
            positions.pop(sym, None)

        # Detect new 52-week highs among PIT members with liquidity
        window_high = close_cap.iloc[max(i - WIN, 0):i]
        for sym in close_cap.columns:
            if sym in positions:
                continue
            if not members.get(sym, False):
                continue
            if dv_now.get(sym, 0) < min_dv:
                continue
            cur = close_cap.iloc[i].get(sym)
            if not cur or not np.isfinite(cur) or cur < min_px:
                continue
            col_win = window_high[sym].dropna()
            if len(col_win) < 50:
                continue
            if cur > col_win.max() and len(positions) < MAX_POSITIONS:
                positions[sym] = (cur, cur)

        # Daily portfolio return: equal weight over open positions
        if positions:
            day_ret = sum(ret_df.iloc[i].get(s, 0.0) for s in positions) / len(positions)
            port_rets.iloc[i] = day_ret
            # Approximate turnover as new openings + closures
            n_changes = len(to_exit)
            to_series.iloc[i] = n_changes / max(len(positions), 1) / 2.0

    net_ret = apply_costs(port_rets, to_series, cost_bps, slip_bps)
    spy = load_price_series("SPY", start, end, ADJ_TOTALRETURN, cache)
    bm = spy["Close"].pct_change(fill_method=None).reindex(net_ret.index)
    ann_to = float(to_series.sum() / max(len(to_series) / TRADING_DAYS, 1))
    return {"returns": net_ret, "benchmark": bm, "description": DESCRIPTION, "turnover_annual": ann_to}
