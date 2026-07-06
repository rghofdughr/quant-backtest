import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import logging
import numpy as np
import pandas as pd
from data import load_price_series, watchlist_symbols, index_constituent_mask, ADJ_CAPITAL, ADJ_TOTALRETURN, compute_dollar_volume
from engine import apply_costs

log = logging.getLogger(__name__)
TRADING_DAYS = 252
DESCRIPTION = "Donchian 50/25 breakout on R1000 PIT equities, long-only, vol-sized. ADJ_CAPITAL for price levels."
ENTRY_DAYS = 50
EXIT_DAYS = 25
VOL_TARGET = 0.10
MAX_POSITIONS = 40

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
    start_ts = pd.Timestamp(start)

    # Position state: {sym: weight}
    positions = {}
    port_rets = pd.Series(0.0, index=trading_idx)
    to_series = pd.Series(0.0, index=trading_idx)

    WARMUP = ENTRY_DAYS + 20  # need enough history for vol estimate

    for i, dt in enumerate(trading_idx):
        if dt < start_ts:
            continue
        if i < WARMUP:
            continue

        # Use yesterday's data for all signals (t-1 info → t execution)
        members = member_df.iloc[i - 1] if i > 0 else member_df.iloc[0]
        dv_now  = dv_df.iloc[i - 1]

        # ---- EXIT CHECK: signal from yesterday's close (no same-bar lookahead) ----
        to_exit = []
        for sym in list(positions.keys()):
            cur_now = close_cap.iloc[i].get(sym)
            if cur_now is None or not np.isfinite(cur_now):
                # Delisted/stopped trading today: exit with whatever return ret_df gives (0 from ffill)
                to_exit.append(sym)
                continue
            cur_prev = close_cap.iloc[i - 1].get(sym) if i > 0 else None
            if cur_prev is None or not np.isfinite(cur_prev):
                continue
            # 25-day low computed through yesterday (excludes today)
            low25 = close_cap[sym].iloc[max(i - EXIT_DAYS - 1, 0):i - 1].min()
            if cur_prev <= low25:
                to_exit.append(sym)

        # ---- P&L: ALL current positions earn today's return (including those exiting) ----
        existing_pos = dict(positions)
        if existing_pos:
            day_ret = sum(existing_pos[s] * ret_df.iloc[i].get(s, 0.0) for s in existing_pos)
            port_rets.iloc[i] = day_ret

        # ---- Remove exits AFTER P&L (exit at today's close, not yesterday's) ----
        for sym in to_exit:
            positions.pop(sym, None)

        # ---- ENTRY CHECK: signal from yesterday's close, symmetric with exits ----
        new_entries = []
        n_open = len(positions)
        if n_open < MAX_POSITIONS:
            for sym in close_cap.columns:
                if sym in positions or n_open >= MAX_POSITIONS:
                    continue
                if not members.get(sym, False):
                    continue
                if dv_now.get(sym, 0) < min_dv:
                    continue
                cur_prev = close_cap.iloc[i - 1].get(sym) if i > 0 else None
                if not cur_prev or not np.isfinite(cur_prev) or cur_prev < min_px:
                    continue
                # 50-day high through yesterday
                hi50 = close_cap[sym].iloc[max(i - ENTRY_DAYS - 1, 0):i - 1].max()
                if cur_prev > hi50:
                    rets_20 = ret_df[sym].iloc[max(i - 21, 0):i - 1]
                    vol_20 = float(rets_20.std(ddof=1)) * np.sqrt(TRADING_DAYS) if len(rets_20) > 5 else 0.20
                    w = min(VOL_TARGET / max(vol_20, 0.05), 1.0 / MAX_POSITIONS)
                    positions[sym] = w
                    new_entries.append(sym)
                    n_open += 1

        n_ch = len(to_exit)
        to_series.iloc[i] = n_ch / max(len(existing_pos) + n_ch, 1) / 2.0

    net_ret = apply_costs(port_rets, to_series, cost_bps, slip_bps)
    spy = load_price_series("SPY", start, end, ADJ_TOTALRETURN, cache)
    bm = spy["Close"].pct_change(fill_method=None).reindex(net_ret.index)
    ann_to = float(to_series.sum() / max(len(to_series) / TRADING_DAYS, 1))
    return {"returns": net_ret, "benchmark": bm, "description": DESCRIPTION, "turnover_annual": ann_to}
