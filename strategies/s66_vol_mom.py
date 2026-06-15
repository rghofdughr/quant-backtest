import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import logging
import numpy as np
import pandas as pd
from data import load_price_series, watchlist_symbols, index_constituent_mask, ADJ_TOTALRETURN, compute_dollar_volume
from engine import portfolio_returns_from_weights, apply_costs

log = logging.getLogger(__name__)
TRADING_DAYS = 252
DESCRIPTION = "Volume-confirmed 12-1m momentum: long top decile only when 50d avg dollar-volume is rising vs 12m ago."

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

    close_map, dv_map, mask_map = {}, {}, {}
    for sym in symbols:
        df = load_price_series(sym, start=start_load, end=end, adjustment=ADJ_TOTALRETURN, cache_dir=cache)
        if df.empty or df["Close"].max() < min_px:
            continue
        close_map[sym] = df["Close"]
        dv_map[sym] = compute_dollar_volume(df)
        m = index_constituent_mask(sym, idx_name, start=start_load, end=end, cache_dir=cache)
        if not m.empty:
            mask_map[sym] = m

    trading_idx = pd.bdate_range(start, end)
    close_df = pd.DataFrame(close_map).reindex(trading_idx, method="ffill")
    dv_df = pd.DataFrame(dv_map).reindex(trading_idx, method="ffill")
    member_df = (pd.DataFrame(mask_map).reindex(trading_idx, method="ffill")
                 .infer_objects(copy=False).fillna(False).astype(bool))
    for col in close_df.columns:
        if col not in member_df.columns:
            member_df[col] = False

    ret_df = close_df.pct_change(fill_method=None)
    reb_dates = pd.date_range(start, end, freq="BME")
    WARMUP = 275  # 12m + 1m skip
    DV_WINDOW = 50
    MOM_LONG = 252
    MOM_SKIP = 21

    weight_schedule = {}
    for rd in reb_dates:
        rd = min(rd, trading_idx[-1])
        pos = trading_idx.searchsorted(rd)
        if pos < WARMUP:
            continue

        dv_now = dv_df.iloc[pos - 1]
        dv_12m_ago = dv_df.iloc[max(pos - MOM_LONG, 0)]
        members = member_df.iloc[pos] if pos < len(member_df) else member_df.iloc[-1]

        valid_cols = [
            c for c in close_df.columns
            if members.get(c, False) and dv_now.get(c, 0) >= min_dv
        ]
        if len(valid_cols) < 20:
            continue

        # 12-1m momentum (skip last month)
        win_end = max(pos - MOM_SKIP, 0)
        win_start = max(pos - MOM_LONG, 0)
        if win_end <= win_start:
            continue

        close_now = close_df.iloc[win_end]
        close_past = close_df.iloc[win_start]
        mom = {}
        for c in valid_cols:
            pnow = close_now.get(c)
            ppast = close_past.get(c)
            if pnow and ppast and ppast > 0 and np.isfinite(pnow) and np.isfinite(ppast):
                # Volume confirmation: 50d avg dv rising vs 12m ago
                dv_n = dv_now.get(c, 0)
                dv_o = dv_12m_ago.get(c, 0)
                if dv_n > dv_o and dv_n >= min_dv:
                    mom[c] = pnow / ppast - 1.0

        if len(mom) < 10:
            continue

        sr = pd.Series(mom)
        top = sr[sr >= sr.quantile(0.90)].index.tolist()
        if not top:
            continue
        weight_schedule[rd] = {s: 1.0 / len(top) for s in top}

    gross_ret, to = portfolio_returns_from_weights(weight_schedule, close_df, start, end)
    net_ret = apply_costs(gross_ret, to, cost_bps, slip_bps)
    spy = load_price_series("SPY", start, end, ADJ_TOTALRETURN, cache)
    bm = spy["Close"].pct_change(fill_method=None).reindex(net_ret.index)
    ann_to = float(to.sum() / max(len(to) / TRADING_DAYS, 1))
    return {"returns": net_ret, "benchmark": bm, "description": DESCRIPTION, "turnover_annual": ann_to}
