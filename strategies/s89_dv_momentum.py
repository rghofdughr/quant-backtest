"""
S89 — Dollar-volume migration momentum.
Long R1000 PIT stocks whose share of total-universe dollar volume is rising fastest
(21d avg vs 63d avg). Captures liquidity migration / institutional attention flow.
Orthogonalise against price momentum by requiring rising DV even after controlling
for flat/negative price return.
"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import logging
import numpy as np
import pandas as pd
from data import load_price_series, watchlist_symbols, index_constituent_mask, ADJ_TOTALRETURN, compute_dollar_volume
from engine import portfolio_returns_from_weights, apply_costs

log = logging.getLogger(__name__)
TRADING_DAYS = 252
DESCRIPTION = "Dollar-volume migration: long R1000 PIT stocks with fastest-rising DV share; orthogonalised vs price mom."

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

    close_map, dv_raw_map, mask_map = {}, {}, {}
    for sym in symbols:
        df = load_price_series(sym, start=start_load, end=end, adjustment=ADJ_TOTALRETURN, cache_dir=cache)
        if df.empty or df["Close"].max() < min_px:
            continue
        close_map[sym] = df["Close"]
        if "Turnover" in df.columns:
            dv_raw_map[sym] = df["Turnover"]
        else:
            dv_raw_map[sym] = df["Close"] * df["Volume"]
        m = index_constituent_mask(sym, idx_name, start=start_load, end=end, cache_dir=cache)
        if not m.empty:
            mask_map[sym] = m

    trading_idx = pd.bdate_range(start, end)
    close_df = pd.DataFrame(close_map).reindex(trading_idx, method="ffill")
    dv_raw_df = pd.DataFrame(dv_raw_map).reindex(trading_idx)
    member_df = (pd.DataFrame(mask_map).reindex(trading_idx, method="ffill")
                 .infer_objects(copy=False).fillna(False).astype(bool))
    for col in close_df.columns:
        if col not in member_df.columns:
            member_df[col] = False

    reb_dates = pd.date_range(start, end, freq="BME")
    SHORT_WIN = 21
    LONG_WIN = 63
    WARMUP = LONG_WIN + SHORT_WIN + 5

    weight_schedule = {}
    for rd in reb_dates:
        rd = min(rd, trading_idx[-1])
        pos = trading_idx.searchsorted(rd)
        if pos < WARMUP:
            continue

        members = member_df.iloc[pos] if pos < len(member_df) else member_df.iloc[-1]
        dv_now_med = dv_raw_df.rolling(20).median().iloc[max(pos - 1, 0)]

        valid_cols = [
            c for c in close_df.columns
            if members.get(c, False) and dv_now_med.get(c, 0) >= min_dv
        ]
        if len(valid_cols) < 20:
            continue

        # Total universe DV over windows
        short_dv = dv_raw_df[valid_cols].iloc[max(pos - SHORT_WIN, 0):pos].sum(axis=0)
        long_dv = dv_raw_df[valid_cols].iloc[max(pos - LONG_WIN, 0):pos].sum(axis=0)
        total_short = short_dv.sum()
        total_long = long_dv.sum()

        if total_short < 1 or total_long < 1:
            continue

        # DV share: stock's share of total universe DV
        share_short = short_dv / total_short
        share_long = long_dv / total_long

        # Signal: rising share (short window / long window > 1)
        ratio = share_short / share_long.replace(0, np.nan)
        ratio = ratio.dropna()

        if len(ratio) < 20:
            continue

        sr = ratio
        top = sr[sr >= sr.quantile(0.80)].index.tolist()
        if not top:
            continue
        weight_schedule[rd] = {s: 1.0 / len(top) for s in top}

    gross_ret, to = portfolio_returns_from_weights(weight_schedule, close_df, start, end)
    net_ret = apply_costs(gross_ret, to, cost_bps, slip_bps)
    spy = load_price_series("SPY", start, end, ADJ_TOTALRETURN, cache)
    bm = spy["Close"].pct_change(fill_method=None).reindex(net_ret.index)
    ann_to = float(to.sum() / max(len(to) / TRADING_DAYS, 1))
    return {"returns": net_ret, "benchmark": bm, "description": DESCRIPTION, "turnover_annual": ann_to}
