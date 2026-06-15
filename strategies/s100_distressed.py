"""
S100 — Distressed-then-recovered (upper-bound study).
Long names that hit a 5-year low within the S&P 500 C&P universe
but did NOT subsequently delist within 2 years (conditioning on survival).
CAUTION: This conditions on future survival. Returns represent an upper bound
on what was achievable — not a deployable strategy. Documented bias.
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
DESCRIPTION = ("Distressed-then-recovered (UPPER BOUND — conditions on survival). "
               "S&P 500 C&P 5yr-low stocks that survive 2yr. Documented look-ahead.")

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
    start_load = "2013-01-01" if is_smoke else "1995-01-01"

    symbols = watchlist_symbols(wl_name)
    if is_smoke:
        symbols = symbols[:200]

    close_map, dv_map, mask_map = {}, {}, {}
    for sym in symbols:
        df = load_price_series(sym, start=start_load, end=end, adjustment=ADJ_TOTALRETURN, cache_dir=cache)
        if df.empty:
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

    FORM_WIN = 1260  # 5 years for "5yr low"
    SURVIVE_WIN = 504  # 2 years for survival check
    HOLD_DAYS = 252   # 1 year hold
    WARMUP = FORM_WIN + SURVIVE_WIN + 10

    reb_dates = pd.date_range(start, end, freq="BME")
    weight_schedule = {}

    for rd in reb_dates:
        rd = min(rd, trading_idx[-1])
        pos = trading_idx.searchsorted(rd)
        if pos < WARMUP:
            continue

        # Survival check uses FUTURE data (this is the look-ahead)
        future_pos = min(pos + SURVIVE_WIN, len(trading_idx) - 1)

        members = member_df.iloc[pos] if pos < len(member_df) else member_df.iloc[-1]
        dv_now = dv_df.iloc[pos - 1]

        valid_cols = [
            c for c in close_df.columns
            if members.get(c, False) and dv_now.get(c, 0) >= min_dv
        ]

        survivors = []
        for c in valid_cols:
            cur = float(close_df.iloc[pos].get(c, np.nan))
            if not np.isfinite(cur) or cur < min_px:
                continue
            # Was this a 5yr low?
            window_5y = close_df[c].iloc[max(pos - FORM_WIN, 0):pos]
            hist_min = float(window_5y.min()) if len(window_5y) > 0 else np.inf
            if not (np.isfinite(hist_min) and cur <= hist_min * 1.10):  # within 10% of 5yr low
                continue
            # Survival check: still has prices 2yr later (LOOK-AHEAD)
            future_data = close_df[c].iloc[pos:future_pos].dropna()
            if len(future_data) < SURVIVE_WIN * 0.7:  # lost >30% of expected data = likely delisted
                continue
            survivors.append(c)

        if not survivors:
            continue
        weight_schedule[rd] = {s: 1.0 / len(survivors) for s in survivors}

    gross_ret, to = portfolio_returns_from_weights(weight_schedule, close_df, start, end)
    net_ret = apply_costs(gross_ret, to, cost_bps, slip_bps)
    spy = load_price_series("SPY", start, end, ADJ_TOTALRETURN, cache)
    bm = spy["Close"].pct_change(fill_method=None).reindex(net_ret.index)
    ann_to = float(to.sum() / max(len(to) / TRADING_DAYS, 1))
    return {"returns": net_ret, "benchmark": bm, "description": DESCRIPTION, "turnover_annual": ann_to}
