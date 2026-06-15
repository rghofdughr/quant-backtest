import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import logging
import numpy as np
import pandas as pd
from data import load_price_series, watchlist_symbols, index_constituent_mask, ADJ_TOTALRETURN, compute_dollar_volume
from engine import portfolio_returns_from_weights, apply_costs

log = logging.getLogger(__name__)
TRADING_DAYS = 252
DESCRIPTION = "Amihud illiquidity premium within R1000 PIT: long top-quintile |ret|/dv; illiquidity capped at tradable floor."

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
        # Raw daily dollar volume (not rolling) for Amihud numerator
        if "Turnover" in df.columns:
            dv_raw_map[sym] = df["Turnover"]
        else:
            dv_raw_map[sym] = df["Close"] * df["Volume"]
        m = index_constituent_mask(sym, idx_name, start=start_load, end=end, cache_dir=cache)
        if not m.empty:
            mask_map[sym] = m

    trading_idx = pd.bdate_range(start, end)
    close_df = pd.DataFrame(close_map).reindex(trading_idx, method="ffill")
    dv_raw_df = pd.DataFrame(dv_raw_map).reindex(trading_idx)  # do NOT ffill raw dv
    member_df = (pd.DataFrame(mask_map).reindex(trading_idx, method="ffill")
                 .infer_objects(copy=False).fillna(False).astype(bool))
    for col in close_df.columns:
        if col not in member_df.columns:
            member_df[col] = False

    ret_df = close_df.pct_change(fill_method=None)
    reb_dates = pd.date_range(start, end, freq="BME")
    WARMUP = 42  # need 2 months
    WIN = 21     # 1-month Amihud window

    weight_schedule = {}
    for rd in reb_dates:
        rd = min(rd, trading_idx[-1])
        pos = trading_idx.searchsorted(rd)
        if pos < WARMUP:
            continue

        members = member_df.iloc[pos] if pos < len(member_df) else member_df.iloc[-1]
        dv_med = dv_raw_df.rolling(20, min_periods=10).median().iloc[max(pos - 1, 0)]

        valid_cols = [
            c for c in close_df.columns
            if members.get(c, False) and dv_med.get(c, 0) >= min_dv
        ]
        if len(valid_cols) < 20:
            continue

        win_rets = ret_df.iloc[max(pos - WIN, 0):pos][valid_cols]
        win_dv = dv_raw_df.iloc[max(pos - WIN, 0):pos][valid_cols]

        amihud = {}
        for c in valid_cols:
            r = win_rets[c].dropna()
            d = win_dv[c].reindex(r.index).replace(0, np.nan).dropna()
            common = r.index.intersection(d.index)
            if len(common) < 10:
                continue
            ratio = r.loc[common].abs() / d.loc[common]
            val = float(ratio.mean())
            if np.isfinite(val) and val > 0:
                amihud[c] = val

        if len(amihud) < 20:
            continue

        sr = pd.Series(amihud)
        # Cap at 80th percentile to avoid untradable penny stocks hiding in R1000
        cap = sr.quantile(0.80)
        sr_capped = sr.clip(upper=cap)
        top = sr_capped[sr_capped >= sr_capped.quantile(0.80)].index.tolist()
        if not top:
            continue
        weight_schedule[rd] = {s: 1.0 / len(top) for s in top}

    gross_ret, to = portfolio_returns_from_weights(weight_schedule, close_df, start, end)
    net_ret = apply_costs(gross_ret, to, cost_bps, slip_bps)
    spy = load_price_series("SPY", start, end, ADJ_TOTALRETURN, cache)
    bm = spy["Close"].pct_change(fill_method=None).reindex(net_ret.index)
    ann_to = float(to.sum() / max(len(to) / TRADING_DAYS, 1))
    return {"returns": net_ret, "benchmark": bm, "description": DESCRIPTION, "turnover_annual": ann_to}
