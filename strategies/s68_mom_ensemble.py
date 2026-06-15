import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import logging
import numpy as np
import pandas as pd
from data import load_price_series, watchlist_symbols, index_constituent_mask, ADJ_TOTALRETURN, compute_dollar_volume
from engine import portfolio_returns_from_weights, apply_costs

log = logging.getLogger(__name__)
TRADING_DAYS = 252
DESCRIPTION = "6/1 + 12/1 momentum ensemble: average rank of 6-1m and 12-1m return; long top decile. Skip-month on both."

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

    reb_dates = pd.date_range(start, end, freq="BME")
    WARMUP = 275
    SKIP = 21
    L6 = 126
    L12 = 252

    weight_schedule = {}
    for rd in reb_dates:
        rd = min(rd, trading_idx[-1])
        pos = trading_idx.searchsorted(rd)
        if pos < WARMUP:
            continue

        members = member_df.iloc[pos] if pos < len(member_df) else member_df.iloc[-1]
        dv_now = dv_df.iloc[pos - 1]

        valid_cols = [
            c for c in close_df.columns
            if members.get(c, False) and dv_now.get(c, 0) >= min_dv
        ]
        if len(valid_cols) < 20:
            continue

        # Price snapshots: now-SKIP, now-SKIP-L6, now-SKIP-L12
        p_skip = close_df.iloc[max(pos - SKIP, 0)]
        p_6m = close_df.iloc[max(pos - SKIP - L6, 0)]
        p_12m = close_df.iloc[max(pos - SKIP - L12, 0)]

        signals = {}
        for c in valid_cols:
            ps = p_skip.get(c)
            p6 = p_6m.get(c)
            p12 = p_12m.get(c)
            if not (ps and p6 and p12 and ps > 0 and p6 > 0 and p12 > 0):
                continue
            if not (np.isfinite(ps) and np.isfinite(p6) and np.isfinite(p12)):
                continue
            mom6 = ps / p6 - 1.0
            mom12 = ps / p12 - 1.0
            signals[c] = (mom6, mom12)

        if len(signals) < 20:
            continue

        sr6 = pd.Series({c: v[0] for c, v in signals.items()})
        sr12 = pd.Series({c: v[1] for c, v in signals.items()})
        rank6 = sr6.rank(pct=True)
        rank12 = sr12.rank(pct=True)
        avg_rank = (rank6 + rank12) / 2.0
        top = avg_rank[avg_rank >= avg_rank.quantile(0.90)].index.tolist()
        if not top:
            continue
        weight_schedule[rd] = {s: 1.0 / len(top) for s in top}

    gross_ret, to = portfolio_returns_from_weights(weight_schedule, close_df, start, end)
    net_ret = apply_costs(gross_ret, to, cost_bps, slip_bps)
    spy = load_price_series("SPY", start, end, ADJ_TOTALRETURN, cache)
    bm = spy["Close"].pct_change(fill_method=None).reindex(net_ret.index)
    ann_to = float(to.sum() / max(len(to) / TRADING_DAYS, 1))
    return {"returns": net_ret, "benchmark": bm, "description": DESCRIPTION, "turnover_annual": ann_to}
