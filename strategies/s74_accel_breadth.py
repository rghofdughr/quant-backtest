import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import logging
import numpy as np
import pandas as pd
from data import load_price_series, watchlist_symbols, index_constituent_mask, ADJ_TOTALRETURN, compute_dollar_volume
from engine import portfolio_returns_from_weights, apply_costs

log = logging.getLogger(__name__)
TRADING_DAYS = 252
DESCRIPTION = "Acceleration momentum (delta-mom) with breadth filter: only trade when >50% of S&P 500 above 200d MA."

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
    SKIP = 21
    MA200 = 200
    # Acceleration: change in 6m momentum vs prior 6m momentum
    L_RECENT = 126   # recent 6m
    L_PRIOR = 126    # prior 6m (non-overlapping)
    WARMUP = SKIP + L_RECENT + L_PRIOR + MA200

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

        # Breadth filter: fraction of universe above 200d MA
        close_now_row = close_df.iloc[pos - 1]
        ma200_row = close_df.iloc[max(pos - MA200, 0):pos].mean()
        above_200 = sum(
            1 for c in valid_cols
            if close_now_row.get(c, 0) > ma200_row.get(c, 0) and np.isfinite(close_now_row.get(c, np.nan))
        )
        breadth = above_200 / max(len(valid_cols), 1)
        if breadth < 0.50:
            # Market breadth too weak — go to cash
            continue

        # Price snapshots for acceleration
        p0 = close_df.iloc[pos - SKIP]              # T-skip
        p1 = close_df.iloc[max(pos - SKIP - L_RECENT, 0)]   # T-skip-6m
        p2 = close_df.iloc[max(pos - SKIP - L_RECENT - L_PRIOR, 0)]  # T-skip-12m

        accel = {}
        for c in valid_cols:
            v0 = p0.get(c)
            v1 = p1.get(c)
            v2 = p2.get(c)
            if not all(v and v > 0 and np.isfinite(v) for v in [v0, v1, v2]):
                continue
            mom_recent = v0 / v1 - 1.0
            mom_prior = v1 / v2 - 1.0
            accel[c] = mom_recent - mom_prior  # positive = accelerating

        if len(accel) < 20:
            continue

        sr = pd.Series(accel)
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
