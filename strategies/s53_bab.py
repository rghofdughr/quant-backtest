import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import logging
import numpy as np
import pandas as pd
from data import load_price_series, watchlist_symbols, index_constituent_mask, ADJ_TOTALRETURN, compute_dollar_volume
from engine import portfolio_returns_from_weights, apply_costs

log = logging.getLogger(__name__)
TRADING_DAYS = 252
DESCRIPTION = "Betting Against Beta (Frazzini-Pedersen 2014). Long low-beta, short high-beta, beta-scaled legs."

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
        symbols = symbols[:100]

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

    spy_df = load_price_series("SPY", start=start_load, end=end, adjustment=ADJ_TOTALRETURN, cache_dir=cache)

    trading_idx = pd.bdate_range(start, end)
    close_df = pd.DataFrame(close_map).reindex(trading_idx, method="ffill")
    dv_df = pd.DataFrame(dv_map).reindex(trading_idx, method="ffill")
    member_df = (pd.DataFrame(mask_map).reindex(trading_idx, method="ffill")
                 .infer_objects(copy=False).fillna(False).astype(bool))
    for col in close_df.columns:
        if col not in member_df.columns:
            member_df[col] = False

    ret_df = close_df.pct_change(fill_method=None)
    spy_ret = spy_df["Close"].reindex(trading_idx, method="ffill").pct_change(fill_method=None)

    reb_dates = pd.date_range(start, end, freq="BME")
    WARMUP = 252
    WIN = 252
    MAX_W = 3.0

    weight_schedule = {}

    for rd in reb_dates:
        rd = min(rd, trading_idx[-1])
        pos = trading_idx.searchsorted(rd)
        if pos < WARMUP:
            continue

        win_start = max(0, pos - WIN)
        stock_wins = ret_df.iloc[win_start:pos]
        spy_wins = spy_ret.iloc[win_start:pos].values

        dv_now = dv_df.iloc[pos - 1] if pos > 0 else dv_df.iloc[0]
        members = member_df.iloc[pos] if pos < len(member_df) else member_df.iloc[-1]

        valid_cols = [
            c for c in close_df.columns
            if members.get(c, False) and dv_now.get(c, 0) >= min_dv
        ]
        if len(valid_cols) < 40:
            continue

        spy_valid = np.isfinite(spy_wins)
        if spy_valid.sum() < 100:
            continue

        spy_v = spy_wins[spy_valid]
        spy_dm = spy_v - spy_v.mean()
        denom_spy = np.dot(spy_dm, spy_dm)
        if denom_spy < 1e-12:
            continue

        betas = {}
        for sym in valid_cols:
            col_vals = stock_wins[sym].values
            col_valid = spy_valid & np.isfinite(col_vals)
            if col_valid.sum() < 100:
                continue
            cv = col_vals[col_valid]
            sv = spy_wins[col_valid]
            spy_dm_loc = sv - sv.mean()
            stock_dm = cv - cv.mean()
            denom_loc = np.dot(spy_dm_loc, spy_dm_loc)
            if denom_loc < 1e-12:
                continue
            beta_i = np.dot(spy_dm_loc, stock_dm) / denom_loc
            if np.isfinite(beta_i) and beta_i > 0:
                betas[sym] = float(beta_i)

        if len(betas) < 40:
            continue

        beta_series = pd.Series(betas)
        q20 = beta_series.quantile(0.20)
        q80 = beta_series.quantile(0.80)

        long_names = beta_series[beta_series <= q20].index.tolist()
        short_names = beta_series[beta_series >= q80].index.tolist()

        if not long_names or not short_names:
            continue

        n_long = len(long_names)
        n_short = len(short_names)

        avg_beta_long = float(beta_series[long_names].mean())
        avg_beta_short = float(beta_series[short_names].mean())

        if avg_beta_long < 1e-6 or avg_beta_short < 1e-6:
            continue

        # Scale weights so portfolio beta = 1 for each leg
        w_long_base = 1.0 / (n_long * avg_beta_long)
        w_short_base = 1.0 / (n_short * avg_beta_short)

        w = {}
        for s in long_names:
            w[s] = float(np.clip(w_long_base, -MAX_W, MAX_W))
        for s in short_names:
            w[s] = float(np.clip(-w_short_base, -MAX_W, MAX_W))

        # Dollar-neutral: rescale so sum of longs = sum of abs(shorts)
        sum_long = sum(v for v in w.values() if v > 0)
        sum_short = sum(abs(v) for v in w.values() if v < 0)
        if sum_long < 1e-9 or sum_short < 1e-9:
            continue
        scale = 1.0
        for s in list(w.keys()):
            if w[s] > 0:
                w[s] = w[s] / sum_long
            else:
                w[s] = w[s] / sum_short

        weight_schedule[rd] = w

    gross_ret, to = portfolio_returns_from_weights(weight_schedule, close_df, start, end)
    net_ret = apply_costs(gross_ret, to, cost_bps, slip_bps)
    spy = load_price_series("SPY", start, end, ADJ_TOTALRETURN, cache)
    bm = spy["Close"].pct_change(fill_method=None).reindex(net_ret.index)
    ann_to = float(to.sum() / max(len(to) / TRADING_DAYS, 1))
    return {"returns": net_ret, "benchmark": bm, "description": DESCRIPTION, "turnover_annual": ann_to}
