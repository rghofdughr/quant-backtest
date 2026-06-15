import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import logging
import numpy as np
import pandas as pd
from data import load_price_series, watchlist_symbols, index_constituent_mask, ADJ_TOTALRETURN, compute_dollar_volume
from engine import portfolio_returns_from_weights, apply_costs

log = logging.getLogger(__name__)
TRADING_DAYS = 252
DESCRIPTION = "ST reversal market-demeaned: 5d return minus S&P 500 equal-weight return; long bottom decile. Weekly rebalance."

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

    # Weekly rebalance on Fridays
    reb_dates = pd.date_range(start, end, freq="W-FRI")
    WIN = 5  # 5-day return

    weight_schedule = {}
    for rd in reb_dates:
        rd_adj = trading_idx[trading_idx <= rd][-1] if any(trading_idx <= rd) else None
        if rd_adj is None:
            continue
        pos = trading_idx.searchsorted(rd_adj)
        if pos < WIN + 1:
            continue

        members = member_df.iloc[pos] if pos < len(member_df) else member_df.iloc[-1]
        dv_now = dv_df.iloc[pos - 1]

        valid_cols = [
            c for c in close_df.columns
            if members.get(c, False) and dv_now.get(c, 0) >= min_dv
        ]
        if len(valid_cols) < 20:
            continue

        # 5-day returns
        p_now = close_df.iloc[pos]
        p_5d = close_df.iloc[max(pos - WIN, 0)]
        ret5 = {}
        for c in valid_cols:
            pn = p_now.get(c)
            pp = p_5d.get(c)
            if pn and pp and pp > 0 and np.isfinite(pn) and np.isfinite(pp):
                ret5[c] = pn / pp - 1.0

        if len(ret5) < 20:
            continue

        # Market demean: subtract equal-weight average
        mkt_ret = np.mean(list(ret5.values()))
        demeaned = {c: v - mkt_ret for c, v in ret5.items()}

        sr = pd.Series(demeaned)
        # Long the bottom decile (largest negative relative return = reversal candidates)
        bottom = sr[sr <= sr.quantile(0.10)].index.tolist()
        if not bottom:
            continue
        weight_schedule[rd_adj] = {s: 1.0 / len(bottom) for s in bottom}

    gross_ret, to = portfolio_returns_from_weights(weight_schedule, close_df, start, end)
    net_ret = apply_costs(gross_ret, to, cost_bps, slip_bps)
    spy = load_price_series("SPY", start, end, ADJ_TOTALRETURN, cache)
    bm = spy["Close"].pct_change(fill_method=None).reindex(net_ret.index)
    ann_to = float(to.sum() / max(len(to) / TRADING_DAYS, 1))
    return {"returns": net_ret, "benchmark": bm, "description": DESCRIPTION, "turnover_annual": ann_to}
