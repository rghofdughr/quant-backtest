import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import logging
import numpy as np
import pandas as pd
from data import load_price_series, watchlist_symbols, index_constituent_mask, ADJ_TOTALRETURN, compute_dollar_volume
from engine import portfolio_returns_from_weights, apply_costs

log = logging.getLogger(__name__)
TRADING_DAYS = 252
DESCRIPTION = "52-week-low proximity signal. Long bottom quintile (price nearest to 52w low). Long-only, equal-weight."

# WARNING: survivorship bias is most severe for names near 52w lows. Norgate includes delisting returns
# which partially mitigates this, but interpret positive results cautiously.

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

    trading_idx = pd.bdate_range(start, end)
    close_df = pd.DataFrame(close_map).reindex(trading_idx, method="ffill")
    dv_df = pd.DataFrame(dv_map).reindex(trading_idx, method="ffill")
    member_df = (pd.DataFrame(mask_map).reindex(trading_idx, method="ffill")
                 .infer_objects(copy=False).fillna(False).astype(bool))
    for col in close_df.columns:
        if col not in member_df.columns:
            member_df[col] = False

    reb_dates = pd.date_range(start, end, freq="BME")
    WARMUP = 252
    LOW_WIN = 252  # 52-week low window in trading days

    weight_schedule = {}

    for rd in reb_dates:
        rd = min(rd, trading_idx[-1])
        pos = trading_idx.searchsorted(rd)
        if pos < WARMUP:
            continue

        win_start = max(0, pos - LOW_WIN)
        price_window = close_df.iloc[win_start:pos]  # last 252 trading days

        dv_now = dv_df.iloc[pos - 1] if pos > 0 else dv_df.iloc[0]
        members = member_df.iloc[pos] if pos < len(member_df) else member_df.iloc[-1]
        close_now = close_df.iloc[pos]

        valid_cols = [
            c for c in close_df.columns
            if members.get(c, False)
            and dv_now.get(c, 0) >= min_dv
            and np.isfinite(close_now.get(c, np.nan))
        ]
        if len(valid_cols) < 20:
            continue

        # Signal: price / rolling 252d low
        low_252 = price_window[valid_cols].min(axis=0)
        cur_px = close_now[valid_cols]

        ratio = cur_px / low_252.replace(0, np.nan)
        ratio = ratio.replace([np.inf, -np.inf], np.nan).dropna()

        if len(ratio) < 20:
            continue

        q20 = ratio.quantile(0.20)
        long_names = ratio[ratio <= q20].index.tolist()

        if not long_names:
            continue

        n_long = len(long_names)
        w = {s: 1.0 / n_long for s in long_names}
        weight_schedule[rd] = w

    gross_ret, to = portfolio_returns_from_weights(weight_schedule, close_df, start, end)
    net_ret = apply_costs(gross_ret, to, cost_bps, slip_bps)
    spy = load_price_series("SPY", start, end, ADJ_TOTALRETURN, cache)
    bm = spy["Close"].pct_change(fill_method=None).reindex(net_ret.index)
    ann_to = float(to.sum() / max(len(to) / TRADING_DAYS, 1))
    return {"returns": net_ret, "benchmark": bm, "description": DESCRIPTION, "turnover_annual": ann_to}
