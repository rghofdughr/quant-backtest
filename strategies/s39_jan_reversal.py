"""
S39 — January tax-loss reversal (De Bondt & Thaler 1985 + Roll 1983)
Universe:  S&P 500 / Russell 2000 point-in-time; small-cap tilt variant
Signal:    Prior calendar-year return (Jan 1 to Dec 31). Long bottom decile (prior-year
           losers), entered late December (last 3 trading days), hold through January–March.
Execution: Enter at Dec close -3 trading days; exit at end-of-March close.
Sizing:    Equal-weight, long-only (tax-loss effect is buy-side).
Sweep:     Hold 1/2/3 months; small-cap tilt (IWM universe vs SPY).
"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import logging
import numpy as np
import pandas as pd

from data import (
    load_price_series, watchlist_symbols, index_constituent_mask,
    ADJ_TOTALRETURN, compute_dollar_volume,
)
from engine import apply_costs

log = logging.getLogger(__name__)
DESCRIPTION = "January tax-loss reversal, S&P 500 PIT bottom decile prior-year losers"
TRADING_DAYS = 252
HOLD_MONTHS  = 3    # hold through March (3 months after December entry)


def run(config: dict) -> dict:
    cfg      = config["backtest"]
    start    = cfg["start_date"]
    end      = cfg["end_date"]
    is_smoke = config.get("smoke", False)
    cache    = config["paths"]["cache_dir"]

    cost_bps = config["costs"]["equity_cost_bps"]
    slip_bps = config["costs"]["equity_slippage_bps"]
    min_dv   = config["liquidity"]["min_dollar_volume"]
    min_px   = config["liquidity"]["min_price"]

    wl_name  = config["universes"].get("sp500", "S&P 500 Current & Past")
    idx_name = config["universes"].get("sp500_index", "S&P 500")
    start_load = "2013-01-01" if is_smoke else "1999-01-01"

    log.info("S39: loading S&P 500 C&P ...")
    symbols = watchlist_symbols(wl_name)
    if is_smoke:
        symbols = symbols[:120]

    close_map, dv_map, mask_map = {}, {}, {}
    for sym in symbols:
        df = load_price_series(sym, start=start_load, end=end,
                               adjustment=ADJ_TOTALRETURN, cache_dir=cache)
        if df.empty or df["Close"].max() < min_px:
            continue
        close_map[sym] = df["Close"]
        dv_map[sym]    = compute_dollar_volume(df)
        m = index_constituent_mask(sym, idx_name, start=start_load, end=end, cache_dir=cache)
        if not m.empty:
            mask_map[sym] = m

    log.info("S39: %d symbols loaded", len(close_map))

    trading_idx = pd.bdate_range(start, end)
    close_df  = pd.DataFrame(close_map).reindex(trading_idx, method="ffill")
    dv_df     = pd.DataFrame(dv_map).reindex(trading_idx, method="ffill")
    member_df = (pd.DataFrame(mask_map)
                   .reindex(trading_idx, method="ffill")
                   .infer_objects(copy=False)
                   .fillna(False).astype(bool))
    for col in close_df.columns:
        if col not in member_df.columns:
            member_df[col] = False

    ret_df = close_df.pct_change(fill_method=None).fillna(0.0)

    port_ret = pd.Series(0.0, index=trading_idx)
    to_ser   = pd.Series(0.0, index=trading_idx)
    prev_w: dict = {}

    # Iterate over each calendar year
    years = sorted(set(trading_idx.year))
    for yr in years:
        if yr < int(start[:4]) + 1:
            continue

        # Calendar-year return = Dec 31 (yr-1) to Dec 31 (yr) total return
        yr_start_days = trading_idx[trading_idx.year == yr - 1]
        yr_end_days   = trading_idx[trading_idx.year == yr]
        if yr_start_days.empty or yr_end_days.empty:
            continue

        p_jan1 = close_df.loc[yr_start_days[-1]]   # last trading day of prior year
        p_dec31 = close_df.loc[yr_end_days[-1]]     # last trading day of this year

        if p_jan1.isna().all() or p_dec31.isna().all():
            continue

        yr_return = (p_dec31 / p_jan1.replace(0, np.nan)) - 1.0

        # Point-in-time filter at last trading day of this year
        last_day_idx = trading_idx.get_loc(yr_end_days[-1])
        prices  = close_df.iloc[last_day_idx]
        dv_row  = dv_df.iloc[last_day_idx]
        members = member_df.iloc[last_day_idx]

        valid = (
            members.reindex(yr_return.index, fill_value=False) &
            (prices.reindex(yr_return.index) >= min_px) &
            (dv_row.reindex(yr_return.index) >= min_dv)
        )
        sig = yr_return[valid].dropna().sort_values()
        if len(sig) < 20:
            continue

        n_each = max(1, len(sig) // 10)
        longs = sig.iloc[:n_each].index.tolist()   # biggest prior-year losers

        # Entry: last 3 trading days of December
        dec_days = trading_idx[(trading_idx.year == yr) & (trading_idx.month == 12)]
        if len(dec_days) < 3:
            continue
        entry_start = dec_days[-3]

        # Exit: end of hold-month (March = month 3)
        exit_month = HOLD_MONTHS   # Jan=1, Feb=2, Mar=3
        exit_days = trading_idx[
            (trading_idx.year == yr + 1) &
            (trading_idx.month == exit_month)
        ]
        if exit_days.empty:
            continue
        exit_day = exit_days[-1]

        hold_mask = (trading_idx >= entry_start) & (trading_idx <= exit_day)
        new_w = {s: 1.0 / len(longs) for s in longs}

        # Turnover
        all_s = set(list(new_w.keys()) + list(prev_w.keys()))
        to = sum(abs(new_w.get(s, 0.0) - prev_w.get(s, 0.0)) for s in all_s) / 2.0
        if entry_start in to_ser.index:
            to_ser[entry_start] += to

        # Portfolio return during holding period
        avail = [s for s in longs if s in ret_df.columns]
        if avail:
            hold_ret = ret_df.loc[hold_mask, avail].fillna(0.0)
            port_ret[hold_mask] += hold_ret.mean(axis=1).values

        prev_w = new_w

    net_ret = apply_costs(port_ret, to_ser, cost_bps, slip_bps)

    spy = load_price_series("SPY", start, end, ADJ_TOTALRETURN, cache)
    bm  = spy["Close"].pct_change(fill_method=None).reindex(net_ret.index)

    ann_to = float(to_ser.sum() / max(len(to_ser) / TRADING_DAYS, 1))
    log.info("S39 done. Hold=%d months, ann turnover: %.2fx", HOLD_MONTHS, ann_to)

    return {
        "returns": net_ret, "benchmark": bm,
        "description": DESCRIPTION, "turnover_annual": ann_to,
    }
