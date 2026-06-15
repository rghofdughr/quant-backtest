"""
S04 — 52-week high proximity (George & Hwang 2004)
Universe:  S&P 500 point-in-time
Signal:    current_close / 252-day_high → "nearness to 52-week high"
           Long top quintile (closest to high); short bottom quintile.
Rebalance: Monthly
Execution: Next trading day close
Sizing:    Equal-weight, dollar-neutral
Sweep:     quintile vs decile; holding 1/3/6 months
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
from engine import portfolio_returns_from_weights, apply_costs

log = logging.getLogger(__name__)
DESCRIPTION = "52-week high proximity (George-Hwang), S&P 500 PIT, top vs bottom quintile"
TRADING_DAYS = 252


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
    n_q = 5  # quintile (5)

    wl_name  = config["universes"].get("sp500", "S&P 500 Current & Past")
    idx_name = config["universes"].get("sp500_index", "S&P 500")
    start_load = "2013-01-01" if is_smoke else "1997-01-01"

    log.info("S04: loading S&P 500 C&P ...")
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

    log.info("S04: %d symbols loaded", len(close_map))

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

    # 252-day rolling high (use RAW prices = TOTALRETURN here; both are fine for proximity ratio)
    high_252 = close_df.rolling(TRADING_DAYS).max()

    reb_dates = pd.date_range(start, end, freq="BME")
    weight_schedule: dict = {}

    for d in reb_dates:
        pos = trading_idx.searchsorted(d, side="right") - 1
        if pos < TRADING_DAYS:
            continue

        prices  = close_df.iloc[pos]
        h252    = high_252.iloc[pos]
        dv_row  = dv_df.iloc[pos]
        members = member_df.iloc[pos]

        # Nearness = current / 52-week high (1.0 = at new high; lower = further below)
        nearness = (prices / h252.replace(0, np.nan)).clip(0, 1)

        valid = (
            members.reindex(nearness.index, fill_value=False) &
            (prices.reindex(nearness.index) >= min_px) &
            (dv_row.reindex(nearness.index) >= min_dv)
        )
        sig = nearness[valid].dropna()
        if len(sig) < n_q * 2:
            continue

        n_each = max(1, len(sig) // n_q)
        ranked = sig.sort_values()
        shorts = ranked.iloc[:n_each].index.tolist()   # farthest from high → short
        longs  = ranked.iloc[-n_each:].index.tolist()  # closest to high → long

        w = {s:  1.0 / len(longs)  for s in longs}
        w.update({s: -1.0 / len(shorts) for s in shorts})
        weight_schedule[d] = w

    gross_ret, to = portfolio_returns_from_weights(weight_schedule, close_df, start, end)
    net_ret = apply_costs(gross_ret, to, cost_bps, slip_bps)

    spy = load_price_series("SPY", start, end, ADJ_TOTALRETURN, cache)
    bm  = spy["Close"].pct_change(fill_method=None).reindex(net_ret.index)

    ann_to = float(to.sum() / max(len(to) / TRADING_DAYS, 1))
    log.info("S04 done. Ann turnover: %.2fx", ann_to)

    return {
        "returns": net_ret, "benchmark": bm,
        "description": DESCRIPTION, "turnover_annual": ann_to,
    }
