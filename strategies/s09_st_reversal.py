"""
S09 — Short-term reversal (Jegadeesh 1990)
Universe:  S&P 500 point-in-time
Signal:    Trailing N-day return; long bottom decile (losers), short top decile (winners)
Rebalance: Weekly (every Friday)
Execution: Monday (1-day lag after signal)
Sizing:    Equal-weight, dollar-neutral (50% long / 50% short)
Note:      High turnover — shows realistic cost drag on a well-known anomaly.
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
DESCRIPTION = "Short-term reversal, S&P 500 PIT, weekly rebalance, top/bottom decile L/S"
TRADING_DAYS = 252


def run(config: dict) -> dict:
    cfg      = config["backtest"]
    start    = cfg["start_date"]
    end      = cfg["end_date"]
    is_smoke = config.get("smoke", False)
    cache    = config["paths"]["cache_dir"]

    s09_cfg   = config.get("strategies", {}).get("s09", {})
    form_days = s09_cfg.get("formation_days", [5])[0]

    cost_bps = config["costs"]["equity_cost_bps"]
    slip_bps = config["costs"]["equity_slippage_bps"]
    min_dv   = config["liquidity"]["min_dollar_volume"]
    min_px   = config["liquidity"]["min_price"]

    wl_name  = config["universes"].get("sp500", "S&P 500 Current & Past")
    idx_name = config["universes"].get("sp500_index", "S&P 500")
    start_load = "2013-01-01" if is_smoke else "1999-01-01"

    log.info("S09: loading S&P 500 C&P ...")
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

    log.info("S09: %d symbols loaded", len(close_map))

    # Everything aligned to business-day grid from here on
    trading_idx = pd.bdate_range(start, end)
    close_df  = pd.DataFrame(close_map).reindex(trading_idx, method="ffill")
    dv_df     = pd.DataFrame(dv_map).reindex(trading_idx, method="ffill")
    member_df = (pd.DataFrame(mask_map)
                   .reindex(trading_idx, method="ffill")
                   .infer_objects(copy=False)
                   .fillna(False)
                   .astype(bool))
    # Extend member_df columns to match close_df (missing masks → always False)
    for col in close_df.columns:
        if col not in member_df.columns:
            member_df[col] = False

    ret_df = close_df.pct_change(fill_method=None).fillna(0.0)

    # Weekly rebalance: every Friday in trading_idx
    fridays = trading_idx[trading_idx.dayofweek == 4]

    port_ret = pd.Series(0.0, index=trading_idx)
    to_ser   = pd.Series(0.0, index=trading_idx)
    prev_w: dict = {}

    for ri, fri in enumerate(fridays):
        fri_pos = trading_idx.searchsorted(fri, side="right") - 1
        if fri_pos < form_days:
            continue

        # Formation return (vectorised over all symbols)
        p_now  = close_df.iloc[fri_pos]
        p_past = close_df.iloc[fri_pos - form_days]
        form_ret = (p_now / p_past.replace(0, np.nan)) - 1.0

        # Filters at this date (all aligned to trading_idx)
        prices  = close_df.iloc[fri_pos]
        dv_row  = dv_df.iloc[fri_pos]
        members = member_df.iloc[fri_pos]

        valid = (
            members.reindex(form_ret.index, fill_value=False) &
            (prices.reindex(form_ret.index) >= min_px) &
            (dv_row.reindex(form_ret.index) >= min_dv)
        )
        signals = form_ret[valid].dropna()
        if len(signals) < 20:
            continue

        n_each = max(1, len(signals) // 10)
        ranked = signals.sort_values()
        longs  = ranked.iloc[:n_each].index.tolist()    # losers → long
        shorts = ranked.iloc[-n_each:].index.tolist()   # winners → short

        # Holding period: days after this Friday through next Friday
        if ri + 1 < len(fridays):
            hold_end = fridays[ri + 1]
        else:
            hold_end = trading_idx[-1]
        hold_mask = (trading_idx > fri) & (trading_idx <= hold_end)

        # Portfolio return
        l_ret = ret_df.loc[hold_mask, [s for s in longs  if s in ret_df.columns]].fillna(0.0)
        s_ret = ret_df.loc[hold_mask, [s for s in shorts if s in ret_df.columns]].fillna(0.0)

        if not l_ret.empty:
            port_ret[hold_mask] += l_ret.mean(axis=1).values / 2.0
        if not s_ret.empty:
            port_ret[hold_mask] -= s_ret.mean(axis=1).values / 2.0

        # Turnover at first holding day
        exec_days = trading_idx[trading_idx > fri]
        if not exec_days.empty:
            new_w = {}
            for s in longs:  new_w[s] =  0.5 / len(longs)
            for s in shorts: new_w[s] = -0.5 / len(shorts)
            all_s = set(list(new_w.keys()) + list(prev_w.keys()))
            to = sum(abs(new_w.get(s, 0.0) - prev_w.get(s, 0.0)) for s in all_s) / 2.0
            to_ser[exec_days[0]] += to
            prev_w = new_w

    net_ret = apply_costs(port_ret, to_ser, cost_bps, slip_bps)

    spy = load_price_series("SPY", start, end, ADJ_TOTALRETURN, cache)
    bm  = spy["Close"].pct_change(fill_method=None).reindex(net_ret.index)

    ann_to = float(to_ser.sum() / max(len(to_ser) / TRADING_DAYS, 1))
    log.info("S09 done (form=%d-day). Ann turnover: %.1fx", form_days, ann_to)

    return {
        "returns": net_ret, "benchmark": bm,
        "description": DESCRIPTION, "turnover_annual": ann_to,
    }
