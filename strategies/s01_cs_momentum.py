"""
S01 — Cross-sectional 12-1 momentum (Jegadeesh-Titman 1993)
Universe:  Russell 1000 point-in-time (C&P watchlist + constituent mask)
Signal:    TOTALRETURN from t-lookback to t-skip (default 252→21)
Portfolio: Long top decile, short bottom decile, equal-weight, dollar-neutral
Rebalance: Monthly (month-end business day)
Execution: Close of rebalance day + 1 (next trading day close)
Sweep:     lookbacks [126,189,252], skips [0,21], deciles vs quintiles
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
DESCRIPTION = "Cross-sectional 12-1 momentum (Jegadeesh-Titman), Russell 1000 PIT"


def _load_universe(config, is_smoke):
    """Load prices + constituent masks for the universe. Returns (close_df, dv_df, member_df)."""
    cache_dir = config.get("paths", {}).get("cache_dir", "cache/parquet")
    start_load = "1998-01-01"
    end_load   = config["backtest"]["end_date"]

    if is_smoke:
        wl_name  = config["universes"].get("sp500", "S&P 500 Current & Past")
        idx_name = config["universes"].get("sp500_index", "S&P 500")
        start_load = "2013-01-01"
        sym_limit  = 150
    else:
        wl_name  = config["universes"].get("russell1000", "Russell 1000 Current & Past")
        idx_name = config["universes"].get("russell1000_index", "Russell 1000")
        sym_limit  = None

    log.info("S01: loading watchlist '%s' ...", wl_name)
    symbols = watchlist_symbols(wl_name)
    if sym_limit:
        symbols = symbols[:sym_limit]
    log.info("S01: %d symbols in watchlist", len(symbols))

    min_dv = config["liquidity"]["min_dollar_volume"]
    min_px = config["liquidity"]["min_price"]

    close_map  = {}
    dv_map     = {}
    mask_map   = {}

    for sym in symbols:
        df = load_price_series(sym, start=start_load, end=end_load,
                               adjustment=ADJ_TOTALRETURN, cache_dir=cache_dir)
        if df.empty or "Close" not in df.columns:
            continue
        # rough pre-filter: must ever pass price and dollar-volume thresholds
        if df["Close"].max() < min_px:
            continue
        close_map[sym] = df["Close"]
        dv_map[sym]    = compute_dollar_volume(df)

        m = index_constituent_mask(sym, idx_name, start=start_load, end=end_load,
                                   cache_dir=cache_dir)
        if not m.empty:
            mask_map[sym] = m

    log.info("S01: %d symbols after loading", len(close_map))

    close_df  = pd.DataFrame(close_map).sort_index()
    dv_df     = pd.DataFrame(dv_map).sort_index()

    # Build membership matrix (forward-fill gaps; daily data may have missing index rows)
    if mask_map:
        member_df = pd.DataFrame(mask_map)
        member_df = (member_df.reindex(close_df.index, method="ffill")
                              .infer_objects(copy=False)
                              .fillna(False)
                              .astype(bool))
    else:
        member_df = pd.DataFrame(False, index=close_df.index, columns=list(close_map.keys()))

    return close_df, dv_df, member_df, idx_name


def _build_weights(close_df, dv_df, member_df, reb_dates, lookback, skip,
                   n_quantile, min_dv, min_px):
    weight_schedule = {}

    for d in reb_dates:
        if d not in close_df.index:
            # Snap to last available date
            avail = close_df.index[close_df.index <= d]
            if avail.empty:
                continue
            snap = avail[-1]
        else:
            snap = d

        row_idx = close_df.index.get_loc(snap)
        if row_idx < lookback:
            continue

        # Price at skip-ago and lookback-ago (vectorised slice)
        p_now   = close_df.iloc[row_idx - skip]  if skip > 0 else close_df.iloc[row_idx]
        p_past  = close_df.iloc[row_idx - lookback]
        mom     = (p_now / p_past) - 1.0  # Series over symbols

        # Liquidity + membership filters at snap date
        prices  = close_df.iloc[row_idx]
        dv_row  = dv_df.iloc[row_idx]
        members = member_df.iloc[row_idx] if snap in member_df.index else pd.Series(False, index=close_df.columns)

        valid = (
            members.reindex(mom.index, fill_value=False) &
            (prices.reindex(mom.index) >= min_px) &
            (dv_row.reindex(mom.index) >= min_dv)
        )
        mom = mom[valid].dropna()

        n_each = max(1, len(mom) // n_quantile)
        if len(mom) < n_each * 2:
            continue

        ranked = mom.sort_values()
        shorts = ranked.iloc[:n_each].index.tolist()
        longs  = ranked.iloc[-n_each:].index.tolist()

        w = {}
        for s in longs:
            w[s] =  1.0 / len(longs)
        for s in shorts:
            w[s] = -1.0 / len(shorts)
        weight_schedule[d] = w

    return weight_schedule


def run(config: dict) -> dict:
    cfg      = config["backtest"]
    start    = cfg["start_date"]
    end      = cfg["end_date"]
    is_smoke = config.get("smoke", False)

    s01_cfg   = config.get("strategies", {}).get("s01", {})
    lookbacks = s01_cfg.get("lookbacks", [252])
    skips     = s01_cfg.get("skip_days", [21])
    n_q       = 10 if s01_cfg.get("decile", True) else 5

    cost_bps = config["costs"]["equity_cost_bps"]
    slip_bps = config["costs"]["equity_slippage_bps"]
    min_dv   = config["liquidity"]["min_dollar_volume"]
    min_px   = config["liquidity"]["min_price"]

    close_df, dv_df, member_df, idx_name = _load_universe(config, is_smoke)

    reb_dates = pd.date_range(start, end, freq="BME")

    # Primary parameter set
    lb, sk = lookbacks[0], skips[0]

    log.info("S01: building weights (lookback=%d, skip=%d, n_quantile=%d) ...", lb, sk, n_q)
    ws = _build_weights(close_df, dv_df, member_df, reb_dates, lb, sk, n_q, min_dv, min_px)

    if not ws:
        log.warning("S01: no valid rebalance periods found.")
        return {"returns": pd.Series(dtype=float), "benchmark": pd.Series(dtype=float),
                "description": DESCRIPTION, "turnover_annual": 0.0}

    log.info("S01: %d rebalance dates with valid portfolios", len(ws))
    gross_ret, to = portfolio_returns_from_weights(ws, close_df, start, end)
    net_ret = apply_costs(gross_ret, to, cost_bps, slip_bps)

    # Benchmark (SPY)
    spy = load_price_series("SPY", start, end, ADJ_TOTALRETURN,
                             config["paths"]["cache_dir"])
    bm = spy["Close"].pct_change().reindex(net_ret.index)

    ann_to = float(to.sum() / max(len(to) / 252, 1))
    log.info("S01 done. Ann turnover: %.1fx", ann_to)

    return {
        "returns":         net_ret,
        "benchmark":       bm,
        "description":     DESCRIPTION,
        "turnover_annual": ann_to,
    }
