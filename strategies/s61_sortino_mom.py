"""
S61 — Downside-deviation momentum (Sortino ratio cross-section)
Universe:  S&P 500 point-in-time
Signal:    Trailing 252-day Sortino ratio per stock. Long top quintile (highest Sortino).
           Sortino = annualized mean return / downside deviation (std of negative returns * sqrt(252)).
Portfolio: Long-only, equal-weight, top quintile.
Rebalance: Monthly.
Warmup:    252 trading days.
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
TRADING_DAYS = 252
DESCRIPTION = "Downside-deviation momentum: long top-quintile Sortino ratio, S&P 500 PIT, monthly"


def run(config: dict) -> dict:
    cfg = config["backtest"]
    start = cfg["start_date"]
    end = cfg["end_date"]
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
        symbols = symbols[:120]

    close_map, dv_map, mask_map = {}, {}, {}
    for sym in symbols:
        df = load_price_series(sym, start=start_load, end=end,
                               adjustment=ADJ_TOTALRETURN, cache_dir=cache)
        if df.empty or df["Close"].max() < min_px:
            continue
        close_map[sym] = df["Close"]
        dv_map[sym] = compute_dollar_volume(df)
        m = index_constituent_mask(sym, idx_name, start=start_load, end=end, cache_dir=cache)
        if not m.empty:
            mask_map[sym] = m

    log.info("S61: %d symbols loaded", len(close_map))

    trading_idx = pd.bdate_range(start, end)
    close_df = pd.DataFrame(close_map).reindex(trading_idx, method="ffill")
    dv_df = pd.DataFrame(dv_map).reindex(trading_idx, method="ffill")
    member_df = (
        pd.DataFrame(mask_map)
        .reindex(trading_idx, method="ffill")
        .infer_objects(copy=False)
        .fillna(False)
        .astype(bool)
    )
    for col in close_df.columns:
        if col not in member_df.columns:
            member_df[col] = False

    ret_df = close_df.pct_change(fill_method=None)

    reb_dates = pd.date_range(start, end, freq="BME")
    WARMUP = TRADING_DAYS  # 252 days warmup
    N_QUANTILE = 5
    WIN = TRADING_DAYS  # 252-day window for Sortino

    weight_schedule = {}

    for rd in reb_dates:
        pos = trading_idx.searchsorted(rd, side="right") - 1
        if pos < WARMUP:
            continue

        win_start = max(0, pos - WIN)
        ret_window = ret_df.iloc[win_start:pos]

        dv_now = dv_df.iloc[pos]
        members = member_df.iloc[pos]

        valid_cols = [
            c for c in close_df.columns
            if members.get(c, False)
            and dv_now.get(c, 0) >= min_dv
        ]
        if len(valid_cols) < N_QUANTILE * 2:
            continue

        # Compute Sortino ratio per stock
        sortino_scores = {}
        for sym in valid_cols:
            if sym not in ret_window.columns:
                continue
            col = ret_window[sym].dropna()
            if len(col) < 60:
                continue
            mean_ret = col.mean() * TRADING_DAYS
            downside = col[col < 0].std() * np.sqrt(TRADING_DAYS)
            sortino = mean_ret / downside if downside > 0 else np.nan
            if np.isfinite(sortino):
                sortino_scores[sym] = sortino

        if len(sortino_scores) < N_QUANTILE * 2:
            continue

        sr_series = pd.Series(sortino_scores).sort_values(ascending=False)
        n_each = max(1, len(sr_series) // N_QUANTILE)
        long_names = sr_series.iloc[:n_each].index.tolist()

        w = {s: 1.0 / len(long_names) for s in long_names}
        weight_schedule[rd] = w

    if not weight_schedule:
        log.warning("S61: no valid rebalance periods found")
        empty = pd.Series(dtype=float)
        return {"returns": empty, "benchmark": empty,
                "description": DESCRIPTION, "turnover_annual": 0.0}

    gross_ret, to = portfolio_returns_from_weights(weight_schedule, close_df, start, end)
    net_ret = apply_costs(gross_ret, to, cost_bps, slip_bps)

    spy = load_price_series("SPY", start, end, ADJ_TOTALRETURN, cache)
    bm = spy["Close"].pct_change(fill_method=None).reindex(net_ret.index)

    ann_to = float(to.sum() / max(len(to) / TRADING_DAYS, 1))
    log.info("S61 done. %d rebalance dates, ann_turnover=%.2fx", len(weight_schedule), ann_to)

    return {
        "returns": net_ret,
        "benchmark": bm,
        "description": DESCRIPTION,
        "turnover_annual": ann_to,
    }
