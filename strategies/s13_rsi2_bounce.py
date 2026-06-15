"""
S13 — RSI(2) oversold bounce (Connors 2009)
Universe:  S&P 500 point-in-time; only trade stocks above their 200-DMA (with-trend filter)
Signal:    RSI(2) < 5 → long at next open; exit when close > 5-day SMA.
Execution: Enter at open of day t+1 after signal at close of t (approx next close).
Sizing:    Equal-weight across all active signals; long-only.
Sweep:     RSI period [2,3,4], entry threshold [5,10], 200-DMA filter on/off
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
DESCRIPTION = "RSI(2) oversold bounce, S&P 500 PIT, above-200DMA filter, exit at 5-DMA"
TRADING_DAYS = 252


def _rsi(series: pd.Series, period: int = 2) -> pd.Series:
    delta = series.diff()
    gain  = delta.clip(lower=0).rolling(period).mean()
    loss  = (-delta.clip(upper=0)).rolling(period).mean()
    rs    = gain / loss.replace(0, np.nan)
    return 100 - 100 / (1 + rs)


def run(config: dict) -> dict:
    cfg      = config["backtest"]
    start    = cfg["start_date"]
    end      = cfg["end_date"]
    is_smoke = config.get("smoke", False)
    cache    = config["paths"]["cache_dir"]

    s13_cfg   = config.get("strategies", {}).get("s13", {})
    rsi_period   = s13_cfg.get("rsi_period", 2)
    rsi_entry    = s13_cfg.get("entry_threshold", 5)
    use_200dma   = s13_cfg.get("regime_filter", True)
    exit_sma_len = 5

    cost_bps = config["costs"]["equity_cost_bps"]
    slip_bps = config["costs"]["equity_slippage_bps"]
    min_dv   = config["liquidity"]["min_dollar_volume"]
    min_px   = config["liquidity"]["min_price"]

    wl_name  = config["universes"].get("sp500", "S&P 500 Current & Past")
    idx_name = config["universes"].get("sp500_index", "S&P 500")
    start_load = "2013-01-01" if is_smoke else "1999-01-01"

    log.info("S13: loading S&P 500 C&P ...")
    symbols = watchlist_symbols(wl_name)
    if is_smoke:
        symbols = symbols[:120]

    close_map = {}
    dv_map    = {}
    mask_map  = {}
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

    log.info("S13: %d symbols loaded", len(close_map))
    close_df  = pd.DataFrame(close_map).sort_index()
    dv_df     = pd.DataFrame(dv_map).sort_index()
    member_df = (pd.DataFrame(mask_map)
                   .reindex(close_df.index, method="ffill")
                   .infer_objects(copy=False)
                   .fillna(False).astype(bool))

    trading_idx = pd.bdate_range(start, end)
    ret_df      = close_df.pct_change(fill_method=None)

    # Pre-compute signals for all symbols
    rsi_df   = close_df.apply(lambda c: _rsi(c, rsi_period))
    sma5_df  = close_df.rolling(exit_sma_len).mean()
    sma200_df = close_df.rolling(200).mean()

    # Entry: RSI(2) < threshold AND (if filter) close > 200-DMA — shifted 1 day
    entry_sig = (rsi_df < rsi_entry)
    if use_200dma:
        entry_sig = entry_sig & (close_df > sma200_df)
    entry_sig = entry_sig.shift(1).fillna(False).astype(bool)

    # Exit: close > 5-DMA — shifted 1 day
    exit_sig = (close_df > sma5_df).shift(1).fillna(False).astype(bool)

    # Build position matrix (state machine per stock)
    position_df = pd.DataFrame(0.0, index=close_df.index, columns=list(close_map.keys()))
    for sym in close_map:
        pos = 0.0
        pos_list = []
        for i in range(len(close_df)):
            if pos == 1.0:
                if exit_sig.iloc[i][sym]:
                    pos = 0.0
            else:
                if entry_sig.iloc[i][sym]:
                    # Apply liquidity filter on the fly
                    dv_val = dv_df.iloc[i].get(sym, 0)
                    mem_val = bool(member_df.iloc[i].get(sym, False)) if sym in member_df.columns else False
                    if dv_val >= min_dv and mem_val:
                        pos = 1.0
            pos_list.append(pos)
        position_df[sym] = pos_list

    position_df = position_df.reindex(trading_idx, method="ffill")

    # Equal-weight active long positions
    n_active = position_df.sum(axis=1).clip(lower=1e-9)
    weight_df = position_df.div(n_active, axis=0)
    weight_df[n_active < 0.5] = 0.0  # full cash when nothing active

    ret_daily = ret_df.reindex(trading_idx, fill_value=0.0).fillna(0.0)
    port_ret  = (weight_df * ret_daily).sum(axis=1)

    to = weight_df.diff().abs().sum(axis=1) / 2.0
    net_ret = apply_costs(port_ret, to, cost_bps, slip_bps)

    spy = load_price_series("SPY", start, end, ADJ_TOTALRETURN, cache)
    bm  = spy["Close"].pct_change(fill_method=None).reindex(net_ret.index)

    ann_to = float(to.sum() / max(len(to) / TRADING_DAYS, 1))
    log.info("S13 done (RSI%d<%d, 200DMA=%s). Ann turnover: %.2fx",
             rsi_period, rsi_entry, use_200dma, ann_to)

    return {
        "returns": net_ret, "benchmark": bm,
        "description": DESCRIPTION, "turnover_annual": ann_to,
    }
