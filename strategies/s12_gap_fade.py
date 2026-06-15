"""
S12 — Overnight gap fade (partial Norgate implementation)
Universe:  S&P 500 C&P PIT; filter for liquid mid/large cap
Signal:    Stocks with large overnight gaps (Open >> prev Close) tend to mean-revert.
           Gap = (Open - prev_Close) / prev_Close.
           Long stocks with large negative gaps (oversold gap-down) if above 200-DMA.
           Short stocks with large positive gaps (overbought gap-up) if below 200-DMA.
Note:      Norgate daily OHLC gives Open; no intraday news filter available.
           Performance may be diluted by earnings gap days.
"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import logging
import numpy as np
import pandas as pd

from data import (load_price_series, watchlist_symbols, index_constituent_mask,
                  compute_dollar_volume, ADJ_TOTALRETURN)
from engine import portfolio_returns_from_weights, apply_costs

log = logging.getLogger(__name__)
DESCRIPTION = "Overnight gap fade, S&P 500 PIT, gap >1.5σ fade, daily hold"
TRADING_DAYS = 252
GAP_Z_THRESH = 1.5    # |gap| > 1.5σ of 63-day gap distribution to trade
MIN_PRICE    = 5.0
MIN_DOLVOL   = 1e6
TOP_N        = 20     # max positions per day


def run(config: dict) -> dict:
    cfg      = config["backtest"]
    start    = cfg["start_date"]
    end      = cfg["end_date"]
    cache    = config["paths"]["cache_dir"]
    cost_bps = config["costs"]["equity_cost_bps"]
    slip_bps = config["costs"]["equity_slippage_bps"]

    univ_name = config["universes"]["sp500"]
    syms      = watchlist_symbols(univ_name)
    if not syms:
        return {"returns": pd.Series(dtype=float), "benchmark": pd.Series(dtype=float),
                "description": DESCRIPTION, "turnover_annual": 0.0}

    trading_idx = pd.bdate_range(start, end)

    close_map, open_map, member_map, dolvol_map = {}, {}, {}, {}
    for sym in syms:
        df = load_price_series(sym, start=start, end=end, adjustment=ADJ_TOTALRETURN, cache_dir=cache)
        if df.empty or "Open" not in df.columns:
            continue
        mask = index_constituent_mask(sym, "S&P 500", start, end, cache)
        close_map[sym]  = df["Close"]
        open_map[sym]   = df["Open"]
        member_map[sym] = mask
        dolvol_map[sym] = compute_dollar_volume(df)

    if len(close_map) < 10:
        log.error("S12: too few symbols with Open price data (%d)", len(close_map))
        return {"returns": pd.Series(dtype=float), "benchmark": pd.Series(dtype=float),
                "description": DESCRIPTION, "turnover_annual": 0.0}

    close_df  = pd.DataFrame(close_map).reindex(trading_idx, method="ffill")
    open_df   = pd.DataFrame(open_map).reindex(trading_idx, method="ffill")
    member_df = pd.DataFrame(member_map).reindex(trading_idx, method="ffill")
    member_df = member_df.infer_objects(copy=False).fillna(False).astype(bool)
    dolvol_df = pd.DataFrame(dolvol_map).reindex(trading_idx, method="ffill")

    gap_df  = (open_df - close_df.shift(1)) / close_df.shift(1)
    gap_df  = gap_df.fillna(0.0)

    # Rolling z-score of gap (63-day window)
    gap_mu  = gap_df.rolling(63).mean()
    gap_std = gap_df.rolling(63).std().replace(0, np.nan)
    gap_z   = (gap_df - gap_mu) / gap_std
    gap_z   = gap_z.fillna(0.0)

    # 200-day DMA filter
    dma200  = close_df.rolling(200).mean()
    above_dma = close_df > dma200

    # Liquidity + price filter
    liq_mask = (dolvol_df >= MIN_DOLVOL) & (close_df >= MIN_PRICE) & member_df

    # Gap fade signal:
    #   Large gap DOWN (z < -thresh) AND above 200-DMA → long (expect bounce)
    #   Large gap UP   (z >  thresh) AND below 200-DMA → short (expect fade)
    long_sig  = (gap_z < -GAP_Z_THRESH) & above_dma  & liq_mask
    short_sig = (gap_z >  GAP_Z_THRESH) & ~above_dma & liq_mask

    # Shift 1: signal is observable at today's open, position taken at open price
    # We approximate with close-to-close and accept the gap-at-open contamination
    long_sig  = (long_sig.astype(float).shift(1).fillna(0.0) > 0)
    short_sig = (short_sig.astype(float).shift(1).fillna(0.0) > 0)

    weight_schedule = {}
    for t in trading_idx:
        longs  = list(close_df.columns[long_sig.loc[t]])
        shorts = list(close_df.columns[short_sig.loc[t]])
        wt = {}
        n_l, n_s = min(len(longs), TOP_N), min(len(shorts), TOP_N)
        for s in longs[:n_l]:
            wt[s] = 0.5 / n_l if n_l else 0.0
        for s in shorts[:n_s]:
            wt[s] = -0.5 / n_s if n_s else 0.0
        if wt:
            weight_schedule[t] = wt

    gross_ret, to_series = portfolio_returns_from_weights(weight_schedule, close_df, start, end)
    net_ret = apply_costs(gross_ret, to_series, cost_bps, slip_bps)

    spy = load_price_series("SPY", start=start, end=end, adjustment=ADJ_TOTALRETURN, cache_dir=cache)
    bm  = spy["Close"].reindex(trading_idx, method="ffill").pct_change(fill_method=None).fillna(0.0) if not spy.empty else gross_ret * 0

    ann_to = float(to_series.sum() / max(len(trading_idx) / TRADING_DAYS, 1))
    log.info("S12 done. Ann turnover: %.2fx", ann_to)

    return {
        "returns": net_ret, "benchmark": bm,
        "description": DESCRIPTION, "turnover_annual": ann_to,
    }
