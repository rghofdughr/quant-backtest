"""
S48 — Gold/copper ratio risk filter on equity allocation
Universe:  SPY (equity) and TLT (bonds) as the two risk states
Signal:    GLD/COPX or GLD/HG ratio. When the ratio's 50-day MA > 200-day MA
           (gold outperforming copper = risk-off), shift to TLT.
           When copper outperforms (ratio downtrend), stay in SPY.
Execution: Daily signal; execute at next close.
Sizing:    Full allocation — 100% SPY or 100% TLT.
Note:      COPX (copper miners ETF) as proxy for copper; HG futures as alternative.
"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import logging
import numpy as np
import pandas as pd

from data import load_price_series, load_futures_series, ADJ_TOTALRETURN
from engine import apply_costs

log = logging.getLogger(__name__)
DESCRIPTION = "Gold/copper ratio risk filter, SPY vs TLT, DMA crossover on GLD/COPX"
TRADING_DAYS = 252


def run(config: dict) -> dict:
    cfg      = config["backtest"]
    start    = cfg["start_date"]
    end      = cfg["end_date"]
    cache    = config["paths"]["cache_dir"]

    cost_bps = config["costs"]["equity_cost_bps"]
    slip_bps = config["costs"]["equity_slippage_bps"]

    spy  = load_price_series("SPY",  start=start, end=end, adjustment=ADJ_TOTALRETURN, cache_dir=cache)
    tlt  = load_price_series("TLT",  start=start, end=end, adjustment=ADJ_TOTALRETURN, cache_dir=cache)
    gld  = load_price_series("GLD",  start=start, end=end, adjustment=ADJ_TOTALRETURN, cache_dir=cache)

    # Copper proxy: try COPX ETF first, fall back to HG futures
    copx = load_price_series("COPX", start=start, end=end, adjustment=ADJ_TOTALRETURN, cache_dir=cache)
    if copx.empty:
        log.info("S48: COPX not available, using HG futures as copper proxy")
        copx = load_futures_series("HG", start=start, end=end, cache_dir=cache)

    if spy.empty or gld.empty or copx.empty:
        log.error("S48: missing required data")
        return {"returns": pd.Series(dtype=float), "benchmark": pd.Series(dtype=float),
                "description": DESCRIPTION, "turnover_annual": 0.0}

    trading_idx = pd.bdate_range(start, end)

    spy_r  = spy["Close"].reindex(trading_idx, method="ffill").pct_change(fill_method=None).fillna(0.0)
    tlt_r  = tlt["Close"].reindex(trading_idx, method="ffill").pct_change(fill_method=None).fillna(0.0) if not tlt.empty else spy_r * 0
    gld_c  = gld["Close"].reindex(trading_idx, method="ffill")
    copx_c = copx["Close"].reindex(trading_idx, method="ffill")

    ratio  = gld_c / copx_c.replace(0, np.nan)
    dma50  = ratio.rolling(50).mean()
    dma200 = ratio.rolling(200).mean()

    # Risk-off (gold > copper, ratio rising) → hold TLT; risk-on → hold SPY
    # Shift 1 day: act on yesterday's signal
    risk_off = ((dma50 > dma200).astype(float).shift(1).fillna(0.0) > 0)

    port_ret = pd.Series(0.0, index=trading_idx)
    port_ret[~risk_off] = spy_r[~risk_off]
    port_ret[risk_off]  = tlt_r[risk_off]

    to = risk_off.astype(float).diff().abs().fillna(0.0)
    net_ret = apply_costs(port_ret, to, cost_bps, slip_bps)

    bm = spy_r.copy()
    risk_off_pct = float(risk_off.mean()) * 100
    ann_to = float(to.sum() / max(len(to) / TRADING_DAYS, 1))
    log.info("S48 done. Risk-off (TLT): %.1f%% of days, ann turnover: %.2fx", risk_off_pct, ann_to)

    return {
        "returns": net_ret, "benchmark": bm,
        "description": DESCRIPTION, "turnover_annual": ann_to,
    }
