"""
S49 — Dollar regime filter on EM equities
Universe:  EEM (Emerging Markets ETF); SPY as risk-off alternative
Signal:    UUP (PowerShares DB USD Index Bullish Fund) or DX futures trend.
           If dollar is in uptrend (50-DMA > 200-DMA), go flat or hold SPY.
           If dollar is in downtrend, hold EEM.
Execution: Monthly signal at month-end; next trading day close.
Sizing:    100% EEM (risk-on) or 100% SPY / cash (risk-off).
"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import logging
import numpy as np
import pandas as pd

from data import load_price_series, load_futures_series, ADJ_TOTALRETURN
from engine import apply_costs

log = logging.getLogger(__name__)
DESCRIPTION = "Dollar regime filter: EEM when USD downtrend, SPY when USD uptrend"
TRADING_DAYS = 252


def run(config: dict) -> dict:
    cfg      = config["backtest"]
    start    = cfg["start_date"]
    end      = cfg["end_date"]
    cache    = config["paths"]["cache_dir"]

    cost_bps = config["costs"]["equity_cost_bps"]
    slip_bps = config["costs"]["equity_slippage_bps"]

    eem = load_price_series("EEM", start=start, end=end, adjustment=ADJ_TOTALRETURN, cache_dir=cache)
    spy = load_price_series("SPY", start=start, end=end, adjustment=ADJ_TOTALRETURN, cache_dir=cache)

    # Dollar proxy: try UUP ETF, fall back to DX futures
    uup = load_price_series("UUP", start=start, end=end, adjustment=ADJ_TOTALRETURN, cache_dir=cache)
    if uup.empty or len(uup) < 210:
        log.info("S49: UUP short/missing, using DX futures as dollar proxy")
        uup = load_futures_series("DX", start=start, end=end, cache_dir=cache)

    if eem.empty or uup.empty:
        return {"returns": pd.Series(dtype=float), "benchmark": pd.Series(dtype=float),
                "description": DESCRIPTION, "turnover_annual": 0.0}

    trading_idx = pd.bdate_range(start, end)
    eem_r  = eem["Close"].reindex(trading_idx, method="ffill").pct_change(fill_method=None).fillna(0.0)
    spy_r  = spy["Close"].reindex(trading_idx, method="ffill").pct_change(fill_method=None).fillna(0.0)
    usd_c  = uup["Close"].reindex(trading_idx, method="ffill")

    dma50  = usd_c.rolling(50).mean()
    dma200 = usd_c.rolling(200).mean()

    # USD uptrend (bad for EM) → hold SPY; USD downtrend → hold EEM
    usd_uptrend = ((dma50 > dma200).astype(float).shift(1).fillna(0.0) > 0)

    port_ret = pd.Series(0.0, index=trading_idx)
    port_ret[~usd_uptrend] = eem_r[~usd_uptrend]   # EEM when USD weak
    port_ret[usd_uptrend]  = spy_r[usd_uptrend]     # SPY when USD strong

    to = usd_uptrend.astype(float).diff().abs().fillna(0.0)
    net_ret = apply_costs(port_ret, to, cost_bps, slip_bps)

    bm = eem_r.copy()
    em_pct = float((~usd_uptrend).mean()) * 100
    ann_to = float(to.sum() / max(len(to) / TRADING_DAYS, 1))
    log.info("S49 done. In EEM: %.1f%% of days, ann turnover: %.2fx", em_pct, ann_to)

    return {
        "returns": net_ret, "benchmark": bm,
        "description": DESCRIPTION, "turnover_annual": ann_to,
    }
