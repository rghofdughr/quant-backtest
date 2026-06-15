"""
S27 — VIX term-structure carry (short vol ETPs)
Universe:  VXX (long VIX futures, 1m/2m blend) as the instrument to short.
           SVXY (short vol ETP) as direct proxy for the strategy.
Signal:    VIX futures are almost always in contango → short VXX earns positive roll.
           Filter: only short VXX when VIX < 30 (not in extreme stress).
           Proxy: long SVXY with VIX regime filter.
Note:      VXX data available post-2009 in Norgate. SVXY post-2012.
           Full VX futures term-structure trade requires individual monthly VX contracts.
"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import logging
import numpy as np
import pandas as pd

from data import load_price_series, load_futures_series, ADJ_TOTALRETURN
from engine import apply_costs

log = logging.getLogger(__name__)
DESCRIPTION = "VIX term-structure carry, short VXX / long SVXY with VIX regime filter"
TRADING_DAYS = 252
VIX_STRESS_THRESH = 30.0


def _load_vix(start: str, end: str, cache: str) -> pd.Series:
    for sym in ["$VIX", "VIX", "^VIX", "VIXY"]:
        df = load_price_series(sym, start=start, end=end, adjustment=ADJ_TOTALRETURN, cache_dir=cache)
        if not df.empty and len(df) > 100:
            return df["Close"]
    df = load_futures_series("VX", start=start, end=end, cache_dir=cache)
    if not df.empty:
        return df["Close"]
    return pd.Series(dtype=float)


def run(config: dict) -> dict:
    cfg      = config["backtest"]
    start    = cfg["start_date"]
    end      = cfg["end_date"]
    cache    = config["paths"]["cache_dir"]
    cost_bps = config["costs"]["equity_cost_bps"]
    slip_bps = config["costs"]["equity_slippage_bps"]

    trading_idx = pd.bdate_range(start, end)

    # Primary: long SVXY (inverse VIX ETP)
    svxy = load_price_series("SVXY", start=start, end=end, adjustment=ADJ_TOTALRETURN, cache_dir=cache)
    # Fallback: short VXX (long VIX ETP) — approximated via negative return
    vxx  = load_price_series("VXX",  start=start, end=end, adjustment=ADJ_TOTALRETURN, cache_dir=cache)

    if svxy.empty and vxx.empty:
        log.warning("S27: neither SVXY nor VXX available — returning empty")
        return {"returns": pd.Series(dtype=float), "benchmark": pd.Series(dtype=float),
                "description": DESCRIPTION, "turnover_annual": 0.0}

    vix_c = _load_vix(start, end, cache)

    spy = load_price_series("SPY", start=start, end=end, adjustment=ADJ_TOTALRETURN, cache_dir=cache)
    spy_r = spy["Close"].reindex(trading_idx, method="ffill").pct_change(fill_method=None).fillna(0.0) if not spy.empty else pd.Series(0.0, index=trading_idx)

    if not svxy.empty:
        inst_r = svxy["Close"].reindex(trading_idx, method="ffill").pct_change(fill_method=None).fillna(0.0)
        log.info("S27: using SVXY as primary instrument")
    else:
        inst_r = -vxx["Close"].reindex(trading_idx, method="ffill").pct_change(fill_method=None).fillna(0.0)
        log.info("S27: using -VXX as proxy (no SVXY)")

    # Regime filter: only hold when VIX < stress threshold
    if not vix_c.empty:
        vix_s = vix_c.reindex(trading_idx, method="ffill")
        low_stress = ((vix_s < VIX_STRESS_THRESH).astype(float).shift(1).fillna(0.0) > 0)
    else:
        low_stress = pd.Series(True, index=trading_idx)
        log.info("S27: no VIX data, holding all days")

    port_ret = low_stress.astype(float) * inst_r
    to = low_stress.astype(float).diff().abs().fillna(0.0)
    net_ret = apply_costs(port_ret, to, cost_bps, slip_bps)

    bm = spy_r
    long_pct = float(low_stress.mean()) * 100
    ann_to = float(to.sum() / max(len(trading_idx) / TRADING_DAYS, 1))
    log.info("S27 done. Active (low stress): %.1f%% of days, ann turnover: %.2fx", long_pct, ann_to)

    return {
        "returns": net_ret, "benchmark": bm,
        "description": DESCRIPTION, "turnover_annual": ann_to,
    }
