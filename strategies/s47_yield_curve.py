"""
S47 — Yield curve slope factor (Fama 1984; Ang & Piazzesi 2003)
Universe:  SPY (equity), TLT (long bonds)
Signal:    10Y−2Y slope proxy via: (TLT returns) − (IEF returns) or ZN/ZT futures spread.
           When the curve is steepening (slope rising, momentum positive) → risk-on: SPY.
           When flattening/inverting → risk-off: TLT.
Execution: Monthly rebalance.
Note:      FRED FRED/T10Y2Y would be ideal; we proxy with ETF returns spread.
           IEF ≈ 7-10Y duration; SHY ≈ 1-3Y duration. Spread ≈ term premium direction.
"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import logging
import numpy as np
import pandas as pd

from data import load_price_series, ADJ_TOTALRETURN
from engine import apply_costs

log = logging.getLogger(__name__)
DESCRIPTION = "Yield curve slope factor, SPY vs TLT, IEF-SHY spread proxy, monthly"
TRADING_DAYS = 252
SLOPE_LOOKBACK = 63   # 3-month slope momentum


def run(config: dict) -> dict:
    cfg      = config["backtest"]
    start    = cfg["start_date"]
    end      = cfg["end_date"]
    cache    = config["paths"]["cache_dir"]
    cost_bps = config["costs"]["equity_cost_bps"]
    slip_bps = config["costs"]["equity_slippage_bps"]

    spy = load_price_series("SPY", start=start, end=end, adjustment=ADJ_TOTALRETURN, cache_dir=cache)
    tlt = load_price_series("TLT", start=start, end=end, adjustment=ADJ_TOTALRETURN, cache_dir=cache)
    ief = load_price_series("IEF", start=start, end=end, adjustment=ADJ_TOTALRETURN, cache_dir=cache)
    shy = load_price_series("SHY", start=start, end=end, adjustment=ADJ_TOTALRETURN, cache_dir=cache)

    if spy.empty:
        return {"returns": pd.Series(dtype=float), "benchmark": pd.Series(dtype=float),
                "description": DESCRIPTION, "turnover_annual": 0.0}

    trading_idx = pd.bdate_range(start, end)
    spy_r = spy["Close"].reindex(trading_idx, method="ffill").pct_change(fill_method=None).fillna(0.0)
    tlt_r = tlt["Close"].reindex(trading_idx, method="ffill").pct_change(fill_method=None).fillna(0.0) if not tlt.empty else pd.Series(0.0, index=trading_idx)

    if not ief.empty and not shy.empty:
        ief_c = ief["Close"].reindex(trading_idx, method="ffill")
        shy_c = shy["Close"].reindex(trading_idx, method="ffill")
        # Log-price spread: IEF (7-10Y) minus SHY (1-3Y) ≈ slope proxy
        slope_proxy = np.log(ief_c) - np.log(shy_c)
    else:
        # Fallback: TLT vs IEF spread
        log.info("S47: SHY/IEF not available, using TLT as long-only slope")
        slope_proxy = None

    if slope_proxy is not None:
        # Slope momentum: is 3m slope change positive (steepening) or negative (flattening)?
        slope_mom = slope_proxy - slope_proxy.shift(SLOPE_LOOKBACK)
        # Steepening → risk-on (hold SPY); flattening/inverted → risk-off (hold TLT)
        risk_on = (slope_mom > 0).shift(1).infer_objects(copy=False).fillna(True).astype(bool)
    else:
        # Without slope data, just hold SPY
        risk_on = pd.Series(True, index=trading_idx)

    # Monthly signal: only change at month starts
    month_mask = pd.Series(False, index=trading_idx)
    month_mask[trading_idx.month != pd.Series(trading_idx.month).shift(1).fillna(-1).values] = True

    regime = pd.Series(True, index=trading_idx)  # default risk-on
    current = True
    for t in trading_idx:
        if month_mask[t]:
            current = bool(risk_on[t])
        regime[t] = current

    port_ret = pd.Series(0.0, index=trading_idx)
    port_ret[regime]  = spy_r[regime]
    port_ret[~regime] = tlt_r[~regime]

    to = regime.astype(float).diff().abs().fillna(0.0)
    net_ret = apply_costs(port_ret, to, cost_bps, slip_bps)

    bm = spy_r.copy()
    risk_on_pct = float(regime.mean()) * 100
    ann_to = float(to.sum() / max(len(to) / TRADING_DAYS, 1))
    log.info("S47 done. Risk-on (SPY): %.1f%% of months, ann turnover: %.2fx", risk_on_pct, ann_to)

    return {
        "returns": net_ret, "benchmark": bm,
        "description": DESCRIPTION, "turnover_annual": ann_to,
    }
