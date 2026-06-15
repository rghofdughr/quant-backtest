"""
S31 — Volatility-targeting overlay on SPY
Universe:  SPY only
Signal:    Scale SPY exposure to target annualised vol using trailing 20-day realised vol.
           Leverage = vol_target / realised_vol; capped at 1.5x.
Execution: Daily; rebalance when leverage changes by more than 5%.
Sizing:    Fractional position in SPY; remainder in cash (0%).
Comparison: Risk-adjusted return vs static buy-and-hold SPY.
"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import logging
import numpy as np
import pandas as pd

from data import load_price_series, ADJ_TOTALRETURN
from engine import apply_costs

log = logging.getLogger(__name__)
DESCRIPTION = "Vol-targeting overlay on SPY, 20-day realized vol, target 10%, max 1.5x"
TRADING_DAYS = 252


def run(config: dict) -> dict:
    cfg      = config["backtest"]
    start    = cfg["start_date"]
    end      = cfg["end_date"]
    cache    = config["paths"]["cache_dir"]

    s31_cfg    = config.get("strategies", {}).get("s31", {})
    vol_look   = s31_cfg.get("vol_lookback", 20)
    vol_target = config.get("sizing", {}).get("vol_target", 0.10)
    lev_cap    = s31_cfg.get("leverage_cap", 1.5)

    cost_bps = config["costs"]["equity_cost_bps"]
    slip_bps = config["costs"]["equity_slippage_bps"]

    spy = load_price_series("SPY", start=start, end=end,
                            adjustment=ADJ_TOTALRETURN, cache_dir=cache)
    if spy.empty:
        return {"returns": pd.Series(dtype=float), "benchmark": pd.Series(dtype=float),
                "description": DESCRIPTION, "turnover_annual": 0.0}

    trading_idx = pd.bdate_range(start, end)
    close = spy["Close"].reindex(trading_idx, method="ffill")
    ret   = close.pct_change(fill_method=None).fillna(0.0)

    # Rolling realised vol
    rvol = ret.rolling(vol_look).std() * np.sqrt(TRADING_DAYS)
    rvol = rvol.replace(0, np.nan).fillna(method="bfill")

    # Target leverage (shifted 1 day: compute from yesterday's vol, apply today)
    target_lev = (vol_target / rvol).clip(upper=lev_cap).shift(1).fillna(0.0)

    port_ret = target_lev * ret

    # Turnover: change in leverage position
    to = target_lev.diff().abs().fillna(0.0)

    net_ret = apply_costs(port_ret, to, cost_bps, slip_bps)

    bm = ret.copy()  # SPY buy-and-hold as benchmark

    ann_to = float(to.sum() / max(len(to) / TRADING_DAYS, 1))
    log.info("S31 done. Avg leverage: %.2fx, ann turnover: %.2fx",
             float(target_lev.mean()), ann_to)

    return {
        "returns": net_ret, "benchmark": bm,
        "description": DESCRIPTION, "turnover_annual": ann_to,
    }
