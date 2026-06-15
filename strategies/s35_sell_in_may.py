"""
S35 — Sell in May / Halloween effect (Bouman & Jacobsen 2002)
Universe:  SPY
Signal:    Long SPY from November through April; flat (or cash) May through October.
Execution: Monthly; enter at month-end, exit at month-end.
Variants:  (1) Long-flat (Nov-Apr) vs buy-and-hold; (2) Add sector tilt: overweight
           consumer discretionary Nov-Apr, consumer staples May-Oct.
"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import logging
import numpy as np
import pandas as pd

from data import load_price_series, ADJ_TOTALRETURN
from engine import apply_costs

log = logging.getLogger(__name__)
DESCRIPTION = "Sell-in-May / Halloween effect, SPY long Nov-Apr, flat May-Oct"
TRADING_DAYS = 252

WINTER_MONTHS = {11, 12, 1, 2, 3, 4}   # November through April


def run(config: dict) -> dict:
    cfg      = config["backtest"]
    start    = cfg["start_date"]
    end      = cfg["end_date"]
    cache    = config["paths"]["cache_dir"]

    cost_bps = config["costs"]["equity_cost_bps"]
    slip_bps = config["costs"]["equity_slippage_bps"]

    spy = load_price_series("SPY", start=start, end=end,
                            adjustment=ADJ_TOTALRETURN, cache_dir=cache)
    xlp = load_price_series("XLP", start=start, end=end,
                            adjustment=ADJ_TOTALRETURN, cache_dir=cache)
    xly = load_price_series("XLY", start=start, end=end,
                            adjustment=ADJ_TOTALRETURN, cache_dir=cache)

    trading_idx = pd.bdate_range(start, end)
    spy_ret = spy["Close"].reindex(trading_idx, method="ffill").pct_change(fill_method=None).fillna(0.0)

    # Simple variant: long SPY in winter months, flat otherwise
    in_winter = pd.Series(spy_ret.index.month, index=trading_idx).isin(WINTER_MONTHS)

    port_ret = in_winter.astype(float) * spy_ret
    to       = in_winter.astype(float).diff().abs().fillna(0.0)
    net_ret  = apply_costs(port_ret, to, cost_bps, slip_bps)

    bm = spy_ret.copy()
    ann_to = float(to.sum() / max(len(to) / TRADING_DAYS, 1))

    log.info("S35 done. In-winter: %.1f%% of days, ann turnover: %.2fx",
             float(in_winter.mean()) * 100, ann_to)

    return {
        "returns": net_ret, "benchmark": bm,
        "description": DESCRIPTION, "turnover_annual": ann_to,
    }
