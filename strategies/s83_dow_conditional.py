"""
S83 — Day-of-week conditional: Monday after a down Friday.
Rationale: Weekend risk-aversion overreaction on Monday after a bad Friday close.
Single pre-registered hypothesis only — no scanning over 25 cells.
"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import logging
import numpy as np
import pandas as pd
from data import load_price_series, ADJ_TOTALRETURN
from engine import apply_costs

log = logging.getLogger(__name__)
TRADING_DAYS = 252
DESCRIPTION = "DOW conditional: long SPY on Monday when prior Friday return was negative. Single pre-registered cell."

def run(config):
    cfg = config["backtest"]
    start, end = cfg["start_date"], cfg["end_date"]
    cache = config["paths"]["cache_dir"]
    cost_bps = config["costs"]["equity_cost_bps"]
    slip_bps = config["costs"]["equity_slippage_bps"]
    start_load = "1997-01-01"

    spy_df = load_price_series("SPY", start=start_load, end=end, adjustment=ADJ_TOTALRETURN, cache_dir=cache)
    if spy_df.empty:
        raise RuntimeError("S83: no SPY data")

    trading_idx = pd.bdate_range(start, end)
    price = spy_df["Close"].reindex(trading_idx, method="ffill")
    ret = price.pct_change(fill_method=None).fillna(0.0)

    port_rets = pd.Series(0.0, index=trading_idx)
    to_series = pd.Series(0.0, index=trading_idx)
    prev_pos = 0.0

    for i, dt in enumerate(trading_idx):
        if dt < pd.Timestamp(start):
            continue
        if i < 1:
            continue

        position = 0.0
        # Monday (weekday 0) after a down Friday (prior trading day's return < 0)
        if dt.weekday() == 0:
            prior_ret = float(ret.iloc[i - 1])
            if np.isfinite(prior_ret) and prior_ret < 0:
                position = 1.0

        port_rets.iloc[i] = position * ret.iloc[i]
        to_series.iloc[i] = abs(position - prev_pos) / 2.0
        prev_pos = position

    net_ret = apply_costs(port_rets, to_series, cost_bps, slip_bps)
    spy = load_price_series("SPY", start, end, ADJ_TOTALRETURN, cache)
    bm = spy["Close"].pct_change(fill_method=None).reindex(net_ret.index)
    ann_to = float(to_series.sum() / max(len(to_series) / TRADING_DAYS, 1))

    active_pct = float((port_rets != 0).mean()) * 100
    log.info("S83: active %.1f%% of days (Mon after down Fri)", active_pct)
    return {"returns": net_ret, "benchmark": bm, "description": DESCRIPTION, "turnover_annual": ann_to}
