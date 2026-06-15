"""
S90 — Credit-spread regime switch.
Use HYG/LQD ratio trend as risk-on / risk-off switch for SPY exposure.
Rising HYG vs LQD = credit spreads tightening = risk-on → long SPY.
Falling = spreads widening = risk-off → flat.
HYG inception: 2007-04-11. LQD inception: 2002-07-26.
Report regime count clearly.
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
DESCRIPTION = "Credit-spread regime: long SPY when HYG/LQD ratio above 63d MA (spreads tightening); else flat. From 2007."
RATIO_MA = 63

def run(config):
    cfg = config["backtest"]
    start, end = cfg["start_date"], cfg["end_date"]
    cache = config["paths"]["cache_dir"]
    cost_bps = config["costs"]["equity_cost_bps"]
    slip_bps = config["costs"]["equity_slippage_bps"]
    start_load = "2007-01-01"  # HYG inception

    hyg_df = load_price_series("HYG", start=start_load, end=end, adjustment=ADJ_TOTALRETURN, cache_dir=cache)
    lqd_df = load_price_series("LQD", start=start_load, end=end, adjustment=ADJ_TOTALRETURN, cache_dir=cache)
    spy_df = load_price_series("SPY", start=start_load, end=end, adjustment=ADJ_TOTALRETURN, cache_dir=cache)
    if hyg_df.empty or lqd_df.empty or spy_df.empty:
        raise RuntimeError("S90: HYG/LQD/SPY data missing (HYG inception 2007)")

    trading_idx = pd.bdate_range(start, end)
    hyg = hyg_df["Close"].reindex(trading_idx, method="ffill")
    lqd = lqd_df["Close"].reindex(trading_idx, method="ffill")
    spy_ret = spy_df["Close"].reindex(trading_idx, method="ffill").pct_change(fill_method=None).fillna(0.0)

    ratio = (hyg / lqd).replace([np.inf, -np.inf], np.nan)
    ratio_ma = ratio.rolling(RATIO_MA, min_periods=RATIO_MA // 2).mean()

    port_rets = pd.Series(0.0, index=trading_idx)
    to_series = pd.Series(0.0, index=trading_idx)
    prev_pos = 0.0

    risk_on_count = 0
    total_count = 0

    for i, dt in enumerate(trading_idx):
        if dt < pd.Timestamp(start):
            continue
        if i < RATIO_MA + 1:
            continue

        r = float(ratio.iloc[i])
        ma = float(ratio_ma.iloc[i])
        if not (np.isfinite(r) and np.isfinite(ma)):
            position = 0.0
        else:
            position = 1.0 if r > ma else 0.0

        total_count += 1
        if position > 0:
            risk_on_count += 1

        port_rets.iloc[i] = position * float(spy_ret.iloc[i])
        to_series.iloc[i] = abs(position - prev_pos) / 2.0
        prev_pos = position

    if total_count > 0:
        log.info("S90: risk-on %.1f%% of days", 100 * risk_on_count / total_count)

    net_ret = apply_costs(port_rets, to_series, cost_bps, slip_bps)
    spy = load_price_series("SPY", start, end, ADJ_TOTALRETURN, cache)
    bm = spy["Close"].pct_change(fill_method=None).reindex(net_ret.index)
    ann_to = float(to_series.sum() / max(len(to_series) / TRADING_DAYS, 1))
    return {"returns": net_ret, "benchmark": bm, "description": DESCRIPTION, "turnover_annual": ann_to}
