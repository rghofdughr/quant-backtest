"""
S84 — FOMC-week calendar drift.
Long SPY across the full FOMC meeting week (Mon through Fri of the week
containing the announcement date). Distinct from S37 (pre-FOMC 24h intraday).
FOMC dates are public/static; no look-ahead.
"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import logging
import numpy as np
import pandas as pd
from data import load_price_series, ADJ_TOTALRETURN
from engine import apply_costs
from calendars import FOMC_DATES

log = logging.getLogger(__name__)
TRADING_DAYS = 252
DESCRIPTION = "FOMC-week calendar: long SPY during the full calendar week of each FOMC announcement. Static dates."

def _fomc_weeks(trading_idx: pd.DatetimeIndex) -> set:
    """Return all trading dates that fall in the same Mon-Sun week as an FOMC date."""
    fomc_ts = set(pd.Timestamp(d) for d in FOMC_DATES)
    fomc_weeks = set()
    for d in fomc_ts:
        # ISO week: Monday=0, Sunday=6
        week_start = d - pd.Timedelta(days=d.weekday())
        week_end = week_start + pd.Timedelta(days=6)
        for td in trading_idx:
            if week_start <= td <= week_end:
                fomc_weeks.add(td)
    return fomc_weeks

def run(config):
    cfg = config["backtest"]
    start, end = cfg["start_date"], cfg["end_date"]
    cache = config["paths"]["cache_dir"]
    cost_bps = config["costs"]["equity_cost_bps"]
    slip_bps = config["costs"]["equity_slippage_bps"]
    start_load = "1997-01-01"

    spy_df = load_price_series("SPY", start=start_load, end=end, adjustment=ADJ_TOTALRETURN, cache_dir=cache)
    if spy_df.empty:
        raise RuntimeError("S84: no SPY data")

    trading_idx = pd.bdate_range(start, end)
    ret = spy_df["Close"].reindex(trading_idx, method="ffill").pct_change(fill_method=None).fillna(0.0)

    fomc_days = _fomc_weeks(trading_idx)
    log.info("S84: %d FOMC-week trading days identified", len(fomc_days))

    port_rets = pd.Series(0.0, index=trading_idx)
    to_series = pd.Series(0.0, index=trading_idx)
    prev_pos = 0.0

    for i, dt in enumerate(trading_idx):
        if dt < pd.Timestamp(start):
            continue
        position = 1.0 if dt in fomc_days else 0.0
        port_rets.iloc[i] = position * ret.iloc[i]
        to_series.iloc[i] = abs(position - prev_pos) / 2.0
        prev_pos = position

    net_ret = apply_costs(port_rets, to_series, cost_bps, slip_bps)
    spy = load_price_series("SPY", start, end, ADJ_TOTALRETURN, cache)
    bm = spy["Close"].pct_change(fill_method=None).reindex(net_ret.index)
    ann_to = float(to_series.sum() / max(len(to_series) / TRADING_DAYS, 1))
    return {"returns": net_ret, "benchmark": bm, "description": DESCRIPTION, "turnover_annual": ann_to}
