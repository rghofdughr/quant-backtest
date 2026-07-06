"""
s117 — Volatility Spike Recovery
Signal: When SPY 5d realized vol spikes above 1.5x its 60d realized vol baseline,
go long SPY for the next 20 trading days (mean reversion in volatility -> equity recovery).
Outside spike recovery windows: cash.
"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import logging
import numpy as np
import pandas as pd
from data import load_price_series, ADJ_TOTALRETURN
from engine import apply_costs

log = logging.getLogger(__name__)
DESCRIPTION = "Vol spike recovery: long SPY 20d after realized vol spikes 1.5x its 60d baseline"

SHORT_WIN  = 5
LONG_WIN   = 60
SPIKE_MULT = 1.5
HOLD_DAYS  = 20

def run(config):
    cfg   = config["backtest"]
    start, end = cfg["start_date"], cfg["end_date"]
    cache = config["paths"]["cache_dir"]
    cost_bps = config["costs"]["equity_cost_bps"]
    slip_bps = config["costs"]["equity_slippage_bps"]

    spy = load_price_series("SPY", start=start, end=end, adjustment=ADJ_TOTALRETURN, cache_dir=cache)
    idx = pd.bdate_range(start, end)
    spy_r = spy["Close"].reindex(idx, method="ffill").pct_change(fill_method=None).fillna(0.0)

    # Annualized realized vol
    rvol_short = spy_r.rolling(SHORT_WIN).std() * np.sqrt(252)
    rvol_long  = spy_r.rolling(LONG_WIN).std() * np.sqrt(252)

    # Spike trigger: yesterday's short vol > SPIKE_MULT * yesterday's long vol
    spike_trigger = (rvol_short.shift(1) > SPIKE_MULT * rvol_long.shift(1)).fillna(False)

    # Extend spike signal forward HOLD_DAYS days
    in_market = pd.Series(False, index=idx)
    hold_remaining = 0
    for i in range(len(idx)):
        if spike_trigger.iloc[i]:
            hold_remaining = HOLD_DAYS
        if hold_remaining > 0:
            in_market.iloc[i] = True
            hold_remaining -= 1

    port_ret = pd.Series(0.0, index=idx)
    port_ret[in_market] = spy_r[in_market]

    to = in_market.astype(float).diff().abs().fillna(0.0)
    net_ret = apply_costs(port_ret, to, cost_bps, slip_bps)
    ann_to = float(to.sum() / max(len(to) / 252, 1))
    pct = float(in_market.mean()) * 100
    log.info("s117: in market %.1f%% of days", pct)
    return {"returns": net_ret, "turnover_annual": ann_to, "description": DESCRIPTION}
