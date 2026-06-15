"""
S93 — Defensive sector rotation based on SPY MA regime.
When SPY < 200d MA: equal-weight XLP, XLU, XLV (defensives).
When SPY > 200d MA: equal-weight XLK, XLY, XLF (growth/cyclical).
Always invested; signal is the SPY trend filter.
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
DESCRIPTION = "Defensive rotation: SPY below 200d MA -> XLP/XLU/XLV; above -> XLK/XLY/XLF. Always invested."
MA_WIN = 200
DEFENSIVES = ["XLP", "XLU", "XLV"]
GROWTH = ["XLK", "XLY", "XLF"]

def run(config):
    cfg = config["backtest"]
    start, end = cfg["start_date"], cfg["end_date"]
    cache = config["paths"]["cache_dir"]
    cost_bps = config["costs"]["equity_cost_bps"]
    slip_bps = config["costs"]["equity_slippage_bps"]
    start_load = "1997-01-01"

    tickers = ["SPY"] + DEFENSIVES + GROWTH
    data = {}
    for tk in tickers:
        df = load_price_series(tk, start=start_load, end=end, adjustment=ADJ_TOTALRETURN, cache_dir=cache)
        if not df.empty:
            data[tk] = df["Close"]

    if "SPY" not in data:
        raise RuntimeError("S93: SPY data missing")

    trading_idx = pd.bdate_range(start, end)
    price_df = pd.DataFrame(data).reindex(trading_idx, method="ffill")
    ret_df = price_df.pct_change(fill_method=None).fillna(0.0)

    port_rets = pd.Series(0.0, index=trading_idx)
    to_series = pd.Series(0.0, index=trading_idx)
    all_sector = DEFENSIVES + GROWTH
    prev_weights = {tk: 0.0 for tk in all_sector}

    for i, dt in enumerate(trading_idx):
        if dt < pd.Timestamp(start):
            continue
        if i < MA_WIN + 1:
            continue

        spy_cur = float(price_df["SPY"].iloc[i])
        spy_ma = float(price_df["SPY"].iloc[max(i - MA_WIN, 0):i].mean())

        if not (np.isfinite(spy_cur) and np.isfinite(spy_ma)):
            weights = {tk: 0.0 for tk in all_sector}
        elif spy_cur > spy_ma:
            # Growth regime
            active = [tk for tk in GROWTH if tk in ret_df.columns]
            n = len(active) if active else 1
            weights = {tk: (1.0 / n if tk in active else 0.0) for tk in all_sector}
        else:
            # Defensive regime
            active = [tk for tk in DEFENSIVES if tk in ret_df.columns]
            n = len(active) if active else 1
            weights = {tk: (1.0 / n if tk in active else 0.0) for tk in all_sector}

        day_ret = sum(weights.get(tk, 0.0) * float(ret_df.iloc[i].get(tk, 0.0)) for tk in all_sector)
        port_rets.iloc[i] = day_ret
        to_day = sum(abs(weights.get(tk, 0.0) - prev_weights.get(tk, 0.0)) for tk in all_sector) / 2.0
        to_series.iloc[i] = to_day
        prev_weights = dict(weights)

    net_ret = apply_costs(port_rets, to_series, cost_bps, slip_bps)
    spy = load_price_series("SPY", start, end, ADJ_TOTALRETURN, cache)
    bm = spy["Close"].pct_change(fill_method=None).reindex(net_ret.index)
    ann_to = float(to_series.sum() / max(len(to_series) / TRADING_DAYS, 1))
    return {"returns": net_ret, "benchmark": bm, "description": DESCRIPTION, "turnover_annual": ann_to}
