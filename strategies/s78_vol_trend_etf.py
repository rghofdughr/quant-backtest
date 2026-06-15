"""
S78 — Vol-targeted trend following on SPY / QQQ / IWM
Each ETF: long when above 200d MA, size to 10% ann vol target.
When below 200d: that leg goes to cash.
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
DESCRIPTION = "Vol-targeted trend on SPY/QQQ/IWM: long each ETF when above 200d MA, sized to 10% vol target; else flat."
ETF_TICKERS = ["SPY", "QQQ", "IWM"]
VOL_TARGET = 0.10
MA_WIN = 200
VOL_WIN = 60   # blend of 20d and 60d

def run(config):
    cfg = config["backtest"]
    start, end = cfg["start_date"], cfg["end_date"]
    cache = config["paths"]["cache_dir"]
    cost_bps = config["costs"]["equity_cost_bps"]
    slip_bps = config["costs"]["equity_slippage_bps"]
    start_load = "2000-01-01"

    etf_data = {}
    for tk in ETF_TICKERS:
        df = load_price_series(tk, start=start_load, end=end, adjustment=ADJ_TOTALRETURN, cache_dir=cache)
        if not df.empty:
            etf_data[tk] = df["Close"]

    if not etf_data:
        raise RuntimeError("S78: no ETF data loaded")

    trading_idx = pd.bdate_range(start, end)
    price_df = pd.DataFrame(etf_data).reindex(trading_idx, method="ffill")
    ret_df = price_df.pct_change(fill_method=None).fillna(0.0)

    port_rets = pd.Series(0.0, index=trading_idx)
    to_series = pd.Series(0.0, index=trading_idx)
    prev_weights = {tk: 0.0 for tk in ETF_TICKERS}

    for i, dt in enumerate(trading_idx):
        if dt < pd.Timestamp(start):
            continue
        if i < MA_WIN + 5:
            continue

        weights = {}
        for tk in ETF_TICKERS:
            if tk not in price_df.columns:
                weights[tk] = 0.0
                continue
            col = price_df[tk]
            cur = float(col.iloc[i])
            ma200 = float(col.iloc[max(i - MA_WIN, 0):i].mean())
            if not (np.isfinite(cur) and np.isfinite(ma200) and cur > ma200):
                weights[tk] = 0.0
                continue
            # Vol estimate: 20d and 60d blend
            r20 = ret_df[tk].iloc[max(i - 20, 0):i]
            r60 = ret_df[tk].iloc[max(i - VOL_WIN, 0):i]
            v20 = float(r20.std(ddof=1)) * np.sqrt(TRADING_DAYS) if len(r20) > 5 else 0.20
            v60 = float(r60.std(ddof=1)) * np.sqrt(TRADING_DAYS) if len(r60) > 10 else 0.20
            vol = (v20 + v60) / 2.0
            w = min(VOL_TARGET / max(vol, 0.04), 2.0)  # cap at 2x leverage per leg
            weights[tk] = w

        # Normalise so total gross exposure <= 1.5
        total = sum(abs(w) for w in weights.values())
        if total > 1.5:
            scale = 1.5 / total
            weights = {k: v * scale for k, v in weights.items()}

        day_ret = sum(weights.get(tk, 0.0) * ret_df[tk].iloc[i] for tk in ETF_TICKERS if tk in ret_df.columns)
        port_rets.iloc[i] = day_ret

        # Turnover
        to_day = sum(abs(weights.get(tk, 0.0) - prev_weights.get(tk, 0.0)) for tk in ETF_TICKERS) / 2.0
        to_series.iloc[i] = to_day
        prev_weights = dict(weights)

    net_ret = apply_costs(port_rets, to_series, cost_bps, slip_bps)
    spy = load_price_series("SPY", start, end, ADJ_TOTALRETURN, cache)
    bm = spy["Close"].pct_change(fill_method=None).reindex(net_ret.index)
    ann_to = float(to_series.sum() / max(len(to_series) / TRADING_DAYS, 1))
    return {"returns": net_ret, "benchmark": bm, "description": DESCRIPTION, "turnover_annual": ann_to}
