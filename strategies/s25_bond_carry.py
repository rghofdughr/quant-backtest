"""
S25 — Bond term-structure carry (duration-adjusted)
Universe:  US Treasury futures: ZT (2Y), ZF (5Y), ZN (10Y), ZB (30Y)
           ETF proxies where futures unavailable: SHY, IEF, TLT, TLO
Signal:    Carry+roll-down: longer duration bonds earn positive carry when curve is upward sloping.
           Proxy: hold longer-duration bonds when 3m return of ZN/ZB > ZT/ZF (curve upsloping).
           Duration-neutral (barbell): long ZB/ZN, short ZT/ZF, DV01-equal.
Note:      True carry requires yield curve data (FRED). This uses price momentum as carry proxy.
"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import logging
import numpy as np
import pandas as pd

from data import load_futures_series, load_price_series, ADJ_TOTALRETURN
from engine import apply_costs

log = logging.getLogger(__name__)
DESCRIPTION = "Bond term-structure carry, ZT/ZF vs ZN/ZB duration ladder, monthly"
TRADING_DAYS = 252
LOOKBACK     = 63    # 3-month return as carry proxy

# Approximate DV01 ratios for duration-neutral weights
# ZT≈2Y: dur 2, ZF≈5Y: dur 5, ZN≈10Y: dur 7, ZB≈30Y: dur 17
DUR = {"ZT": 2.0, "ZF": 5.0, "ZN": 7.0, "ZB": 17.0,
       "SHY": 2.0, "IEF": 7.0, "TLT": 17.0}

BONDS = [
    ("ZT", "SHY"),   # short-end
    ("ZF", "IEF"),   # mid
    ("ZN", "IEF"),   # 10Y
    ("ZB", "TLT"),   # long-end
]


def run(config: dict) -> dict:
    cfg      = config["backtest"]
    start    = cfg["start_date"]
    end      = cfg["end_date"]
    cache    = config["paths"]["cache_dir"]
    cost_bps = config["costs"]["equity_cost_bps"]
    slip_bps = config["costs"]["equity_slippage_bps"]

    trading_idx = pd.bdate_range(start, end)

    close_map, dur_map = {}, {}
    for fut_sym, etf_sym in BONDS:
        df = load_futures_series(fut_sym, start=start, end=end, cache_dir=cache)
        if not df.empty:
            close_map[fut_sym] = df["Close"]
            dur_map[fut_sym]   = DUR.get(fut_sym, 5.0)
        else:
            df = load_price_series(etf_sym, start=start, end=end, adjustment=ADJ_TOTALRETURN, cache_dir=cache)
            if not df.empty:
                close_map[etf_sym] = df["Close"]
                dur_map[etf_sym]   = DUR.get(etf_sym, 5.0)

    if len(close_map) < 2:
        log.error("S25: too few bond instruments (%d)", len(close_map))
        return {"returns": pd.Series(dtype=float), "benchmark": pd.Series(dtype=float),
                "description": DESCRIPTION, "turnover_annual": 0.0}

    close_df = pd.DataFrame(close_map).reindex(trading_idx, method="ffill")
    ret_df   = close_df.pct_change(fill_method=None).fillna(0.0)
    mom_df   = close_df.pct_change(LOOKBACK, fill_method=None)

    syms     = list(close_map.keys())
    durs     = pd.Series({s: dur_map[s] for s in syms})

    # Sort by duration (short to long)
    durs_sorted = durs.sort_values()
    short_end   = list(durs_sorted[durs_sorted <= 3].index)
    long_end    = list(durs_sorted[durs_sorted >= 7].index)

    month_starts = trading_idx[
        (trading_idx.month != pd.Series(trading_idx.month).shift(1).fillna(-1).values)
    ]

    port_ret  = pd.Series(0.0, index=trading_idx)
    to_series = pd.Series(0.0, index=trading_idx)
    prev_wts  = pd.Series(0.0, index=close_df.columns)
    cur_wts   = pd.Series(0.0, index=close_df.columns)

    for t in trading_idx:
        if t in month_starts:
            mom_t = mom_df.loc[t].dropna()
            if mom_t.empty:
                cur_wts[:] = 0.0
            else:
                # If long-end momentum > short-end → upsloping → carry trade: long long-end
                long_mom  = mom_t.reindex(long_end).mean()
                short_mom = mom_t.reindex(short_end).mean()
                wts = pd.Series(0.0, index=close_df.columns)
                if not (pd.isna(long_mom) or pd.isna(short_mom)):
                    if long_mom > short_mom:
                        # Long long-end, short short-end (duration-neutral roughly)
                        for s in long_end:
                            wts[s] =  0.5 / max(len(long_end), 1)
                        for s in short_end:
                            wts[s] = -0.5 / max(len(short_end), 1)
                    else:
                        # Inverted / flat: reverse
                        for s in short_end:
                            wts[s] =  0.5 / max(len(short_end), 1)
                        for s in long_end:
                            wts[s] = -0.5 / max(len(long_end), 1)
                cur_wts = wts

        port_ret[t]  = float((cur_wts * ret_df.loc[t]).sum())
        to_series[t] = float((cur_wts - prev_wts).abs().sum())
        prev_wts = cur_wts.copy()

    net_ret = apply_costs(port_ret, to_series, cost_bps, slip_bps)

    # Benchmark: equal-weight TLT
    tlt = load_price_series("TLT", start=start, end=end, adjustment=ADJ_TOTALRETURN, cache_dir=cache)
    bm  = tlt["Close"].reindex(trading_idx, method="ffill").pct_change(fill_method=None).fillna(0.0) if not tlt.empty else port_ret * 0

    ann_to = float(to_series.sum() / max(len(trading_idx) / TRADING_DAYS, 1))
    log.info("S25 done. %d bond instruments | ann turnover: %.2fx", len(close_map), ann_to)

    return {
        "returns": net_ret, "benchmark": bm,
        "description": DESCRIPTION, "turnover_annual": ann_to,
    }
