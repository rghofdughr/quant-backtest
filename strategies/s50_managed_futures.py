"""
S50 — Managed futures (diversified trend-following)
Universe:  Verified available futures: ES, CL, NG, DX, EMD, HG, ZS, SB
           + ETF proxies for missing markets: QQQ (NQ), TLT (bonds), GLD, SLV, EEM, DBC
Signal:    Ensemble of 3 time-series momentum lookbacks (1m/3m/12m), equal-weighted.
           Each instrument: sign(trend) × vol-targeted position, sum normalized to 1.
Execution: Monthly rebalance (first trading day of month).
Sizing:    Target portfolio vol ~10%; each market's position ∝ signal / realized_vol.
"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import logging
import numpy as np
import pandas as pd

from data import load_price_series, load_futures_series, ADJ_TOTALRETURN
from engine import apply_costs

log = logging.getLogger(__name__)
DESCRIPTION = "Managed futures: diversified trend ensemble, vol-targeted, monthly rebalance"
TRADING_DAYS = 252
PORT_VOL_TARGET = 0.10   # 10% annual target
MAX_LEV_PER_MKT = 1.5   # cap individual position

FUTURES_SYMS = ["ES", "CL", "NG", "DX", "EMD", "HG", "ZS", "SB"]
ETF_SYMS     = ["QQQ", "TLT", "GLD", "SLV", "EEM", "DBC"]
LOOKBACKS    = [21, 63, 252]   # 1m / 3m / 12m in trading days


def _load_all(config: dict) -> dict:
    """Load price series for futures and ETF proxies. Returns {sym: close_series}."""
    cfg   = config["backtest"]
    start = cfg["start_date"]
    end   = cfg["end_date"]
    cache = config["paths"]["cache_dir"]
    out   = {}
    for sym in FUTURES_SYMS:
        df = load_futures_series(sym, start=start, end=end, cache_dir=cache)
        if not df.empty:
            out[sym] = df["Close"]
        else:
            log.debug("S50: futures %s not available, skipping", sym)
    for sym in ETF_SYMS:
        df = load_price_series(sym, start=start, end=end,
                               adjustment=ADJ_TOTALRETURN, cache_dir=cache)
        if not df.empty:
            out[sym] = df["Close"]
        else:
            log.debug("S50: ETF %s not available, skipping", sym)
    return out


def run(config: dict) -> dict:
    cfg      = config["backtest"]
    start    = cfg["start_date"]
    end      = cfg["end_date"]
    cost_bps = config["costs"]["equity_cost_bps"]
    slip_bps = config["costs"]["equity_slippage_bps"]

    close_map = _load_all(config)
    if len(close_map) < 3:
        log.error("S50: insufficient markets (%d available)", len(close_map))
        return {"returns": pd.Series(dtype=float), "benchmark": pd.Series(dtype=float),
                "description": DESCRIPTION, "turnover_annual": 0.0}

    trading_idx = pd.bdate_range(start, end)
    close_df    = pd.DataFrame(close_map).reindex(trading_idx, method="ffill")
    ret_df      = close_df.pct_change(fill_method=None).fillna(0.0)

    # Realized vol (21-day) per market for position sizing
    rvol_df = ret_df.rolling(21).std() * np.sqrt(TRADING_DAYS)
    rvol_df = rvol_df.replace(0, np.nan).ffill().fillna(0.02)

    # Ensemble signal: average across lookbacks, then take sign
    n = len(trading_idx)
    signal_df = pd.DataFrame(0.0, index=trading_idx, columns=close_df.columns)
    for lb in LOOKBACKS:
        mom = close_df.pct_change(lb, fill_method=None)
        # +1 if positive, -1 if negative, 0 if NaN
        signal_df += np.sign(mom.fillna(0.0)) / len(LOOKBACKS)

    # Shift by 1 day for execution lag
    signal_df = signal_df.shift(1).fillna(0.0)

    # Vol-targeted weights: w_i = signal_i × (port_vol / rvol_i) / n_markets
    # Then normalize so portfolio vol ~ PORT_VOL_TARGET
    n_mkts = close_df.shape[1]
    raw_wts = signal_df * (PORT_VOL_TARGET / rvol_df) / np.sqrt(n_mkts)
    raw_wts = raw_wts.clip(-MAX_LEV_PER_MKT, MAX_LEV_PER_MKT)

    # Monthly rebalance: only allow weight changes on first business day of each month
    month_starts = trading_idx[
        (trading_idx.month != pd.DatetimeIndex(
            [trading_idx[max(i-1,0)] for i in range(len(trading_idx))]
        ).month)
    ]
    # Build scheduled weight matrix (hold through month)
    sched_wts = pd.DataFrame(0.0, index=trading_idx, columns=close_df.columns)
    prev_end = 0
    for i, ms in enumerate(month_starts):
        ms_loc = trading_idx.get_loc(ms)
        if i == 0:
            sched_wts.iloc[0:ms_loc] = 0.0
        next_ms = month_starts[i+1] if i+1 < len(month_starts) else trading_idx[-1]
        next_loc = trading_idx.get_loc(next_ms) if next_ms in trading_idx else len(trading_idx)
        # Use signal available at month start for the whole month
        w = raw_wts.iloc[ms_loc].values
        sched_wts.iloc[ms_loc:next_loc] = w

    # Portfolio return
    port_ret = (sched_wts * ret_df).sum(axis=1)

    # Turnover: sum of |weight changes| at rebalances
    wt_diff = sched_wts.diff().abs().sum(axis=1).fillna(0.0)
    net_ret  = apply_costs(port_ret, wt_diff, cost_bps, slip_bps)

    # Benchmark: equal-weight buy-hold of all markets
    bm = ret_df.mean(axis=1)

    ann_to  = float(wt_diff.sum() / max(len(trading_idx) / TRADING_DAYS, 1))
    n_avail = len(close_map)
    log.info("S50 done. %d markets | ann turnover: %.2fx", n_avail, ann_to)

    return {
        "returns": net_ret, "benchmark": bm,
        "description": DESCRIPTION, "turnover_annual": ann_to,
    }
