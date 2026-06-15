"""
S02 — Time-series (absolute) momentum across futures + ETFs (Moskowitz-Ooi-Pedersen 2012)
Universe:  20 liquid futures + ETFs (equities, rates, FX, commodities)
Signal:    If trailing lookback return > 0 → long; else flat (or short)
Sizing:    Vol-target each instrument to 10% ann, then equal-risk-weight portfolio
Rebalance: Monthly (month-end business day)
Execution: Next trading day close
Sweep:     lookbacks [63=3mo, 126=6mo, 252=12mo], long-flat vs long-short
"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import logging
import numpy as np
import pandas as pd

from data import load_price_series, load_futures_series, ADJ_TOTALRETURN
from engine import apply_costs

log = logging.getLogger(__name__)
DESCRIPTION = "Time-series (absolute) momentum, futures + ETFs, vol-targeted"

# Native Norgate continuous futures (verified available)
FUTURES = [
    "ES",   # S&P 500
    "CL",   # Crude Oil (WTI)
    "NG",   # Natural Gas
    "DX",   # US Dollar Index
    "EMD",  # S&P MidCap 400
    "HG",   # Copper
    "ZS",   # Soybeans
    "SB",   # Sugar
]
# ETF proxies for markets not in Norgate futures subscription
ETFS = [
    "QQQ",  # Nasdaq 100 (proxy for NQ)
    "IWM",  # Russell 2000 (proxy for RTY)
    "TLT",  # 20yr Treasury (proxy for ZB)
    "IEF",  # 7-10yr Treasury (proxy for ZN)
    "GLD",  # Gold (proxy for GC)
    "SLV",  # Silver (supplements SI)
    "FXE",  # Euro (proxy for 6E)
    "EEM",  # Emerging markets
    "DBC",  # Broad commodities
    "CORN", # Corn (proxy for ZC)
]

VOL_TARGET_PER_ASSET = 0.10   # annualized vol target per asset
TRADING_DAYS = 252


def _load_all(config, is_smoke) -> dict[str, pd.DataFrame]:
    cache_dir = config["paths"]["cache_dir"]
    start     = config["backtest"]["start_date"]
    end       = config["backtest"]["end_date"]

    symbols = FUTURES + ETFS
    if is_smoke:
        symbols = ["ES", "CL", "DX", "GLD", "TLT", "SPY", "IEF", "EEM"]

    data = {}
    for sym in symbols:
        try:
            if sym.startswith("&"):
                df = load_futures_series(sym, start=start, end=end, cache_dir=cache_dir)
            else:
                df = load_price_series(sym, start=start, end=end,
                                       adjustment=ADJ_TOTALRETURN, cache_dir=cache_dir)
            if not df.empty and "Close" in df.columns and len(df) > 100:
                data[sym] = df
        except Exception as e:
            log.warning("S02: could not load %s: %s", sym, e)

    log.info("S02: loaded %d instruments", len(data))
    return data


def run(config: dict) -> dict:
    cfg      = config["backtest"]
    start    = cfg["start_date"]
    end      = cfg["end_date"]
    is_smoke = config.get("smoke", False)

    s02_cfg   = config.get("strategies", {}).get("s02", {})
    lookbacks = s02_cfg.get("lookbacks", [252])
    long_only = not s02_cfg.get("long_short", False)
    vol_tgt   = s02_cfg.get("vol_target", VOL_TARGET_PER_ASSET)

    cost_bps = config["costs"]["equity_cost_bps"]
    slip_bps = config["costs"]["equity_slippage_bps"]

    data = _load_all(config, is_smoke)
    if not data:
        return {"returns": pd.Series(dtype=float), "benchmark": pd.Series(dtype=float),
                "description": DESCRIPTION, "turnover_annual": 0.0}

    # Wide close and return DataFrames
    close_df = pd.DataFrame({s: df["Close"] for s, df in data.items()}).sort_index()
    ret_df   = close_df.pct_change(fill_method=None)

    trading_idx = pd.bdate_range(start, end)
    close_df    = close_df.reindex(trading_idx, method="ffill")
    ret_df      = ret_df.reindex(trading_idx).fillna(0.0)

    reb_dates = pd.date_range(start, end, freq="BME")

    lookback = lookbacks[0]
    vol_look = min(63, lookback)

    port_ret = pd.Series(0.0, index=trading_idx)
    to_ser   = pd.Series(0.0, index=trading_idx)

    prev_positions: dict[str, float] = {}

    for ri, reb_d in enumerate(reb_dates):
        # Find last close ≤ reb_d
        avail = close_df.index[close_df.index <= reb_d]
        if avail.empty:
            continue
        snap = avail[-1]
        row  = close_df.index.get_loc(snap)

        if row < lookback:
            continue

        # Determine holding period (next trading day → day before next rebalance execution)
        future = trading_idx[trading_idx > reb_d]
        if future.empty:
            break
        hold_start = future[0]

        if ri + 1 < len(reb_dates):
            nf = trading_idx[trading_idx > reb_dates[ri + 1]]
            hold_end = nf[0] if not nf.empty else trading_idx[-1]
        else:
            hold_end = trading_idx[-1]

        hold_mask = (trading_idx >= hold_start) & (trading_idx <= hold_end)

        # Signal: trailing-lookback return for each instrument
        past_close = close_df.iloc[row - lookback]
        curr_close = close_df.iloc[row]
        ts_signal  = ((curr_close / past_close) - 1.0).dropna()

        # Vol estimate (annualized realized vol over vol_look days)
        vol_window = ret_df.iloc[max(0, row - vol_look) : row + 1]
        realized_vol = vol_window.std() * np.sqrt(TRADING_DAYS)
        realized_vol = realized_vol.replace(0, np.nan)

        # Position = direction × (vol_target / realized_vol), capped at 1.5
        positions: dict[str, float] = {}
        for sym in ts_signal.index:
            if sym not in realized_vol.index or np.isnan(realized_vol[sym]):
                continue
            direction = 1.0 if ts_signal[sym] > 0 else (0.0 if long_only else -1.0)
            pos = direction * (vol_tgt / realized_vol[sym])
            pos = max(-1.5, min(1.5, pos))  # cap individual leverage
            if pos != 0.0:
                positions[sym] = pos

        # Normalise to target portfolio vol (approx equal risk, sum of abs weights ≤ leverage_cap)
        n_active = len(positions)
        if n_active == 0:
            prev_positions = {}
            continue
        raw_sum = sum(abs(v) for v in positions.values())
        lev_cap = config.get("sizing", {}).get("leverage_cap", 2.0)
        if raw_sum > lev_cap:
            scale = lev_cap / raw_sum
            positions = {s: v * scale for s, v in positions.items()}

        # Turnover
        all_s = set(list(positions.keys()) + list(prev_positions.keys()))
        to = sum(abs(positions.get(s, 0.0) - prev_positions.get(s, 0.0)) for s in all_s) / 2.0
        if hold_start in to_ser.index:
            to_ser[hold_start] += to

        # Portfolio return during holding period
        syms = list(positions.keys())
        wts  = np.array([positions[s] for s in syms])
        avail_syms = [s for s in syms if s in ret_df.columns]
        wts_avail  = np.array([positions[s] for s in avail_syms])

        hold_ret = ret_df.loc[hold_mask, avail_syms]
        port_ret[hold_mask] += (hold_ret * wts_avail).sum(axis=1).values

        prev_positions = dict(positions)

    net_ret = apply_costs(port_ret, to_ser, cost_bps, slip_bps)

    spy = load_price_series("SPY", start, end, ADJ_TOTALRETURN, config["paths"]["cache_dir"])
    bm  = spy["Close"].pct_change().reindex(net_ret.index)

    ann_to = float(to_ser.sum() / max(len(to_ser) / TRADING_DAYS, 1))
    log.info("S02 done. Ann turnover: %.2fx", ann_to)

    return {
        "returns":         net_ret,
        "benchmark":       bm,
        "description":     DESCRIPTION,
        "turnover_annual": ann_to,
    }
