"""
S63 — Range-expansion Donchian breakout on index ETFs (no futures, no roll contamination)
Universe:  5 index ETFs: SPY, QQQ, IWM, EFA, EEM
Signal:    Long when Close > N-day high (excluding today). Exit when Close < exit_N-day low.
           Entry channel default N=55; exit channel default exit_N=20.
Sizing:    Vol-target each ETF at 10% annual vol. Cap per-instrument weight at 1.5.
           Monthly re-scaling of vol targets; daily signal checks.
Benchmark: SPY.
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
DESCRIPTION = "Donchian breakout (N=55/exit=20) on 5 index ETFs, vol-targeted, long-only"

ETFS = ["SPY", "QQQ", "IWM", "EFA", "EEM"]
VOL_TARGET_PER_ETF = 0.10   # 10% ann vol per instrument
MAX_LEV = 1.5               # cap per-instrument


def _run_breakout(close: pd.Series, high: pd.Series, low: pd.Series,
                  entry_n: int, exit_n: int) -> pd.Series:
    """
    State-machine Donchian breakout. Returns daily position: +1 (long) or 0 (flat).
    Uses shifted channels: compare today's close to yesterday's N-day high/low.
    """
    n = len(close)
    # Shifted so today's close is compared to prior N-day range
    entry_high = high.rolling(entry_n).max().shift(1)
    exit_low = low.rolling(exit_n).min().shift(1)

    positions = np.zeros(n)
    direction = 0  # 0 = flat, 1 = long
    warmup = max(entry_n, exit_n) + 2

    for i in range(warmup, n):
        c = close.iloc[i]
        eh = entry_high.iloc[i]
        xl = exit_low.iloc[i]

        if np.isnan(eh) or np.isnan(xl):
            positions[i] = 0.0
            continue

        if direction == 0:
            # Enter long if close breaks above N-day high
            if c > eh:
                direction = 1
        elif direction == 1:
            # Exit if close drops below exit_N-day low
            if c < xl:
                direction = 0

        positions[i] = float(direction)

    return pd.Series(positions, index=close.index)


def run(config: dict) -> dict:
    cfg = config["backtest"]
    start = cfg["start_date"]
    end = cfg["end_date"]
    cache = config["paths"]["cache_dir"]

    s63_cfg = config.get("strategies", {}).get("s63", {})
    entry_n = int(s63_cfg.get("N", 55))
    exit_n = int(s63_cfg.get("exit_N", 20))
    k = float(s63_cfg.get("k", 0.0))  # breakout filter (0 = pure Donchian)
    vol_look = int(s63_cfg.get("vol_lookback", 63))  # for vol scaling

    cost_bps = config["costs"]["equity_cost_bps"]
    slip_bps = config["costs"]["equity_slippage_bps"]

    # Load from well before start for warmup
    load_start = "1993-01-01"
    ohlcv = {}
    for sym in ETFS:
        df = load_price_series(sym, start=load_start, end=end,
                               adjustment=ADJ_TOTALRETURN, cache_dir=cache)
        if not df.empty and "Close" in df.columns:
            ohlcv[sym] = df

    if not ohlcv:
        empty = pd.Series(dtype=float)
        return {"returns": empty, "benchmark": empty,
                "description": DESCRIPTION, "turnover_annual": 0.0}

    full_idx = pd.bdate_range(load_start, end)
    trading_idx = pd.bdate_range(start, end)

    # Build combined daily signal and weight series
    port_ret_series = pd.Series(0.0, index=full_idx)
    to_series = pd.Series(0.0, index=full_idx)

    weight_dfs = {}
    ret_dfs = {}

    for sym, df in ohlcv.items():
        close = df["Close"].reindex(full_idx, method="ffill")
        # Use High/Low if available; otherwise fall back to Close for channel computation
        if "High" in df.columns and "Low" in df.columns:
            high = df["High"].reindex(full_idx, method="ffill")
            low = df["Low"].reindex(full_idx, method="ffill")
        else:
            high = close
            low = close

        # Adjust entry with k factor: close > (1+k) * N-day high
        if k != 0.0:
            entry_high_adj = high.rolling(entry_n).max().shift(1) * (1.0 + k)
            exit_low_adj = low.rolling(exit_n).min().shift(1)
            n = len(close)
            positions = np.zeros(n)
            direction = 0
            warmup = max(entry_n, exit_n) + 2
            for i in range(warmup, n):
                c = close.iloc[i]
                eh = entry_high_adj.iloc[i]
                xl = exit_low_adj.iloc[i]
                if np.isnan(eh) or np.isnan(xl):
                    positions[i] = 0.0
                    continue
                if direction == 0:
                    if c > eh:
                        direction = 1
                elif direction == 1:
                    if c < xl:
                        direction = 0
                positions[i] = float(direction)
            raw_pos = pd.Series(positions, index=full_idx)
        else:
            raw_pos = _run_breakout(close, high, low, entry_n, exit_n)

        # Vol-targeting: monthly re-scaling
        ret = close.pct_change(fill_method=None).fillna(0.0)
        rvol = ret.rolling(vol_look).std() * np.sqrt(TRADING_DAYS)
        # Vol scale = vol_target / rvol, lagged 1 day, capped at MAX_LEV
        vol_scale = (VOL_TARGET_PER_ETF / rvol.replace(0, np.nan)).clip(upper=MAX_LEV).shift(1).fillna(0.0)

        # Final weight = position (0 or 1) * vol_scale
        weight = raw_pos * vol_scale

        weight_dfs[sym] = weight
        ret_dfs[sym] = ret

    weight_df = pd.DataFrame(weight_dfs).reindex(full_idx).fillna(0.0)
    ret_df = pd.DataFrame(ret_dfs).reindex(full_idx).fillna(0.0)

    port_ret_full = (weight_df * ret_df).sum(axis=1)
    to_full = weight_df.diff().abs().sum(axis=1).fillna(0.0)

    # Restrict to backtest window
    port_ret = port_ret_full.reindex(trading_idx).fillna(0.0)
    to = to_full.reindex(trading_idx).fillna(0.0)

    net_ret = apply_costs(port_ret, to, cost_bps, slip_bps)

    spy = load_price_series("SPY", start, end, ADJ_TOTALRETURN, cache)
    bm = spy["Close"].reindex(trading_idx, method="ffill").pct_change(fill_method=None).fillna(0.0)

    ann_to = float(to.sum() / max(len(trading_idx) / TRADING_DAYS, 1))

    # Log position stats
    avg_lev = float(weight_df.reindex(trading_idx).abs().sum(axis=1).mean())
    n_etfs_in = int((weight_df.reindex(trading_idx) > 0).any().sum())
    log.info(
        "S63 done. entry_N=%d, exit_N=%d, k=%.3f. Avg leverage=%.2fx, ETFs ever long=%d, ann_to=%.2fx",
        entry_n, exit_n, k, avg_lev, n_etfs_in, ann_to,
    )

    return {
        "returns": net_ret,
        "benchmark": bm,
        "description": DESCRIPTION,
        "turnover_annual": ann_to,
    }
