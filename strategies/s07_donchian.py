"""
S07 — Donchian channel breakout (Turtle Trader style)
Universe:  8 verified Norgate continuous futures
Signal:    Long on N-day high breakout; short on N-day low breakout.
           Exit long when price falls below exit-window low; vice versa.
Sizing:    Vol-targeted per market: weight = sign × (vol_target / realized_vol), capped at max_lev.
           Realized vol = 20-day rolling std of daily returns, annualized.
Execution: Signal uses shifted channels (no look-ahead). Next-close approximation.
"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import logging
import numpy as np
import pandas as pd

from data import load_futures_series, load_price_series, ADJ_TOTALRETURN
from engine import apply_costs

log = logging.getLogger(__name__)
DESCRIPTION = "Donchian channel breakout (Turtle), vol-targeted, 8 futures"

FUTURES = ["ES", "CL", "NG", "DX", "EMD", "HG", "ZS", "SB"]
TRADING_DAYS = 252
VOL_TARGET   = 0.10   # 10% annualized portfolio vol target
MAX_LEV      = 1.5    # max weight per market


def _donchian_weights(close: pd.Series, entry_days: int, exit_days: int,
                      vol_target_per_mkt: float, max_lev: float) -> pd.Series:
    """
    State machine: track long/short/flat per market.
    Returns a weight time-series (signed portfolio fraction) for one market.
    """
    n = len(close)
    entry_high = close.rolling(entry_days).max().shift(1)
    entry_low  = close.rolling(entry_days).min().shift(1)
    exit_high  = close.rolling(exit_days).max().shift(1)
    exit_low   = close.rolling(exit_days).min().shift(1)

    # Realized vol for position sizing (annualized daily std, lagged 1 day)
    rvol = close.pct_change(fill_method=None).rolling(20).std() * np.sqrt(TRADING_DAYS)
    rvol = rvol.shift(1)

    weights   = np.zeros(n)
    direction = 0   # +1 long, -1 short, 0 flat
    warmup    = max(entry_days, exit_days, 21)

    for i in range(warmup, n):
        c  = close.iloc[i]
        eh = entry_high.iloc[i]
        el = entry_low.iloc[i]
        xh = exit_high.iloc[i]
        xl = exit_low.iloc[i]
        rv = rvol.iloc[i]

        if np.isnan(eh) or np.isnan(el):
            weights[i] = 0.0
            continue

        # Update direction via state machine
        if direction > 0:
            if c < xl:
                direction = 0
                if c < el:
                    direction = -1
        elif direction < 0:
            if c > xh:
                direction = 0
                if c > eh:
                    direction = 1
        else:
            if c > eh:
                direction = 1
            elif c < el:
                direction = -1

        if direction != 0 and not np.isnan(rv) and rv > 0:
            weights[i] = direction * min(vol_target_per_mkt / rv, max_lev)
        else:
            weights[i] = 0.0

    return pd.Series(weights, index=close.index)


def run(config: dict) -> dict:
    cfg      = config["backtest"]
    start    = cfg["start_date"]
    end      = cfg["end_date"]
    is_smoke = config.get("smoke", False)
    cache    = config["paths"]["cache_dir"]

    s07_cfg  = config.get("strategies", {}).get("s07", {})
    entry    = s07_cfg.get("entry_days", [20])[0]
    exit_d   = s07_cfg.get("exit_days",  [10])[0]

    cost_bps = config["costs"]["equity_cost_bps"]
    slip_bps = config["costs"]["equity_slippage_bps"]

    syms        = FUTURES[:4] if is_smoke else FUTURES
    trading_idx = pd.bdate_range(start, end)
    n_mkts      = len(syms)
    vt_per_mkt  = VOL_TARGET / max(n_mkts, 1)

    weight_map = {}
    ret_map    = {}
    for sym in syms:
        try:
            df = load_futures_series(sym, start=start, end=end, cache_dir=cache)
            if df.empty or len(df) < entry + 30:
                continue
            c = df["Close"].reindex(trading_idx, method="ffill")
            w = _donchian_weights(c, entry, exit_d, vt_per_mkt, MAX_LEV)
            weight_map[sym] = w
            ret_map[sym]    = c.pct_change(fill_method=None).fillna(0.0)
        except Exception as e:
            log.warning("S07: %s failed: %s", sym, e)

    if not weight_map:
        log.error("S07: no instruments loaded")
        return {"returns": pd.Series(dtype=float), "benchmark": pd.Series(dtype=float),
                "description": DESCRIPTION, "turnover_annual": 0.0}

    weight_df = pd.DataFrame(weight_map).reindex(trading_idx).fillna(0.0)
    ret_df    = pd.DataFrame(ret_map).reindex(trading_idx).fillna(0.0)

    port_ret  = (weight_df * ret_df).sum(axis=1)
    to_series = weight_df.diff().abs().sum(axis=1).fillna(0.0)
    net       = apply_costs(port_ret, to_series, cost_bps, slip_bps)

    spy = load_price_series("SPY", start, end, ADJ_TOTALRETURN, cache)
    bm  = spy["Close"].reindex(trading_idx, method="ffill").pct_change(fill_method=None).fillna(0.0) if not spy.empty else net * 0

    ann_to = float(to_series.sum() / max(len(trading_idx) / TRADING_DAYS, 1))
    log.info("S07 done (entry=%d, exit=%d, %d mkts). Ann turnover: %.2fx",
             entry, exit_d, len(weight_map), ann_to)

    return {
        "returns": net, "benchmark": bm,
        "description": DESCRIPTION, "turnover_annual": ann_to,
    }
