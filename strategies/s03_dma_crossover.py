"""
S03 — Dual moving-average crossover with optional volatility filter
Universe:  SPY + 11 SPDR sector ETFs + selected futures
Signal:    Long when fast-DMA > slow-DMA; optionally also require 20d realised vol
           to be below its trailing 1-year median.
Execution: Daily signal at close; execute at next open (approximated as next close).
Sizing:    Equal-weight across long positions (fractional sizing, no shorting).
Sweep:     (fast,slow) in [(10,50),(20,100),(50,200)]; with/without vol filter.
"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import logging
import numpy as np
import pandas as pd

from data import load_price_series, load_futures_series, ADJ_TOTALRETURN
from engine import apply_costs

log = logging.getLogger(__name__)
DESCRIPTION = "Dual MA crossover with vol filter, SPY + sectors + futures"

SECTORS = ["XLK", "XLF", "XLE", "XLV", "XLI", "XLY", "XLP", "XLB", "XLU"]
FUTURES_SYM = ["ES", "CL", "NG", "DX"]   # verified available in Norgate
BASE_ETF    = ["SPY"]

PARAM_SETS = [(50, 200), (20, 100), (10, 50)]
TRADING_DAYS = 252


def _load_instruments(config, is_smoke):
    cache_dir = config["paths"]["cache_dir"]
    start     = config["backtest"]["start_date"]
    end       = config["backtest"]["end_date"]

    etfs = BASE_ETF + SECTORS
    futs = [] if is_smoke else FUTURES_SYM

    prices = {}
    for sym in etfs:
        df = load_price_series(sym, start=start, end=end,
                               adjustment=ADJ_TOTALRETURN, cache_dir=cache_dir)
        if not df.empty:
            prices[sym] = df["Close"]
    for sym in futs:
        try:
            df = load_futures_series(sym, start=start, end=end, cache_dir=cache_dir)
            if not df.empty:
                prices[sym] = df["Close"]
        except Exception as e:
            log.warning("S03: %s load failed: %s", sym, e)

    log.info("S03: %d instruments loaded", len(prices))
    return prices


def _crossover_signal(close: pd.Series, fast: int, slow: int) -> pd.Series:
    """1 when fast-DMA > slow-DMA, 0 otherwise (shifted 1 day for look-ahead safety)."""
    dma_fast = close.rolling(fast).mean()
    dma_slow = close.rolling(slow).mean()
    signal   = (dma_fast > dma_slow).astype(float)
    return signal.shift(1)  # trade on next day


def _vol_filter(close: pd.Series, vol_window: int = 20, vol_lookback: int = 252) -> pd.Series:
    """1 when 20d realised vol is below its 1-year median, 0 otherwise (shifted 1 day)."""
    rvol  = close.pct_change().rolling(vol_window).std() * np.sqrt(TRADING_DAYS)
    med   = rvol.rolling(vol_lookback).median()
    filt  = (rvol < med).astype(float)
    return filt.shift(1)


def _strategy_returns(prices: dict, fast: int, slow: int, use_vol_filter: bool,
                      trading_idx: pd.DatetimeIndex,
                      cost_bps: float, slip_bps: float) -> tuple[pd.Series, pd.Series]:
    """Compute portfolio return series for one parameter set."""
    signal_df = pd.DataFrame(index=trading_idx)
    for sym, close in prices.items():
        c = close.reindex(trading_idx, method="ffill")
        sig = _crossover_signal(c, fast, slow)
        if use_vol_filter:
            sig = sig * _vol_filter(c)
        signal_df[sym] = sig

    # Equal-weight across active long signals; 0 if nothing active
    n_active = signal_df.sum(axis=1).clip(lower=1e-9)
    weights  = signal_df.div(n_active, axis=0)   # rows sum to 1 when any active

    # Daily returns
    ret_df = pd.DataFrame({s: prices[s].reindex(trading_idx).pct_change(fill_method=None)
                            for s in prices}).reindex(trading_idx).fillna(0.0)

    port_ret = (weights * ret_df).sum(axis=1)

    # Turnover (weight changes day-over-day)
    to = weights.diff().abs().sum(axis=1) / 2.0

    net = apply_costs(port_ret, to, cost_bps, slip_bps)
    return net, to


def run(config: dict) -> dict:
    cfg      = config["backtest"]
    start    = cfg["start_date"]
    end      = cfg["end_date"]
    is_smoke = config.get("smoke", False)

    cost_bps = config["costs"]["equity_cost_bps"]
    slip_bps = config["costs"]["equity_slippage_bps"]

    prices       = _load_instruments(config, is_smoke)
    trading_idx  = pd.bdate_range(start, end)

    # Primary: (50,200) with vol filter
    fast, slow = PARAM_SETS[0]
    net_ret, to = _strategy_returns(prices, fast, slow, use_vol_filter=True,
                                    trading_idx=trading_idx,
                                    cost_bps=cost_bps, slip_bps=slip_bps)

    spy = load_price_series("SPY", start, end, ADJ_TOTALRETURN, config["paths"]["cache_dir"])
    bm  = spy["Close"].pct_change().reindex(net_ret.index)

    ann_to = float(to.sum() / max(len(to) / TRADING_DAYS, 1))
    log.info("S03 done (fast=%d, slow=%d, vol_filter=True). Ann turnover: %.2fx",
             fast, slow, ann_to)

    return {
        "returns":         net_ret,
        "benchmark":       bm,
        "description":     DESCRIPTION,
        "turnover_annual": ann_to,
    }
