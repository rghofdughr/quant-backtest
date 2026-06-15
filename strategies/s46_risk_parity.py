"""
S46 — Risk parity vs 60/40 benchmark
Universe:  SPY, TLT, GLD, DBC, VNQ (or simple SPY/AGG pair)
Signal:    Inverse-volatility weights so each asset contributes equally to portfolio risk.
           Optional: full risk-parity via correlation matrix (Cholesky).
Rebalance: Monthly
Execution: Next trading day close
Compare:   vs 60/40 (SPY/TLT) and equal-weight.
"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import logging
import numpy as np
import pandas as pd

from data import load_price_series, ADJ_TOTALRETURN
from engine import portfolio_returns_from_weights, apply_costs

log = logging.getLogger(__name__)
DESCRIPTION = "Risk parity across SPY/TLT/GLD/DBC/VNQ, inverse-vol weighted, monthly rebalance"
TRADING_DAYS = 252

DEFAULT_ASSETS = ["SPY", "TLT", "GLD", "DBC", "VNQ"]


def _inv_vol_weights(returns_df: pd.DataFrame, lookback: int = 63) -> dict:
    """Inverse-vol weights: w_i = (1/vol_i) / sum(1/vol_j)."""
    vols = returns_df.iloc[-lookback:].std() * np.sqrt(TRADING_DAYS)
    inv_vol = (1.0 / vols.replace(0, np.nan)).dropna()
    if inv_vol.empty:
        return {}
    total = inv_vol.sum()
    return {s: float(w / total) for s, w in inv_vol.items()}


def _risk_parity_weights(returns_df: pd.DataFrame, lookback: int = 63) -> dict:
    """
    Full risk-parity via equal risk contribution (ERC).
    Iterative solution (Newton-Raphson) to find w such that
    w_i × (Σw)_i = w_j × (Σw)_j for all i,j.
    Falls back to inverse-vol on failure.
    """
    Σ = returns_df.iloc[-lookback:].cov().values * TRADING_DAYS
    n = len(Σ)
    if n < 2 or np.any(np.isnan(Σ)):
        return _inv_vol_weights(returns_df, lookback)

    w = np.ones(n) / n
    for _ in range(500):
        Σw = Σ @ w
        port_vol = np.sqrt(w @ Σw)
        rc = w * Σw / port_vol          # risk contribution per asset
        rc_target = port_vol / n        # equal target
        grad = 2 * (rc - rc_target)
        w -= 0.01 * grad
        w = np.maximum(w, 1e-6)
        w /= w.sum()
        if np.max(np.abs(rc - rc_target)) < 1e-6:
            break

    syms = returns_df.columns.tolist()
    return {s: float(w[i]) for i, s in enumerate(syms)}


def run(config: dict) -> dict:
    cfg      = config["backtest"]
    start    = cfg["start_date"]
    end      = cfg["end_date"]
    cache    = config["paths"]["cache_dir"]

    assets     = config.get("strategies", {}).get("s46", {}).get("assets", DEFAULT_ASSETS)
    use_erc    = config.get("strategies", {}).get("s46", {}).get("full_risk_parity", False)
    vol_look   = 63

    cost_bps = config["costs"]["equity_cost_bps"]
    slip_bps = config["costs"]["equity_slippage_bps"]

    prices: dict[str, pd.Series] = {}
    for sym in assets:
        df = load_price_series(sym, start=start, end=end,
                               adjustment=ADJ_TOTALRETURN, cache_dir=cache)
        if not df.empty and "Close" in df.columns:
            prices[sym] = df["Close"]

    log.info("S46: loaded %d assets: %s", len(prices), list(prices.keys()))

    trading_idx = pd.bdate_range(start, end)
    close_df  = pd.DataFrame(prices).reindex(trading_idx, method="ffill")
    ret_df    = close_df.pct_change(fill_method=None).fillna(0.0)

    reb_dates = pd.date_range(start, end, freq="BME")
    weight_schedule: dict = {}

    for d in reb_dates:
        pos = trading_idx.searchsorted(d, side="right") - 1
        if pos < vol_look:
            continue
        ret_window = ret_df.iloc[max(0, pos - vol_look) : pos + 1]
        avail = ret_window.dropna(axis=1, how="all").columns.tolist()
        if len(avail) < 2:
            continue
        ret_w = ret_window[avail]

        if use_erc:
            w = _risk_parity_weights(ret_w, vol_look)
        else:
            w = _inv_vol_weights(ret_w, vol_look)

        if w:
            weight_schedule[d] = w

    gross_ret, to = portfolio_returns_from_weights(weight_schedule, close_df, start, end)
    net_ret = apply_costs(gross_ret, to, cost_bps, slip_bps)

    spy = load_price_series("SPY", start, end, ADJ_TOTALRETURN, cache)
    bm  = spy["Close"].pct_change(fill_method=None).reindex(net_ret.index)

    ann_to = float(to.sum() / max(len(to) / TRADING_DAYS, 1))
    log.info("S46 done (ERC=%s). Ann turnover: %.2fx", use_erc, ann_to)

    return {
        "returns": net_ret, "benchmark": bm,
        "description": DESCRIPTION, "turnover_annual": ann_to,
    }
