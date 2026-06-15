"""
S59 — Vol-of-vol SPY timing overlay
Universe:  SPY only
Signal:    Compute 21-day realized vol of SPY daily returns, then compute
           63-day rolling std of that vol series (= vol-of-vol).
           Scale SPY exposure inversely: target = min(1.5, vol_target / vov).
           When vov is above its own 252-day rolling 90th percentile, go to 0.
Execution: Daily signal (1-day lag). Rebalance daily.
Benchmark: SPY.
Compare to S31 (plain vol targeting).
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
DESCRIPTION = "Vol-of-vol SPY timing: scale exposure inversely to vov; 0 when vov > 90th pct"


def run(config: dict) -> dict:
    cfg = config["backtest"]
    start = cfg["start_date"]
    end = cfg["end_date"]
    cache = config["paths"]["cache_dir"]

    s59_cfg = config.get("strategies", {}).get("s59", {})
    # Optional override for vol_target
    vol_target_override = s59_cfg.get("vol_target", None)

    cost_bps = config["costs"]["equity_cost_bps"]
    slip_bps = config["costs"]["equity_slippage_bps"]

    # Load SPY from well before start for warmup (252 + 63 + 21 ≈ 336 days)
    load_start = "1993-01-01"
    spy = load_price_series("SPY", start=load_start, end=end,
                            adjustment=ADJ_TOTALRETURN, cache_dir=cache)
    if spy.empty:
        empty = pd.Series(dtype=float)
        return {"returns": empty, "benchmark": empty,
                "description": DESCRIPTION, "turnover_annual": 0.0}

    full_idx = pd.bdate_range(load_start, end)
    close = spy["Close"].reindex(full_idx, method="ffill")
    ret_full = close.pct_change(fill_method=None).fillna(0.0)

    # Step 1: 21-day realized vol (annualized)
    rvol = ret_full.rolling(21).std() * np.sqrt(TRADING_DAYS)

    # Step 2: vol-of-vol = 63-day rolling std of rvol
    vov = rvol.rolling(63).std()

    # Determine vol_target for the exposure formula
    # Use median of vov over full history (pre-computed from available data)
    if vol_target_override is not None:
        vol_target = float(vol_target_override)
    else:
        # Median of vov over the full available series (use all history up to end)
        vol_target = float(vov.median())

    # 252-day rolling 90th percentile of vov (for stress cutoff)
    vov_p90 = vov.rolling(252).quantile(0.90)

    # Compute raw exposure (un-lagged), then lag by 1 day
    raw_exposure = (vol_target / vov.replace(0, np.nan)).clip(upper=1.5).fillna(0.0)
    # Zero out when vov > its own 252-day 90th pct (stress regime)
    stress_mask = (vov > vov_p90).fillna(False)
    raw_exposure[stress_mask] = 0.0

    # Apply 1-day lag: today's exposure is set from yesterday's signal
    exposure = raw_exposure.shift(1).fillna(0.0)

    # Restrict to backtest window
    trading_idx = pd.bdate_range(start, end)
    exposure_bt = exposure.reindex(trading_idx, method="ffill").fillna(0.0)
    ret_bt = ret_full.reindex(trading_idx, method="ffill").fillna(0.0)

    port_ret = exposure_bt * ret_bt

    # Turnover = abs daily change in exposure (fraction of portfolio)
    to = exposure_bt.diff().abs().fillna(0.0)

    net_ret = apply_costs(port_ret, to, cost_bps, slip_bps)

    bm = ret_bt.copy()
    ann_to = float(to.sum() / max(len(trading_idx) / TRADING_DAYS, 1))

    log.info(
        "S59 done. vol_target=%.4f, avg_exposure=%.2fx, stress_days=%d, ann_turnover=%.2fx",
        vol_target,
        float(exposure_bt.mean()),
        int(stress_mask.reindex(trading_idx, method="ffill").fillna(False).sum()),
        ann_to,
    )

    return {
        "returns": net_ret,
        "benchmark": bm,
        "description": DESCRIPTION,
        "turnover_annual": ann_to,
    }
