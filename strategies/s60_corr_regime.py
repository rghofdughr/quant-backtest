"""
S60 — Correlation regime switch
Universe:  SPY exposure gated by average pairwise correlation of 11 SPDR sector ETFs.
Signal:    Rolling 63-day average pairwise corr across C(11,2)=55 pairs.
           Z-score vs trailing 252-day mean/std. When z > threshold → go to 0 exposure.
           Otherwise full SPY exposure.
Lead-lag:  Signal uses TODAY's correlation to set TOMORROW's exposure (1-day lag).
Benchmark: SPY.
"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import logging
import numpy as np
import pandas as pd
from itertools import combinations
from data import load_price_series, ADJ_TOTALRETURN
from engine import apply_costs

log = logging.getLogger(__name__)
TRADING_DAYS = 252
DESCRIPTION = "Correlation regime: full SPY when avg sector corr z-score <= threshold, else cash"

SECTORS = ["XLK", "XLF", "XLE", "XLI", "XLC", "XLY", "XLP", "XLV", "XLRE", "XLB", "XLU"]
PAIRS = list(combinations(range(len(SECTORS)), 2))  # 55 pairs


def _avg_pairwise_corr(ret_window: pd.DataFrame) -> float:
    """Compute average pairwise correlation across all pairs (ignoring NaN pairs)."""
    vals = []
    mat = ret_window.values  # shape (T, n_sectors)
    for i, j in PAIRS:
        col_i = mat[:, i]
        col_j = mat[:, j]
        # Only use rows where both are finite
        mask = np.isfinite(col_i) & np.isfinite(col_j)
        if mask.sum() < 20:
            continue
        c_i = col_i[mask]
        c_j = col_j[mask]
        std_i = np.std(c_i, ddof=1)
        std_j = np.std(c_j, ddof=1)
        if std_i < 1e-12 or std_j < 1e-12:
            continue
        corr = np.dot(c_i - c_i.mean(), c_j - c_j.mean()) / ((len(c_i) - 1) * std_i * std_j)
        vals.append(float(np.clip(corr, -1.0, 1.0)))
    return float(np.mean(vals)) if vals else np.nan


def run(config: dict) -> dict:
    cfg = config["backtest"]
    start = cfg["start_date"]
    end = cfg["end_date"]
    cache = config["paths"]["cache_dir"]

    s60_cfg = config.get("strategies", {}).get("s60", {})
    z_threshold = float(s60_cfg.get("z_threshold", 1.5))
    corr_window = int(s60_cfg.get("corr_window", 63))
    zscore_window = int(s60_cfg.get("zscore_window", 252))

    cost_bps = config["costs"]["equity_cost_bps"]
    slip_bps = config["costs"]["equity_slippage_bps"]

    # Load sector ETFs from well before start for warmup
    load_start = "1998-01-01"
    sector_close = {}
    for sym in SECTORS:
        df = load_price_series(sym, start=load_start, end=end,
                               adjustment=ADJ_TOTALRETURN, cache_dir=cache)
        if not df.empty and "Close" in df.columns:
            sector_close[sym] = df["Close"]

    if len(sector_close) < 5:
        log.warning("S60: only %d sector ETFs loaded (need ≥5)", len(sector_close))
        empty = pd.Series(dtype=float)
        return {"returns": empty, "benchmark": empty,
                "description": DESCRIPTION, "turnover_annual": 0.0}

    full_idx = pd.bdate_range(load_start, end)
    sector_df = pd.DataFrame(sector_close).reindex(full_idx, method="ffill")
    ret_df = sector_df.pct_change(fill_method=None)

    n = len(full_idx)
    avg_corr_series = pd.Series(np.nan, index=full_idx)

    # Vectorized rolling average pairwise correlation using pandas corr
    # For each day d, compute corr on the window [d-corr_window+1 : d+1]
    # Use a rolling approach: compute rolling corr matrix per pair
    # For efficiency, compute each pair's rolling correlation and average
    log.info("S60: computing rolling pairwise correlations (%d pairs) ...", len(PAIRS))

    pair_corr_dfs = []
    sector_list = list(sector_df.columns)
    for i, j in PAIRS:
        sym_i = sector_list[i] if i < len(sector_list) else None
        sym_j = sector_list[j] if j < len(sector_list) else None
        if sym_i is None or sym_j is None:
            continue
        if sym_i not in ret_df.columns or sym_j not in ret_df.columns:
            continue
        rc = ret_df[sym_i].rolling(corr_window, min_periods=max(20, corr_window // 2)).corr(ret_df[sym_j])
        pair_corr_dfs.append(rc)

    if not pair_corr_dfs:
        empty = pd.Series(dtype=float)
        return {"returns": empty, "benchmark": empty,
                "description": DESCRIPTION, "turnover_annual": 0.0}

    corr_stack = pd.concat(pair_corr_dfs, axis=1)
    avg_corr_series = corr_stack.mean(axis=1)

    # Z-score of avg_corr vs trailing zscore_window mean/std
    rolling_mean = avg_corr_series.rolling(zscore_window, min_periods=zscore_window // 2).mean()
    rolling_std = avg_corr_series.rolling(zscore_window, min_periods=zscore_window // 2).std()
    zscore = (avg_corr_series - rolling_mean) / rolling_std.replace(0, np.nan)

    # Signal: 1 = full SPY exposure, 0 = cash
    # 1-day lag: today's zscore sets tomorrow's exposure
    raw_exposure = (zscore <= z_threshold).astype(float).fillna(0.0)
    exposure = raw_exposure.shift(1).fillna(0.0)

    # Load SPY for returns
    spy = load_price_series("SPY", start=load_start, end=end,
                            adjustment=ADJ_TOTALRETURN, cache_dir=cache)
    if spy.empty:
        empty = pd.Series(dtype=float)
        return {"returns": empty, "benchmark": empty,
                "description": DESCRIPTION, "turnover_annual": 0.0}

    spy_close = spy["Close"].reindex(full_idx, method="ffill")
    spy_ret = spy_close.pct_change(fill_method=None).fillna(0.0)

    # Restrict to backtest window
    trading_idx = pd.bdate_range(start, end)
    exposure_bt = exposure.reindex(trading_idx, method="ffill").fillna(0.0)
    ret_bt = spy_ret.reindex(trading_idx, method="ffill").fillna(0.0)

    port_ret = exposure_bt * ret_bt

    # Turnover = abs daily change in exposure (binary: 0 or 1)
    to = exposure_bt.diff().abs().fillna(0.0)
    net_ret = apply_costs(port_ret, to, cost_bps, slip_bps)

    bm = ret_bt.copy()
    ann_to = float(to.sum() / max(len(trading_idx) / TRADING_DAYS, 1))

    cash_pct = float((exposure_bt == 0.0).mean())
    log.info(
        "S60 done. z_threshold=%.1f, in-cash=%.1f%%, ann_turnover=%.2fx",
        z_threshold, cash_pct * 100, ann_to,
    )

    return {
        "returns": net_ret,
        "benchmark": bm,
        "description": DESCRIPTION,
        "turnover_annual": ann_to,
    }
