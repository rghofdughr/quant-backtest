"""
S65 — Sector dispersion timing gate on S08-style sector rotation
Mechanics:
1. Compute monthly cross-sectional dispersion = std of 1-month returns across 11 SPDR sector ETFs.
2. When dispersion > rolling median of all prior months → take S08-style positions (top-3 by 3m return).
3. When dispersion <= median → go flat (cash).
Comparison: vs ungated S08.
Note: XLRE started in 2015; handled gracefully (NaN excluded from ranking).
"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import logging
import numpy as np
import pandas as pd
from data import load_price_series, ADJ_TOTALRETURN
from engine import portfolio_returns_from_weights, apply_costs

log = logging.getLogger(__name__)
TRADING_DAYS = 252
DESCRIPTION = "S08 sector rotation with dispersion on/off gate: active when cross-sector dispersion > median"

SECTORS = ["XLK", "XLF", "XLE", "XLI", "XLC", "XLY", "XLP", "XLV", "XLRE", "XLB", "XLU"]

# S08-style parameters (replicated here)
TOP_N = 3
LOOKBACK_MONTHS = 3
ABS_MOM_CASH_FILTER = True   # same default as S08


def run(config: dict) -> dict:
    cfg = config["backtest"]
    start = cfg["start_date"]
    end = cfg["end_date"]
    cache = config["paths"]["cache_dir"]

    s65_cfg = config.get("strategies", {}).get("s65", {})
    top_n = int(s65_cfg.get("top_n", TOP_N))
    lb_months = int(s65_cfg.get("lookback_months", LOOKBACK_MONTHS))
    abs_mom = bool(s65_cfg.get("abs_momentum_cash_filter", ABS_MOM_CASH_FILTER))

    cost_bps = config["costs"]["equity_cost_bps"]
    slip_bps = config["costs"]["equity_slippage_bps"]

    # Load from well before start for warmup
    load_start = "1998-01-01"
    sector_close = {}
    for sym in SECTORS:
        df = load_price_series(sym, start=load_start, end=end,
                               adjustment=ADJ_TOTALRETURN, cache_dir=cache)
        if not df.empty and "Close" in df.columns:
            sector_close[sym] = df["Close"]

    avail_sectors = list(sector_close.keys())
    log.info("S65: loaded %d sector ETFs", len(avail_sectors))

    if len(avail_sectors) < 3:
        empty = pd.Series(dtype=float)
        return {"returns": empty, "benchmark": empty,
                "description": DESCRIPTION, "turnover_annual": 0.0}

    # Build price DataFrame on full history
    full_idx = pd.bdate_range(load_start, end)
    close_full = pd.DataFrame(sector_close).reindex(full_idx, method="ffill")

    # For portfolio_returns_from_weights we need a close_df on the backtest window
    trading_idx = pd.bdate_range(start, end)
    close_df = close_full.reindex(trading_idx, method="ffill")

    lb_days = lb_months * 21  # approximate trading days per month

    reb_dates = pd.date_range(start, end, freq="BME")

    # --- Step 1: Compute monthly dispersion series ---
    # At each month-end, compute std of 1-month returns across available sectors
    dispersion_list = []

    for d in reb_dates:
        avail_d = close_full.index[close_full.index <= d]
        if len(avail_d) < 22:
            continue
        snap_idx = len(avail_d) - 1
        p_now = close_full.iloc[snap_idx]
        p_prev = close_full.iloc[max(0, snap_idx - 21)]
        monthly_rets = (p_now / p_prev - 1.0).dropna()
        if len(monthly_rets) < 3:
            continue
        disp = float(monthly_rets.std())
        dispersion_list.append({"date": d, "dispersion": disp})

    if len(dispersion_list) < 4:
        log.warning("S65: insufficient dispersion history")
        empty = pd.Series(dtype=float)
        return {"returns": empty, "benchmark": empty,
                "description": DESCRIPTION, "turnover_annual": 0.0}

    disp_df = pd.DataFrame(dispersion_list).set_index("date")["dispersion"]

    # --- Step 2: Build S08-style weight schedule with dispersion gate ---
    weight_schedule = {}

    for i, d in enumerate(reb_dates):
        avail_d = close_full.index[close_full.index <= d]
        if len(avail_d) < lb_days + 2:
            continue
        if d not in disp_df.index:
            continue

        # Dispersion gate: is today's dispersion > rolling median of PRIOR months?
        prior_disp = disp_df[disp_df.index < d]
        if len(prior_disp) < 2:
            # Not enough history for median — skip (conservative)
            weight_schedule[d] = {}
            continue

        rolling_median = float(prior_disp.median())
        cur_disp = disp_df[d]

        if cur_disp <= rolling_median:
            # Low dispersion → go flat
            weight_schedule[d] = {}
            continue

        # High dispersion → take S08 positions (top N by lookback return)
        snap_row = len(avail_d) - 1
        p_now = close_full.iloc[snap_row]
        p_past = close_full.iloc[max(0, snap_row - lb_days)]

        returns = ((p_now / p_past) - 1.0).dropna().sort_values(ascending=False)
        if returns.empty:
            weight_schedule[d] = {}
            continue

        # Absolute momentum cash filter (same as S08)
        if abs_mom and returns.iloc[0] <= 0:
            weight_schedule[d] = {}
            continue

        selected = returns.iloc[:top_n].index.tolist()
        if not selected:
            weight_schedule[d] = {}
            continue

        w = {s: 1.0 / len(selected) for s in selected}
        weight_schedule[d] = w

    if not weight_schedule:
        log.warning("S65: no valid rebalance periods found")
        empty = pd.Series(dtype=float)
        return {"returns": empty, "benchmark": empty,
                "description": DESCRIPTION, "turnover_annual": 0.0}

    gross_ret, to = portfolio_returns_from_weights(weight_schedule, close_df, start, end)
    net_ret = apply_costs(gross_ret, to, cost_bps, slip_bps)

    spy = load_price_series("SPY", start, end, ADJ_TOTALRETURN, cache)
    bm = spy["Close"].pct_change(fill_method=None).reindex(net_ret.index)

    n_active = sum(1 for w in weight_schedule.values() if w)
    n_total = len(weight_schedule)
    active_pct = n_active / max(n_total, 1) * 100
    ann_to = float(to.sum() / max(len(to) / TRADING_DAYS, 1))

    log.info(
        "S65 done. top_n=%d, lb=%dm, active=%d/%d months (%.0f%%), ann_to=%.2fx",
        top_n, lb_months, n_active, n_total, active_pct, ann_to,
    )

    return {
        "returns": net_ret,
        "benchmark": bm,
        "description": DESCRIPTION,
        "turnover_annual": ann_to,
    }
