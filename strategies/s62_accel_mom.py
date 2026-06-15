"""
S62 -- Acceleration momentum (change-of-momentum)
Universe:  S&P 500 point-in-time
Signal:    recent_mom - prior_mom where:
             recent_mom = close[t-21] / close[t-147] - 1  (6 months ending 1 month ago)
             prior_mom  = close[t-147] / close[t-273] - 1 (6 months before that)
           Positive signal = momentum is accelerating.
Portfolio: Long top quintile, short bottom quintile. Dollar-neutral, equal-weight.
Rebalance: Monthly. Warmup: 273 trading days.
"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import logging
import numpy as np
import pandas as pd
from data import (
    load_price_series, watchlist_symbols, index_constituent_mask,
    ADJ_TOTALRETURN, compute_dollar_volume,
)
from engine import portfolio_returns_from_weights, apply_costs

log = logging.getLogger(__name__)
TRADING_DAYS = 252
DESCRIPTION = "Acceleration momentum (delta-mom): long top quintile, short bottom quintile, S&P 500 PIT"

# Lookback offsets in trading days
SKIP = 21       # 1-month skip
MID = 147       # 7-month offset (defines boundary between recent and prior halves)
LONG = 273      # 13-month offset (start of prior 6-month window)
WARMUP = LONG + 5


def run(config: dict) -> dict:
    cfg = config["backtest"]
    start = cfg["start_date"]
    end = cfg["end_date"]
    is_smoke = config.get("smoke", False)
    cache = config["paths"]["cache_dir"]

    cost_bps = config["costs"]["equity_cost_bps"]
    slip_bps = config["costs"]["equity_slippage_bps"]
    min_dv = config["liquidity"]["min_dollar_volume"]
    min_px = config["liquidity"]["min_price"]

    wl_name = config["universes"].get("sp500", "S&P 500 Current & Past")
    idx_name = config["universes"].get("sp500_index", "S&P 500")
    start_load = "2013-01-01" if is_smoke else "1997-01-01"

    symbols = watchlist_symbols(wl_name)
    if is_smoke:
        symbols = symbols[:120]

    close_map, dv_map, mask_map = {}, {}, {}
    for sym in symbols:
        df = load_price_series(sym, start=start_load, end=end,
                               adjustment=ADJ_TOTALRETURN, cache_dir=cache)
        if df.empty or df["Close"].max() < min_px:
            continue
        close_map[sym] = df["Close"]
        dv_map[sym] = compute_dollar_volume(df)
        m = index_constituent_mask(sym, idx_name, start=start_load, end=end, cache_dir=cache)
        if not m.empty:
            mask_map[sym] = m

    log.info("S62: %d symbols loaded", len(close_map))

    trading_idx = pd.bdate_range(start, end)
    close_df = pd.DataFrame(close_map).reindex(trading_idx, method="ffill")
    dv_df = pd.DataFrame(dv_map).reindex(trading_idx, method="ffill")
    member_df = (
        pd.DataFrame(mask_map)
        .reindex(trading_idx, method="ffill")
        .infer_objects(copy=False)
        .fillna(False)
        .astype(bool)
    )
    for col in close_df.columns:
        if col not in member_df.columns:
            member_df[col] = False

    reb_dates = pd.date_range(start, end, freq="BME")
    N_QUANTILE = 5

    # --- Spot-check turnover estimate ---
    # Sample a few rebalance dates and estimate avg monthly turnover
    spot_dates = []
    for rd in reb_dates:
        pos = trading_idx.searchsorted(rd, side="right") - 1
        if pos >= WARMUP:
            spot_dates.append((rd, pos))
        if len(spot_dates) >= 6:
            break

    prev_long_set = set()
    prev_short_set = set()
    turnover_estimates = []

    def _get_portfolio(pos):
        if pos < WARMUP:
            return [], []
        dv_now = dv_df.iloc[pos]
        members = member_df.iloc[pos]
        valid = [
            c for c in close_df.columns
            if members.get(c, False) and dv_now.get(c, 0) >= min_dv
        ]
        if len(valid) < N_QUANTILE * 4:
            return [], []
        px_now = close_df.iloc[pos - SKIP]
        px_mid = close_df.iloc[pos - MID]
        px_old = close_df.iloc[pos - LONG]
        accel = {}
        for sym in valid:
            p0 = px_now.get(sym, np.nan)
            p1 = px_mid.get(sym, np.nan)
            p2 = px_old.get(sym, np.nan)
            if np.isfinite(p0) and np.isfinite(p1) and np.isfinite(p2) and p1 > 0 and p2 > 0:
                recent = p0 / p1 - 1.0
                prior = p1 / p2 - 1.0
                accel[sym] = recent - prior
        if len(accel) < N_QUANTILE * 4:
            return [], []
        s = pd.Series(accel).sort_values()
        n_each = max(1, len(s) // N_QUANTILE)
        return s.iloc[:n_each].index.tolist(), s.iloc[-n_each:].index.tolist()

    for rd, pos in spot_dates:
        sh, lo = _get_portfolio(pos)
        cur_s, cur_l = set(sh), set(lo)
        if prev_long_set or prev_short_set:
            chg = len(cur_l.symmetric_difference(prev_long_set)) + \
                  len(cur_s.symmetric_difference(prev_short_set))
            total_held = max(len(prev_long_set) + len(prev_short_set), 1)
            turnover_estimates.append(chg / total_held)
        prev_long_set, prev_short_set = cur_l, cur_s

    if turnover_estimates:
        avg_monthly_to = float(np.mean(turnover_estimates))
        est_annual_to = avg_monthly_to * 12
        print(f"[S62] Turnover spot-check: avg monthly turnover ~ {avg_monthly_to:.1%}, "
              f"estimated annual ~ {est_annual_to:.1f}x")
        if est_annual_to > 100:
            print(f"[S62] WARNING: estimated annual turnover {est_annual_to:.1f}x exceeds 100x -- "
                  f"costs will be very significant")

    # --- Build weight schedule ---
    weight_schedule = {}

    for rd in reb_dates:
        pos = trading_idx.searchsorted(rd, side="right") - 1
        if pos < WARMUP:
            continue

        dv_now = dv_df.iloc[pos]
        members = member_df.iloc[pos]
        valid_cols = [
            c for c in close_df.columns
            if members.get(c, False) and dv_now.get(c, 0) >= min_dv
        ]
        if len(valid_cols) < N_QUANTILE * 4:
            continue

        # Acceleration signal
        px_now = close_df.iloc[pos - SKIP]   # price 1 month ago
        px_mid = close_df.iloc[pos - MID]    # price 7 months ago
        px_old = close_df.iloc[pos - LONG]   # price 13 months ago

        accel = {}
        for sym in valid_cols:
            p0 = px_now.get(sym, np.nan)
            p1 = px_mid.get(sym, np.nan)
            p2 = px_old.get(sym, np.nan)
            if (np.isfinite(p0) and np.isfinite(p1) and np.isfinite(p2)
                    and p1 > 0 and p2 > 0):
                recent_mom = p0 / p1 - 1.0
                prior_mom = p1 / p2 - 1.0
                accel[sym] = recent_mom - prior_mom

        if len(accel) < N_QUANTILE * 4:
            continue

        accel_series = pd.Series(accel).sort_values()
        n_each = max(1, len(accel_series) // N_QUANTILE)

        short_names = accel_series.iloc[:n_each].index.tolist()   # lowest accel → short
        long_names = accel_series.iloc[-n_each:].index.tolist()   # highest accel → long

        w = {}
        for s in long_names:
            w[s] = 1.0 / len(long_names)
        for s in short_names:
            w[s] = w.get(s, 0.0) - 1.0 / len(short_names)
        weight_schedule[rd] = w

    if not weight_schedule:
        log.warning("S62: no valid rebalance periods found")
        empty = pd.Series(dtype=float)
        return {"returns": empty, "benchmark": empty,
                "description": DESCRIPTION, "turnover_annual": 0.0}

    gross_ret, to = portfolio_returns_from_weights(weight_schedule, close_df, start, end)
    net_ret = apply_costs(gross_ret, to, cost_bps, slip_bps)

    spy = load_price_series("SPY", start, end, ADJ_TOTALRETURN, cache)
    bm = spy["Close"].pct_change(fill_method=None).reindex(net_ret.index)

    ann_to = float(to.sum() / max(len(to) / TRADING_DAYS, 1))
    log.info("S62 done. %d rebalance dates, ann_turnover=%.2fx", len(weight_schedule), ann_to)

    return {
        "returns": net_ret,
        "benchmark": bm,
        "description": DESCRIPTION,
        "turnover_annual": ann_to,
    }
