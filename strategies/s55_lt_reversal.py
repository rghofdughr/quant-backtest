import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import logging
import numpy as np
import pandas as pd
from collections import deque
from data import load_price_series, watchlist_symbols, index_constituent_mask, ADJ_TOTALRETURN, compute_dollar_volume
from engine import portfolio_returns_from_weights, apply_costs

log = logging.getLogger(__name__)
TRADING_DAYS = 252
DESCRIPTION = "Long-horizon reversal (DeBondt-Thaler). Long past 5yr losers, short past winners. Overlapping 12-cohort construction."

def run(config):
    cfg = config["backtest"]
    start, end = cfg["start_date"], cfg["end_date"]
    is_smoke = config.get("smoke", False)
    cache = config["paths"]["cache_dir"]
    cost_bps = config["costs"]["equity_cost_bps"]
    slip_bps = config["costs"]["equity_slippage_bps"]
    min_dv = config["liquidity"]["min_dollar_volume"]
    min_px = config["liquidity"]["min_price"]
    wl_name = config["universes"].get("sp500", "S&P 500 Current & Past")
    idx_name = config["universes"].get("sp500_index", "S&P 500")
    start_load = "2013-01-01" if is_smoke else "1994-01-01"

    symbols = watchlist_symbols(wl_name)
    if is_smoke:
        symbols = symbols[:100]

    close_map, dv_map, mask_map = {}, {}, {}
    for sym in symbols:
        df = load_price_series(sym, start=start_load, end=end, adjustment=ADJ_TOTALRETURN, cache_dir=cache)
        if df.empty or df["Close"].max() < min_px:
            continue
        close_map[sym] = df["Close"]
        dv_map[sym] = compute_dollar_volume(df)
        m = index_constituent_mask(sym, idx_name, start=start_load, end=end, cache_dir=cache)
        if not m.empty:
            mask_map[sym] = m

    trading_idx = pd.bdate_range(start, end)
    close_df = pd.DataFrame(close_map).reindex(trading_idx, method="ffill")
    dv_df = pd.DataFrame(dv_map).reindex(trading_idx, method="ffill")
    member_df = (pd.DataFrame(mask_map).reindex(trading_idx, method="ffill")
                 .infer_objects(copy=False).fillna(False).astype(bool))
    for col in close_df.columns:
        if col not in member_df.columns:
            member_df[col] = False

    # Need full history (start_load → end) for lookback
    full_idx = pd.bdate_range(start_load, end)
    close_full = pd.DataFrame(close_map).reindex(full_idx, method="ffill")
    dv_full = pd.DataFrame(dv_map).reindex(full_idx, method="ffill")
    member_full = (pd.DataFrame(mask_map).reindex(full_idx, method="ffill")
                   .infer_objects(copy=False).fillna(False).astype(bool))
    for col in close_full.columns:
        if col not in member_full.columns:
            member_full[col] = False

    reb_dates = pd.date_range(start, end, freq="BME")
    # Warmup: 1323 trading days = 63 months
    WARMUP = 1323
    # Signal window: rows [pos-1260 : pos-273]
    LT_FAR = 1260   # ~60 months ago
    LT_NEAR = 273   # ~13 months ago (skip recent 12 months)
    MAX_COHORTS = 12

    cohort_deque = deque(maxlen=MAX_COHORTS)
    weight_schedule = {}

    for rd in reb_dates:
        rd = min(rd, trading_idx[-1])
        # Position in full_idx
        pos_full = full_idx.searchsorted(rd)
        if pos_full < WARMUP:
            continue

        dv_now = dv_full.iloc[pos_full - 1] if pos_full > 0 else dv_full.iloc[0]
        members = member_full.iloc[pos_full] if pos_full < len(member_full) else member_full.iloc[-1]

        valid_cols = [
            c for c in close_full.columns
            if members.get(c, False) and dv_now.get(c, 0) >= min_dv
        ]

        # Compute long-horizon return: from pos-LT_FAR to pos-LT_NEAR
        far_pos = max(0, pos_full - LT_FAR)
        near_pos = max(0, pos_full - LT_NEAR)
        if near_pos <= far_pos:
            continue

        close_far = close_full.iloc[far_pos]
        close_near = close_full.iloc[near_pos]

        lt_ret = {}
        for c in valid_cols:
            cf = close_far.get(c, np.nan)
            cn = close_near.get(c, np.nan)
            if np.isfinite(cf) and np.isfinite(cn) and cf > 0:
                lt_ret[c] = cn / cf - 1.0

        if len(lt_ret) < 20:
            cohort_deque.append({})
        else:
            lt_series = pd.Series(lt_ret)
            q20 = lt_series.quantile(0.20)
            q80 = lt_series.quantile(0.80)
            long_names = lt_series[lt_series <= q20].index.tolist()
            short_names = lt_series[lt_series >= q80].index.tolist()

            if not long_names or not short_names:
                cohort_deque.append({})
            else:
                n_long = len(long_names)
                n_short = len(short_names)
                cohort_w = {}
                for s in long_names:
                    cohort_w[s] = 1.0 / n_long
                for s in short_names:
                    cohort_w[s] = cohort_w.get(s, 0.0) - 1.0 / n_short
                cohort_deque.append(cohort_w)

        # Average all current cohorts (each gets 1/MAX_COHORTS weight)
        n_cohorts = len(cohort_deque)
        if n_cohorts == 0:
            continue

        combined = {}
        for cohort in cohort_deque:
            for sym, wt in cohort.items():
                combined[sym] = combined.get(sym, 0.0) + wt / MAX_COHORTS

        # Remove zero weights
        combined = {s: w for s, w in combined.items() if abs(w) > 1e-9}
        weight_schedule[rd] = combined

    gross_ret, to = portfolio_returns_from_weights(weight_schedule, close_df, start, end)
    net_ret = apply_costs(gross_ret, to, cost_bps, slip_bps)
    spy = load_price_series("SPY", start, end, ADJ_TOTALRETURN, cache)
    bm = spy["Close"].pct_change(fill_method=None).reindex(net_ret.index)
    ann_to = float(to.sum() / max(len(to) / TRADING_DAYS, 1))
    return {"returns": net_ret, "benchmark": bm, "description": DESCRIPTION, "turnover_annual": ann_to}
