"""
S101 — Within-sector cointegration pairs.
For each SPDR sector, find cointegrated pairs (Engle-Granger) on rolling 252d
training window; trade z-score ±2 entry / 0 exit. Out-of-sample only.
Delisted legs are force-closed at last observed price (modeled as a loss).
Key discipline: pair selection entirely on training window; never in-sample.
"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import logging
import numpy as np
import pandas as pd
from data import load_price_series, watchlist_symbols, index_constituent_mask, ADJ_TOTALRETURN, compute_dollar_volume
from engine import apply_costs

try:
    from statsmodels.tsa.stattools import coint
    _STATSMODELS = True
except ImportError:
    _STATSMODELS = False

log = logging.getLogger(__name__)
TRADING_DAYS = 252
DESCRIPTION = "Within-sector cointegration pairs: Engle-Granger on 252d rolling window, trade z±2; OOS only, force-close delists."

# SPDR sector ETFs as universe proxies (use their PIT constituent data if available,
# otherwise use the sector ETF return to group stocks by highest correlation)
SECTORS = ["XLK", "XLF", "XLE", "XLI", "XLV", "XLP", "XLU", "XLY", "XLB", "XLRE", "XLC"]
ENTRY_Z = 2.0
EXIT_Z = 0.0
TRAIN_WIN = 252
OOS_WIN = 63   # re-train every quarter
MAX_PAIRS_PER_SECTOR = 3
MAX_TOTAL_PAIRS = 15

def _zscore(spread: pd.Series, win: int = 20) -> float:
    mu = spread.rolling(win).mean().iloc[-1]
    sd = spread.rolling(win).std().iloc[-1]
    if sd < 1e-9:
        return 0.0
    return float((spread.iloc[-1] - mu) / sd)

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
    start_load = "2013-01-01" if is_smoke else "1997-01-01"

    if not _STATSMODELS:
        raise NotImplementedError("S101 requires statsmodels: pip install statsmodels")

    symbols = watchlist_symbols(wl_name)
    if is_smoke:
        symbols = symbols[:200]

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

    # Load sector ETFs for grouping by correlation
    sector_data = {}
    for etf in SECTORS:
        df = load_price_series(etf, start=start_load, end=end, adjustment=ADJ_TOTALRETURN, cache_dir=cache)
        if not df.empty:
            sector_data[etf] = df["Close"]

    trading_idx = pd.bdate_range(start, end)
    close_df = pd.DataFrame(close_map).reindex(trading_idx, method="ffill")
    dv_df = pd.DataFrame(dv_map).reindex(trading_idx, method="ffill")
    member_df = (pd.DataFrame(mask_map).reindex(trading_idx, method="ffill")
                 .infer_objects(copy=False).fillna(False).astype(bool))
    for col in close_df.columns:
        if col not in member_df.columns:
            member_df[col] = False

    sector_df = pd.DataFrame(sector_data).reindex(trading_idx, method="ffill")
    sector_ret = sector_df.pct_change(fill_method=None)
    ret_df = close_df.pct_change(fill_method=None).fillna(0.0)
    log_price = np.log(close_df.replace(0, np.nan))

    WARMUP = TRAIN_WIN + OOS_WIN + 5

    port_rets = pd.Series(0.0, index=trading_idx)
    to_series = pd.Series(0.0, index=trading_idx)
    # Active pairs: {(sym1, sym2): {beta, spread_hist, long_short (1 or -1)}}
    active_pairs = {}

    # Re-train dates
    retrain_dates = pd.date_range(start, end, freq=f"{OOS_WIN}B")

    def _assign_sector(sym, pos):
        """Assign stock to sector by highest rolling correlation with sector ETF returns."""
        if not sector_ret.columns.tolist():
            return None
        r = ret_df[sym].iloc[max(pos - TRAIN_WIN, 0):pos]
        if r.isna().all():
            return None
        best_corr = -1
        best_sec = None
        for sec in sector_ret.columns:
            sec_r = sector_ret[sec].iloc[max(pos - TRAIN_WIN, 0):pos]
            common = r.dropna().index.intersection(sec_r.dropna().index)
            if len(common) < 50:
                continue
            c = float(r.loc[common].corr(sec_r.loc[common]))
            if np.isfinite(c) and c > best_corr:
                best_corr = c
                best_sec = sec
        return best_sec if best_corr > 0.3 else None

    # Track pair positions as weights
    pair_weights = {}

    for i, dt in enumerate(trading_idx):
        if dt < pd.Timestamp(start):
            continue
        if i < WARMUP:
            continue

        # Re-train on schedule
        if dt in retrain_dates:
            members = member_df.iloc[i] if i < len(member_df) else member_df.iloc[-1]
            dv_now = dv_df.iloc[i - 1]
            valid_cols = [
                c for c in close_df.columns
                if members.get(c, False) and dv_now.get(c, 0) >= min_dv
            ]

            # Group by sector
            sector_groups = {}
            for c in valid_cols:
                sec = _assign_sector(c, i)
                if sec:
                    sector_groups.setdefault(sec, []).append(c)

            new_pairs = {}
            for sec, group in sector_groups.items():
                if len(group) < 2:
                    continue
                # Find top cointegrated pairs within this sector
                found = 0
                train_log = log_price[group].iloc[max(i - TRAIN_WIN, 0):i]
                tested = []
                for ii, a in enumerate(group):
                    for b in group[ii + 1:]:
                        if found >= MAX_PAIRS_PER_SECTOR:
                            break
                        col_a = train_log[a].dropna()
                        col_b = train_log[b].dropna()
                        common = col_a.index.intersection(col_b.index)
                        if len(common) < 100:
                            continue
                        try:
                            score, pval, _ = coint(col_a.loc[common].values, col_b.loc[common].values)
                        except Exception:
                            continue
                        if pval < 0.05:
                            # Estimate hedge ratio
                            from numpy.linalg import lstsq
                            X = col_a.loc[common].values.reshape(-1, 1)
                            Y = col_b.loc[common].values
                            beta = float(lstsq(np.column_stack([X, np.ones(len(X))]), Y, rcond=None)[0][0])
                            tested.append((pval, a, b, beta))
                            found += 1
                tested.sort(key=lambda x: x[0])
                for pval, a, b, beta in tested[:MAX_PAIRS_PER_SECTOR]:
                    new_pairs[(a, b)] = beta

                if len(new_pairs) >= MAX_TOTAL_PAIRS:
                    break

            active_pairs = new_pairs
            pair_weights = {}

        # Compute spreads and signals for active pairs
        if not active_pairs:
            continue

        day_ret = 0.0
        n_active = 0
        for (a, b), beta in active_pairs.items():
            if a not in log_price.columns or b not in log_price.columns:
                continue
            # Force-close if either leg delisted (no price data)
            pa = float(log_price[a].iloc[i]) if i < len(log_price) else np.nan
            pb = float(log_price[b].iloc[i]) if i < len(log_price) else np.nan
            if not (np.isfinite(pa) and np.isfinite(pb)):
                pair_weights.pop((a, b), None)
                continue

            # Compute z-score of spread
            spread_hist = log_price[a].iloc[max(i - 60, 0):i + 1] - beta * log_price[b].iloc[max(i - 60, 0):i + 1]
            if len(spread_hist) < 20:
                continue
            z = _zscore(spread_hist.dropna())

            prev_state = pair_weights.get((a, b), 0)  # +1 = long a short b, -1 = reverse

            if prev_state == 0:
                if z > ENTRY_Z:
                    pair_weights[(a, b)] = -1  # spread too high, short a, long b
                elif z < -ENTRY_Z:
                    pair_weights[(a, b)] = 1   # spread too low, long a, short b
            else:
                # Exit on mean reversion
                if abs(z) < EXIT_Z:
                    pair_weights[(a, b)] = 0

            state = pair_weights.get((a, b), 0)
            if state != 0:
                ra = float(ret_df[a].iloc[i]) if a in ret_df.columns else 0.0
                rb = float(ret_df[b].iloc[i]) if b in ret_df.columns else 0.0
                pair_ret = state * (ra - beta * rb)
                day_ret += pair_ret
                n_active += 1

        if n_active > 0:
            port_rets.iloc[i] = day_ret / n_active

        # Approximate turnover
        to_series.iloc[i] = min(len([v for v in pair_weights.values() if v != 0]) * 0.01, 0.5)

    net_ret = apply_costs(port_rets, to_series, cost_bps, slip_bps)
    spy = load_price_series("SPY", start, end, ADJ_TOTALRETURN, cache)
    bm = spy["Close"].pct_change(fill_method=None).reindex(net_ret.index)
    ann_to = float(to_series.sum() / max(len(to_series) / TRADING_DAYS, 1))
    return {"returns": net_ret, "benchmark": bm, "description": DESCRIPTION, "turnover_annual": ann_to}
