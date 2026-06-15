import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import logging
import numpy as np
import pandas as pd
from data import load_price_series, watchlist_symbols, index_constituent_mask, ADJ_TOTALRETURN, compute_dollar_volume
from engine import portfolio_returns_from_weights, apply_costs

log = logging.getLogger(__name__)
TRADING_DAYS = 252
DESCRIPTION = "Cross-sectional skewness anomaly (momentum-neutralized). Long low-skew, short high-skew quintiles."

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

    ret_df = close_df.pct_change(fill_method=None)
    reb_dates = pd.date_range(start, end, freq="BME")
    WARMUP = 252
    SKEW_WIN = 250
    MOM_LONG = 252
    MOM_SHORT = 21

    weight_schedule = {}

    for rd in reb_dates:
        rd = min(rd, trading_idx[-1])
        pos = trading_idx.searchsorted(rd)
        if pos < WARMUP:
            continue

        # Skewness window
        skew_start = max(0, pos - SKEW_WIN)
        window_rets = ret_df.iloc[skew_start:pos]  # shape (SKEW_WIN, n_syms)

        # Momentum: 12-1 month
        mom_start = max(0, pos - MOM_LONG)
        mom_end = max(0, pos - MOM_SHORT)
        if mom_end <= mom_start:
            continue
        close_now = close_df.iloc[mom_end]
        close_then = close_df.iloc[mom_start]
        momentum = (close_now / close_then - 1.0)

        # Liquidity and membership filter
        dv_now = dv_df.iloc[pos - 1] if pos > 0 else dv_df.iloc[0]
        members = member_df.iloc[pos] if pos < len(member_df) else member_df.iloc[-1]

        valid_cols = [
            c for c in close_df.columns
            if members.get(c, False)
            and dv_now.get(c, 0) >= min_dv
            and not np.isnan(momentum.get(c, np.nan))
        ]

        if len(valid_cols) < 20:
            continue

        # Compute winsorized skewness
        w_rets = window_rets[valid_cols].clip(-0.15, 0.15)
        # Need at least 30 non-NaN days per stock
        n_valid = w_rets.notna().sum(axis=0)
        valid_cols = [c for c in valid_cols if n_valid[c] >= 30]
        if len(valid_cols) < 20:
            continue

        w_rets = w_rets[valid_cols]
        raw_skew = w_rets.skew(axis=0).values  # shape (n,)

        mom_vals = momentum[valid_cols].values.astype(float)
        log_price = np.log(close_df.iloc[pos][valid_cols].values.astype(float) + 1e-12)

        # Neutralize skew on momentum and log_price via cross-sectional regression
        n = len(valid_cols)
        X = np.column_stack([mom_vals, log_price, np.ones(n)])

        # Drop rows with NaN in X or raw_skew
        valid_rows = (
            np.isfinite(raw_skew) &
            np.isfinite(mom_vals) &
            np.isfinite(log_price)
        )
        if valid_rows.sum() < 20:
            continue

        X_fit = X[valid_rows]
        y_fit = raw_skew[valid_rows]
        coef = np.linalg.lstsq(X_fit, y_fit, rcond=None)[0]
        residuals = np.full(n, np.nan)
        residuals[valid_rows] = y_fit - X_fit @ coef

        valid_cols_arr = np.array(valid_cols)
        res_series = pd.Series(residuals, index=valid_cols_arr)
        res_series = res_series.dropna()

        if len(res_series) < 20:
            continue

        q20 = res_series.quantile(0.20)
        q80 = res_series.quantile(0.80)

        long_names = res_series[res_series <= q20].index.tolist()
        short_names = res_series[res_series >= q80].index.tolist()

        if not long_names or not short_names:
            continue

        n_long = len(long_names)
        n_short = len(short_names)
        w = {}
        for s in long_names:
            w[s] = 1.0 / n_long
        for s in short_names:
            w[s] = -1.0 / n_short

        weight_schedule[rd] = w

    gross_ret, to = portfolio_returns_from_weights(weight_schedule, close_df, start, end)
    net_ret = apply_costs(gross_ret, to, cost_bps, slip_bps)
    spy = load_price_series("SPY", start, end, ADJ_TOTALRETURN, cache)
    bm = spy["Close"].pct_change(fill_method=None).reindex(net_ret.index)
    ann_to = float(to.sum() / max(len(to) / TRADING_DAYS, 1))
    return {"returns": net_ret, "benchmark": bm, "description": DESCRIPTION, "turnover_annual": ann_to}
