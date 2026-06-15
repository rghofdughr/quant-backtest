"""
S05 — Residual momentum (Blitz-Huij-Martens 2011)
Universe:  Russell 1000 point-in-time (reuses S01 universe infrastructure)
Signal:    12-1 momentum computed on OLS residuals from rolling 36-month regression
           on Fama-French 3 factors proxied via ETFs:
             MKT = SPY excess return
             SMB = IWM - SPY  (small-minus-big proxy)
             HML = IWD - IWF  (value-minus-growth proxy)
Rebalance: Monthly
Execution: Next trading day close
Sizing:    Equal-weight, dollar-neutral, long top decile / short bottom decile
Compare:   Sharpe vs raw 12-1 momentum (S01) — residual should have lower vol.
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
DESCRIPTION = "Residual 12-1 momentum (FF3 de-trended), Russell 1000 PIT, long/short decile"
TRADING_DAYS = 252
FF_LOOKBACK  = 36   # months for rolling factor regression
MOM_LOOKBACK = 12   # months for momentum signal
MOM_SKIP     = 1    # skip last month


def _compute_ff_proxies(cache: str, start: str, end: str) -> pd.DataFrame:
    """Build monthly FF3 factor returns from ETF proxies."""
    etfs = {"SPY": "mkt", "IWM": "smb_raw", "IWD": "hml_raw", "IWF": "hml_neg"}
    monthly = {}
    for sym, label in etfs.items():
        df = load_price_series(sym, start="1997-01-01", end=end,
                               adjustment=ADJ_TOTALRETURN, cache_dir=cache)
        if not df.empty:
            c = df["Close"].resample("ME").last()
            monthly[sym] = c.pct_change(fill_method=None)

    ff = pd.DataFrame(monthly)
    ff["MKT"] = ff["SPY"]
    ff["SMB"] = ff["IWM"] - ff["SPY"]    # small minus big
    ff["HML"] = ff["IWD"] - ff["IWF"]   # value minus growth
    return ff[["MKT", "SMB", "HML"]].dropna()


def _rolling_residuals(monthly_ret: pd.Series, ff: pd.DataFrame,
                       reg_lookback: int = 36) -> pd.Series:
    """
    For each month t, regress stock's returns [t-36 .. t-1] on FF factors.
    Return fitted residuals at each month t (only for months with full lookback).
    """
    aligned = pd.concat([monthly_ret, ff], axis=1).dropna()
    if len(aligned) < reg_lookback + 2:
        return pd.Series(dtype=float, name=monthly_ret.name)

    y   = aligned.iloc[:, 0].values
    X   = aligned.iloc[:, 1:].values  # (n, 3)
    n   = len(y)
    resid = np.full(n, np.nan)

    for i in range(reg_lookback, n):
        y_f = y[i - reg_lookback:i]
        X_f = np.column_stack([np.ones(reg_lookback), X[i - reg_lookback:i]])
        try:
            betas, _, _, _ = np.linalg.lstsq(X_f, y_f, rcond=None)
            resid[i] = y[i] - float(np.array([1.0, *X[i]]) @ betas)
        except Exception:
            pass

    return pd.Series(resid, index=aligned.index, name=monthly_ret.name)


def run(config: dict) -> dict:
    cfg      = config["backtest"]
    start    = cfg["start_date"]
    end      = cfg["end_date"]
    is_smoke = config.get("smoke", False)
    cache    = config["paths"]["cache_dir"]

    cost_bps = config["costs"]["equity_cost_bps"]
    slip_bps = config["costs"]["equity_slippage_bps"]
    min_dv   = config["liquidity"]["min_dollar_volume"]
    min_px   = config["liquidity"]["min_price"]

    if is_smoke:
        wl_name  = config["universes"].get("sp500", "S&P 500 Current & Past")
        idx_name = config["universes"].get("sp500_index", "S&P 500")
        start_load = "2008-01-01"
        sym_limit  = 80
    else:
        wl_name  = config["universes"].get("russell1000", "Russell 1000 Current & Past")
        idx_name = config["universes"].get("russell1000_index", "Russell 1000")
        start_load = "1997-01-01"
        sym_limit  = None

    log.info("S05: loading FF proxies ...")
    ff = _compute_ff_proxies(cache, start_load, end)

    log.info("S05: loading universe %s ...", wl_name)
    symbols = watchlist_symbols(wl_name)
    if sym_limit:
        symbols = symbols[:sym_limit]

    # Load monthly returns for each symbol
    close_map, dv_map, mask_map = {}, {}, {}
    monthly_ret_map: dict[str, pd.Series] = {}

    for sym in symbols:
        df = load_price_series(sym, start=start_load, end=end,
                               adjustment=ADJ_TOTALRETURN, cache_dir=cache)
        if df.empty or df["Close"].max() < min_px:
            continue
        close_map[sym] = df["Close"]
        dv_map[sym]    = compute_dollar_volume(df)
        mo = df["Close"].resample("ME").last().pct_change(fill_method=None)
        monthly_ret_map[sym] = mo

        m = index_constituent_mask(sym, idx_name, start=start_load, end=end, cache_dir=cache)
        if not m.empty:
            mask_map[sym] = m

    log.info("S05: %d symbols loaded, computing residuals ...", len(close_map))

    # Compute rolling OLS residuals for each stock
    resid_map: dict[str, pd.Series] = {}
    for sym, mo_ret in monthly_ret_map.items():
        r = _rolling_residuals(mo_ret, ff, FF_LOOKBACK)
        if not r.dropna().empty:
            resid_map[sym] = r

    log.info("S05: residuals computed for %d symbols", len(resid_map))

    # Build daily infrastructure for filtering and weight construction
    trading_idx = pd.bdate_range(start, end)
    close_df  = pd.DataFrame(close_map).reindex(trading_idx, method="ffill")
    dv_df     = pd.DataFrame(dv_map).reindex(trading_idx, method="ffill")
    member_df = (pd.DataFrame(mask_map)
                   .reindex(trading_idx, method="ffill")
                   .infer_objects(copy=False)
                   .fillna(False).astype(bool))
    for col in close_df.columns:
        if col not in member_df.columns:
            member_df[col] = False

    # Monthly rebalance dates
    reb_dates = pd.date_range(start, end, freq="BME")
    weight_schedule: dict = {}

    for d in reb_dates:
        pos = trading_idx.searchsorted(d, side="right") - 1
        if pos < 0:
            continue

        prices  = close_df.iloc[pos]
        dv_row  = dv_df.iloc[pos]
        members = member_df.iloc[pos]

        # Get residual momentum signal: resid at (t-skip) / cumulative resid from (t-lookback) to (t-skip)
        mo_d = pd.Timestamp(d).to_period("M").to_timestamp("M")  # end-of-month snap
        signals = {}
        for sym, resid in resid_map.items():
            r = resid[resid.index <= mo_d].dropna()
            if len(r) < MOM_LOOKBACK:
                continue
            # Cumulative residual return from t-12 to t-1 (skip last month)
            r_window = r.iloc[-(MOM_LOOKBACK):-(MOM_SKIP) if MOM_SKIP > 0 else None]
            signals[sym] = float((1 + r_window).prod() - 1)

        # Apply filters
        valid_syms = [
            s for s in signals
            if s in members.index and bool(members.get(s, False))
            and s in prices.index and prices.get(s, 0) >= min_px
            and s in dv_row.index and dv_row.get(s, 0) >= min_dv
        ]
        sig_valid = {s: signals[s] for s in valid_syms}
        if len(sig_valid) < 20:
            continue

        ranked = sorted(sig_valid.items(), key=lambda x: x[1])
        n_each = max(1, len(ranked) // 10)
        shorts = [s for s, _ in ranked[:n_each]]
        longs  = [s for s, _ in ranked[-n_each:]]

        w = {s:  1.0 / len(longs)  for s in longs}
        w.update({s: -1.0 / len(shorts) for s in shorts})
        weight_schedule[d] = w

    if not weight_schedule:
        return {"returns": pd.Series(dtype=float), "benchmark": pd.Series(dtype=float),
                "description": DESCRIPTION, "turnover_annual": 0.0}

    gross_ret, to = portfolio_returns_from_weights(weight_schedule, close_df, start, end)
    net_ret = apply_costs(gross_ret, to, cost_bps, slip_bps)

    spy = load_price_series("SPY", start, end, ADJ_TOTALRETURN, cache)
    bm  = spy["Close"].pct_change(fill_method=None).reindex(net_ret.index)

    ann_to = float(to.sum() / max(len(to) / TRADING_DAYS, 1))
    log.info("S05 done. %d rebalance dates, ann turnover: %.2fx",
             len(weight_schedule), ann_to)

    return {
        "returns": net_ret, "benchmark": bm,
        "description": DESCRIPTION, "turnover_annual": ann_to,
    }
