"""
s124 — Betting Against Beta (Frazzini & Pedersen 2014)
Signal:    60-month rolling OLS beta vs SPY
Long:      bottom beta decile, levered up so portfolio beta = 1
Short:     top beta decile, de-levered so portfolio beta = 1
Net:       long-biased (net notional > 0), beta-neutral
Rebalance: Monthly
Universe:  Russell 1000 PIT

Note: beta-neutrality means equal dollar notional per leg scaled by 1/avg_beta.
      Low-beta leg is levered ~1.5-2x; high-beta leg is de-levered ~0.6-0.8x.
"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import logging
import numpy as np
import pandas as pd
from data import load_price_series, watchlist_symbols, index_constituent_mask, ADJ_TOTALRETURN, compute_dollar_volume
from engine import portfolio_returns_from_weights, apply_costs

log = logging.getLogger(__name__)
DESCRIPTION = "Betting Against Beta: long low-beta / short high-beta, leverage-equalized (Frazzini-Pedersen 2014)"

BETA_LOOKBACK = 1260   # 60 months
DECILE        = 0.10
MIN_STOCKS    = 40
MIN_HISTORY   = 252    # require at least 1yr per stock
LOAD_START    = "1994-01-01"


def run(config):
    cfg      = config["backtest"]
    start, end = cfg["start_date"], cfg["end_date"]
    cache    = config["paths"]["cache_dir"]
    cost_bps = config["costs"]["equity_cost_bps"]
    slip_bps = config["costs"]["equity_slippage_bps"]
    min_dv   = config["liquidity"]["min_dollar_volume"]
    min_px   = config["liquidity"]["min_price"]
    wl_name  = config["universes"]["russell1000"]
    idx_name = config["universes"]["russell1000_index"]

    symbols = watchlist_symbols(wl_name)
    close_map, dv_map, mask_map = {}, {}, {}
    for sym in symbols:
        df = load_price_series(sym, start=LOAD_START, end=end,
                               adjustment=ADJ_TOTALRETURN, cache_dir=cache)
        if df.empty or df["Close"].max() < min_px:
            continue
        close_map[sym] = df["Close"]
        dv_map[sym]    = compute_dollar_volume(df)
        m = index_constituent_mask(sym, idx_name, start=LOAD_START, end=end, cache_dir=cache)
        if not m.empty:
            mask_map[sym] = m

    full_idx  = pd.bdate_range(LOAD_START, end)
    close_df  = pd.DataFrame(close_map).reindex(full_idx, method="ffill")
    dv_df     = pd.DataFrame(dv_map).reindex(full_idx, method="ffill")
    member_df = (pd.DataFrame(mask_map).reindex(full_idx, method="ffill")
                 .infer_objects(copy=False).fillna(False).astype(bool))
    for col in close_df.columns:
        if col not in member_df.columns:
            member_df[col] = False

    # Load SPY returns on full index
    spy_df  = load_price_series("SPY", start=LOAD_START, end=end,
                                adjustment=ADJ_TOTALRETURN, cache_dir=cache)
    spy_cls = spy_df["Close"].reindex(full_idx, method="ffill")
    spy_ret = spy_cls.pct_change(fill_method=None).fillna(0.0)

    # Precompute all stock returns
    ret_df  = close_df.pct_change(fill_method=None).fillna(0.0)

    weight_schedule = {}
    backtest_idx    = pd.bdate_range(start, end)
    reb_dates       = pd.date_range(start, end, freq="BME")

    for reb_d in reb_dates:
        pos = full_idx.searchsorted(reb_d, side="right") - 1
        if pos < BETA_LOOKBACK + 5:
            continue

        members = member_df.iloc[pos]
        dv_now  = dv_df.iloc[pos]
        px_now  = close_df.iloc[pos]
        valid   = [c for c in close_df.columns
                   if members.get(c, False)
                   and dv_now.get(c, 0.0) >= min_dv
                   and px_now.get(c, 0.0) >= min_px
                   and c in ret_df.columns]
        if len(valid) < MIN_STOCKS:
            continue

        # Vectorised 60-month beta computation
        window_rets = ret_df[valid].iloc[pos - BETA_LOOKBACK: pos]  # (T, N)
        spy_window  = spy_ret.iloc[pos - BETA_LOOKBACK: pos].values  # (T,)

        spy_c   = spy_window - spy_window.mean()
        spy_var = float(spy_c.var())
        if spy_var < 1e-12:
            continue

        # Only include stocks with sufficient non-NaN history
        data = window_rets.copy()
        data = data.loc[:, data.notna().sum() >= MIN_HISTORY]
        if data.shape[1] < MIN_STOCKS:
            continue

        data = data.fillna(0.0)
        stock_c = data.values - data.values.mean(axis=0)          # (T, N)
        cov_vec = (stock_c.T @ spy_c) / (len(spy_window) - 1)     # (N,)
        betas   = pd.Series(cov_vec / spy_var, index=data.columns)

        # Shrink betas toward 1 (Vasicek / Frazzini-Pedersen standard)
        betas = 0.6 * betas + 0.4 * 1.0

        lo_cut  = betas.quantile(DECILE)
        hi_cut  = betas.quantile(1 - DECILE)
        longs_b = betas[betas <= lo_cut]   # low-beta decile
        shorts_b = betas[betas >= hi_cut]  # high-beta decile

        if len(longs_b) == 0 or len(shorts_b) == 0:
            continue

        avg_beta_L = float(longs_b.mean())
        avg_beta_H = float(shorts_b.mean())
        if avg_beta_L < 0.05 or avg_beta_H < 0.05:
            continue

        # Equal-weight within leg, then scale by 1/avg_beta for neutrality
        scale_L = 1.0 / avg_beta_L
        scale_H = 1.0 / avg_beta_H

        w = {}
        for s in longs_b.index:
            w[s]  =  scale_L / len(longs_b)
        for s in shorts_b.index:
            w[s] = w.get(s, 0.0) - scale_H / len(shorts_b)

        weight_schedule[reb_d] = w

    gross_ret, to = portfolio_returns_from_weights(weight_schedule, close_df, start, end)
    net_ret = apply_costs(gross_ret, to, cost_bps, slip_bps)
    ann_to  = float(to.sum() / max(len(to) / 252, 1))
    return {"returns": net_ret, "turnover_annual": ann_to, "description": DESCRIPTION}
