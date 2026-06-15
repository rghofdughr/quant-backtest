"""
S26 — Dividend yield + sustainability (partial Norgate)
Universe:  S&P 500 C&P PIT
Signal:    12-month trailing dividend yield from Norgate total-return vs. price-return.
           Proxy: div_yield ≈ (TOTALRETURN return) − (price return) over rolling 252 days.
           Long top quintile of div yield with positive 5-yr dividend growth (proxy: TR/PR trend).
Note:      Payout ratio requires EPS (Sharadar/Compustat). Here we use yield + growth proxy.
"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import logging
import numpy as np
import pandas as pd

from data import (load_price_series, watchlist_symbols, index_constituent_mask,
                  compute_dollar_volume, ADJ_TOTALRETURN, ADJ_CAPITAL)
from engine import portfolio_returns_from_weights, apply_costs

log = logging.getLogger(__name__)
DESCRIPTION = "Dividend yield + sustainability factor, S&P 500 PIT, long high-yield quintile"
TRADING_DAYS = 252
DIV_LOOKBACK = 252     # 1-year to compute annual dividend yield proxy
GROWTH_LOOKBACK = 252 * 3  # 3-year to assess yield growth trend


def run(config: dict) -> dict:
    cfg      = config["backtest"]
    start    = cfg["start_date"]
    end      = cfg["end_date"]
    cache    = config["paths"]["cache_dir"]
    cost_bps = config["costs"]["equity_cost_bps"]
    slip_bps = config["costs"]["equity_slippage_bps"]

    univ_name = config["universes"]["sp500"]
    syms      = watchlist_symbols(univ_name)
    if not syms:
        return {"returns": pd.Series(dtype=float), "benchmark": pd.Series(dtype=float),
                "description": DESCRIPTION, "turnover_annual": 0.0}

    # Need to load back further for dividend growth lookback
    from datetime import datetime, timedelta
    start_ext = (datetime.fromisoformat(start) - timedelta(days=GROWTH_LOOKBACK + 30)).strftime("%Y-%m-%d")

    trading_idx = pd.bdate_range(start, end)

    tr_close_map, px_close_map, member_map = {}, {}, {}
    for sym in syms:
        df_tr = load_price_series(sym, start=start_ext, end=end, adjustment=ADJ_TOTALRETURN, cache_dir=cache)
        df_px = load_price_series(sym, start=start_ext, end=end, adjustment=ADJ_CAPITAL, cache_dir=cache)
        if df_tr.empty or df_px.empty:
            continue
        mask = index_constituent_mask(sym, "S&P 500", start, end, cache)
        tr_close_map[sym] = df_tr["Close"]
        px_close_map[sym] = df_px["Close"]
        member_map[sym]   = mask

    if len(tr_close_map) < 10:
        log.error("S26: too few symbols (%d)", len(tr_close_map))
        return {"returns": pd.Series(dtype=float), "benchmark": pd.Series(dtype=float),
                "description": DESCRIPTION, "turnover_annual": 0.0}

    ext_idx = pd.bdate_range(start_ext, end)
    tr_df   = pd.DataFrame(tr_close_map).reindex(ext_idx, method="ffill")
    px_df   = pd.DataFrame(px_close_map).reindex(ext_idx, method="ffill")
    memb_df = pd.DataFrame(member_map).reindex(trading_idx, method="ffill")
    memb_df = memb_df.infer_objects(copy=False).fillna(False).astype(bool)

    # Dividend yield proxy: cumulative TR return − cumulative PR return over 252 days
    # div_yield_proxy ≈ (TR[t] / TR[t-252]) / (PR[t] / PR[t-252]) - 1
    tr_1yr = tr_df / tr_df.shift(DIV_LOOKBACK)
    px_1yr = px_df / px_df.shift(DIV_LOOKBACK)
    div_yield = (tr_1yr / px_1yr.replace(0, np.nan) - 1).clip(lower=0)  # div yield can't be negative

    # Dividend growth proxy: slope of annual div yield over 3 years
    div_growth = div_yield - div_yield.shift(GROWTH_LOOKBACK)

    # Restrict to in-sample dates
    div_yield_is = div_yield.reindex(trading_idx)
    div_growth_is = div_growth.reindex(trading_idx)

    weight_schedule = {}
    rebal_dates = trading_idx[
        (trading_idx.month != pd.Series(trading_idx.month).shift(1).fillna(-1).values)
    ]

    for t in rebal_dates:
        dy  = div_yield_is.loc[t]
        dg  = div_growth_is.loc[t]
        mem = memb_df.loc[t]
        valid = mem & dy.notna() & dg.notna()
        if valid.sum() < 20:
            continue
        dy_v  = dy[valid]
        dg_v  = dg[valid]
        # Quintile on yield; within that, filter for growing dividends
        q80 = dy_v.quantile(0.80)
        top = dy_v[dy_v >= q80].index.tolist()
        # Sustainability screen: positive dividend growth
        top_sust = [s for s in top if dg_v[s] >= 0]
        if not top_sust:
            top_sust = top  # fallback: all top-quintile
        w = 1.0 / len(top_sust)
        weight_schedule[t] = {s: w for s in top_sust}

    gross_ret, to_series = portfolio_returns_from_weights(weight_schedule, tr_df.reindex(trading_idx), start, end)
    net_ret = apply_costs(gross_ret, to_series, cost_bps, slip_bps)

    spy = load_price_series("SPY", start=start, end=end, adjustment=ADJ_TOTALRETURN, cache_dir=cache)
    bm  = spy["Close"].reindex(trading_idx, method="ffill").pct_change(fill_method=None).fillna(0.0) if not spy.empty else gross_ret * 0

    ann_to = float(to_series.sum() / max(len(trading_idx) / TRADING_DAYS, 1))
    log.info("S26 done. Ann turnover: %.2fx", ann_to)

    return {
        "returns": net_ret, "benchmark": bm,
        "description": DESCRIPTION, "turnover_annual": ann_to,
    }
