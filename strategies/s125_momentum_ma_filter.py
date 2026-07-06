"""
s125 — Momentum with Market Regime Filter
Signal:    12-1 month cross-sectional momentum (Jegadeesh-Titman), but
           only hold when SPY is above its 200-day SMA.
           Otherwise: go to cash.
Long:      top decile by 12-1m return when SPY > 200d MA
Short:     bottom decile when SPY > 200d MA (long-short variant)
Cash:      when SPY < 200d MA
Rebalance: Monthly
Universe:  Russell 1000 PIT

Rationale: Momentum crashes happen in volatile, recovery-phase markets.
           The 200-day MA acts as a "crash avoidance" filter (Daniel & Moskowitz 2016).
"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import logging
import numpy as np
import pandas as pd
from data import load_price_series, watchlist_symbols, index_constituent_mask, ADJ_TOTALRETURN, compute_dollar_volume
from engine import portfolio_returns_from_weights, apply_costs

log = logging.getLogger(__name__)
DESCRIPTION = "Momentum + 200d MA regime filter: hold momentum only when SPY above 200d SMA"

MOM_LOOKBACK = 252    # 12 months
MOM_SKIP     = 21     # skip last month
MA_WINDOW    = 200    # 200-day MA for regime filter
DECILE       = 0.10
MIN_STOCKS   = 30
LONG_SHORT   = True   # True = long bottom too (hedged), False = long only


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
        df = load_price_series(sym, start="1997-01-01", end=end,
                               adjustment=ADJ_TOTALRETURN, cache_dir=cache)
        if df.empty or df["Close"].max() < min_px:
            continue
        close_map[sym] = df["Close"]
        dv_map[sym]    = compute_dollar_volume(df)
        m = index_constituent_mask(sym, idx_name, start="1997-01-01", end=end, cache_dir=cache)
        if not m.empty:
            mask_map[sym] = m

    trading_idx = pd.bdate_range(start, end)
    close_df  = pd.DataFrame(close_map).reindex(trading_idx, method="ffill")
    dv_df     = pd.DataFrame(dv_map).reindex(trading_idx, method="ffill")
    member_df = (pd.DataFrame(mask_map).reindex(trading_idx, method="ffill")
                 .infer_objects(copy=False).fillna(False).astype(bool))
    for col in close_df.columns:
        if col not in member_df.columns:
            member_df[col] = False

    # SPY for regime filter
    spy_df  = load_price_series("SPY", start="1997-01-01", end=end,
                                adjustment=ADJ_TOTALRETURN, cache_dir=cache)
    spy_cls = spy_df["Close"].reindex(trading_idx, method="ffill")
    spy_ma  = spy_cls.rolling(MA_WINDOW).mean()

    # Pre-compute momentum for all stocks
    mom_df = close_df.pct_change(MOM_LOOKBACK - MOM_SKIP) / close_df.pct_change(MOM_SKIP).add(1) - 1
    # Simpler: 12m return skipping last month
    mom_df = (close_df.shift(MOM_SKIP) / close_df.shift(MOM_LOOKBACK) - 1)

    reb_dates = pd.date_range(start, end, freq="BME")
    weight_schedule = {}

    for d in reb_dates:
        pos = trading_idx.searchsorted(d, side="right") - 1
        if pos < MOM_LOOKBACK + 5:
            continue

        # Regime check: SPY above 200d MA?
        spy_price = spy_cls.iloc[pos]
        spy_ma_val = spy_ma.iloc[pos]
        if pd.isna(spy_ma_val) or spy_price < spy_ma_val:
            # Market below MA → go to cash
            weight_schedule[d] = {}
            continue

        members = member_df.iloc[pos]
        dv_now  = dv_df.iloc[pos]
        px_now  = close_df.iloc[pos]
        valid   = [c for c in close_df.columns
                   if members.get(c, False)
                   and dv_now.get(c, 0.0) >= min_dv
                   and px_now.get(c, 0.0) >= min_px]
        if len(valid) < MIN_STOCKS:
            continue

        mom = mom_df.iloc[pos][valid].dropna()
        if len(mom) < MIN_STOCKS:
            continue

        hi_cut = mom.quantile(1 - DECILE)
        lo_cut = mom.quantile(DECILE)
        longs  = mom[mom >= hi_cut].index.tolist()   # winners
        shorts = mom[mom <= lo_cut].index.tolist()   # losers

        if not longs:
            continue

        w = {s: 1.0 / len(longs) for s in longs}
        if LONG_SHORT and shorts:
            for s in shorts:
                w[s] = w.get(s, 0.0) - 1.0 / len(shorts)
        weight_schedule[d] = w

    gross_ret, to = portfolio_returns_from_weights(weight_schedule, close_df, start, end)
    net_ret = apply_costs(gross_ret, to, cost_bps, slip_bps)
    ann_to  = float(to.sum() / max(len(to) / 252, 1))
    return {"returns": net_ret, "turnover_annual": ann_to, "description": DESCRIPTION}
