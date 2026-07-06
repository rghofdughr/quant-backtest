"""
s126 — Industry Relative Strength
Signal:    Two-step filter.
           Step 1: Rank GICS Industry Groups (Level 2) by equal-weight 12-1m momentum.
                   Select top tertile of industries.
           Step 2: Within those industries, rank individual stocks by 12-1m momentum.
                   Buy top quintile of stocks.
Rebalance: Monthly
Universe:  Russell 1000 PIT
Note:      GICS classifications are current (not PIT) — mild lookahead on reclassified stocks.
           Moskowitz & Grinblatt (1999): industry momentum explains much of stock momentum.
"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import logging
import numpy as np
import pandas as pd
import norgatedata
from data import load_price_series, watchlist_symbols, index_constituent_mask, ADJ_TOTALRETURN, compute_dollar_volume
from engine import portfolio_returns_from_weights, apply_costs

log = logging.getLogger(__name__)
DESCRIPTION = "Industry Relative Strength: stock momentum within strongest GICS industry groups"

MOM_LOOKBACK  = 252
MOM_SKIP      = 21
IND_TERTILE   = 0.67   # top 1/3 of industries
STOCK_QUINTILE = 0.80  # top quintile of stocks within selected industries
MIN_STOCKS    = 10


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

    # Build GICS Level-2 mapping (current classification, not PIT)
    gics_map = {}
    for sym in symbols:
        try:
            g = norgatedata.classification_at_level(sym, 'GICS', 'Name', 2)
            if g:
                gics_map[sym] = g
        except Exception:
            pass
    log.info("s126: %d symbols with GICS classification", len(gics_map))

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

    # Pre-compute momentum (12m skip 1m)
    mom_df = close_df.shift(MOM_SKIP) / close_df.shift(MOM_LOOKBACK) - 1

    reb_dates = pd.date_range(start, end, freq="BME")
    weight_schedule = {}

    for d in reb_dates:
        pos = trading_idx.searchsorted(d, side="right") - 1
        if pos < MOM_LOOKBACK + 5:
            continue

        members = member_df.iloc[pos]
        dv_now  = dv_df.iloc[pos]
        px_now  = close_df.iloc[pos]
        valid   = [c for c in close_df.columns
                   if members.get(c, False)
                   and dv_now.get(c, 0.0) >= min_dv
                   and px_now.get(c, 0.0) >= min_px
                   and c in gics_map]
        if len(valid) < MIN_STOCKS:
            continue

        stock_mom = mom_df.iloc[pos][valid].dropna()
        if len(stock_mom) < MIN_STOCKS:
            continue

        # Step 1: industry momentum = equal-weight average of constituent stock returns
        ind_moms = {}
        for sym, mom in stock_mom.items():
            ind = gics_map[sym]
            if ind not in ind_moms:
                ind_moms[ind] = []
            ind_moms[ind].append(mom)
        ind_avg = {ind: np.mean(vals) for ind, vals in ind_moms.items() if len(vals) >= 3}

        if len(ind_avg) < 3:
            continue

        ind_series = pd.Series(ind_avg)
        threshold  = ind_series.quantile(IND_TERTILE)
        strong_industries = set(ind_series[ind_series >= threshold].index)

        # Step 2: stocks within strong industries, top quintile by own momentum
        eligible = [s for s in stock_mom.index if gics_map.get(s) in strong_industries]
        if len(eligible) < MIN_STOCKS:
            continue

        eligible_mom = stock_mom[eligible]
        cutoff = eligible_mom.quantile(STOCK_QUINTILE)
        picks  = eligible_mom[eligible_mom >= cutoff].index.tolist()
        if not picks:
            continue

        weight_schedule[d] = {s: 1.0 / len(picks) for s in picks}

    gross_ret, to = portfolio_returns_from_weights(weight_schedule, close_df, start, end)
    net_ret = apply_costs(gross_ret, to, cost_bps, slip_bps)
    ann_to  = float(to.sum() / max(len(to) / 252, 1))
    return {"returns": net_ret, "turnover_annual": ann_to, "description": DESCRIPTION}
