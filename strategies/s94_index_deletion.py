"""
S94 — Index deletion drift / reversal.
Detect S&P 500 deletions (1->0 flips in constituent timeseries).
Test: long the deleted stocks for 20 trading days post-deletion (reversal hypothesis).
Excludes stocks that are delisted within 5 days of deletion (bankruptcy/M&A forced exits).
Limitation: Norgate gives effective date, not announcement. Documented in description.
"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import logging
import numpy as np
import pandas as pd
from data import load_price_series, watchlist_symbols, index_constituent_mask, ADJ_TOTALRETURN
from engine import apply_costs

log = logging.getLogger(__name__)
TRADING_DAYS = 252
DESCRIPTION = "S&P 500 deletion reversal: long deleted stocks for 20d post-effective-date. Excludes delist-within-5d."
HOLD_DAYS = 20
DELIST_GRACE = 5

def run(config):
    cfg = config["backtest"]
    start, end = cfg["start_date"], cfg["end_date"]
    is_smoke = config.get("smoke", False)
    cache = config["paths"]["cache_dir"]
    cost_bps = config["costs"]["equity_cost_bps"]
    slip_bps = config["costs"]["equity_slippage_bps"]
    min_px = config["liquidity"]["min_price"]
    wl_name = config["universes"].get("sp500", "S&P 500 Current & Past")
    idx_name = config["universes"].get("sp500_index", "S&P 500")
    start_load = "2013-01-01" if is_smoke else "1999-01-01"

    symbols = watchlist_symbols(wl_name)
    if is_smoke:
        symbols = symbols[:200]

    close_map, mask_map = {}, {}
    for sym in symbols:
        df = load_price_series(sym, start=start_load, end=end, adjustment=ADJ_TOTALRETURN, cache_dir=cache)
        if df.empty:
            continue
        close_map[sym] = df["Close"]
        m = index_constituent_mask(sym, idx_name, start=start_load, end=end, cache_dir=cache)
        if not m.empty:
            mask_map[sym] = m

    trading_idx = pd.bdate_range(start, end)
    close_df = pd.DataFrame(close_map).reindex(trading_idx, method="ffill")
    member_df = (pd.DataFrame(mask_map).reindex(trading_idx, method="ffill")
                 .infer_objects(copy=False).fillna(False).astype(bool))

    # Detect deletions: 1->0 transitions
    deletions = []
    for sym, mask in mask_map.items():
        m_idx = mask.reindex(trading_idx, method="ffill").fillna(False).astype(bool)
        changes = m_idx.astype(int).diff()
        del_dates = trading_idx[changes.values == -1]
        for d in del_dates:
            deletions.append((d, sym))
    deletions = sorted(deletions)
    log.info("S94: detected %d deletions", len(deletions))

    ret_df = close_df.pct_change(fill_method=None).fillna(0.0)
    position_df = pd.DataFrame(0.0, index=trading_idx, columns=list(close_map.keys()))

    for del_date, sym in deletions:
        if del_date < pd.Timestamp(start) or sym not in ret_df.columns:
            continue
        start_idx = trading_idx.searchsorted(del_date)
        if start_idx >= len(trading_idx):
            continue

        # Exclude stocks that delist within DELIST_GRACE days (forced exits)
        end_check = min(start_idx + DELIST_GRACE, len(trading_idx))
        post_data = close_df[sym].iloc[start_idx:end_check].dropna()
        if len(post_data) < DELIST_GRACE and len(post_data) < 2:
            log.debug("S94: %s excluded (delist within %d days of deletion)", sym, DELIST_GRACE)
            continue

        end_idx = min(start_idx + HOLD_DAYS, len(trading_idx))
        position_df.loc[trading_idx[start_idx:end_idx], sym] += 1.0

    n_active = position_df.abs().sum(axis=1).clip(lower=1e-9)
    weight_df = position_df.div(n_active, axis=0)
    weight_df[(n_active < 0.5).values] = 0.0

    port_ret = (weight_df * ret_df.reindex(trading_idx).fillna(0.0)).sum(axis=1)
    to = weight_df.diff().abs().sum(axis=1) / 2.0
    net_ret = apply_costs(port_ret, to, cost_bps, slip_bps)

    spy = load_price_series("SPY", start, end, ADJ_TOTALRETURN, cache)
    bm = spy["Close"].pct_change(fill_method=None).reindex(net_ret.index)
    ann_to = float(to.sum() / max(len(to) / TRADING_DAYS, 1))
    active_pct = float((position_df.abs().sum(axis=1) > 0).mean()) * 100
    log.info("S94 done. Active: %.1f%% of days", active_pct)
    return {"returns": net_ret, "benchmark": bm, "description": DESCRIPTION, "turnover_annual": ann_to}
