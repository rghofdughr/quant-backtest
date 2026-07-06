"""
S81 — January small-cap reversal with survivorship-free PIT universe.
Within R2000 PIT, long the bottom decile by November return.
Hold through December + January. Critically includes delisted stocks.
S39 failed because it used a survivorship-biased universe; this uses C&P.

BUG FIX (2026-06-16): Original code used portfolio_returns_from_weights()
with a weight_schedule containing one entry PER TRADING DAY in Dec+Jan.
That function uses += accumulation and computes hold periods as:
  hold_end[i] = first trading day after reb_dates[i+1]
This means each interior hold day appears in TWO consecutive hold periods
(double-counted via +=). Far worse: the last Jan-31 entry had its hold_end
set to the NEXT December window start (~10 months later), silently
accumulating ~10 months of phantom returns with the wrong weights.
Over 24 years this compounded into the reported 149% CAGR / 456% vol.

Fix: directly assign weighted returns to hold days using vectorised pandas
with a plain = (not +=), so each trading day gets exactly one year's signal.
"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import logging
import numpy as np
import pandas as pd
from data import load_price_series, watchlist_symbols, index_constituent_mask, ADJ_TOTALRETURN, compute_dollar_volume
from engine import apply_costs

log = logging.getLogger(__name__)
TRADING_DAYS = 252
DESCRIPTION = "Jan small-cap reversal (PIT survivorship-free): R2000 C&P bottom-decile Nov return, hold Dec+Jan."


def run(config):
    cfg = config["backtest"]
    start, end = cfg["start_date"], cfg["end_date"]
    is_smoke = config.get("smoke", False)
    cache = config["paths"]["cache_dir"]
    cost_bps = config["costs"]["equity_cost_bps"]
    slip_bps = config["costs"]["equity_slippage_bps"]
    min_dv = config["liquidity"]["min_dollar_volume"]
    wl_name = config["universes"].get("russell2000", "Russell 2000 Current & Past")
    idx_name = "Russell 2000"
    start_load = "2013-01-01" if is_smoke else "1997-01-01"

    symbols = watchlist_symbols(wl_name)
    if is_smoke:
        symbols = symbols[:300]

    close_map, dv_map, mask_map = {}, {}, {}
    for sym in symbols:
        df = load_price_series(sym, start=start_load, end=end, adjustment=ADJ_TOTALRETURN, cache_dir=cache)
        if df.empty:
            continue
        close_map[sym] = df["Close"]
        dv_map[sym] = compute_dollar_volume(df)
        m = index_constituent_mask(sym, idx_name, start=start_load, end=end, cache_dir=cache)
        if not m.empty:
            mask_map[sym] = m

    log.info("S81: %d symbols loaded", len(close_map))

    trading_idx = pd.bdate_range(start, end)
    close_df = pd.DataFrame(close_map).reindex(trading_idx, method="ffill")
    dv_df = pd.DataFrame(dv_map).reindex(trading_idx, method="ffill")
    member_df = (pd.DataFrame(mask_map).reindex(trading_idx, method="ffill")
                 .infer_objects(copy=False).fillna(False).astype(bool))
    for col in close_df.columns:
        if col not in member_df.columns:
            member_df[col] = False

    ret_df = close_df.pct_change(fill_method=None).fillna(0.0)
    port_rets = pd.Series(0.0, index=trading_idx)
    to_series = pd.Series(0.0, index=trading_idx)

    start_yr = pd.Timestamp(start).year
    end_yr = pd.Timestamp(end).year

    for year in range(start_yr, end_yr + 1):
        # Signal: last trading day of November
        nov_end = pd.Timestamp(f"{year}-11-30")
        sig_candidates = trading_idx[trading_idx <= nov_end]
        if len(sig_candidates) == 0:
            continue
        sig_date = sig_candidates[-1]
        sig_pos = trading_idx.searchsorted(sig_date)
        if sig_pos < 21:
            continue

        members_sig = member_df.iloc[sig_pos] if sig_pos < len(member_df) else member_df.iloc[-1]
        dv_sig = dv_df.iloc[sig_pos - 1]
        p_now = close_df.iloc[sig_pos]
        p_month_ago = close_df.iloc[max(sig_pos - 21, 0)]

        valid_cols = [
            c for c in close_df.columns
            if members_sig.get(c, False) and dv_sig.get(c, 0) >= min_dv
        ]

        nov_rets = {}
        for c in valid_cols:
            pn = p_now.get(c)
            pp = p_month_ago.get(c)
            if pn and pp and pp > 0 and np.isfinite(pn) and np.isfinite(pp):
                nov_rets[c] = pn / pp - 1.0

        if len(nov_rets) < 20:
            continue

        sr = pd.Series(nov_rets)
        bottom = sr[sr <= sr.quantile(0.10)].index.tolist()
        if not bottom:
            continue

        w_syms = [s for s in bottom if s in ret_df.columns]
        if not w_syms:
            continue

        # Hold Dec of this year through Jan of next year
        hold_start = pd.Timestamp(f"{year}-12-01")
        hold_end = pd.Timestamp(f"{year + 1}-01-31")
        hold_days = trading_idx[(trading_idx >= hold_start) & (trading_idx <= hold_end)]
        if not len(hold_days):
            continue

        # FIXED: direct assignment with =, not +=, no overlap accumulation.
        # Each trading day in Dec-Jan gets exactly this year's signal — once.
        avail = [s for s in w_syms if s in ret_df.columns]
        if avail:
            hold_rets = ret_df.loc[hold_days, avail]
            port_rets.loc[hold_days] = hold_rets.mean(axis=1)   # equal weight

        # Turnover: full portfolio turns over at entry and exit each year
        to_series[hold_days[0]] += 0.5   # entering
        to_series[hold_days[-1]] += 0.5  # exiting

    net_ret = apply_costs(port_rets, to_series, cost_bps, slip_bps)
    spy = load_price_series("SPY", start, end, ADJ_TOTALRETURN, cache)
    bm = spy["Close"].pct_change(fill_method=None).reindex(net_ret.index)
    ann_to = float(to_series.sum() / max(len(to_series) / TRADING_DAYS, 1))
    return {"returns": net_ret, "benchmark": bm, "description": DESCRIPTION, "turnover_annual": ann_to}
