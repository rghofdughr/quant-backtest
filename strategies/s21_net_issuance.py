"""
S21 — Net share issuance anomaly (Pontiff & Woodgate 2008)
Universe:  Large/mid cap (S&P 500 C&P)
Signal:    YoY change in shares outstanding (dilution → sell; buybacks → buy).
           CAVEAT: Norgate provides current shares outstanding (scalar), NOT a
           point-in-time timeseries. This implementation uses split-adjust ratio
           changes as a PROXY: split-adjusted price / unadjusted price change
           captures stock splits but NOT dilutive issuances.
           For a proper implementation, use Sharadar Core US Fundamentals which
           provides historical shares outstanding as a PIT quarterly timeseries.
Rebalance: Quarterly (after earnings season, March/June/Sep/Dec)
Execution: Next trading day close
Status:    PARTIAL — split-ratio proxy only; see NOTE above.
"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import logging
import warnings
import numpy as np
import pandas as pd

from data import (
    load_price_series, watchlist_symbols, index_constituent_mask,
    ADJ_TOTALRETURN, ADJ_NONE, compute_dollar_volume,
)
from engine import portfolio_returns_from_weights, apply_costs

log = logging.getLogger(__name__)
DESCRIPTION = ("Net share issuance proxy (split-adjust ratio), S&P 500 PIT — "
               "PARTIAL: proxy detects splits, not dilutive issuances; "
               "full implementation requires Sharadar fundamentals.")
TRADING_DAYS = 252


def _approx_shares_change(sym: str, cache: str, end: str) -> pd.Series:
    """
    Approximate YoY shares-outstanding change from split-adjust ratio.
    Returns a quarterly series of 'dilution proxy' (positive = more shares).
    """
    df_adj  = load_price_series(sym, start="1997-01-01", end=end,
                                adjustment=ADJ_TOTALRETURN, cache_dir=cache)
    df_raw  = load_price_series(sym, start="1997-01-01", end=end,
                                adjustment=ADJ_NONE, cache_dir=cache)

    if df_adj.empty or df_raw.empty:
        return pd.Series(dtype=float)

    # Adjust ratio = adj_close / raw_close; changes indicate splits
    ratio = (df_adj["Close"] / df_raw["Close"].replace(0, np.nan)).dropna()
    ratio_qtr = ratio.resample("QE").last()
    # YoY change in adjust ratio ≈ cumulative splits over the year
    # Positive change → more shares (dilution); negative → buybacks/reverse splits
    yoy = ratio_qtr.pct_change(4)   # 4 quarters = 1 year
    return yoy


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

    wl_name  = config["universes"].get("sp500", "S&P 500 Current & Past")
    idx_name = config["universes"].get("sp500_index", "S&P 500")
    start_load = "2013-01-01" if is_smoke else "1998-01-01"

    log.info("S21: loading S&P 500 C&P (split-ratio issuance proxy) ...")
    symbols = watchlist_symbols(wl_name)
    if is_smoke:
        symbols = symbols[:80]

    close_map, dv_map, mask_map, issuance_map = {}, {}, {}, {}

    for sym in symbols:
        df = load_price_series(sym, start=start_load, end=end,
                               adjustment=ADJ_TOTALRETURN, cache_dir=cache)
        if df.empty or df["Close"].max() < min_px:
            continue
        close_map[sym] = df["Close"]
        dv_map[sym]    = compute_dollar_volume(df)

        iss = _approx_shares_change(sym, cache, end)
        if not iss.dropna().empty:
            issuance_map[sym] = iss

        m = index_constituent_mask(sym, idx_name, start=start_load, end=end, cache_dir=cache)
        if not m.empty:
            mask_map[sym] = m

    log.info("S21: %d symbols, %d with issuance proxy", len(close_map), len(issuance_map))

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

    # Quarterly rebalance (end of March, June, Sep, Dec)
    reb_dates = pd.date_range(start, end, freq="QE")
    weight_schedule: dict = {}

    for d in reb_dates:
        pos = trading_idx.searchsorted(d, side="right") - 1
        if pos < 0:
            continue

        prices  = close_df.iloc[pos]
        dv_row  = dv_df.iloc[pos]
        members = member_df.iloc[pos]

        signals = {}
        for sym, iss in issuance_map.items():
            iss_avail = iss[iss.index <= pd.Timestamp(d)]
            if iss_avail.empty or np.isnan(iss_avail.iloc[-1]):
                continue
            signals[sym] = float(iss_avail.iloc[-1])

        valid = {
            s: v for s, v in signals.items()
            if bool(members.get(s, False))
            and prices.get(s, 0) >= min_px
            and dv_row.get(s, 0) >= min_dv
        }
        if len(valid) < 20:
            continue

        ranked = sorted(valid.items(), key=lambda x: x[1])
        n_each = max(1, len(ranked) // 5)
        # Short high issuers (positive ΔShares), long repurchasers (negative ΔShares)
        longs  = [s for s, _ in ranked[:n_each]]    # lowest ΔShares = repurchasers
        shorts = [s for s, _ in ranked[-n_each:]]   # highest ΔShares = issuers

        w = {s:  1.0 / len(longs)  for s in longs}
        w.update({s: -1.0 / len(shorts) for s in shorts})
        weight_schedule[d] = w

    if not weight_schedule:
        warnings.warn("S21: no valid rebalance periods. Issuance proxy may lack signal.")
        return {"returns": pd.Series(dtype=float), "benchmark": pd.Series(dtype=float),
                "description": DESCRIPTION, "turnover_annual": 0.0}

    gross_ret, to = portfolio_returns_from_weights(weight_schedule, close_df, start, end)
    net_ret = apply_costs(gross_ret, to, cost_bps, slip_bps)

    spy = load_price_series("SPY", start, end, ADJ_TOTALRETURN, cache)
    bm  = spy["Close"].pct_change(fill_method=None).reindex(net_ret.index)

    ann_to = float(to.sum() / max(len(to) / TRADING_DAYS, 1))
    log.info("S21 done. Ann turnover: %.2fx (proxy signal only)", ann_to)

    return {
        "returns": net_ret, "benchmark": bm,
        "description": DESCRIPTION, "turnover_annual": ann_to,
    }
