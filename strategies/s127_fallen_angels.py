"""
s127 — Fallen Angels
Signal:    Stocks in R1000 that have fallen 50%+ from their 52-week high.
           These are "quality" large-caps undergoing stress — mean reversion candidates.
Long:      All qualifying stocks, equal-weighted basket
Cash:      When no qualifying stocks meet threshold
Rebalance: Monthly
Universe:  Russell 1000 PIT

Note: No profitability filter (fundamental data unavailable). Pure price-based.
      "Fallen angel" in fixed income means downgraded bonds; here adapted to equity.
      Conceptual base: contrarian / mean reversion in large-cap distressed names.
"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import logging
import numpy as np
import pandas as pd
from data import load_price_series, watchlist_symbols, index_constituent_mask, ADJ_TOTALRETURN, compute_dollar_volume
from engine import portfolio_returns_from_weights, apply_costs

log = logging.getLogger(__name__)
DESCRIPTION = "Fallen Angels: R1000 stocks down 50%+ from 52-week high, equal-weight basket"

HIGH_WINDOW  = 252    # 52-week high lookback
THRESHOLD    = -0.50  # must be at least 50% below 52wk high
MAX_HOLDINGS = 40     # cap to avoid extremely thin baskets
MIN_HOLDINGS = 5


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

    # Pre-compute rolling 52-week high and discount from high
    high_df     = close_df.rolling(HIGH_WINDOW).max()
    discount_df = close_df / high_df - 1   # 0 = at high, -0.5 = 50% below high

    reb_dates = pd.date_range(start, end, freq="BME")
    weight_schedule = {}

    for d in reb_dates:
        pos = trading_idx.searchsorted(d, side="right") - 1
        if pos < HIGH_WINDOW + 5:
            continue

        members  = member_df.iloc[pos]
        dv_now   = dv_df.iloc[pos]
        px_now   = close_df.iloc[pos]
        disc_now = discount_df.iloc[pos]

        valid = [c for c in close_df.columns
                 if members.get(c, False)
                 and dv_now.get(c, 0.0) >= min_dv
                 and px_now.get(c, 0.0) >= min_px]
        if not valid:
            continue

        disc_valid = disc_now[valid].dropna()
        # Fallen angels: below threshold
        fallen = disc_valid[disc_valid <= THRESHOLD].sort_values()

        if len(fallen) < MIN_HOLDINGS:
            weight_schedule[d] = {}
            continue

        # Cap to MAX_HOLDINGS deepest fallen (most beaten-down first)
        picks = fallen.iloc[:MAX_HOLDINGS].index.tolist()
        weight_schedule[d] = {s: 1.0 / len(picks) for s in picks}

    gross_ret, to = portfolio_returns_from_weights(weight_schedule, close_df, start, end)
    net_ret = apply_costs(gross_ret, to, cost_bps, slip_bps)
    ann_to  = float(to.sum() / max(len(to) / 252, 1))
    return {"returns": net_ret, "turnover_annual": ann_to, "description": DESCRIPTION}
