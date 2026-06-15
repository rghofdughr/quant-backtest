import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import logging
import numpy as np
import pandas as pd
from data import load_price_series, ADJ_TOTALRETURN
from engine import portfolio_returns_from_weights, apply_costs

log = logging.getLogger(__name__)
TRADING_DAYS = 252
DESCRIPTION = "Cross-asset ETF momentum. Long top-3, short bottom-3 of 9 ETFs on 12-1 month return. Monthly rebalance."

ETFS = ["SPY", "EFA", "EEM", "TLT", "IEF", "GLD", "DBC", "VNQ", "HYG"]


def run(config):
    cfg = config["backtest"]
    start, end = cfg["start_date"], cfg["end_date"]
    is_smoke = config.get("smoke", False)
    cache = config["paths"]["cache_dir"]
    cost_bps = config["costs"]["equity_cost_bps"]
    slip_bps = config["costs"]["equity_slippage_bps"]
    start_load = "2013-01-01" if is_smoke else "2000-01-01"

    MOM_LONG = 252   # 12 months
    MOM_SHORT = 21   # skip 1 month
    WARMUP = MOM_LONG + 10

    # Load all ETFs
    etf_close = {}
    for sym in ETFS:
        df = load_price_series(sym, start=start_load, end=end, adjustment=ADJ_TOTALRETURN, cache_dir=cache)
        if not df.empty:
            etf_close[sym] = df["Close"]

    if len(etf_close) < 3:
        empty = pd.Series(dtype=float)
        return {"returns": empty, "benchmark": empty, "description": DESCRIPTION, "turnover_annual": 0.0}

    # Align on common business days from start_load to end
    full_idx = pd.bdate_range(start_load, end)
    close_full = pd.DataFrame(etf_close).reindex(full_idx, method="ffill")

    # Build close_df on the backtest window for portfolio_returns_from_weights
    trading_idx = pd.bdate_range(start, end)
    close_df = close_full.reindex(trading_idx, method="ffill")

    reb_dates = pd.date_range(start, end, freq="BME")
    weight_schedule = {}

    for rd in reb_dates:
        rd = min(rd, trading_idx[-1])
        pos_full = full_idx.searchsorted(rd)
        if pos_full < WARMUP:
            continue

        long_pos = max(0, pos_full - MOM_LONG)
        short_pos = max(0, pos_full - MOM_SHORT)
        if short_pos <= long_pos:
            continue

        px_now = close_full.iloc[short_pos]
        px_then = close_full.iloc[long_pos]

        mom = {}
        for sym in ETFS:
            if sym not in close_full.columns:
                continue
            p0 = px_then.get(sym, np.nan)
            p1 = px_now.get(sym, np.nan)
            if np.isfinite(p0) and np.isfinite(p1) and p0 > 0:
                mom[sym] = p1 / p0 - 1.0

        if len(mom) < 6:
            continue

        mom_series = pd.Series(mom).sort_values(ascending=False)
        ranked = mom_series.index.tolist()
        n = len(ranked)

        # Long top 3, short bottom 3, flat middle
        n_long = min(3, n // 3)
        n_short = min(3, n // 3)
        long_names = ranked[:n_long]
        short_names = ranked[-n_short:]

        w = {}
        for s in long_names:
            w[s] = 1.0 / n_long
        for s in short_names:
            w[s] = w.get(s, 0.0) - 1.0 / n_short

        weight_schedule[rd] = w

    gross_ret, to = portfolio_returns_from_weights(weight_schedule, close_df, start, end)
    net_ret = apply_costs(gross_ret, to, cost_bps, slip_bps)
    spy_df = load_price_series("SPY", start, end, ADJ_TOTALRETURN, cache)
    bm = spy_df["Close"].pct_change(fill_method=None).reindex(net_ret.index)

    # Compute turnover as sum of abs weight changes across rebalances / years
    n_years = max((pd.Timestamp(end) - pd.Timestamp(start)).days / 365.25, 1)
    ann_to = float(to.sum() / max(len(to) / TRADING_DAYS, 1))
    return {"returns": net_ret, "benchmark": bm, "description": DESCRIPTION, "turnover_annual": ann_to}
