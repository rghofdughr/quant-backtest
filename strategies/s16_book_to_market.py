"""
S16 — Book-to-market deciles (value factor)
Reference: Fama & French (1992)
Universe:  Russell 1000 C&P PIT
Signal:    Annual book value / market cap. Long top quintile (high B/M = value).
           4-month lag for filing delay (annual reports). July rebalance standard.
Data:      REQUIRES Sharadar Core US Fundamentals (Nasdaq Data Link) or Compustat.
           Norgate does NOT provide book value. Stub returns empty.
Stub:      Strategy logic is complete. Plug in Sharadar by implementing
           data.FundamentalsStub.book_to_market(symbols, date) -> pd.Series
"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import logging
import pandas as pd

from data import FundamentalsStub, load_price_series, watchlist_symbols, ADJ_TOTALRETURN
from engine import portfolio_returns_from_weights, apply_costs

log = logging.getLogger(__name__)
DESCRIPTION = "Book-to-market value decile, Russell 1000 PIT [STUB — needs Sharadar]"
TRADING_DAYS = 252


def run(config: dict) -> dict:
    cfg      = config["backtest"]
    start    = cfg["start_date"]
    end      = cfg["end_date"]
    cache    = config["paths"]["cache_dir"]
    cost_bps = config["costs"]["equity_cost_bps"]
    slip_bps = config["costs"]["equity_slippage_bps"]

    univ_name = config["universes"]["russell1000"]
    syms      = watchlist_symbols(univ_name)
    if not syms:
        return {"returns": pd.Series(dtype=float), "benchmark": pd.Series(dtype=float),
                "description": DESCRIPTION, "turnover_annual": 0.0}

    trading_idx = pd.bdate_range(start, end)
    fund = FundamentalsStub()

    # July rebalance dates (standard Fama-French)
    rebal_dates = [t for t in trading_idx if t.month == 7 and t.day <= 7]

    weight_schedule = {}
    for t in rebal_dates:
        try:
            bm_series = fund.book_to_market(syms, t)
        except NotImplementedError:
            log.warning("S16: book_to_market data not available — plug in Sharadar. Skipping.")
            break
        top_value = bm_series[bm_series >= bm_series.quantile(0.80)].index.tolist()
        if top_value:
            weight_schedule[t] = {s: 1.0 / len(top_value) for s in top_value}

    if not weight_schedule:
        log.info("S16: no weights (fundamentals data stub not implemented)")
        spy = load_price_series("SPY", start=start, end=end, adjustment=ADJ_TOTALRETURN, cache_dir=cache)
        bm  = spy["Close"].reindex(trading_idx, method="ffill").pct_change(fill_method=None).fillna(0.0) if not spy.empty else pd.Series(0.0, index=trading_idx)
        return {"returns": pd.Series(0.0, index=trading_idx), "benchmark": bm,
                "description": DESCRIPTION, "turnover_annual": 0.0}

    close_map = {}
    for sym in syms:
        df = load_price_series(sym, start=start, end=end, adjustment=ADJ_TOTALRETURN, cache_dir=cache)
        if not df.empty:
            close_map[sym] = df["Close"]
    close_df = pd.DataFrame(close_map).reindex(trading_idx, method="ffill")

    gross_ret, to_series = portfolio_returns_from_weights(weight_schedule, close_df, start, end)
    net_ret = apply_costs(gross_ret, to_series, cost_bps, slip_bps)

    spy = load_price_series("SPY", start=start, end=end, adjustment=ADJ_TOTALRETURN, cache_dir=cache)
    bm  = spy["Close"].reindex(trading_idx, method="ffill").pct_change(fill_method=None).fillna(0.0) if not spy.empty else gross_ret * 0
    ann_to = float(to_series.sum() / max(len(trading_idx) / TRADING_DAYS, 1))

    return {"returns": net_ret, "benchmark": bm, "description": DESCRIPTION, "turnover_annual": ann_to}
