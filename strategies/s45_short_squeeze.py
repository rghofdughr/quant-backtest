"""
S45 — Short-squeeze candidates
Reference: Asquith, Pathak & Ritter (2005); Cohen, Diether & Malloy (2007)
Universe:  Russell 2000 C&P PIT (small-caps most vulnerable to squeezes)
Signal:    Short interest ratio (SIR = shares_short / avg_daily_volume) > 10 days,
           combined with 20-day price momentum > 0.
           High SIR + positive momentum → squeeze candidate → long.
Data:      Short interest: FINRA biweekly downloads (free, ~2-week lag).
           FundamentalsStub.short_interest(symbols, date) → Series of SIR values.
           Momentum component (Norgate): fully implemented below.
Note:      FINRA biweekly short interest file at finra.org; requires parsing and DB build.
           S&P 500 short interest history available via IEX Cloud or similar.
"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import logging
import numpy as np
import pandas as pd

from data import FundamentalsStub, load_price_series, watchlist_symbols, ADJ_TOTALRETURN
from engine import portfolio_returns_from_weights, apply_costs

log = logging.getLogger(__name__)
DESCRIPTION = "Short-squeeze candidates: high SIR + momentum [STUB for SIR — needs FINRA]"
TRADING_DAYS = 252
MOM_LOOKBACK = 20
SIR_THRESH   = 10.0   # days to cover


def run(config: dict) -> dict:
    cfg      = config["backtest"]
    start    = cfg["start_date"]
    end      = cfg["end_date"]
    cache    = config["paths"]["cache_dir"]
    cost_bps = config["costs"]["equity_cost_bps"]
    slip_bps = config["costs"]["equity_slippage_bps"]

    univ_name = config["universes"]["russell2000"]
    syms      = watchlist_symbols(univ_name)
    trading_idx = pd.bdate_range(start, end)

    spy = load_price_series("SPY", start=start, end=end, adjustment=ADJ_TOTALRETURN, cache_dir=cache)
    bm  = spy["Close"].reindex(trading_idx, method="ffill").pct_change(fill_method=None).fillna(0.0) if not spy.empty else pd.Series(0.0, index=trading_idx)

    fund = FundamentalsStub()
    sir_available = False
    try:
        fund.short_interest(syms[0] if syms else "SPY", pd.Timestamp(start).date())
        sir_available = True
    except NotImplementedError:
        log.warning("S45: short_interest stub not implemented. Needs FINRA biweekly data.")

    if not sir_available:
        # Momentum-only proxy: high recent momentum in small caps as squeeze proxy
        log.info("S45: falling back to momentum-only proxy (no SIR data)")
        close_map = {}
        for sym in syms[:200]:  # limit for smoke-test speed
            df = load_price_series(sym, start=start, end=end, adjustment=ADJ_TOTALRETURN, cache_dir=cache)
            if not df.empty:
                close_map[sym] = df["Close"]

        if not close_map:
            return {"returns": pd.Series(0.0, index=trading_idx), "benchmark": bm,
                    "description": DESCRIPTION, "turnover_annual": 0.0}

        close_df = pd.DataFrame(close_map).reindex(trading_idx, method="ffill")
        mom_df   = close_df.pct_change(MOM_LOOKBACK, fill_method=None)

        weight_schedule = {}
        month_starts = trading_idx[
            (trading_idx.month != pd.Series(trading_idx.month).shift(1).fillna(-1).values)
        ]
        for t in month_starts:
            ranks = mom_df.loc[t].dropna()
            if ranks.empty:
                continue
            top = ranks.nlargest(20).index.tolist()
            weight_schedule[t] = {s: 1.0 / len(top) for s in top}

        gross_ret, to_series = portfolio_returns_from_weights(weight_schedule, close_df, start, end)
        net_ret = apply_costs(gross_ret, to_series, cost_bps, slip_bps)
        ann_to  = float(to_series.sum() / max(len(trading_idx) / TRADING_DAYS, 1))

        return {"returns": net_ret, "benchmark": bm,
                "description": DESCRIPTION + " [MOM PROXY — plug in FINRA SIR data]",
                "turnover_annual": ann_to}

    return {"returns": pd.Series(0.0, index=trading_idx), "benchmark": bm,
            "description": DESCRIPTION, "turnover_annual": 0.0}
