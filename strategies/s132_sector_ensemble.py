"""
S132 -- Sector Ensemble: rank-average across lookbacks instead of single parameter
Universe:  11 SPDR sector ETFs
Signal:    For each sector, compute rank (1=best) across 5 lookbacks (1/3/6/9/12m).
           Composite score = mean rank. Hold top-3 by composite score.
Cash filter: only invest if best sector has positive return in majority (>=3) of lookbacks.
Rationale: stops betting on a single lookback draw; ensemble is more robust OOS.
"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import logging
import numpy as np
import pandas as pd

from data import load_price_series, ADJ_TOTALRETURN
from engine import portfolio_returns_from_weights, apply_costs

log = logging.getLogger(__name__)
DESCRIPTION = "Sector ensemble: composite rank across 1/3/6/9/12m lookbacks, top-3"

SECTORS   = ["XLK", "XLF", "XLE", "XLV", "XLI", "XLY", "XLP", "XLRE", "XLB", "XLU", "XLC"]
LOOKBACKS = [21, 63, 126, 189, 252]   # 1m, 3m, 6m, 9m, 12m
TOP_N     = 3
VOTE_THRESHOLD = 3    # majority of 5 lookbacks must show positive return for cash filter
TRADING_DAYS = 252


def run(config: dict) -> dict:
    cfg      = config["backtest"]
    start    = cfg["start_date"]
    end      = cfg["end_date"]
    cache    = config["paths"]["cache_dir"]
    cost_bps = config["costs"]["equity_cost_bps"]
    slip_bps = config["costs"]["equity_slippage_bps"]

    prices = {}
    for sym in SECTORS:
        df = load_price_series(sym, start="1998-01-01", end=end,
                               adjustment=ADJ_TOTALRETURN, cache_dir=cache)
        if not df.empty and "Close" in df.columns:
            prices[sym] = df["Close"]

    close_df = pd.DataFrame(prices).sort_index()
    reb_dates = pd.date_range(start, end, freq="BME")
    weight_schedule: dict = {}

    max_lb = max(LOOKBACKS)

    for d in reb_dates:
        avail = close_df.index[close_df.index <= d]
        if len(avail) < max_lb + 2:
            continue
        row = len(avail) - 1

        # Compute return for each sector at each lookback
        rank_matrix = {}   # {sym: [rank_lb1, rank_lb2, ...]}
        ret_matrix  = {}   # {sym: [ret_lb1, ret_lb2, ...]}

        p_now = close_df.iloc[row]

        for lb in LOOKBACKS:
            p_past = close_df.iloc[max(0, row - lb)]
            lb_rets = {}
            for sym in SECTORS:
                if sym in p_now.index and sym in p_past.index:
                    pn, pp = p_now[sym], p_past[sym]
                    if pd.notna(pn) and pd.notna(pp) and pp > 0:
                        lb_rets[sym] = pn / pp - 1.0

            if not lb_rets:
                continue

            # Rank: 1=best (highest return)
            sorted_syms = sorted(lb_rets, key=lb_rets.get, reverse=True)
            for i, sym in enumerate(sorted_syms):
                rank_matrix.setdefault(sym, []).append(i + 1)
                ret_matrix.setdefault(sym, []).append(lb_rets[sym])

        if len(rank_matrix) < TOP_N:
            weight_schedule[d] = {}
            continue

        # Composite score = mean rank (lower = better)
        composite = {sym: np.mean(ranks) for sym, ranks in rank_matrix.items()
                     if len(ranks) >= len(LOOKBACKS) // 2 + 1}  # need at least majority of lookbacks

        if len(composite) < TOP_N:
            weight_schedule[d] = {}
            continue

        # Sort by composite score (ascending = best)
        ranked = sorted(composite.items(), key=lambda x: x[1])
        top_sym = ranked[0][0]

        # Cash filter: best composite sector must show positive return in >= VOTE_THRESHOLD lookbacks
        top_rets = ret_matrix.get(top_sym, [])
        positive_votes = sum(1 for r in top_rets if r > 0)
        if positive_votes < VOTE_THRESHOLD:
            weight_schedule[d] = {}
            continue

        selected = [sym for sym, _ in ranked[:TOP_N]]
        weight_schedule[d] = {s: 1.0 / len(selected) for s in selected}

    if not weight_schedule:
        return {"returns": pd.Series(dtype=float), "description": DESCRIPTION, "turnover_annual": 0.0}

    gross_ret, to = portfolio_returns_from_weights(weight_schedule, close_df, start, end)
    net_ret  = apply_costs(gross_ret, to, cost_bps, slip_bps)
    ann_to   = float(to.sum() / max(len(to) / TRADING_DAYS, 1))

    log.info("S132 done. Ann turnover: %.2fx", ann_to)
    return {"returns": net_ret, "description": DESCRIPTION, "turnover_annual": ann_to}
