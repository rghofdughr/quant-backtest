"""
S133 -- Credit-Gated Sector Ensemble
Signal:  Same sector ensemble as S132 (composite rank across 1/3/6/9/12m).
Gate:    HYG/LQD ratio vs 63d MA (from S90). When credit is stressed (ratio < MA),
         go flat. When credit is tightening (ratio > MA), full sector ensemble.
Rationale: credit spreads are a LEADING indicator of equity regime transitions.
           Replaces S128's slow momentum cash filter (9m lookback) with a faster
           fundamental regime indicator. Expected: sharper drawdown cuts.
"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import logging
import numpy as np
import pandas as pd

from data import load_price_series, ADJ_TOTALRETURN
from engine import portfolio_returns_from_weights, apply_costs

log = logging.getLogger(__name__)
DESCRIPTION = "Credit-gated sector ensemble: sector rotation only when HYG/LQD > 63d MA"

SECTORS   = ["XLK", "XLF", "XLE", "XLV", "XLI", "XLY", "XLP", "XLRE", "XLB", "XLU", "XLC"]
LOOKBACKS = [21, 63, 126, 189, 252]
TOP_N     = 3
VOTE_THRESHOLD = 3
CREDIT_MA_DAYS = 63
TRADING_DAYS   = 252


def run(config: dict) -> dict:
    cfg      = config["backtest"]
    start    = cfg["start_date"]
    end      = cfg["end_date"]
    cache    = config["paths"]["cache_dir"]
    cost_bps = config["costs"]["equity_cost_bps"]
    slip_bps = config["costs"]["equity_slippage_bps"]

    # --- Load credit signal (HYG/LQD) ---
    credit_start = "2007-01-01"
    hyg_df = load_price_series("HYG", start=credit_start, end=end, adjustment=ADJ_TOTALRETURN, cache_dir=cache)
    lqd_df = load_price_series("LQD", start=credit_start, end=end, adjustment=ADJ_TOTALRETURN, cache_dir=cache)
    if hyg_df.empty or lqd_df.empty:
        raise RuntimeError("S133: HYG or LQD data missing")

    trading_idx = pd.bdate_range(start, end)
    hyg = hyg_df["Close"].reindex(trading_idx, method="ffill")
    lqd = lqd_df["Close"].reindex(trading_idx, method="ffill")
    ratio    = (hyg / lqd).replace([np.inf, -np.inf], np.nan)
    ratio_ma = ratio.rolling(CREDIT_MA_DAYS, min_periods=CREDIT_MA_DAYS // 2).mean()

    def credit_risk_on(d):
        if d not in ratio.index:
            return True  # default to on when no data
        r  = ratio.get(d, np.nan)
        ma = ratio_ma.get(d, np.nan)
        if not (np.isfinite(r) and np.isfinite(ma)):
            return True
        return bool(r > ma)

    # --- Load sector prices ---
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

    risk_on_count = 0

    for d in reb_dates:
        avail = close_df.index[close_df.index <= d]
        if len(avail) < max_lb + 2:
            continue
        row = len(avail) - 1

        # Check credit gate first
        if not credit_risk_on(d):
            weight_schedule[d] = {}
            continue

        risk_on_count += 1

        # Sector ensemble (same logic as S132)
        p_now = close_df.iloc[row]
        rank_matrix = {}
        ret_matrix  = {}

        for lb in LOOKBACKS:
            p_past  = close_df.iloc[max(0, row - lb)]
            lb_rets = {}
            for sym in SECTORS:
                if sym in p_now.index and sym in p_past.index:
                    pn, pp = p_now[sym], p_past[sym]
                    if pd.notna(pn) and pd.notna(pp) and pp > 0:
                        lb_rets[sym] = pn / pp - 1.0
            if not lb_rets:
                continue
            sorted_syms = sorted(lb_rets, key=lb_rets.get, reverse=True)
            for i, sym in enumerate(sorted_syms):
                rank_matrix.setdefault(sym, []).append(i + 1)
                ret_matrix.setdefault(sym, []).append(lb_rets[sym])

        if len(rank_matrix) < TOP_N:
            weight_schedule[d] = {}
            continue

        composite = {sym: np.mean(ranks) for sym, ranks in rank_matrix.items()
                     if len(ranks) >= len(LOOKBACKS) // 2 + 1}
        if len(composite) < TOP_N:
            weight_schedule[d] = {}
            continue

        ranked  = sorted(composite.items(), key=lambda x: x[1])
        top_sym = ranked[0][0]
        top_rets = ret_matrix.get(top_sym, [])
        if sum(1 for r in top_rets if r > 0) < VOTE_THRESHOLD:
            weight_schedule[d] = {}
            continue

        selected = [sym for sym, _ in ranked[:TOP_N]]
        weight_schedule[d] = {s: 1.0 / len(selected) for s in selected}

    n_reb = sum(1 for d in weight_schedule if weight_schedule[d])
    n_tot = len(weight_schedule)
    log.info("S133: credit gate passed %.0f%% of rebalances (%d/%d)",
             100 * risk_on_count / max(n_tot, 1), risk_on_count, n_tot)

    if not weight_schedule:
        return {"returns": pd.Series(dtype=float), "description": DESCRIPTION, "turnover_annual": 0.0}

    gross_ret, to = portfolio_returns_from_weights(weight_schedule, close_df, start, end)
    net_ret  = apply_costs(gross_ret, to, cost_bps, slip_bps)
    ann_to   = float(to.sum() / max(len(to) / TRADING_DAYS, 1))

    log.info("S133 done. Ann turnover: %.2fx", ann_to)
    return {"returns": net_ret, "description": DESCRIPTION, "turnover_annual": ann_to}
