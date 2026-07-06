"""
S131 -- Regime-Gated: TS momentum signal gates sector rotation vs multi-asset
Signal: 1m return on full multi-asset universe; fraction_positive > 0.5 = risk-on
Risk-on  (majority trending up):   sector rotation, 9m lookback, top-3 sectors
Risk-off (majority trending down):  TS momentum on full universe, 1m, long-flat, vol-targeted
Rationale: when macro breadth is positive, concentrate in best equity sectors;
           when macro breadth is negative, follow cross-asset trends defensively.
"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import logging
import numpy as np
import pandas as pd

from data import load_price_series, ADJ_TOTALRETURN
from engine import portfolio_returns_from_weights, apply_costs

log = logging.getLogger(__name__)
DESCRIPTION = "Regime-gated: sector rotation when risk-on, TS momentum when risk-off"

SECTORS = ["XLK", "XLF", "XLE", "XLV", "XLI", "XLY", "XLP", "XLRE", "XLB", "XLU", "XLC"]

# Multi-asset universe for the regime signal (same as S02)
MACRO_UNIVERSE = [
    "ES", "CL", "NG", "DX", "EMD", "HG", "ZS", "SB",
    "QQQ", "IWM", "TLT", "IEF", "GLD", "SLV", "FXE", "EEM", "DBC", "CORN",
]

SECTOR_LB_DAYS   = 9 * 21   # 9m for sector ranking
SECTOR_TOP_N     = 3
TS_LOOKBACK      = 21       # 1m for regime signal and TS momentum
VOL_TARGET       = 0.10
VOL_LOOK         = 63
LEV_CAP          = 2.0
RISK_ON_THRESH   = 0.50     # fraction of assets with positive 1m return
TRADING_DAYS     = 252


def run(config: dict) -> dict:
    cfg      = config["backtest"]
    start    = cfg["start_date"]
    end      = cfg["end_date"]
    cache    = config["paths"]["cache_dir"]
    cost_bps = config["costs"]["equity_cost_bps"]
    slip_bps = config["costs"]["equity_slippage_bps"]

    all_syms = list(dict.fromkeys(SECTORS + MACRO_UNIVERSE))

    prices = {}
    for sym in all_syms:
        try:
            df = load_price_series(sym, start="1998-01-01", end=end,
                                   adjustment=ADJ_TOTALRETURN, cache_dir=cache)
            if not df.empty and "Close" in df.columns:
                prices[sym] = df["Close"]
        except Exception as e:
            log.warning("S131: could not load %s: %s", sym, e)

    if not prices:
        return {"returns": pd.Series(dtype=float), "description": DESCRIPTION, "turnover_annual": 0.0}

    close_df = pd.DataFrame(prices).sort_index()
    ret_df   = close_df.pct_change(fill_method=None)

    reb_dates      = pd.date_range(start, end, freq="BME")
    weight_schedule: dict = {}
    risk_on_count  = 0

    for d in reb_dates:
        avail = close_df.index[close_df.index <= d]
        if len(avail) < SECTOR_LB_DAYS + 2:
            continue
        row = len(avail) - 1

        # --- Regime signal: 1m TS on macro universe ---
        avail_macro = [s for s in MACRO_UNIVERSE if s in close_df.columns]
        p_now_m  = close_df[avail_macro].iloc[row]
        p_past_m = close_df[avail_macro].iloc[row - TS_LOOKBACK]
        ts_m     = ((p_now_m / p_past_m) - 1.0).dropna()

        n_pos = int((ts_m > 0).sum())
        n_tot = len(ts_m)
        frac_pos = n_pos / n_tot if n_tot > 0 else 0.0
        risk_on  = frac_pos > RISK_ON_THRESH

        if risk_on:
            risk_on_count += 1
            # --- Sector rotation ---
            avail_sec = [s for s in SECTORS if s in close_df.columns]
            p_now_s   = close_df[avail_sec].iloc[row].dropna()
            p_past_s  = close_df[avail_sec].iloc[max(0, row - SECTOR_LB_DAYS)].dropna()
            common    = p_now_s.index.intersection(p_past_s.index)
            if common.empty:
                weight_schedule[d] = {}
                continue
            rets     = ((p_now_s[common] / p_past_s[common]) - 1.0).sort_values(ascending=False)
            selected = rets.iloc[:SECTOR_TOP_N].index.tolist()
            weight_schedule[d] = {s: 1.0 / len(selected) for s in selected}

        else:
            # --- TS momentum on full macro universe: 1m, long-flat, vol-targeted ---
            vol_window   = ret_df[avail_macro].iloc[max(0, row - VOL_LOOK):row + 1]
            realized_vol = vol_window.std() * np.sqrt(TRADING_DAYS)
            realized_vol = realized_vol.replace(0, np.nan)

            positions: dict = {}
            for sym in ts_m.index:
                if sym not in realized_vol.index or pd.isna(realized_vol[sym]):
                    continue
                if ts_m[sym] > 0:
                    pos = min(VOL_TARGET / realized_vol[sym], 1.5)
                    positions[sym] = pos

            raw_sum = sum(abs(v) for v in positions.values())
            if raw_sum > LEV_CAP:
                scale = LEV_CAP / raw_sum
                positions = {s: v * scale for s, v in positions.items()}

            weight_schedule[d] = positions

    if not weight_schedule:
        return {"returns": pd.Series(dtype=float), "description": DESCRIPTION, "turnover_annual": 0.0}

    n_reb = len(weight_schedule)
    log.info("S131: risk-on %.0f%% of rebalances", 100 * risk_on_count / max(n_reb, 1))

    gross_ret, to = portfolio_returns_from_weights(weight_schedule, close_df, start, end)
    net_ret = apply_costs(gross_ret, to, cost_bps, slip_bps)
    ann_to  = float(to.sum() / max(len(to) / TRADING_DAYS, 1))

    log.info("S131 done. Ann turnover: %.2fx", ann_to)
    return {
        "returns":         net_ret,
        "description":     DESCRIPTION,
        "turnover_annual": ann_to,
    }
