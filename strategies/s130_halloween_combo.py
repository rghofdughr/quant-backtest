"""
S130 -- Halloween Combo: Sector Rotation in winter + TS Momentum in summer
Winter (Oct-Apr): hold top-3 SPDR sectors by 9m momentum (S08-style)
Summer (May-Sep): TS momentum on non-equity assets, 1m lookback, long-flat, vol-targeted
Rationale: equities outperform Oct-Apr; non-equity trends exist year-round.
"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import logging
import numpy as np
import pandas as pd

from data import load_price_series, ADJ_TOTALRETURN
from engine import portfolio_returns_from_weights, apply_costs

log = logging.getLogger(__name__)
DESCRIPTION = "Halloween combo: sector rotation Oct-Apr, TS momentum May-Sep"

SECTORS = ["XLK", "XLF", "XLE", "XLV", "XLI", "XLY", "XLP", "XLRE", "XLB", "XLU", "XLC"]

# Non-equity assets for summer TS momentum (no equity indices)
SUMMER_UNIVERSE = [
    "TLT", "IEF",           # bonds
    "GLD", "SLV",           # precious metals
    "DBC", "CORN",          # broad commodities
    "FXE",                  # FX (Euro)
    "CL", "NG",             # energy futures
    "HG", "ZS", "SB",      # industrial/agricultural
    "DX",                   # dollar index
]

WINTER_MONTHS    = {10, 11, 12, 1, 2, 3, 4}
SECTOR_LB_DAYS   = 9 * 21   # 9 months
SECTOR_TOP_N     = 3
TS_LOOKBACK      = 21       # 1 month
VOL_TARGET       = 0.10
VOL_LOOK         = 63       # 3m for vol estimate
LEV_CAP          = 2.0
TRADING_DAYS     = 252


def run(config: dict) -> dict:
    cfg      = config["backtest"]
    start    = cfg["start_date"]
    end      = cfg["end_date"]
    cache    = config["paths"]["cache_dir"]
    cost_bps = config["costs"]["equity_cost_bps"]
    slip_bps = config["costs"]["equity_slippage_bps"]

    all_syms = SECTORS + SUMMER_UNIVERSE

    prices = {}
    for sym in all_syms:
        try:
            df = load_price_series(sym, start="1998-01-01", end=end,
                                   adjustment=ADJ_TOTALRETURN, cache_dir=cache)
            if not df.empty and "Close" in df.columns:
                prices[sym] = df["Close"]
        except Exception as e:
            log.warning("S130: could not load %s: %s", sym, e)

    if not prices:
        return {"returns": pd.Series(dtype=float), "description": DESCRIPTION, "turnover_annual": 0.0}

    close_df = pd.DataFrame(prices).sort_index()
    ret_df   = close_df.pct_change(fill_method=None)

    reb_dates      = pd.date_range(start, end, freq="BME")
    weight_schedule: dict = {}

    for d in reb_dates:
        avail = close_df.index[close_df.index <= d]
        if len(avail) < SECTOR_LB_DAYS + 2:
            continue
        row   = len(avail) - 1
        month = d.month

        if month in WINTER_MONTHS:
            # --- Sector rotation: top-3 by 9m return ---
            avail_sec = [s for s in SECTORS if s in close_df.columns]
            p_now   = close_df[avail_sec].iloc[row].dropna()
            p_past  = close_df[avail_sec].iloc[max(0, row - SECTOR_LB_DAYS)].dropna()
            common  = p_now.index.intersection(p_past.index)
            if common.empty:
                weight_schedule[d] = {}
                continue
            rets     = ((p_now[common] / p_past[common]) - 1.0).sort_values(ascending=False)
            selected = rets.iloc[:SECTOR_TOP_N].index.tolist()
            weight_schedule[d] = {s: 1.0 / len(selected) for s in selected}

        else:
            # --- TS momentum on non-equity universe: 1m, long-flat, vol-targeted ---
            avail_sum = [s for s in SUMMER_UNIVERSE if s in close_df.columns]
            if row < TS_LOOKBACK:
                weight_schedule[d] = {}
                continue

            p_now  = close_df[avail_sum].iloc[row]
            p_past = close_df[avail_sum].iloc[row - TS_LOOKBACK]
            ts_sig = ((p_now / p_past) - 1.0).dropna()

            vol_window   = ret_df[avail_sum].iloc[max(0, row - VOL_LOOK):row + 1]
            realized_vol = vol_window.std() * np.sqrt(TRADING_DAYS)
            realized_vol = realized_vol.replace(0, np.nan)

            positions: dict = {}
            for sym in ts_sig.index:
                if sym not in realized_vol.index or pd.isna(realized_vol[sym]):
                    continue
                if ts_sig[sym] > 0:
                    pos = min(VOL_TARGET / realized_vol[sym], 1.5)
                    positions[sym] = pos

            raw_sum = sum(abs(v) for v in positions.values())
            if raw_sum > LEV_CAP:
                scale = LEV_CAP / raw_sum
                positions = {s: v * scale for s, v in positions.items()}

            weight_schedule[d] = positions

    if not weight_schedule:
        return {"returns": pd.Series(dtype=float), "description": DESCRIPTION, "turnover_annual": 0.0}

    gross_ret, to = portfolio_returns_from_weights(weight_schedule, close_df, start, end)
    net_ret = apply_costs(gross_ret, to, cost_bps, slip_bps)
    ann_to  = float(to.sum() / max(len(to) / TRADING_DAYS, 1))

    log.info("S130 done. Ann turnover: %.2fx", ann_to)
    return {
        "returns":         net_ret,
        "description":     DESCRIPTION,
        "turnover_annual": ann_to,
    }
