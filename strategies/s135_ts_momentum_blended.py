"""
S135 -- TS Momentum: 1m direction signal + 63d vol sizing (reduce turnover)
Problem with S129 (1m signal): 11.7x turnover because vol_look=min(63,21)=21d,
so position SIZES are noisy. Direction is right; sizing is jumpy.
Fix: keep 21d momentum signal for direction, use 63d vol estimate for sizing.
This decouples fast direction from slow sizing → stable positions, lower churn.
Also adds 12m agreement confirmation: only go long if 12m is also positive.
This prevents chasing very short-term blips that the slow signal disagrees with.
"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import logging
import numpy as np
import pandas as pd

from data import load_price_series, ADJ_TOTALRETURN
from engine import apply_costs

log = logging.getLogger(__name__)
DESCRIPTION = "TS momentum: 1m signal + 12m confirmation + 63d vol sizing"

FUTURES = ["ES", "CL", "NG", "DX", "EMD", "HG", "ZS", "SB"]
ETFS    = ["QQQ", "IWM", "TLT", "IEF", "GLD", "SLV", "FXE", "EEM", "DBC", "CORN"]

SIGNAL_LB  = 21    # 1m direction signal
CONFIRM_LB = 252   # 12m confirmation: only long if both agree
VOL_LOOK   = 63    # 63d vol estimate (decoupled from signal)
VOL_TARGET = 0.10
LEV_CAP    = 2.0
TRADING_DAYS = 252


def run(config: dict) -> dict:
    cfg      = config["backtest"]
    start    = cfg["start_date"]
    end      = cfg["end_date"]
    cache    = config["paths"]["cache_dir"]
    cost_bps = config["costs"]["equity_cost_bps"]
    slip_bps = config["costs"]["equity_slippage_bps"]

    all_syms = FUTURES + ETFS
    data = {}
    for sym in all_syms:
        try:
            df = load_price_series(sym, start=start, end=end,
                                   adjustment=ADJ_TOTALRETURN, cache_dir=cache)
            if not df.empty and "Close" in df.columns and len(df) > 100:
                data[sym] = df
        except Exception as e:
            log.warning("S135: could not load %s: %s", sym, e)

    if not data:
        return {"returns": pd.Series(dtype=float), "description": DESCRIPTION, "turnover_annual": 0.0}

    close_df = pd.DataFrame({s: df["Close"] for s, df in data.items()}).sort_index()
    ret_df   = close_df.pct_change(fill_method=None)

    trading_idx = pd.bdate_range(start, end)
    close_df    = close_df.reindex(trading_idx, method="ffill")
    ret_df      = ret_df.reindex(trading_idx).fillna(0.0)

    reb_dates = pd.date_range(start, end, freq="BME")
    port_ret  = pd.Series(0.0, index=trading_idx)
    to_ser    = pd.Series(0.0, index=trading_idx)
    prev_positions: dict = {}

    for ri, reb_d in enumerate(reb_dates):
        avail = close_df.index[close_df.index <= reb_d]
        if avail.empty:
            continue
        snap = avail[-1]
        row  = close_df.index.get_loc(snap)

        if row < CONFIRM_LB:
            continue

        future = trading_idx[trading_idx > reb_d]
        if future.empty:
            break
        hold_start = future[0]

        if ri + 1 < len(reb_dates):
            nf = trading_idx[trading_idx > reb_dates[ri + 1]]
            hold_end = nf[0] if not nf.empty else trading_idx[-1]
        else:
            hold_end = trading_idx[-1]

        hold_mask = (trading_idx >= hold_start) & (trading_idx <= hold_end)

        # Fast signal (1m)
        sig_fast = ((close_df.iloc[row] / close_df.iloc[row - SIGNAL_LB]) - 1.0).dropna()
        # Slow confirmation (12m)
        sig_slow = ((close_df.iloc[row] / close_df.iloc[row - CONFIRM_LB]) - 1.0).dropna()

        # Vol estimate: 63d (independent of signal lookback)
        vol_window   = ret_df.iloc[max(0, row - VOL_LOOK):row + 1]
        realized_vol = vol_window.std() * np.sqrt(TRADING_DAYS)
        realized_vol = realized_vol.replace(0, np.nan)

        positions: dict = {}
        for sym in sig_fast.index:
            if sym not in sig_slow.index:
                continue
            if sym not in realized_vol.index or pd.isna(realized_vol[sym]):
                continue
            # Only long if BOTH fast and slow agree positive
            if sig_fast[sym] > 0 and sig_slow[sym] > 0:
                pos = min(VOL_TARGET / realized_vol[sym], 1.5)
                positions[sym] = pos

        raw_sum = sum(abs(v) for v in positions.values())
        if raw_sum > LEV_CAP:
            scale = LEV_CAP / raw_sum
            positions = {s: v * scale for s, v in positions.items()}

        all_s = set(list(positions.keys()) + list(prev_positions.keys()))
        to = sum(abs(positions.get(s, 0.0) - prev_positions.get(s, 0.0)) for s in all_s) / 2.0
        if hold_start in to_ser.index:
            to_ser[hold_start] += to

        syms      = list(positions.keys())
        avail_sym = [s for s in syms if s in ret_df.columns]
        wts_arr   = np.array([positions[s] for s in avail_sym])

        hold_ret  = ret_df.loc[hold_mask, avail_sym]
        port_ret[hold_mask] += (hold_ret * wts_arr).sum(axis=1).values

        prev_positions = dict(positions)

    net_ret = apply_costs(port_ret, to_ser, cost_bps, slip_bps)
    ann_to  = float(to_ser.sum() / max(len(to_ser) / TRADING_DAYS, 1))

    log.info("S135 done. Ann turnover: %.2fx", ann_to)
    return {"returns": net_ret, "description": DESCRIPTION, "turnover_annual": ann_to}
