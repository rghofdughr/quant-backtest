"""
s118 — Low-Vol Sector Rotation
Signal: When SPY 20d realized vol > 20% ann -> hold defensive sectors (XLU, XLP, XLV equal weight).
        When vol <= 20% -> hold SPY. Monthly rebalance.
"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import logging
import numpy as np
import pandas as pd
from data import load_price_series, ADJ_TOTALRETURN
from engine import apply_costs

log = logging.getLogger(__name__)
DESCRIPTION = "Low-vol rotation: XLU/XLP/XLV defensive mix when SPY vol > 20%, SPY when calm"

VOL_THRESHOLD = 0.20  # 20% annualized
DEFENSIVES    = ["XLU", "XLP", "XLV"]

def run(config):
    cfg   = config["backtest"]
    start, end = cfg["start_date"], cfg["end_date"]
    cache = config["paths"]["cache_dir"]
    cost_bps = config["costs"]["equity_cost_bps"]
    slip_bps = config["costs"]["equity_slippage_bps"]

    prices = {}
    for sym in ["SPY"] + DEFENSIVES:
        df = load_price_series(sym, start=start, end=end, adjustment=ADJ_TOTALRETURN, cache_dir=cache)
        if not df.empty:
            prices[sym] = df["Close"]

    idx = pd.bdate_range(start, end)
    close_df = pd.DataFrame({s: c.reindex(idx, method="ffill") for s, c in prices.items()})
    ret_df = close_df.pct_change(fill_method=None).fillna(0.0)

    rvol_20d = ret_df["SPY"].rolling(20).std() * np.sqrt(252)
    high_vol = (rvol_20d.shift(1) > VOL_THRESHOLD).fillna(False)

    # Monthly: use month-end signal, hold all month
    monthly_hv = high_vol.resample("ME").last().reindex(idx, method="ffill").fillna(False)

    def_cols = [s for s in DEFENSIVES if s in ret_df.columns]
    n_def = len(def_cols)

    port_ret = pd.Series(0.0, index=idx)
    port_ret[~monthly_hv] = ret_df.loc[~monthly_hv, "SPY"]
    if def_cols:
        port_ret[monthly_hv] = ret_df.loc[monthly_hv, def_cols].mean(axis=1)

    # Turnover: switches between SPY and defensive mix (two-way ~ 1 trade per change)
    to = monthly_hv.astype(float).diff().abs().fillna(0.0)
    net_ret = apply_costs(port_ret, to, cost_bps, slip_bps)
    ann_to = float(to.sum() / max(len(to) / 252, 1))
    return {"returns": net_ret, "turnover_annual": ann_to, "description": DESCRIPTION}
