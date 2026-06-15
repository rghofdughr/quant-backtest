"""
S77 — Dual Momentum (Gary Antonacci / GEM style)
Relative: rank R1000 members by 12-1m return; select top decile.
Absolute: hold a member only if its own 12m return > BIL proxy (cash rate).
If top decile fails absolute filter, go to BIL (or flat if BIL unavailable before 2007).
"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import logging
import numpy as np
import pandas as pd
from data import load_price_series, watchlist_symbols, index_constituent_mask, ADJ_TOTALRETURN, compute_dollar_volume
from engine import portfolio_returns_from_weights, apply_costs

log = logging.getLogger(__name__)
TRADING_DAYS = 252
DESCRIPTION = "Dual momentum (GEM): top-decile 12-1m momentum, held only when > BIL (cash) 12m return; else flat."

def run(config):
    cfg = config["backtest"]
    start, end = cfg["start_date"], cfg["end_date"]
    is_smoke = config.get("smoke", False)
    cache = config["paths"]["cache_dir"]
    cost_bps = config["costs"]["equity_cost_bps"]
    slip_bps = config["costs"]["equity_slippage_bps"]
    min_dv = config["liquidity"]["min_dollar_volume"]
    min_px = config["liquidity"]["min_price"]
    wl_name = config["universes"].get("russell1000", "Russell 1000 Current & Past")
    idx_name = config["universes"].get("russell1000_index", "Russell 1000")
    start_load = "2013-01-01" if is_smoke else "1997-01-01"

    symbols = watchlist_symbols(wl_name)
    if is_smoke:
        symbols = symbols[:150]

    close_map, dv_map, mask_map = {}, {}, {}
    for sym in symbols:
        df = load_price_series(sym, start=start_load, end=end, adjustment=ADJ_TOTALRETURN, cache_dir=cache)
        if df.empty or df["Close"].max() < min_px:
            continue
        close_map[sym] = df["Close"]
        dv_map[sym] = compute_dollar_volume(df)
        m = index_constituent_mask(sym, idx_name, start=start_load, end=end, cache_dir=cache)
        if not m.empty:
            mask_map[sym] = m

    # BIL (1-3m T-bill ETF) as absolute momentum hurdle; inception 2007
    try:
        bil_df = load_price_series("BIL", start=start_load, end=end, adjustment=ADJ_TOTALRETURN, cache_dir=cache)
        bil_close = bil_df["Close"] if not bil_df.empty else None
    except Exception:
        bil_close = None

    trading_idx = pd.bdate_range(start, end)
    close_df = pd.DataFrame(close_map).reindex(trading_idx, method="ffill")
    dv_df = pd.DataFrame(dv_map).reindex(trading_idx, method="ffill")
    member_df = (pd.DataFrame(mask_map).reindex(trading_idx, method="ffill")
                 .infer_objects(copy=False).fillna(False).astype(bool))
    for col in close_df.columns:
        if col not in member_df.columns:
            member_df[col] = False

    bil_series = bil_close.reindex(trading_idx, method="ffill") if bil_close is not None else None

    reb_dates = pd.date_range(start, end, freq="BME")
    MOM_WIN = 252
    SKIP = 21
    WARMUP = MOM_WIN + SKIP + 5

    weight_schedule = {}
    for rd in reb_dates:
        rd = min(rd, trading_idx[-1])
        pos = trading_idx.searchsorted(rd)
        if pos < WARMUP:
            continue

        members = member_df.iloc[pos] if pos < len(member_df) else member_df.iloc[-1]
        dv_now = dv_df.iloc[pos - 1]

        valid_cols = [
            c for c in close_df.columns
            if members.get(c, False) and dv_now.get(c, 0) >= min_dv
        ]
        if len(valid_cols) < 20:
            continue

        p_skip = close_df.iloc[pos - SKIP]
        p_12m = close_df.iloc[max(pos - SKIP - MOM_WIN, 0)]

        # BIL 12m return
        if bil_series is not None:
            bil_now = bil_series.iloc[pos - SKIP]
            bil_12m = bil_series.iloc[max(pos - SKIP - MOM_WIN, 0)]
            cash_hurdle = float(bil_now / bil_12m - 1.0) if (bil_12m and bil_12m > 0) else 0.0
        else:
            cash_hurdle = 0.0

        # Rank by 12-1m return
        mom = {}
        for c in valid_cols:
            pn = p_skip.get(c)
            pp = p_12m.get(c)
            if pn and pp and pp > 0 and np.isfinite(pn) and np.isfinite(pp):
                mom[c] = pn / pp - 1.0

        if len(mom) < 20:
            continue

        sr = pd.Series(mom)
        top = sr[sr >= sr.quantile(0.90)].index.tolist()
        # Absolute filter: keep only those beating the cash hurdle
        passed = [c for c in top if mom[c] > cash_hurdle]
        if not passed:
            weight_schedule[rd] = {}  # cash
            continue
        weight_schedule[rd] = {s: 1.0 / len(passed) for s in passed}

    gross_ret, to = portfolio_returns_from_weights(weight_schedule, close_df, start, end)
    net_ret = apply_costs(gross_ret, to, cost_bps, slip_bps)
    spy = load_price_series("SPY", start, end, ADJ_TOTALRETURN, cache)
    bm = spy["Close"].pct_change(fill_method=None).reindex(net_ret.index)
    ann_to = float(to.sum() / max(len(to) / TRADING_DAYS, 1))
    return {"returns": net_ret, "benchmark": bm, "description": DESCRIPTION, "turnover_annual": ann_to}
