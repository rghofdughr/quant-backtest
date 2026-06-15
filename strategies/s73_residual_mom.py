import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import logging
import numpy as np
import pandas as pd
from data import load_price_series, watchlist_symbols, index_constituent_mask, ADJ_TOTALRETURN, compute_dollar_volume
from engine import portfolio_returns_from_weights, apply_costs

log = logging.getLogger(__name__)
TRADING_DAYS = 252
DESCRIPTION = "Residual 12-1m momentum: beta to SPX estimated rolling on prior 36m; rank residuals; long top decile."

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
    start_load = "2013-01-01" if is_smoke else "1995-01-01"

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

    spy_df = load_price_series("SPY", start=start_load, end=end, adjustment=ADJ_TOTALRETURN, cache_dir=cache)

    trading_idx = pd.bdate_range(start, end)
    close_df = pd.DataFrame(close_map).reindex(trading_idx, method="ffill")
    dv_df = pd.DataFrame(dv_map).reindex(trading_idx, method="ffill")
    member_df = (pd.DataFrame(mask_map).reindex(trading_idx, method="ffill")
                 .infer_objects(copy=False).fillna(False).astype(bool))
    for col in close_df.columns:
        if col not in member_df.columns:
            member_df[col] = False

    ret_df = close_df.pct_change(fill_method=None)
    spy_ret = spy_df["Close"].reindex(trading_idx, method="ffill").pct_change(fill_method=None)

    reb_dates = pd.date_range(start, end, freq="BME")
    BETA_WIN = 756   # 36m rolling for beta estimation
    MOM_WIN = 252
    SKIP = 21
    WARMUP = BETA_WIN + 10

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

        # Beta estimation window: pos-BETA_WIN to pos-SKIP
        beta_end = max(pos - SKIP, 0)
        beta_start = max(beta_end - BETA_WIN, 0)
        spy_b = spy_ret.iloc[beta_start:beta_end].values
        spy_finite = np.isfinite(spy_b)

        # Momentum window: pos-MOM_WIN-SKIP to pos-SKIP
        mom_end = beta_end
        mom_start = max(mom_end - MOM_WIN, 0)

        # Prices for computing 12-1m return
        p_now = close_df.iloc[pos - SKIP]
        p_past = close_df.iloc[max(pos - SKIP - MOM_WIN, 0)]

        residual_moms = {}
        spy_v = spy_b[spy_finite]
        spy_dm = spy_v - spy_v.mean() if len(spy_v) > 1 else None

        for c in valid_cols:
            pn = p_now.get(c)
            pp = p_past.get(c)
            if not (pn and pp and pp > 0 and np.isfinite(pn) and np.isfinite(pp)):
                continue

            # Beta from beta window
            if spy_dm is not None and len(spy_dm) > 30:
                stk_b = ret_df[c].iloc[beta_start:beta_end].values
                stk_finite = np.isfinite(stk_b) & spy_finite[:len(stk_b)]
                if stk_finite.sum() > 30:
                    sv = spy_b[stk_finite[:len(spy_b)]]
                    xv = stk_b[stk_finite[:len(stk_b)]]
                    # Trim to same length
                    n = min(len(sv), len(xv))
                    sv, xv = sv[:n], xv[:n]
                    denom = np.sum((sv - sv.mean()) ** 2)
                    beta = np.dot(sv - sv.mean(), xv - xv.mean()) / denom if denom > 1e-12 else 1.0
                else:
                    beta = 1.0
            else:
                beta = 1.0

            # Residual momentum: total 12-1m return minus beta * SPX 12-1m
            spy_mom = spy_ret.iloc[mom_start:mom_end]
            spy_cum = float((1 + spy_mom.fillna(0)).prod() - 1)
            stock_tot = pn / pp - 1.0
            resid_mom = stock_tot - beta * spy_cum
            if np.isfinite(resid_mom):
                residual_moms[c] = resid_mom

        if len(residual_moms) < 20:
            continue

        sr = pd.Series(residual_moms)
        top = sr[sr >= sr.quantile(0.90)].index.tolist()
        if not top:
            continue
        weight_schedule[rd] = {s: 1.0 / len(top) for s in top}

    gross_ret, to = portfolio_returns_from_weights(weight_schedule, close_df, start, end)
    net_ret = apply_costs(gross_ret, to, cost_bps, slip_bps)
    spy = load_price_series("SPY", start, end, ADJ_TOTALRETURN, cache)
    bm = spy["Close"].pct_change(fill_method=None).reindex(net_ret.index)
    ann_to = float(to.sum() / max(len(to) / TRADING_DAYS, 1))
    return {"returns": net_ret, "benchmark": bm, "description": DESCRIPTION, "turnover_annual": ann_to}
