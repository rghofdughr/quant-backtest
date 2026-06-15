"""
S102 — ETF vs constituent basket arbitrage.
Trade XLK (and optionally other sectors) against a PIT-reconstructed
equal-weight basket of its current members when the spread dislocates >1.5 stdev.
Critical: basket membership is point-in-time from index_constituent_timeseries.
"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import logging
import numpy as np
import pandas as pd
from data import load_price_series, watchlist_symbols, index_constituent_mask, ADJ_TOTALRETURN, compute_dollar_volume
from engine import apply_costs

log = logging.getLogger(__name__)
TRADING_DAYS = 252
DESCRIPTION = "ETF-vs-basket arb: XLK vs PIT constituent basket; trade spread dislocation >1.5σ; PIT membership enforced."
SECTORS = ["XLK", "XLF", "XLE"]  # test on 3 liquid sectors
ENTRY_Z = 1.5
EXIT_Z = 0.0
SPREAD_WIN = 60   # z-score window

def run(config):
    cfg = config["backtest"]
    start, end = cfg["start_date"], cfg["end_date"]
    is_smoke = config.get("smoke", False)
    cache = config["paths"]["cache_dir"]
    cost_bps = config["costs"]["equity_cost_bps"]
    slip_bps = config["costs"]["equity_slippage_bps"]
    min_dv = config["liquidity"]["min_dollar_volume"]
    min_px = config["liquidity"]["min_price"]
    wl_name = config["universes"].get("sp500", "S&P 500 Current & Past")
    start_load = "2013-01-01" if is_smoke else "1997-01-01"

    symbols = watchlist_symbols(wl_name)
    if is_smoke:
        symbols = symbols[:300]

    close_map, dv_map, mask_map = {}, {}, {}
    for sym in symbols:
        df = load_price_series(sym, start=start_load, end=end, adjustment=ADJ_TOTALRETURN, cache_dir=cache)
        if df.empty or df["Close"].max() < min_px:
            continue
        close_map[sym] = df["Close"]
        dv_map[sym] = compute_dollar_volume(df)
        # Load membership for each sector ETF (their constituent watchlists)
        for etf in SECTORS:
            idx_key = f"{etf}_index"
            idx_name_try = f"{etf}"
            m = index_constituent_mask(sym, idx_name_try, start=start_load, end=end, cache_dir=cache)
            if not m.empty:
                mask_map.setdefault(etf, {})[sym] = m

    etf_data = {}
    for etf in SECTORS:
        df = load_price_series(etf, start=start_load, end=end, adjustment=ADJ_TOTALRETURN, cache_dir=cache)
        if not df.empty:
            etf_data[etf] = df["Close"]

    if not etf_data:
        raise RuntimeError("S102: no sector ETF data loaded")

    trading_idx = pd.bdate_range(start, end)
    close_df = pd.DataFrame(close_map).reindex(trading_idx, method="ffill")
    dv_df = pd.DataFrame(dv_map).reindex(trading_idx, method="ffill")
    etf_df = pd.DataFrame(etf_data).reindex(trading_idx, method="ffill")

    # Build member DataFrames per sector
    sector_members = {}
    for etf, m_dict in mask_map.items():
        if m_dict:
            mdf = (pd.DataFrame(m_dict).reindex(trading_idx, method="ffill")
                   .infer_objects(copy=False).fillna(False).astype(bool))
            sector_members[etf] = mdf

    ret_df = close_df.pct_change(fill_method=None).fillna(0.0)
    etf_ret = etf_df.pct_change(fill_method=None).fillna(0.0)

    # Compute daily log-spread: log(ETF / basket) for each sector
    log_spreads = {}
    for etf in SECTORS:
        if etf not in etf_df.columns or etf not in sector_members:
            continue
        mdf = sector_members[etf]
        basket_prices = []
        for i in range(len(trading_idx)):
            members_i = mdf.iloc[i] if i < len(mdf) else mdf.iloc[-1]
            active = [c for c in mdf.columns if members_i.get(c, False)
                      and dv_df.iloc[max(i - 1, 0)].get(c, 0) >= min_dv]
            if not active:
                basket_prices.append(np.nan)
            else:
                px = close_df[active].iloc[i].dropna()
                basket_prices.append(float(px.mean()) if len(px) > 0 else np.nan)
        basket = pd.Series(basket_prices, index=trading_idx)
        etf_px = etf_df[etf]
        spread = np.log((etf_px / basket).replace([np.inf, -np.inf], np.nan))
        log_spreads[etf] = spread

    spread_df = pd.DataFrame(log_spreads)

    port_rets = pd.Series(0.0, index=trading_idx)
    to_series = pd.Series(0.0, index=trading_idx)
    state = {etf: 0 for etf in SECTORS}  # +1 = long basket short ETF, -1 = reverse

    WARMUP = SPREAD_WIN + 5

    for i, dt in enumerate(trading_idx):
        if dt < pd.Timestamp(start):
            continue
        if i < WARMUP:
            continue

        day_ret = 0.0
        n_active = 0

        for etf in SECTORS:
            if etf not in spread_df.columns or etf not in etf_ret.columns:
                continue
            sp_hist = spread_df[etf].iloc[max(i - SPREAD_WIN, 0):i + 1].dropna()
            if len(sp_hist) < SPREAD_WIN // 2:
                continue
            z = float((sp_hist.iloc[-1] - sp_hist.mean()) / max(sp_hist.std(ddof=1), 1e-9))
            if not np.isfinite(z):
                continue

            if state[etf] == 0:
                if z > ENTRY_Z:
                    state[etf] = 1   # ETF expensive vs basket: short ETF, long basket
                elif z < -ENTRY_Z:
                    state[etf] = -1  # ETF cheap vs basket: long ETF, short basket
            else:
                if abs(z) < EXIT_Z:
                    state[etf] = 0

            s = state[etf]
            if s != 0:
                # ETF leg return
                etf_r = float(etf_ret[etf].iloc[i])
                # Basket leg: equal weight of current members
                mdf = sector_members.get(etf)
                basket_r = 0.0
                if mdf is not None:
                    members_i = mdf.iloc[i] if i < len(mdf) else mdf.iloc[-1]
                    active = [c for c in mdf.columns if members_i.get(c, False)
                              and dv_df.iloc[max(i - 1, 0)].get(c, 0) >= min_dv]
                    if active:
                        basket_r = float(ret_df[active].iloc[i].mean())
                pair_ret = s * (basket_r - etf_r)
                day_ret += pair_ret
                n_active += 1

        if n_active > 0:
            port_rets.iloc[i] = day_ret / n_active

        changes = sum(1 for etf in SECTORS if abs(state.get(etf, 0)) > 0) * 0.02
        to_series.iloc[i] = min(changes, 0.3)

    net_ret = apply_costs(port_rets, to_series, cost_bps, slip_bps)
    spy = load_price_series("SPY", start, end, ADJ_TOTALRETURN, cache)
    bm = spy["Close"].pct_change(fill_method=None).reindex(net_ret.index)
    ann_to = float(to_series.sum() / max(len(to_series) / TRADING_DAYS, 1))
    return {"returns": net_ret, "benchmark": bm, "description": DESCRIPTION, "turnover_annual": ann_to}
