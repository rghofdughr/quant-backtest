import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import logging
import numpy as np
import pandas as pd
from data import load_price_series, watchlist_symbols, index_constituent_mask, ADJ_TOTALRETURN, compute_dollar_volume
from engine import portfolio_returns_from_weights, apply_costs

log = logging.getLogger(__name__)
TRADING_DAYS = 252
DESCRIPTION = "Volume-weighted momentum (Lee-Swaminathan). Double sort on 12-1 momentum x volume trend."

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
    idx_name = config["universes"].get("sp500_index", "S&P 500")
    start_load = "2013-01-01" if is_smoke else "1997-01-01"

    symbols = watchlist_symbols(wl_name)
    if is_smoke:
        symbols = symbols[:100]

    close_map, dv_map, mask_map, vol_map = {}, {}, {}, {}
    for sym in symbols:
        df = load_price_series(sym, start=start_load, end=end, adjustment=ADJ_TOTALRETURN, cache_dir=cache)
        if df.empty or df["Close"].max() < min_px:
            continue
        close_map[sym] = df["Close"]
        dv_map[sym] = compute_dollar_volume(df)
        if "Volume" in df.columns:
            vol_map[sym] = df["Volume"]
        m = index_constituent_mask(sym, idx_name, start=start_load, end=end, cache_dir=cache)
        if not m.empty:
            mask_map[sym] = m

    trading_idx = pd.bdate_range(start, end)
    close_df = pd.DataFrame(close_map).reindex(trading_idx, method="ffill")
    dv_df = pd.DataFrame(dv_map).reindex(trading_idx, method="ffill")
    vol_df = pd.DataFrame(vol_map).reindex(trading_idx, method="ffill")
    member_df = (pd.DataFrame(mask_map).reindex(trading_idx, method="ffill")
                 .infer_objects(copy=False).fillna(False).astype(bool))
    for col in close_df.columns:
        if col not in member_df.columns:
            member_df[col] = False

    reb_dates = pd.date_range(start, end, freq="BME")
    WARMUP = 252
    MOM_LONG = 252   # 12 months
    MOM_SHORT = 21   # skip 1 month
    VOL_SHORT = 63   # recent volume window
    VOL_LONG = 252   # baseline volume window
    MIN_LEG = 5

    weight_schedule = {}

    for rd in reb_dates:
        rd = min(rd, trading_idx[-1])
        pos = trading_idx.searchsorted(rd)
        if pos < WARMUP:
            continue

        dv_now = dv_df.iloc[pos - 1] if pos > 0 else dv_df.iloc[0]
        members = member_df.iloc[pos] if pos < len(member_df) else member_df.iloc[-1]

        valid_cols = [
            c for c in close_df.columns
            if members.get(c, False) and dv_now.get(c, 0) >= min_dv
        ]
        if len(valid_cols) < 40:
            continue

        # Momentum: 12-1 month
        mom_long_pos = max(0, pos - MOM_LONG)
        mom_short_pos = max(0, pos - MOM_SHORT)
        if mom_short_pos <= mom_long_pos:
            continue

        close_now = close_df.iloc[mom_short_pos]
        close_then = close_df.iloc[mom_long_pos]

        # Volume trend: avg(last 63d volume) / avg(last 252d volume)
        vol_short_pos = max(0, pos - VOL_SHORT)
        vol_long_pos = max(0, pos - VOL_LONG)

        signals = {}
        for c in valid_cols:
            cf = close_then.get(c, np.nan)
            cn = close_now.get(c, np.nan)
            if not (np.isfinite(cf) and np.isfinite(cn) and cf > 0):
                continue
            mom = cn / cf - 1.0

            # Volume trend
            if c not in vol_df.columns:
                continue
            vol_recent = vol_df[c].iloc[vol_short_pos:pos]
            vol_base = vol_df[c].iloc[vol_long_pos:pos]
            avg_recent = vol_recent.dropna().mean()
            avg_base = vol_base.dropna().mean()
            if not (np.isfinite(avg_recent) and np.isfinite(avg_base) and avg_base > 0):
                continue
            vol_trend = avg_recent / avg_base
            signals[c] = (mom, vol_trend)

        if len(signals) < 40:
            continue

        mom_series = pd.Series({c: v[0] for c, v in signals.items()})
        vol_trend_series = pd.Series({c: v[1] for c, v in signals.items()})

        q20_mom = mom_series.quantile(0.20)
        q80_mom = mom_series.quantile(0.80)

        # Double sort: LONG = top momentum quintile AND high volume trend (>1.0)
        # SHORT = bottom momentum quintile AND low volume trend (<1.0)
        long_cands = mom_series[mom_series >= q80_mom].index
        short_cands = mom_series[mom_series <= q20_mom].index

        long_names = [c for c in long_cands if vol_trend_series.get(c, 0) > 1.0]
        short_names = [c for c in short_cands if vol_trend_series.get(c, 1.0) < 1.0]

        if len(long_names) < MIN_LEG or len(short_names) < MIN_LEG:
            continue

        n_long = len(long_names)
        n_short = len(short_names)
        w = {}
        for s in long_names:
            w[s] = 1.0 / n_long
        for s in short_names:
            w[s] = -1.0 / n_short

        weight_schedule[rd] = w

    gross_ret, to = portfolio_returns_from_weights(weight_schedule, close_df, start, end)
    net_ret = apply_costs(gross_ret, to, cost_bps, slip_bps)
    spy = load_price_series("SPY", start, end, ADJ_TOTALRETURN, cache)
    bm = spy["Close"].pct_change(fill_method=None).reindex(net_ret.index)
    ann_to = float(to.sum() / max(len(to) / TRADING_DAYS, 1))
    return {"returns": net_ret, "benchmark": bm, "description": DESCRIPTION, "turnover_annual": ann_to}
