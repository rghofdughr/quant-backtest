import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import logging
import numpy as np
import pandas as pd
from data import load_price_series, ADJ_TOTALRETURN
from engine import portfolio_returns_from_weights, apply_costs

log = logging.getLogger(__name__)
TRADING_DAYS = 252
DESCRIPTION = "ADX-gated time-series momentum on 10 ETFs. Vol-targeted at 10% annual per instrument."

ETFS = ["SPY", "QQQ", "IWM", "TLT", "IEF", "GLD", "SLV", "FXE", "EEM", "DBC"]
VOL_TARGET = 0.10
VOL_WIN = 63


def wilder_smooth(series: np.ndarray, period: int) -> np.ndarray:
    """Compute Wilder's smoothed moving average. series must be 1-D float array."""
    n = len(series)
    result = np.full(n, np.nan)
    # Find first index with enough non-NaN values
    valid = np.isfinite(series)
    # Seed: sum of first `period` finite values starting from the first valid index
    count = 0
    seed_sum = 0.0
    seed_idx = -1
    for i in range(n):
        if valid[i]:
            seed_sum += series[i]
            count += 1
            if count == period:
                seed_idx = i
                break
    if seed_idx < 0:
        return result
    result[seed_idx] = seed_sum
    alpha = (period - 1) / period  # multiplier for previous value (13/14 for period=14)
    for i in range(seed_idx + 1, n):
        if valid[i]:
            result[i] = alpha * result[i - 1] + series[i]
        else:
            result[i] = result[i - 1]
    return result


def compute_adx(df: pd.DataFrame, period: int = 14) -> pd.Series:
    """
    Compute ADX from OHLCV DataFrame with columns High, Low, Close.
    Returns a Series of ADX values indexed like df.
    """
    high = df["High"].values.astype(float)
    low = df["Low"].values.astype(float)
    close = df["Close"].values.astype(float)
    n = len(df)

    tr = np.full(n, np.nan)
    dm_pos = np.full(n, np.nan)
    dm_neg = np.full(n, np.nan)

    for i in range(1, n):
        prev_close = close[i - 1]
        prev_high = high[i - 1]
        prev_low = low[i - 1]

        tr_val = max(
            high[i] - low[i],
            abs(high[i] - prev_close),
            abs(low[i] - prev_close),
        )
        tr[i] = tr_val

        up_move = high[i] - prev_high
        down_move = prev_low - low[i]

        if up_move > down_move and up_move > 0:
            dm_pos[i] = up_move
        else:
            dm_pos[i] = 0.0

        if down_move > up_move and down_move > 0:
            dm_neg[i] = down_move
        else:
            dm_neg[i] = 0.0

    atr = wilder_smooth(tr, period)
    s_dm_pos = wilder_smooth(dm_pos, period)
    s_dm_neg = wilder_smooth(dm_neg, period)

    with np.errstate(divide="ignore", invalid="ignore"):
        di_pos = np.where(atr > 0, 100.0 * s_dm_pos / atr, np.nan)
        di_neg = np.where(atr > 0, 100.0 * s_dm_neg / atr, np.nan)
        dx = np.where(
            (di_pos + di_neg) > 0,
            100.0 * np.abs(di_pos - di_neg) / (di_pos + di_neg),
            np.nan,
        )

    adx = wilder_smooth(dx, period)
    return pd.Series(adx, index=df.index)


def run(config):
    cfg = config["backtest"]
    start, end = cfg["start_date"], cfg["end_date"]
    is_smoke = config.get("smoke", False)
    cache = config["paths"]["cache_dir"]
    cost_bps = config["costs"]["equity_cost_bps"]
    slip_bps = config["costs"]["equity_slippage_bps"]
    start_load = "2013-01-01" if is_smoke else "2000-01-01"
    adx_threshold = config.get("s54", {}).get("adx_threshold", 25.0)

    MOM_WIN = 63  # trailing momentum window (days)
    ADX_PERIOD = 14
    WARMUP = ADX_PERIOD * 3 + MOM_WIN + 10

    # Load all ETFs
    etf_data = {}
    for sym in ETFS:
        df = load_price_series(sym, start=start_load, end=end, adjustment=ADJ_TOTALRETURN, cache_dir=cache)
        if not df.empty:
            etf_data[sym] = df

    if not etf_data:
        empty = pd.Series(dtype=float)
        return {"returns": empty, "benchmark": empty, "description": DESCRIPTION, "turnover_annual": 0.0}

    trading_idx = pd.bdate_range(start, end)

    # Build close_df for portfolio_returns_from_weights
    close_map = {sym: df["Close"] for sym, df in etf_data.items()}
    close_df = pd.DataFrame(close_map).reindex(trading_idx, method="ffill")

    # Pre-compute ADX for each ETF on full loaded history
    adx_map = {}
    ret_map = {}
    vol_map = {}
    for sym, df in etf_data.items():
        df_full = df.reindex(pd.bdate_range(start_load, end), method="ffill")
        adx_s = compute_adx(df_full, period=ADX_PERIOD)
        adx_map[sym] = adx_s
        ret_s = df_full["Close"].pct_change(fill_method=None)
        ret_map[sym] = ret_s
        # Rolling vol
        vol_map[sym] = ret_s.rolling(VOL_WIN).std() * np.sqrt(TRADING_DAYS)

    reb_dates = pd.date_range(start, end, freq="BME")
    weight_schedule = {}

    for rd in reb_dates:
        rd = min(rd, trading_idx[-1])
        w = {}
        for sym in ETFS:
            if sym not in etf_data:
                continue
            adx_s = adx_map[sym]
            ret_s = ret_map[sym]
            vol_s = vol_map[sym]

            # Get values at rebalance date (or nearest prior date)
            try:
                adx_val = adx_s.asof(rd)
                # Trailing 63-day return
                pos_full = ret_s.index.searchsorted(rd)
                if pos_full < WARMUP:
                    continue
                # mom: price ratio over last MOM_WIN days
                close_series = pd.DataFrame(etf_data[sym]["Close"]).reindex(
                    pd.bdate_range(start_load, end), method="ffill"
                )["Close"]
                now_px = close_series.asof(rd)
                ago_px_idx = max(0, pos_full - MOM_WIN)
                ago_px = close_series.iloc[ago_px_idx]
                if ago_px <= 0 or not np.isfinite(ago_px) or not np.isfinite(now_px):
                    continue
                mom_signal = now_px / ago_px - 1.0

                vol_val = vol_s.asof(rd)
                if not np.isfinite(vol_val) or vol_val < 1e-6:
                    continue
                if not np.isfinite(adx_val):
                    continue

                # Gate: ADX > threshold
                if adx_val <= adx_threshold:
                    continue

                # Direction: long if momentum > 0, else flat
                if mom_signal <= 0:
                    continue

                # Vol-target sizing
                weight = VOL_TARGET / vol_val
                w[sym] = float(weight)
            except Exception:
                continue

        weight_schedule[rd] = w

    gross_ret, to = portfolio_returns_from_weights(weight_schedule, close_df, start, end)
    net_ret = apply_costs(gross_ret, to, cost_bps, slip_bps)
    spy_df = load_price_series("SPY", start, end, ADJ_TOTALRETURN, cache)
    bm = spy_df["Close"].pct_change(fill_method=None).reindex(net_ret.index)
    ann_to = float(to.sum() / max(len(to) / TRADING_DAYS, 1))
    return {"returns": net_ret, "benchmark": bm, "description": DESCRIPTION, "turnover_annual": ann_to}
