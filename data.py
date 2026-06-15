"""
data.py — Norgate Data loaders with parquet caching.

Cache key: cache/parquet/{symbol}_{adj}.parquet
All functions return pandas DataFrames/Series with DatetimeIndex.
"""

from __future__ import annotations
import hashlib
import logging
import os
from datetime import date, datetime
from pathlib import Path
from typing import List, Optional

import numpy as np
import pandas as pd

try:
    import norgatedata
    NORGATE_OK = norgatedata.status()
except Exception:
    NORGATE_OK = False
    norgatedata = None  # type: ignore

log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Cache helpers
# ---------------------------------------------------------------------------

def _cache_path(cache_dir: str, key: str) -> Path:
    p = Path(cache_dir) / f"{key}.parquet"
    p.parent.mkdir(parents=True, exist_ok=True)
    return p


def _load_cache(path: Path) -> Optional[pd.DataFrame]:
    if path.exists():
        try:
            return pd.read_parquet(path)
        except Exception:
            path.unlink(missing_ok=True)
    return None


def _save_cache(df: pd.DataFrame, path: Path) -> None:
    df.to_parquet(path, index=True)


def _safe_key(symbol: str, suffix: str) -> str:
    return f"{symbol.replace('&', 'CONT').replace('/', '_')}_{suffix}"


# ---------------------------------------------------------------------------
# Equity price loader
# ---------------------------------------------------------------------------

ADJ_TOTALRETURN = "TOTALRETURN"
ADJ_CAPITAL     = "CAPITAL"
ADJ_NONE        = "NONE"

_ADJ_MAP = {
    ADJ_TOTALRETURN: norgatedata.StockPriceAdjustmentType.TOTALRETURN if NORGATE_OK else None,
    ADJ_CAPITAL:     norgatedata.StockPriceAdjustmentType.CAPITAL     if NORGATE_OK else None,
    ADJ_NONE:        norgatedata.StockPriceAdjustmentType.NONE        if NORGATE_OK else None,
}


def load_price_series(
    symbol: str,
    start: str = "2000-01-01",
    end: str   = "2024-12-31",
    adjustment: str = ADJ_TOTALRETURN,
    cache_dir: str  = "cache/parquet",
    force_refresh: bool = False,
) -> pd.DataFrame:
    """
    Returns OHLCV DataFrame for *symbol* with DatetimeIndex.
    Columns: Open, High, Low, Close, Volume, Turnover (unadjusted $ volume).
    Always pulls TOTALRETURN for equities unless overridden.
    """
    key = _safe_key(symbol, adjustment)
    path = _cache_path(cache_dir, key)

    if not force_refresh:
        cached = _load_cache(path)
        if cached is not None:
            mask = (cached.index >= pd.Timestamp(start)) & (cached.index <= pd.Timestamp(end))
            return cached.loc[mask]

    if not NORGATE_OK:
        raise RuntimeError("Norgate not connected — cannot pull live data.")

    adj_type = _ADJ_MAP.get(adjustment, _ADJ_MAP[ADJ_TOTALRETURN])
    try:
        df = norgatedata.price_timeseries(
            symbol,
            stock_price_adjustment_setting=adj_type,
            padding_setting=norgatedata.PaddingType.NONE,
            start_date=date.fromisoformat("2000-01-01"),   # always cache full history
            timeseriesformat="pandas-dataframe",
        )
    except (ValueError, Exception) as exc:
        log.warning("Symbol %s not in Norgate subscription (%s)", symbol, exc)
        return pd.DataFrame()
    if df is None or df.empty:
        log.warning("No data returned for %s", symbol)
        return pd.DataFrame()

    df.index = pd.to_datetime(df.index)

    # Also attach unadjusted dollar volume (Close_unadj * Volume) for liquidity
    try:
        df_raw = norgatedata.price_timeseries(
            symbol,
            stock_price_adjustment_setting=norgatedata.StockPriceAdjustmentType.NONE,
            padding_setting=norgatedata.PaddingType.NONE,
            start_date=date.fromisoformat("2000-01-01"),
            timeseriesformat="pandas-dataframe",
        )
        if df_raw is not None and not df_raw.empty:
            df_raw.index = pd.to_datetime(df_raw.index)
            df["Turnover"] = df_raw["Close"] * df_raw["Volume"]
    except Exception:
        df["Turnover"] = df["Close"] * df["Volume"]

    _save_cache(df, path)

    mask = (df.index >= pd.Timestamp(start)) & (df.index <= pd.Timestamp(end))
    return df.loc[mask]


# ---------------------------------------------------------------------------
# Universe / watchlist loaders
# ---------------------------------------------------------------------------

def watchlist_symbols(watchlist_name: str) -> List[str]:
    """Return all symbols (current + past) in a Norgate watchlist."""
    if not NORGATE_OK:
        raise RuntimeError("Norgate not connected.")
    return norgatedata.watchlist_symbols(watchlist_name)


def index_constituent_mask(
    symbol: str,
    index_name: str,
    start: str = "2000-01-01",
    end: str   = "2024-12-31",
    cache_dir: str = "cache/parquet",
) -> pd.Series:
    """
    Returns a boolean Series indexed by trading date: True when *symbol*
    was a member of *index_name* on that date.  Point-in-time safe.
    """
    key = _safe_key(symbol, f"constituent_{index_name.replace(' ', '_')}")
    path = _cache_path(cache_dir, key)

    cached = _load_cache(path)
    if cached is not None:
        s = cached.iloc[:, 0]
        return s[(s.index >= pd.Timestamp(start)) & (s.index <= pd.Timestamp(end))].astype(bool)

    if not NORGATE_OK:
        raise RuntimeError("Norgate not connected.")

    ts = norgatedata.index_constituent_timeseries(symbol, index_name)
    if ts is None or len(ts) == 0:
        return pd.Series(dtype=bool)

    df = pd.DataFrame(ts)
    df.columns = ["Date", "Member"]
    df["Date"] = pd.to_datetime(df["Date"])
    df = df.set_index("Date").sort_index()
    df["Member"] = df["Member"].astype(bool)

    _save_cache(df, path)

    s = df["Member"]
    return s[(s.index >= pd.Timestamp(start)) & (s.index <= pd.Timestamp(end))]


# ---------------------------------------------------------------------------
# Point-in-time universe builder
# ---------------------------------------------------------------------------

def build_pit_universe(
    watchlist_name: str,
    index_name: str,
    start: str = "2000-01-01",
    end: str   = "2024-12-31",
    min_dollar_volume: float = 1e6,
    min_price: float = 5.0,
    cache_dir: str = "cache/parquet",
    price_adj: str = ADJ_TOTALRETURN,
) -> dict[str, pd.DataFrame]:
    """
    Loads prices for every symbol in *watchlist_name* that was ever a member
    of *index_name*.  Returns {symbol: price_df} — filtered by liquidity
    criteria but NOT by index membership on any given date (that's applied
    per-strategy at signal time).

    Call get_pit_members(date, member_masks) to get the live members.
    """
    symbols = watchlist_symbols(watchlist_name)
    log.info("Watchlist '%s': %d symbols", watchlist_name, len(symbols))

    price_map: dict[str, pd.DataFrame] = {}
    for sym in symbols:
        df = load_price_series(sym, start, end, price_adj, cache_dir)
        if df.empty:
            continue
        # Basic liquidity check (at least some days pass)
        if "Turnover" in df.columns:
            median_dv = df["Turnover"].rolling(20).median().dropna()
            if median_dv.empty or median_dv.max() < min_dollar_volume:
                continue
        if df["Close"].max() < min_price:
            continue
        price_map[sym] = df

    log.info("After liquidity filter: %d symbols", len(price_map))
    return price_map


def get_pit_members(
    as_of: pd.Timestamp,
    member_masks: dict[str, pd.Series],
    price_map: dict[str, pd.DataFrame],
    min_dollar_volume: float = 1e6,
    min_price: float = 5.0,
) -> List[str]:
    """
    Return symbols that were index members on *as_of*, pass price/liquidity
    filters on *as_of*, and have price data available.
    """
    result = []
    for sym, mask in member_masks.items():
        if sym not in price_map:
            continue
        # Check membership
        prior = mask[mask.index <= as_of]
        if prior.empty or not bool(prior.iloc[-1]):
            continue
        df = price_map[sym]
        row = df[df.index <= as_of]
        if row.empty:
            continue
        row = row.iloc[-1]
        if row["Close"] < min_price:
            continue
        if "Turnover" in df.columns:
            dv_window = df["Turnover"][df.index <= as_of].iloc[-20:]
            if len(dv_window) < 5 or dv_window.median() < min_dollar_volume:
                continue
        result.append(sym)
    return result


# ---------------------------------------------------------------------------
# Futures loader
# ---------------------------------------------------------------------------

def load_futures_series(
    symbol: str,
    start: str = "2000-01-01",
    end: str   = "2024-12-31",
    back_adjusted: bool = True,
    cache_dir: str = "cache/parquet",
) -> pd.DataFrame:
    """
    Continuous futures contract (e.g. '&ES', '&CL').
    back_adjusted=True → uses back-adjusted (TOTALRETURN equivalent).
    """
    adj = ADJ_TOTALRETURN if back_adjusted else ADJ_NONE
    return load_price_series(symbol, start, end, adj, cache_dir)


# ---------------------------------------------------------------------------
# ETF / single-symbol convenience
# ---------------------------------------------------------------------------

def load_etf(
    symbol: str,
    start: str = "2000-01-01",
    end: str   = "2024-12-31",
    cache_dir: str = "cache/parquet",
) -> pd.DataFrame:
    return load_price_series(symbol, start, end, ADJ_TOTALRETURN, cache_dir)


# ---------------------------------------------------------------------------
# Return computation helpers
# ---------------------------------------------------------------------------

def daily_returns(df: pd.DataFrame, col: str = "Close") -> pd.Series:
    return df[col].pct_change()


def load_dividends(
    symbol: str,
    cache_dir: str = "cache/parquet",
) -> pd.DataFrame:
    """
    Return dividend history for *symbol* as a DataFrame indexed by ex-date.
    Columns: ExDate (index), Amount.
    Cached per symbol.
    """
    key = _safe_key(symbol, "dividends")
    path = _cache_path(cache_dir, key)

    cached = _load_cache(path)
    if cached is not None:
        return cached

    if not NORGATE_OK:
        raise RuntimeError("Norgate not connected.")

    try:
        raw = norgatedata.dividends(symbol, timeseriesformat="pandas-dataframe")
    except Exception as exc:
        log.warning("Dividends for %s failed: %s", symbol, exc)
        return pd.DataFrame(columns=["Amount"])

    if raw is None or raw.empty:
        return pd.DataFrame(columns=["Amount"])

    # Norgate returns columns like 'Ex-Dividend Date', 'Dividend'
    raw.index = pd.to_datetime(raw.index)
    # Normalise column name
    amount_col = [c for c in raw.columns if "div" in c.lower() or "amount" in c.lower()]
    if not amount_col:
        amount_col = [raw.columns[0]]
    df = raw[[amount_col[0]]].rename(columns={amount_col[0]: "Amount"})
    df.index.name = "ExDate"

    _save_cache(df, path)
    return df


def compute_dollar_volume(df: pd.DataFrame) -> pd.Series:
    """20-day rolling median dollar volume."""
    if "Turnover" in df.columns:
        return df["Turnover"].rolling(20).median()
    return (df["Close"] * df["Volume"]).rolling(20).median()


# ---------------------------------------------------------------------------
# Stub loader interface for missing fundamentals
# ---------------------------------------------------------------------------

class FundamentalsStub:
    """
    Placeholder for a fundamentals vendor (Sharadar / Compustat / SEC EDGAR).
    Strategies that need fundamentals call these methods; the caller raises
    NotImplementedError unless a real provider is wired in.
    """

    REQUIRED_VENDOR = "Sharadar (via Nasdaq Data Link) or Compustat"

    @staticmethod
    def book_to_market(symbol: str, as_of: date) -> Optional[float]:
        raise NotImplementedError(
            f"Book-to-market requires {FundamentalsStub.REQUIRED_VENDOR}."
        )

    @staticmethod
    def earnings_yield(symbol: str, as_of: date) -> Optional[float]:
        raise NotImplementedError(
            f"Earnings yield (E/P) requires {FundamentalsStub.REQUIRED_VENDOR}."
        )

    @staticmethod
    def gross_profitability(symbol: str, as_of: date) -> Optional[float]:
        raise NotImplementedError(
            f"Gross profitability requires {FundamentalsStub.REQUIRED_VENDOR}."
        )

    @staticmethod
    def piotroski_fscore(symbol: str, fiscal_year: int) -> Optional[int]:
        raise NotImplementedError(
            f"F-score requires {FundamentalsStub.REQUIRED_VENDOR}."
        )

    @staticmethod
    def accruals(symbol: str, fiscal_year: int) -> Optional[float]:
        raise NotImplementedError(
            f"Accruals require {FundamentalsStub.REQUIRED_VENDOR}."
        )

    @staticmethod
    def ev_ebitda(symbol: str, as_of: date) -> Optional[float]:
        raise NotImplementedError(
            f"EV/EBITDA requires {FundamentalsStub.REQUIRED_VENDOR}."
        )

    @staticmethod
    def earnings_surprise(symbol: str, announce_date: date) -> Optional[float]:
        raise NotImplementedError(
            "EPS surprise (SUE) requires Zacks or Sharadar Earnings Estimates."
        )

    @staticmethod
    def short_interest(symbol: str, as_of: date) -> Optional[float]:
        raise NotImplementedError(
            "Short interest requires FINRA / Nasdaq biweekly data or a vendor."
        )

    @staticmethod
    def insider_transactions(symbol: str, start: date, end: date):
        raise NotImplementedError(
            "Insider transactions require SEC Form 4 data (EDGAR or a vendor)."
        )

    @staticmethod
    def ma_deal(symbol: str, as_of: date):
        raise NotImplementedError(
            "M&A deal data requires a deal database (Refinitiv, Bloomberg, or vendor)."
        )

    @staticmethod
    def analyst_revisions(symbol: str, as_of: date):
        raise NotImplementedError(
            "Analyst revisions require I/B/E/S or similar estimate-revision vendor."
        )


fundamentals = FundamentalsStub()
