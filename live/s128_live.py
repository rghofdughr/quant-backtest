"""
s128_live.py  - Monthly sector rotation signal + CSV logger
Strategy: 9-month relative momentum on 11 SPDR sector ETFs, hold top-4 equal-weight.
Cash filter: if #1 sector 9m return <= 0, hold 100% cash.
Lookback: 189 trading days (frozen parameter -do not change).

Run on the last trading day of each month, after market close (4 PM ET).
Usage:
    python s128_live.py           # normal run (logs only on month-end)
    python s128_live.py --force   # force log + signal regardless of date (for testing)
    python s128_live.py --verify  # run data verification checks only
"""

import csv
import sys
import warnings
from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd
import yfinance as yf

warnings.filterwarnings("ignore")

# -- Strategy parameters (FROZEN -do not modify) ------------------------------
SECTORS  = ["XLK", "XLF", "XLE", "XLV", "XLI", "XLY", "XLP", "XLRE", "XLB", "XLU", "XLC"]
LB_DAYS  = 189    # 9-month lookback in trading days
TOP_N    = 4      # hold top-4 sectors
# ------------------------------------------------------------------------------

LOG_FILE      = Path(__file__).parent / "s128_log.csv"
DOWNLOAD_DAYS = LB_DAYS + 80   # ~270 trading days (~13.5 months) for buffer

LOG_FIELDS = (
    ["run_date", "data_through"]
    + [f"{s}_9m" for s in SECTORS]
    + [f"rank_{i}" for i in range(1, 12)]
    + ["target", "cash_filter", "paper_fill"]
)


# -- Data fetching -------------------------------------------------------------

def _period_str(n_trading_days):
    """Convert trading-day count to a yfinance period string with generous buffer."""
    months = int(n_trading_days / 21) + 4
    return f"{months}mo"


def fetch_close(tickers, n_days, adjusted=True):
    """
    Download adjusted (or raw) closing prices for all tickers.
    Returns a DataFrame indexed by date, columns = tickers.
    auto_adjust=True: prices include dividends reinvested (total return proxy).
    auto_adjust=False: raw split-adjusted prices only, no dividends.
    """
    period = _period_str(n_days)
    raw = yf.download(
        tickers,
        period=period,
        auto_adjust=adjusted,
        progress=False,
    )
    # yfinance 1.x returns MultiIndex columns: (field, ticker)
    if isinstance(raw.columns, pd.MultiIndex):
        close = raw["Close"].copy()
    else:
        close = raw[["Close"]].copy() if "Close" in raw.columns else raw.copy()

    close.index = pd.to_datetime(close.index).normalize()
    return close.dropna(how="all")


# -- Data verification ---------------------------------------------------------

def verify_data(close_adj, close_raw):
    """
    Run two checks and print results:
    1. XLU adjusted vs raw 9m return (confirms dividends are included).
    2. Row count per ticker (flags anything with < LB_DAYS + 5 trading days).
    Returns list of tickers with insufficient history.
    """
    print()
    print("-" * 60)
    print("DATA VERIFICATION")
    print("-" * 60)

    # -- Check 1: XLU dividend adjustment -------------------------
    print("\n  Check 1 -XLU 9m return: adjusted vs raw")
    print("  (XLU pays ~3-4%/yr in dividends; the two numbers should differ)")
    for label, df in [("adjusted (auto_adjust=True)", close_adj),
                      ("raw     (auto_adjust=False)", close_raw)]:
        col = df["XLU"] if "XLU" in df.columns else None
        if col is None:
            print(f"    XLU {label}: NOT FOUND")
            continue
        s = col.dropna()
        if len(s) < LB_DAYS + 1:
            print(f"    XLU {label}: only {len(s)} rows -too short to compute")
            continue
        ret = float(s.iloc[-1] / s.iloc[-LB_DAYS - 1] - 1)
        print(f"    XLU {label}: {ret:+.3%}")

    adj_xlu  = close_adj["XLU"].dropna() if "XLU" in close_adj.columns else pd.Series()
    raw_xlu  = close_raw["XLU"].dropna() if "XLU" in close_raw.columns else pd.Series()
    if len(adj_xlu) >= LB_DAYS + 1 and len(raw_xlu) >= LB_DAYS + 1:
        adj_ret = float(adj_xlu.iloc[-1] / adj_xlu.iloc[-LB_DAYS - 1] - 1)
        raw_ret = float(raw_xlu.iloc[-1] / raw_xlu.iloc[-LB_DAYS - 1] - 1)
        diff    = adj_ret - raw_ret
        if abs(diff) < 0.002:
            print(f"\n  *** WARNING: adjusted and raw returns differ by only {diff:+.3%}.")
            print( "  *** Dividends may NOT be included. Signal will be subtly wrong.")
            print( "  *** Consider using Norgate data for this strategy instead.")
        else:
            print(f"\n  OK -difference of {diff:+.3%} confirms dividends are included.")

    # -- Check 2: row counts per ticker ---------------------------
    print(f"\n  Check 2 -Trading days of adjusted history per ticker")
    print(f"  (need {LB_DAYS + 5}+ rows; XLRE launched Oct 2015, XLC launched Jun 2018)")
    short = []
    for sym in SECTORS:
        n = int(close_adj[sym].dropna().count()) if sym in close_adj.columns else 0
        status = "OK" if n >= LB_DAYS + 5 else "SHORT"
        flag   = "  <<< WARNING" if status == "SHORT" else ""
        print(f"    {sym:<6}  {n:>4} days  [{status}]{flag}")
        if status == "SHORT":
            short.append(sym)

    if short:
        print(f"\n  WARNING: {short} excluded from ranking due to short history.")
    else:
        print(f"\n  All 11 tickers have sufficient history.")

    print("-" * 60)
    return short


# -- Signal computation ---------------------------------------------------------

def compute_signal(close):
    """
    Compute 9m trailing total-return rank and target portfolio.

    Uses exactly LB_DAYS price steps:
        p_past = close[-LB_DAYS - 1]   (189 trading days ago)
        p_now  = close[-1]              (today / latest close)
    Returns (mom_series sorted desc, target_list, cash_filter_bool).
    """
    # Only use tickers with enough history
    valid_cols = [c for c in close.columns if close[c].dropna().count() >= LB_DAYS + 2]
    df = close[valid_cols].dropna()   # drop rows where ANY remaining ticker is NaN

    if len(df) < LB_DAYS + 2:
        raise ValueError(
            f"Need {LB_DAYS + 2} clean rows, only have {len(df)}. "
            "Try running after market close or check your internet connection."
        )

    p_now  = df.iloc[-1]
    p_past = df.iloc[-LB_DAYS - 1]
    mom    = ((p_now / p_past) - 1.0).sort_values(ascending=False)

    cash_filter = bool(mom.iloc[0] <= 0)
    target = ["CASH"] if cash_filter else mom.index[:TOP_N].tolist()
    return mom, target, cash_filter


# -- CSV logger -----------------------------------------------------------------

def append_log(run_date, data_through, mom, target, cash_filter):
    """Append one row to s128_log.csv. Creates the file + header if it doesn't exist."""
    row = {
        "run_date":     str(run_date),
        "data_through": str(data_through),
        "target":       ",".join(target),
        "cash_filter":  str(cash_filter),
        "paper_fill":   "",   # you fill this in after executing your paper trades
    }
    for sym in SECTORS:
        row[f"{sym}_9m"] = f"{float(mom[sym]):.6f}" if sym in mom.index else ""
    for i, sym in enumerate(mom.index, 1):
        row[f"rank_{i}"] = sym

    write_header = not LOG_FILE.exists()
    with open(LOG_FILE, "a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=LOG_FIELDS)
        if write_header:
            writer.writeheader()
        writer.writerow(row)
    print(f"\n  OK Logged to {LOG_FILE}")
    print(f"    Fill in the 'paper_fill' column after you execute your paper trades.")


# -- Month-end detection --------------------------------------------------------

def is_last_trading_day_of_month():
    """Return True if today is the last business day of the current calendar month."""
    today = pd.Timestamp.today().normalize()
    month_end    = today.replace(day=1) + pd.offsets.MonthEnd(0)
    last_bday    = month_end - pd.offsets.BDay(0)   # roll back if month-end is weekend
    # More reliable: get all bdays in month, check if today is the last
    bdays = pd.bdate_range(today.replace(day=1), month_end)
    return len(bdays) > 0 and today == bdays[-1]


def last_trading_day_of_month():
    today = pd.Timestamp.today().normalize()
    month_end = today.replace(day=1) + pd.offsets.MonthEnd(0)
    bdays = pd.bdate_range(today.replace(day=1), month_end)
    return bdays[-1].date() if len(bdays) > 0 else None


# -- Main -----------------------------------------------------------------------

def main():
    force  = "--force"  in sys.argv
    verify = "--verify" in sys.argv

    print("=" * 62)
    print("  S128 SECTOR ROTATION -LIVE SIGNAL")
    print("=" * 62)

    today = pd.Timestamp.today().normalize()
    is_month_end = is_last_trading_day_of_month()

    if not is_month_end and not force:
        print(f"\n  Today ({today.date()}) is NOT the last trading day of the month.")
        print(f"  Next signal date: {last_trading_day_of_month()}")
        print(f"\n  Running in PREVIEW mode -signal shown but NOT logged.")
        print(f"  To log anyway: python s128_live.py --force")
        will_log = False
    else:
        will_log = True

    # -- Fetch data -------------------------------------------------
    print(f"\nFetching {DOWNLOAD_DAYS} trading days of data for {len(SECTORS)} ETFs...")
    close_adj = fetch_close(SECTORS, DOWNLOAD_DAYS, adjusted=True)
    close_raw = fetch_close(SECTORS, DOWNLOAD_DAYS, adjusted=False)

    if close_adj.empty:
        print("\nERROR: yfinance returned no data.")
        print("Check your internet connection. If yfinance is consistently failing,")
        print("see the comment at the bottom of this file for free alternatives.")
        return

    data_through = close_adj.index[-1].date()
    print(f"  Latest data date: {data_through}")
    print(f"  Total rows: {len(close_adj)}")

    # -- Verify -----------------------------------------------------
    short_tickers = verify_data(close_adj, close_raw)

    if verify:
        return   # verification-only mode stops here

    # -- Compute signal ---------------------------------------------
    clean = close_adj.drop(columns=short_tickers, errors="ignore")
    mom, target, cash_filter = compute_signal(clean)

    print()
    print("-" * 60)
    print("9-MONTH MOMENTUM RANKING")
    print("-" * 60)
    for i, (sym, ret) in enumerate(mom.items(), 1):
        hold = "  <<< HOLD" if sym in target else ""
        print(f"  {i:>2}. {sym:<6}  {ret:>+7.2%}{hold}")

    print()
    print("-" * 60)
    print("TARGET PORTFOLIO")
    print("-" * 60)
    if cash_filter:
        print(f"  CASH FILTER ACTIVE")
        print(f"  Top sector: {mom.index[0]} at {mom.iloc[0]:+.2%} (negative -go to cash)")
        print(f"  Action: 100% cash / money market")
    else:
        print(f"  Cash filter: inactive (top sector {mom.index[0]} = {mom.iloc[0]:+.2%})")
        print()
        for sym in target:
            print(f"  {sym:<6}  25.00%")

    # -- Log --------------------------------------------------------
    if will_log:
        append_log(today.date(), data_through, mom, target, cash_filter)
    else:
        print(f"\n  (preview mode -not logged)")

    print()
    print("=" * 62)
    print("  Next run: last trading day of next month, after 4 PM ET")
    print("=" * 62)


if __name__ == "__main__":
    main()


# -- Free data source alternatives if yfinance breaks --------------------------
#
# 1. Stooq (no key needed, free):
#    import pandas_datareader as pdr
#    df = pdr.get_data_stooq("XLK.US", start="2024-01-01")
#    -Note: Stooq prices are NOT dividend-adjusted; you lose the total-return signal.
#      For a dividend-heavy ETF like XLU, this matters. Use yfinance if it's available.
#
# 2. EODHD (free tier, 20 req/day):
#    https://eodhd.com -requires free API key, gives adjusted prices.
#
# 3. Your Norgate subscription (most accurate -same data as the backtest):
#    from data import load_price_series, ADJ_TOTALRETURN
#    Norgate's ADJ_TOTALRETURN is the ground truth; yfinance approximates it.
#    If you want perfect alignment with the backtest, use Norgate.
