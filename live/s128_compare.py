"""
s128_compare.py  —  Live vs backtest comparison helper
Reads s128_log.csv and computes actual monthly returns for the paper trades.

Usage:
    python s128_compare.py

Requirements: at least 2 rows in s128_log.csv (i.e. two monthly signals logged).

What it shows:
  - Month-by-month: target portfolio, return earned, vs SPY
  - Running totals: cumulative return, Sharpe, hit rate
  - Drift flag: if live Sharpe deviates > 0.5 from expected ~0.5-1.0, investigate.
"""

import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import yfinance as yf

warnings.filterwarnings("ignore")

LOG_FILE = Path(__file__).parent / "s128_log.csv"
SECTORS  = ["XLK", "XLF", "XLE", "XLV", "XLI", "XLY", "XLP", "XLRE", "XLB", "XLU", "XLC"]


# -- Data fetching -------------------------------------------------------------

def fetch_monthly_returns(tickers, start, end):
    """
    Fetch adjusted prices for tickers from start to end date.
    Returns daily return DataFrame.
    """
    df = yf.download(
        tickers,
        start=str(start),
        end=str(pd.Timestamp(end) + pd.Timedelta(days=5)),   # buffer for end date
        auto_adjust=True,
        progress=False,
    )
    if isinstance(df.columns, pd.MultiIndex):
        close = df["Close"]
    else:
        close = df[["Close"]] if "Close" in df.columns else df

    close.index = pd.to_datetime(close.index).normalize()
    return close


def period_return(close, syms, date_from, date_to):
    """
    Compute equal-weight return of syms from date_from close to date_to close.
    Both dates must be trading days in close.index (or we snap to nearest).
    """
    avail = close.index
    # Snap to nearest available trading day
    from_idx = avail.searchsorted(pd.Timestamp(date_from))
    to_idx   = avail.searchsorted(pd.Timestamp(date_to))
    from_idx = min(from_idx, len(avail) - 1)
    to_idx   = min(to_idx, len(avail) - 1)

    if from_idx >= to_idx:
        return float("nan")

    p0 = close.iloc[from_idx]
    p1 = close.iloc[to_idx]
    valid_syms = [s for s in syms if s in close.columns and pd.notna(p0.get(s)) and pd.notna(p1.get(s))]
    if not valid_syms:
        return float("nan")
    ret_each = [(float(p1[s]) / float(p0[s]) - 1.0) for s in valid_syms]
    return float(np.mean(ret_each))


# -- Main ----------------------------------------------------------------------

def main():
    print("=" * 68)
    print("  S128 LIVE PAPER TRADE COMPARISON")
    print("=" * 68)

    # -- Load log ----------------------------------------------------
    if not LOG_FILE.exists():
        print(f"\nNo log file found at {LOG_FILE}")
        print("Run s128_live.py on at least two consecutive month-ends first.")
        return

    log = pd.read_csv(LOG_FILE, parse_dates=["run_date", "data_through"])
    print(f"\nLog rows: {len(log)}  (from {log['run_date'].min().date()} to {log['run_date'].max().date()})")

    if len(log) < 2:
        print("\nNeed at least 2 logged months to compute a return.")
        print("Come back after next month-end.")
        return

    # -- Fetch prices for full period --------------------------------
    all_syms  = SECTORS + ["SPY"]
    date_min  = log["data_through"].min()
    date_max  = log["data_through"].max()
    print(f"\nFetching prices {date_min.date()} → {date_max.date()} for {all_syms}...")
    close = fetch_monthly_returns(all_syms, date_min, date_max)
    if close.empty:
        print("ERROR: could not fetch price data. Check internet connection.")
        return
    print(f"  {len(close)} trading days loaded.")

    # -- Compute month-by-month returns ------------------------------
    print()
    print("-" * 68)
    print(f"  {'Month':<12} {'Target':<22} {'Strat ret':>10} {'SPY ret':>10} {'vs SPY':>8}")
    print("-" * 68)

    results = []
    for i in range(len(log) - 1):
        row_now  = log.iloc[i]
        row_next = log.iloc[i + 1]

        d_from = row_now["data_through"]
        d_to   = row_next["data_through"]
        target = str(row_now["target"]).split(",")

        label = str(row_now["data_through"].date())

        if target == ["CASH"]:
            strat_ret = 0.0   # cash earns 0% (conservative — no T-bill credit)
            target_str = "CASH"
        else:
            strat_ret  = period_return(close, target, d_from, d_to)
            target_str = ",".join(target)

        spy_ret = period_return(close, ["SPY"], d_from, d_to)
        vs_spy  = strat_ret - spy_ret if not np.isnan(strat_ret) else float("nan")

        results.append({"label": label, "target": target_str,
                        "strat": strat_ret, "spy": spy_ret, "vs_spy": vs_spy})

        strat_s = f"{strat_ret:>+9.2%}" if not np.isnan(strat_ret) else "     N/A"
        spy_s   = f"{spy_ret:>+9.2%}"   if not np.isnan(spy_ret)   else "     N/A"
        vs_s    = f"{vs_spy:>+7.2%}"    if not np.isnan(vs_spy)    else "    N/A"
        print(f"  {label:<12} {target_str:<22} {strat_s} {spy_s} {vs_s}")

    # -- Summary stats ------------------------------------------------
    df_r = pd.DataFrame(results).dropna(subset=["strat", "spy"])
    if df_r.empty:
        print("\nNo complete return periods to summarize yet.")
        return

    strats = df_r["strat"].values
    spys   = df_r["spy"].values
    n = len(strats)

    def annualised_sr(rets):
        if len(rets) < 2:
            return float("nan")
        mu, sd = np.mean(rets), np.std(rets, ddof=1)
        return float(mu / sd * np.sqrt(12)) if sd > 1e-10 else float("nan")

    cum_strat = float(np.prod(1 + strats) - 1)
    cum_spy   = float(np.prod(1 + spys)   - 1)
    sr_live   = annualised_sr(strats)
    hit_rate  = float(np.mean(strats > spys))

    print("-" * 68)
    print(f"\n  Months tracked : {n}")
    print(f"  Cumulative strat: {cum_strat:>+8.2%}")
    print(f"  Cumulative SPY  : {cum_spy:>+8.2%}")
    print(f"  vs SPY (cumul.) : {cum_strat - cum_spy:>+8.2%}")
    print(f"  Annualised Sharpe (live, monthly): {sr_live:>+.3f}"
          if not np.isnan(sr_live) else "  Annualised Sharpe: N/A (need more months)")
    print(f"  Beat SPY rate   : {hit_rate:.0%}  ({int(hit_rate*n)}/{n} months)")

    # -- Drift flag ---------------------------------------------------
    print()
    print("-" * 68)
    print("  DRIFT CHECK")
    print("-" * 68)
    EXPECTED_SR_LO = 0.3    # lower bound; anything below this after 6+ months is a flag
    EXPECTED_SR_HI = 1.5    # upper bound; suspiciously high could mean data error

    if n < 6:
        print(f"  Only {n} months — too early for a reliable Sharpe estimate.")
        print(f"  SE(Sharpe, {n}mo) ~= +/-{1/np.sqrt(n/12):.2f}. Revisit after 6+ months.")
    elif np.isnan(sr_live):
        print("  Cannot compute Sharpe — check for NaN returns.")
    elif sr_live < EXPECTED_SR_LO:
        print(f"  *** DRIFT FLAG: live Sharpe {sr_live:+.3f} is below expected floor {EXPECTED_SR_LO}.")
        print( "  *** Check: (1) are you using total-return prices?")
        print( "             (2) does your live execution match the signal dates?")
        print( "             (3) compare live sector returns to s128_log.csv.")
    elif sr_live > EXPECTED_SR_HI:
        print(f"  *** DATA CHECK: live Sharpe {sr_live:+.3f} is unusually high.")
        print( "  *** Verify prices aren't stale or double-adjusted.")
    else:
        print(f"  Live Sharpe {sr_live:+.3f} is within expected range [{EXPECTED_SR_LO}, {EXPECTED_SR_HI}].")
        print( "  No drift detected.")

    print()
    print("  Reminder: SE(Sharpe) is large over short windows.")
    print("  After 6 months: SE ~+/-0.82.  After 12 months: SE ~+/-0.58.")
    print("  Use this to catch big implementation errors, not to judge performance.")
    print()
    print("=" * 68)


if __name__ == "__main__":
    main()
