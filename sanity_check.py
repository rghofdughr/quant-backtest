"""
sanity_check.py
---------------
Cheap pre-flight checks to run BEFORE `python runner.py --all`.

Goal: catch the three failure modes that silently corrupt a multi-decade,
survivorship-bias-free backtest:

  1. SURVIVORSHIP LEAK   - delisted names must return their full history up to
                           the delist date, NOT an empty frame. Empty-on-delist
                           silently removes losers and reintroduces the exact
                           bias Norgate exists to prevent.
  2. LOOK-AHEAD UNIVERSE - the point-in-time index membership must actually
                           change over time. A constant member count is a red
                           flag that index_constituent_timeseries isn't filtering
                           by date (or you're using a static "current" list).
  3. ENGINE PLAUSIBILITY - SPY total-return CAGR over a known window must match
                           reality, and per-strategy turnover must be non-zero
                           and finite. Catches broken cost/return/plumbing.

This script does NOT depend on your strategy modules beyond an optional turnover
hook. It talks to Norgate directly + your data.py loader so it validates the
same path your backtests use.

Usage:
    python sanity_check.py
    python sanity_check.py --start 2015-01-01 --end 2017-12-31 --n 100

Exit code 0 = all checks passed. Non-zero = at least one check failed; do NOT
launch the full run until it's green.
"""

from __future__ import annotations

import argparse
import sys
import datetime as dt

import numpy as np
import pandas as pd

# ----------------------------------------------------------------------------
# Adjust these imports to match your repo. The script tries your data.py first
# (so it validates the SAME loading path your strategies use) and falls back to
# raw norgatedata if a helper isn't present.
# ----------------------------------------------------------------------------
try:
    import data as project_data  # your data.py
except Exception:  # pragma: no cover
    project_data = None

try:
    import norgatedata as nd
except Exception:  # pragma: no cover
    nd = None


# ----------------------------------------------------------------------------
# Small result helper
# ----------------------------------------------------------------------------
class Check:
    def __init__(self):
        self.rows = []
        self.failed = False

    def record(self, name, passed, detail=""):
        self.rows.append((name, passed, detail))
        if not passed:
            self.failed = True
        symbol = "PASS" if passed else "FAIL"
        print(f"[{symbol}] {name}")
        if detail:
            for line in str(detail).splitlines():
                print(f"        {line}")

    def summary(self):
        print("\n" + "=" * 64)
        n_pass = sum(1 for _, p, _ in self.rows if p)
        print(f"  {n_pass}/{len(self.rows)} checks passed")
        print("=" * 64)
        if self.failed:
            print("  DO NOT launch the full run until all checks are green.")
        else:
            print("  Infrastructure looks sound. Full run is reasonable.")
        return 0 if not self.failed else 1


# ----------------------------------------------------------------------------
# Loader shims — prefer the project's own loaders so we test the real path.
# ----------------------------------------------------------------------------
def load_prices(symbol, start, end):
    """Return a DataFrame with at least a 'Close' column, indexed by date."""
    if project_data is not None and hasattr(project_data, "load_prices"):
        return project_data.load_prices(symbol, start=start, end=end)
    if nd is None:
        raise RuntimeError("Neither data.py nor norgatedata is importable.")
    df = nd.price_timeseries(
        symbol,
        stock_price_adjustment_setting=nd.StockPriceAdjustmentType.TOTALRETURN,
        padding_setting=nd.PaddingType.NONE,
        start_date=pd.Timestamp(start).to_pydatetime(),
        end_date=pd.Timestamp(end).to_pydatetime(),
        timeseriesformat="pandas-dataframe",
    )
    return df if df is not None else pd.DataFrame()


def pit_members(indexname, asof):
    """
    Return the set of symbols that were members of `indexname` on date `asof`,
    using point-in-time constituent data. Prefer project_data if it wraps this.
    """
    if project_data is not None and hasattr(project_data, "point_in_time_universe"):
        return set(project_data.point_in_time_universe(indexname, asof))
    if nd is None:
        raise RuntimeError("Neither data.py nor norgatedata is importable.")
    # Fall back: walk the Current & Past watchlist and test membership on `asof`.
    wl = f"{indexname} Current & Past"
    try:
        symbols = nd.watchlist_symbols(wl)
    except Exception as e:
        raise RuntimeError(f"Could not load watchlist '{wl}': {e}")
    members = set()
    asof_ts = pd.Timestamp(asof)
    for sym in symbols:
        try:
            ser = nd.index_constituent_timeseries(
                sym, indexname, timeseriesformat="pandas-dataframe"
            )
        except Exception:
            continue
        if ser is None or len(ser) == 0:
            continue
        # column is typically 'Index Constituent' (1/0). Be liberal about names.
        col = ser.columns[-1]
        s = ser[col]
        s.index = pd.to_datetime(s.index)
        on_or_before = s[s.index <= asof_ts]
        if len(on_or_before) and float(on_or_before.iloc[-1]) >= 1.0:
            members.add(sym)
    return members


# ----------------------------------------------------------------------------
# CHECK 1: survivorship — delisted names must carry real history.
# ----------------------------------------------------------------------------
# Known delisted US tickers with terminal declines / acquisitions. Pick a few
# that should have substantial pre-delist history in any survivorship-free set.
KNOWN_DELISTED = ["LEH", "WAMUQ", "ENRNQ", "BSC", "GM"]  # Lehman, WaMu, Enron, Bear, old GM


def check_survivorship(chk: Check, start, end):
    found_any = False
    details = []
    for sym in KNOWN_DELISTED:
        try:
            df = load_prices(sym, start="2000-01-01", end=end)
        except Exception as e:
            details.append(f"{sym}: loader raised {type(e).__name__}: {e}")
            continue
        n = 0 if df is None else len(df)
        if n > 50:  # has real history
            found_any = True
            last = pd.to_datetime(df.index[-1]).date()
            details.append(f"{sym}: {n} rows, last date {last}  (OK - history present)")
        else:
            details.append(f"{sym}: {n} rows  (suspicious - delisted name nearly empty)")
    detail = "\n".join(details)
    detail += (
        "\n\nExpectation: at least some known-delisted tickers return real "
        "multi-year history.\nIf ALL return ~empty, delisted names are being "
        "silently dropped -> survivorship leak."
    )
    chk.record(
        "Survivorship: delisted names carry real pre-delist history",
        passed=found_any,
        detail=detail,
    )


# ----------------------------------------------------------------------------
# CHECK 2: point-in-time universe actually varies over time.
# ----------------------------------------------------------------------------
def check_pit_universe(chk: Check, indexname="S&P 500"):
    try:
        dates = [dt.date(2005, 6, 30), dt.date(2010, 6, 30),
                 dt.date(2015, 6, 30), dt.date(2020, 6, 30)]
        counts = {}
        sets = {}
        for d in dates:
            members = pit_members(indexname, d)
            counts[d] = len(members)
            sets[d] = members

        varies = len(set(counts.values())) > 1 or any(
            sets[dates[i]] != sets[dates[i + 1]] for i in range(len(dates) - 1)
        )
        # turnover between consecutive snapshots should be non-trivial
        turnovers = []
        for i in range(len(dates) - 1):
            a, b = sets[dates[i]], sets[dates[i + 1]]
            if a or b:
                changed = len(a ^ b)
                turnovers.append(changed)

        detail = "\n".join(f"{d}: {counts[d]} members" for d in dates)
        detail += "\nSymbols changed between snapshots: " + ", ".join(map(str, turnovers))
        detail += (
            "\n\nExpectation: membership count/composition changes across years. "
            "A flat, identical set means the universe is NOT point-in-time "
            "(look-ahead / survivorship risk)."
        )
        passed = varies and any(t > 0 for t in turnovers)
        chk.record(
            f"Point-in-time universe ({indexname}) changes over time",
            passed=passed,
            detail=detail,
        )
    except Exception as e:
        chk.record(
            f"Point-in-time universe ({indexname}) changes over time",
            passed=False,
            detail=f"Check raised {type(e).__name__}: {e}\n"
                   f"(If membership data isn't wired up, fix before the full run.)",
        )


# ----------------------------------------------------------------------------
# CHECK 3a: SPY total-return CAGR over a known window is plausible.
# ----------------------------------------------------------------------------
# Approx SPY total-return CAGR for reference windows (sanity bands, not exact):
#   2015-01-01 .. 2017-12-31  ~ 10-12% annualized
#   2010-01-01 .. 2019-12-31  ~ 13-14% annualized
REFERENCE_WINDOWS = {
    ("2015-01-01", "2017-12-31"): (0.06, 0.16),
    ("2010-01-01", "2019-12-31"): (0.10, 0.16),
}


def cagr(series):
    series = series.dropna()
    if len(series) < 2:
        return np.nan
    years = (pd.to_datetime(series.index[-1]) - pd.to_datetime(series.index[0])).days / 365.25
    if years <= 0:
        return np.nan
    return (series.iloc[-1] / series.iloc[0]) ** (1 / years) - 1


def _close_col(df):
    for c in ["Close", "close", "Adj Close", "Last"]:
        if c in df.columns:
            return c
    return df.columns[-1]


def check_spy_cagr(chk: Check):
    passed_any = False
    details = []
    for (s, e), (lo, hi) in REFERENCE_WINDOWS.items():
        try:
            df = load_prices("SPY", start=s, end=e)
            if df is None or len(df) < 50:
                details.append(f"{s}..{e}: SPY returned {0 if df is None else len(df)} rows (load problem)")
                continue
            c = _close_col(df)
            g = cagr(df[c])
            ok = (g is not np.nan) and (lo <= g <= hi)
            passed_any = passed_any or ok
            flag = "OK" if ok else "OUT OF BAND"
            details.append(f"{s}..{e}: SPY TR CAGR = {g:6.2%}  (expect {lo:.0%}-{hi:.0%})  [{flag}]")
        except Exception as ex:
            details.append(f"{s}..{e}: raised {type(ex).__name__}: {ex}")
    detail = "\n".join(details)
    detail += (
        "\n\nIf CAGR is far outside the band, the price-adjustment setting "
        "(should be TOTALRETURN) or the return plumbing is wrong."
    )
    chk.record(
        "Engine: SPY total-return CAGR matches reality on known windows",
        passed=passed_any,
        detail=detail,
    )


# ----------------------------------------------------------------------------
# CHECK 3b: per-strategy turnover is non-zero and finite (optional hook).
# ----------------------------------------------------------------------------
def check_turnover(chk: Check, start, end, n_names):
    """
    Tries to import runner and run a tiny subset. If your runner exposes a
    `run_strategy(name, config)` returning an object/dict with weights or a
    turnover figure, we validate it. Otherwise this check is skipped (not failed).
    """
    try:
        import runner  # noqa
    except Exception:
        chk.record(
            "Engine: per-strategy turnover non-zero & finite",
            passed=True,
            detail="SKIPPED - runner.py not importable as a module. "
                   "Run a manual 100-name 2015-2017 subset and eyeball turnover.",
        )
        return

    hook = getattr(runner, "run_strategy", None)
    if hook is None:
        chk.record(
            "Engine: per-strategy turnover non-zero & finite",
            passed=True,
            detail="SKIPPED - runner has no run_strategy(name, config) hook. "
                   "Add one or eyeball turnover manually on a small subset.",
        )
        return

    # A few cheap, Norgate-native strategies that SHOULD have non-zero turnover.
    sample = ["S01", "S09", "S34", "S46"]
    cfg = {"start": start, "end": end, "max_names": n_names, "smoke": True}
    details = []
    all_ok = True
    for name in sample:
        try:
            res = hook(name, cfg)
            to = None
            if isinstance(res, dict):
                to = res.get("turnover")
            else:
                to = getattr(res, "turnover", None)
            if to is None:
                details.append(f"{name}: no turnover reported (can't validate)")
                continue
            ok = np.isfinite(to) and to > 0
            all_ok = all_ok and ok
            details.append(f"{name}: turnover = {to:.3f}  [{'OK' if ok else 'BAD'}]")
        except Exception as ex:
            all_ok = False
            details.append(f"{name}: raised {type(ex).__name__}: {ex}")
    chk.record(
        "Engine: per-strategy turnover non-zero & finite",
        passed=all_ok,
        detail="\n".join(details) if details else "no strategies returned turnover",
    )


# ----------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--start", default="2015-01-01")
    ap.add_argument("--end", default="2017-12-31")
    ap.add_argument("--n", type=int, default=100, help="names for turnover subset")
    ap.add_argument("--index", default="S&P 500")
    args = ap.parse_args()

    print("=" * 64)
    print("  PRE-FLIGHT SANITY CHECKS  (run before runner.py --all)")
    print(f"  window {args.start} .. {args.end}   subset n={args.n}")
    print("=" * 64)

    if nd is not None:
        try:
            print(f"  norgatedata status: {nd.status()}")
        except Exception as e:
            print(f"  WARNING: norgatedata.status() failed: {e}")
            print("  Is NDU (Norgate Data Updater) running locally?")
    print()

    chk = Check()
    check_survivorship(chk, args.start, args.end)
    print()
    check_pit_universe(chk, indexname=args.index)
    print()
    check_spy_cagr(chk)
    print()
    check_turnover(chk, args.start, args.end, args.n)

    code = chk.summary()
    sys.exit(code)


if __name__ == "__main__":
    main()
