"""
Probe 2 -- exact API surface for the concentration backtest.
"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import norgatedata as ng
import pandas as pd
from datetime import date

def section(t): print(f"\n{'='*60}\n  {t}\n{'='*60}")

# 1. Available watchlists
section("1. Watchlists")
try:
    wl = ng.watchlists()
    print(f"  Total: {len(wl)}")
    sp_wl = [w for w in wl if 'S&P' in w or 'sp' in w.lower() or 'sector' in w.lower() or 'Select' in w]
    print(f"  S&P/Sector-related ({len(sp_wl)}):")
    for w in sorted(sp_wl):
        print(f"    '{w}'")
except Exception as e:
    print(f"  FAILED: {e}")

# 2. S&P 500 symbols via watchlist
section("2. S&P 500 watchlist symbols (count)")
for wl_name in ['S&P 500 Current & Past', 'S&P 500', 'S&P500']:
    try:
        syms = ng.watchlist_symbols(wl_name)
        print(f"  '{wl_name}': {len(syms)} symbols, first 5: {syms[:5]}")
    except Exception as e:
        print(f"  '{wl_name}': FAILED - {e}")

# 3. index_constituent_timeseries for one symbol
section("3. index_constituent_timeseries for AAPL in S&P 500")
for idx_name in ['S&P 500 Current & Past', 'S&P 500']:
    try:
        ts = ng.index_constituent_timeseries('AAPL', idx_name,
                                              timeseriesformat='pandas-dataframe')
        print(f"  '{idx_name}': OK, shape={ts.shape}, cols={ts.columns.tolist()}")
        print(f"  tail:\n{ts.tail(5)}")
        break
    except Exception as e:
        print(f"  '{idx_name}': FAILED - {e}")

# 4. classification / GICS
section("4. classification('AAPL')")
try:
    c = ng.classification('AAPL')
    print(f"  type={type(c)}, val={c}")
except Exception as e:
    print(f"  FAILED: {e}")

section("4b. classification_at_level('AAPL', level)")
for level in [1, 2, 3, 4, 'Sector', 'Industry Group', 'Industry', 'Sub-Industry']:
    try:
        c = ng.classification_at_level('AAPL', level)
        print(f"  level={level}: '{c}'")
    except Exception as e:
        print(f"  level={level}: FAILED - {e}")

# 5. Shares outstanding
section("5. sharesoutstanding('AAPL')")
try:
    so = ng.sharesoutstanding('AAPL', timeseriesformat='pandas-dataframe')
    print(f"  OK: shape={so.shape}, cols={so.columns.tolist() if hasattr(so,'columns') else 'Series'}")
    print(f"  tail:\n{so.tail(3)}")
except Exception as e:
    print(f"  FAILED: {e}")
    try:
        so = ng.sharesoutstanding('AAPL')
        print(f"  Without timeseriesformat: type={type(so)}")
        if hasattr(so, '__len__'): print(f"  first: {list(so)[:3]}")
    except Exception as e2:
        print(f"  Also failed: {e2}")

# 6. unadjusted_close_timeseries
section("6. unadjusted_close_timeseries('AAPL')")
try:
    uc = ng.unadjusted_close_timeseries('AAPL', timeseriesformat='pandas-dataframe')
    print(f"  OK: shape={uc.shape}")
    print(f"  tail:\n{uc.tail(3)}")
except Exception as e:
    print(f"  FAILED: {e}")

# 7. Check sector-level watchlists for SPDR sectors
section("7. Sector-specific watchlist symbols (checking XLK-related)")
for wl_name in ['XLK', 'S&P Technology', 'Technology Select Sector',
                 'S&P 500 Technology Sector', 'GICS Sector 45']:
    try:
        syms = ng.watchlist_symbols(wl_name)
        print(f"  '{wl_name}': {len(syms)} symbols")
    except Exception as e:
        print(f"  '{wl_name}': not found")

# 8. Check what PIT looks like for constituent at a specific date
section("8. PIT constituent check: AAPL in S&P 500 at 2020-01-31")
try:
    ts = ng.index_constituent_timeseries('AAPL', 'S&P 500 Current & Past',
                                          timeseriesformat='pandas-dataframe')
    ts.index = pd.to_datetime(ts.index)
    snap = ts[ts.index <= pd.Timestamp('2020-01-31')]
    if not snap.empty:
        val = snap.iloc[-1].values[0]
        print(f"  AAPL was in S&P 500 on 2020-01-31: {bool(val)}")
    print(f"  ts columns: {ts.columns.tolist()}")
    print(f"  ts dtype: {ts.dtypes.tolist()}")
except Exception as e:
    print(f"  FAILED: {e}")

# 9. Quick test of GICS for multiple symbols to understand sector mapping
section("9. GICS sectors for sample stocks")
for sym in ['AAPL', 'JPM', 'XOM', 'JNJ', 'GE', 'AMZN', 'PG', 'PLD', 'LIN', 'NEE', 'GOOGL']:
    try:
        c = ng.classification_at_level(sym, 1)
        print(f"  {sym}: {c}")
    except Exception as e:
        print(f"  {sym}: FAILED")

# 10. fundamental() function
section("10. fundamental('AAPL', 'marketcap') or similar")
for field in ['marketcap', 'MarketCap', 'shares_outstanding', 'SharesOutstanding',
              'shares', 'market_cap']:
    try:
        f = ng.fundamental('AAPL', field, timeseriesformat='pandas-dataframe')
        if f is not None and not (hasattr(f, 'empty') and f.empty):
            print(f"  field='{field}' OK: {type(f)}")
            if hasattr(f, 'tail'): print(f"  tail:\n{f.tail(3)}")
            break
    except Exception as e:
        print(f"  field='{field}': {e}")

print("\nDone.")
