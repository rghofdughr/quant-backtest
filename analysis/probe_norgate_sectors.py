"""
Probe Norgate to understand what's available for the concentration backtest:
1. S&P 500 PIT constituents
2. Sector-level index constituents (SPDR-specific vs GICS-based)
3. Market cap data availability
4. GICS sector classification
"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import norgatedata as ng
from datetime import date
import pandas as pd

TEST_DATE = date(2020, 1, 31)
TEST_SYM  = "AAPL"

def section(title):
    print(f"\n{'='*60}")
    print(f"  {title}")
    print('='*60)

# ── 1. S&P 500 PIT constituents ──────────────────────────────────────────────
section("1. S&P 500 PIT constituents")
try:
    sp500 = ng.index_constituents("S&P 500 Current & Past", as_of_date=TEST_DATE)
    print(f"  OK: {len(sp500)} symbols at {TEST_DATE}")
    print(f"  First 10: {sp500[:10]}")
except Exception as e:
    print(f"  FAILED: {e}")
    # Try alternate name
    try:
        sp500 = ng.index_constituents("S&P 500", as_of_date=TEST_DATE)
        print(f"  OK (alt name): {len(sp500)} symbols")
    except Exception as e2:
        print(f"  Also failed alt: {e2}")

# ── 2. Sector-specific index names ───────────────────────────────────────────
section("2. Sector index names (SPDR / GICS)")
sector_candidates = [
    "S&P 500 Technology",
    "S&P Technology Select Sector",
    "S&P 500 (Technology Sector)",
    "GICS Technology",
    "XLK",
    "S&P 500 Financials",
    "S&P Financial Select Sector",
    "S&P 500 Energy",
    "S&P 500 Health Care",
    "S&P 500 Industrials",
    "S&P 500 Consumer Discretionary",
    "S&P 500 Consumer Staples",
    "S&P 500 Real Estate",
    "S&P 500 Materials",
    "S&P 500 Utilities",
    "S&P 500 Communication Services",
    "S&P 500 Information Technology",
]
for name in sector_candidates:
    try:
        c = ng.index_constituents(name, as_of_date=TEST_DATE)
        print(f"  FOUND '{name}': {len(c)} symbols -> {c[:5]}")
    except Exception as e:
        print(f"  NOT FOUND '{name}'")

# ── 3. Market cap via market_cap_timeseries ───────────────────────────────────
section("3. Market cap: market_cap_timeseries")
try:
    mc = ng.market_cap_timeseries(TEST_SYM, pandas_dataframe=True)
    if mc is not None and not mc.empty:
        print(f"  OK: {mc.columns.tolist()}")
        print(f"  Sample: {mc.tail(3)}")
        print(f"  At {TEST_DATE}: {mc[mc.index <= pd.Timestamp(TEST_DATE)].iloc[-1].values}")
    else:
        print("  returned empty")
except Exception as e:
    print(f"  FAILED: {e}")

# ── 4. Market cap via price_timeseries extra fields ───────────────────────────
section("4. Market cap via price_timeseries extra fields")
try:
    ts = ng.price_timeseries(
        TEST_SYM,
        stock_price_adjustment_setting=ng.StockPriceAdjustmentType.NONE,
        pandas_dataframe=True,
    )
    print(f"  price_timeseries columns: {ts.columns.tolist()}")
    print(f"  tail:\n{ts.tail(3)}")
except Exception as e:
    print(f"  FAILED: {e}")

# ── 5. 'Shares Issued' field availability ────────────────────────────────────
section("5. Shares Issued / Market Capitalisation field")
for field_try in [['Close', 'Volume', 'Shares Issued'],
                  ['Close', 'Unadjusted Close', 'Market Capitalisation'],
                  ['Close', 'Turnover', 'Market Cap']]:
    try:
        ts = ng.price_timeseries(
            TEST_SYM,
            stock_price_adjustment_setting=ng.StockPriceAdjustmentType.NONE,
            fields=field_try,
            pandas_dataframe=True,
        )
        print(f"  OK with fields={field_try}: cols={ts.columns.tolist()}")
        print(f"  tail:\n{ts.tail(3)}")
        break
    except Exception as e:
        print(f"  FAILED fields={field_try}: {e}")

# ── 6. GICS sector via security_details ──────────────────────────────────────
section("6. GICS sector via security_details")
try:
    det = ng.security_details(TEST_SYM)
    print(f"  AAPL details keys: {list(det.keys()) if isinstance(det, dict) else type(det)}")
    print(f"  AAPL details: {det}")
except Exception as e:
    print(f"  FAILED: {e}")

# ── 7. Check a known historical delisted symbol ───────────────────────────────
section("7. Delisted stock availability (LEHM = Lehman Brothers)")
for sym in ["LEHM", "ENRN", "WCOM", "LM"]:
    try:
        ts = ng.price_timeseries(
            sym,
            stock_price_adjustment_setting=ng.StockPriceAdjustmentType.TOTALRETURN,
            pandas_dataframe=True,
        )
        if ts is not None and not ts.empty:
            print(f"  {sym}: OK, {len(ts)} rows, last={ts.index[-1].date()}")
        else:
            print(f"  {sym}: empty series")
    except Exception as e:
        print(f"  {sym}: FAILED - {e}")

# ── 8. List available index names (sample) ───────────────────────────────────
section("8. Available index names from Norgate")
try:
    idx_list = ng.index_names()
    sp_related = [n for n in idx_list if 'S&P' in n or 'SP' in n or 'Select' in n or 'Sector' in n]
    print(f"  Total indices: {len(idx_list)}")
    print(f"  S&P/Sector-related ({len(sp_related)}):")
    for n in sorted(sp_related)[:40]:
        print(f"    '{n}'")
except Exception as e:
    print(f"  FAILED: {e}")

print("\nDone.")
