"""
analysis/run_new_batch.py
Run and score all 20 new strategies (s103-s122).
Outputs IS/OOS Sharpe, CAGR, MDD, turnover for each.
"""
import sys, os, time, yaml, importlib
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import pandas as pd
from pathlib import Path

REPO = Path(__file__).parent.parent
with open(REPO / "config.yaml") as f:
    config = yaml.safe_load(f)
config["paths"]["cache_dir"] = str(REPO / "cache/parquet")
config["backtest"]["start_date"] = "2000-01-01"
config["backtest"]["end_date"]   = "2024-12-31"

IS_START  = "2000-01-03"
IS_END    = "2017-06-30"
OOS_START = "2017-07-03"
OOS_END   = "2024-12-31"

def sr(ret, s, e):
    r = ret.loc[s:e].dropna()
    if len(r) < 10 or r.std() < 1e-10: return float("nan")
    return float(r.mean() / r.std() * np.sqrt(252))

def cagr(ret, s, e):
    r = ret.loc[s:e].dropna()
    if len(r) < 10: return float("nan")
    return float((1 + r).prod() ** (252 / len(r)) - 1)

def mdd(ret, s, e):
    r = ret.loc[s:e].dropna()
    if len(r) < 2: return float("nan")
    cum = (1 + r).cumprod()
    return float((cum / cum.cummax() - 1).min())

NEW_STRATEGIES = [
    ("s103", "yield_curve_regime",   "Yield Curve Regime",      "MACRO"),
    ("s104", "gold_regime",          "Gold Regime",             "MACRO"),
    ("s105", "oil_regime",           "Oil/Energy Regime",       "MACRO"),
    ("s106", "global_momentum",      "Global Momentum",         "MACRO"),
    ("s107", "real_estate",          "Real Estate (VNQ)",       "MACRO"),
    ("s108", "weekly_reversal",      "Weekly Reversal (R1k)",   "REVERSION"),
    ("s109", "sector_rsi",           "Sector RSI Reversion",    "REVERSION"),
    ("s110", "monthly_reversal",     "Monthly Reversal (R1k)",  "REVERSION"),
    ("s111", "etf_zscore",           "ETF Z-score Reversion",   "REVERSION"),
    ("s112", "turn_of_month",        "Turn of Month",           "SEASONAL"),
    ("s113", "january_barometer",    "January Barometer",       "SEASONAL"),
    ("s114", "preholiday",           "Pre-Holiday",             "SEASONAL"),
    ("s115", "year_end_reversal",    "Year-End Reversal (R1k)", "SEASONAL"),
    ("s116", "weekday_effect",       "Weekday Effect",          "SEASONAL"),
    ("s117", "vol_spike_recovery",   "Vol Spike Recovery",      "VOLATILITY"),
    ("s118", "low_vol_rotation",     "Low-Vol Rotation",        "VOLATILITY"),
    ("s119", "vol_compression",      "Vol Compression",         "VOLATILITY"),
    ("s120", "ema_cross_multiasset", "EMA Cross Multi-Asset",   "MACRO"),
    ("s121", "breadth_thrust",       "Breadth Thrust",          "MACRO"),
    ("s122", "52wk_high_proximity",  "52wk High Proximity(R1k)","REVERSION"),
]

print("=" * 100)
print("  NEW STRATEGY BATCH RESULTS (s103-s122)")
print("=" * 100)
print()
print(f"  {'ID':5s}  {'Name':26s}  {'Cat':10s}  {'IS SR':>7}  {'OOS SR':>7}  {'IS CAGR':>8}  {'OOS CAGR':>9}  {'OOS MDD':>8}  {'TO/yr':>6}  {'Time':>5s}")
print(f"  {'-'*105}")

results = {}
for sid, stem, name, category in NEW_STRATEGIES:
    t0 = time.time()
    try:
        mod = importlib.import_module(f"strategies.{sid}_{stem}")
        r = mod.run(config)
        ret = r.get("returns", pd.Series(dtype=float))
        if ret is None or ret.empty:
            print(f"  {sid:5s}  {name:26s}  {category:10s}  EMPTY")
            continue

        is_sr    = sr(ret,   IS_START,  IS_END)
        oos_sr   = sr(ret,   OOS_START, OOS_END)
        is_cagr  = cagr(ret, IS_START,  IS_END)
        oos_cagr = cagr(ret, OOS_START, OOS_END)
        oos_mdd  = mdd(ret,  OOS_START, OOS_END)
        ann_to  = r.get("turnover_annual", float("nan"))
        elapsed = time.time() - t0

        results[sid] = {"name": name, "cat": category,
                        "is_sr": is_sr, "oos_sr": oos_sr,
                        "is_cagr": is_cagr, "oos_cagr": oos_cagr,
                        "oos_mdd": oos_mdd, "ann_to": ann_to}

        def fs(v): return f"{v:+.3f}" if not np.isnan(v) else "  nan"
        def fp(v): return f"{v*100:+.1f}%" if not np.isnan(v) else "  nan"

        print(f"  {sid:5s}  {name:26s}  {category:10s}  {fs(is_sr):>7}  {fs(oos_sr):>7}  {fp(is_cagr):>8}  {fp(oos_cagr):>9}  {fp(oos_mdd):>8}  {ann_to:>6.1f}x  {elapsed:>4.0f}s")

    except Exception as e:
        elapsed = time.time() - t0
        print(f"  {sid:5s}  {name:26s}  {category:10s}  ERROR: {e}  ({elapsed:.0f}s)")

print()
print("=" * 100)
print("  SORTED BY OOS SHARPE (top performers)")
print("=" * 100)
print()
print(f"  {'ID':5s}  {'Name':26s}  {'Cat':10s}  {'IS SR':>7}  {'OOS SR':>7}  {'OOS CAGR':>9}  {'OOS MDD':>8}")
print(f"  {'-'*88}")

def fs(v): return f"{v:+.3f}" if not np.isnan(v) else "  nan"
def fp(v): return f"{v*100:+.1f}%" if not np.isnan(v) else "  nan"

for sid, d in sorted(results.items(), key=lambda x: -(x[1]["oos_sr"] or float("-inf"))):
    print(f"  {sid:5s}  {d['name']:26s}  {d['cat']:10s}  {fs(d['is_sr']):>7}  {fs(d['oos_sr']):>7}  {fp(d['oos_cagr']):>9}  {fp(d['oos_mdd']):>8}")

print()
print("Threshold for book consideration: OOS SR >= 0.70, then redundancy test")
