"""
analysis/run_batch2.py
Run and score s123-s127 (price-based strategies from Graham/Frazzini/Moskowitz list).
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

STRATEGIES = [
    ("s123", "long_run_reversal",    "Long-Run Reversal (4yr)"),
    ("s124", "bab",                  "Betting Against Beta"),
    ("s125", "momentum_ma_filter",   "Momentum + MA Filter"),
    ("s126", "industry_rs",          "Industry Relative Strength"),
    ("s127", "fallen_angels",        "Fallen Angels"),
]

print("=" * 100)
print("  BATCH 2: s123-s127 RESULTS")
print("=" * 100)
print()
print(f"  {'ID':5s}  {'Name':28s}  {'IS SR':>7}  {'OOS SR':>7}  {'IS CAGR':>8}  {'OOS CAGR':>9}  {'OOS MDD':>8}  {'TO/yr':>6}  {'Time':>5s}")
print(f"  {'-'*105}")

results = {}
for sid, stem, name in STRATEGIES:
    t0 = time.time()
    try:
        mod = importlib.import_module(f"strategies.{sid}_{stem}")
        r   = mod.run(config)
        ret = r.get("returns", pd.Series(dtype=float))
        if ret is None or ret.empty:
            print(f"  {sid:5s}  {name:28s}  EMPTY")
            continue

        is_sr    = sr(ret,   IS_START,  IS_END)
        oos_sr   = sr(ret,   OOS_START, OOS_END)
        is_cagr  = cagr(ret, IS_START,  IS_END)
        oos_cagr = cagr(ret, OOS_START, OOS_END)
        oos_mdd  = mdd(ret,  OOS_START, OOS_END)
        ann_to   = r.get("turnover_annual", float("nan"))
        elapsed  = time.time() - t0

        results[sid] = dict(name=name, is_sr=is_sr, oos_sr=oos_sr,
                            is_cagr=is_cagr, oos_cagr=oos_cagr,
                            oos_mdd=oos_mdd, ann_to=ann_to)

        def fs(v): return f"{v:+.3f}" if not (v is None or np.isnan(v)) else "  nan"
        def fp(v): return f"{v*100:+.1f}%" if not (v is None or np.isnan(v)) else "  nan"

        print(f"  {sid:5s}  {name:28s}  {fs(is_sr):>7}  {fs(oos_sr):>7}  "
              f"{fp(is_cagr):>8}  {fp(oos_cagr):>9}  {fp(oos_mdd):>8}  "
              f"{ann_to if ann_to == ann_to else 0:>6.1f}x  {elapsed:>4.0f}s")

    except Exception as e:
        elapsed = time.time() - t0
        print(f"  {sid:5s}  {name:28s}  ERROR: {e}  ({elapsed:.0f}s)")

print()
print("=" * 100)
print("  SORTED BY OOS SHARPE")
print("=" * 100)
print()
print(f"  {'ID':5s}  {'Name':28s}  {'IS SR':>7}  {'OOS SR':>7}  {'OOS CAGR':>9}  {'OOS MDD':>8}")
print(f"  {'-'*80}")
for sid, d in sorted(results.items(), key=lambda x: -(x[1]["oos_sr"] if x[1]["oos_sr"]==x[1]["oos_sr"] else -99)):
    def fs(v): return f"{v:+.3f}" if not (v is None or np.isnan(v)) else "  nan"
    def fp(v): return f"{v*100:+.1f}%" if not (v is None or np.isnan(v)) else "  nan"
    print(f"  {sid:5s}  {d['name']:28s}  {fs(d['is_sr']):>7}  {fs(d['oos_sr']):>7}  "
          f"{fp(d['oos_cagr']):>9}  {fp(d['oos_mdd']):>8}")

print()
print("Threshold for redundancy testing: OOS SR >= 0.70")
