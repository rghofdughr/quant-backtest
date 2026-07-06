"""
analysis/deep_dive.py
Deep-dive parameter analysis for s08, s35, s02.
  S08: grid sweep top_n x lookback_months x cash_filter
  S35: per-year table, variants (flat/TLT/window sweep)
  S02: lookback sweep (3/6/12m), ensemble, long-short vs long-flat
"""
import sys, os, yaml, importlib, copy
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np
import pandas as pd
from pathlib import Path

REPO = Path(__file__).parent.parent
with open(REPO / "config.yaml") as f:
    BASE = yaml.safe_load(f)
BASE["paths"]["cache_dir"] = str(REPO / "cache/parquet")
BASE["backtest"]["start_date"] = "2000-01-01"
BASE["backtest"]["end_date"]   = "2024-12-31"

IS_START, IS_END   = "2000-01-03", "2017-06-30"
OOS_START, OOS_END = "2017-07-03", "2024-12-31"

def sr(ret, s, e):
    r = ret.loc[s:e].dropna()
    if len(r) < 10 or r.std() < 1e-10: return float("nan")
    return float(r.mean() / r.std() * np.sqrt(252))

def cagr_f(ret, s, e):
    r = ret.loc[s:e].dropna()
    if len(r) < 10: return float("nan")
    return float((1 + r).prod() ** (252 / len(r)) - 1)

def mdd_f(ret, s, e):
    r = ret.loc[s:e].dropna()
    if len(r) < 2: return float("nan")
    cum = (1 + r).cumprod()
    return float((cum / cum.cummax() - 1).min())

def annual_rets(ret):
    r = (1 + ret).resample("YE").prod() - 1
    return {int(t.year): float(v) for t, v in r.items()}

SEP = "=" * 72

# ==============================================================================
# S08 -- SECTOR ROTATION SWEEP
# ==============================================================================
print()
print(SEP)
print("  S08 -- SECTOR ROTATION: PARAMETER SWEEP")
print(SEP)

from data import load_price_series, ADJ_TOTALRETURN
from engine import portfolio_returns_from_weights, apply_costs

SECTORS = ["XLK","XLF","XLE","XLV","XLI","XLY","XLP","XLRE","XLB","XLU","XLC"]
cache = BASE["paths"]["cache_dir"]
s08_prices = {}
for sym in SECTORS:
    df = load_price_series(sym, start="1998-01-01", end="2024-12-31",
                           adjustment=ADJ_TOTALRETURN, cache_dir=cache)
    if not df.empty:
        s08_prices[sym] = df["Close"]

close_s08 = pd.DataFrame(s08_prices).sort_index()
cost_bps = BASE["costs"]["equity_cost_bps"]
slip_bps = BASE["costs"]["equity_slippage_bps"]
reb_dates = pd.date_range("2000-01-01", "2024-12-31", freq="BME")

def run_s08_params(top_n, lb_months, cash_filter):
    lb_days = lb_months * 21
    ws = {}
    for d in reb_dates:
        avail_d = close_s08.index[close_s08.index <= d]
        if len(avail_d) < lb_days + 2: continue
        row = len(avail_d) - 1
        rets = ((close_s08.iloc[row] / close_s08.iloc[max(0, row - lb_days)]) - 1
                ).dropna().sort_values(ascending=False)
        if rets.empty: continue
        if cash_filter and rets.iloc[0] <= 0:
            ws[d] = {}; continue
        ws[d] = {s: 1.0/top_n for s in rets.iloc[:top_n].index.tolist()}
    gross, to = portfolio_returns_from_weights(ws, close_s08, "2000-01-01", "2024-12-31")
    return apply_costs(gross, to, cost_bps, slip_bps)

top_ns = [1, 2, 3, 4, 5]
lbs    = [1, 2, 3, 6, 9, 12]

for cf_on in [True, False]:
    tag = "cash_filter=ON" if cf_on else "cash_filter=OFF"
    print(f"\n  Grid (IS SR / OOS SR) -- {tag}\n")
    print(f"  {'LB':>4}", end="")
    for n in top_ns: print(f"   top-{n}(IS/OOS)", end="")
    print()
    print(f"  {'-'*4}", end="")
    for n in top_ns: print(f"  ---------------", end="")
    print()

    best_oos = -99; best_params = None; best_ret = None
    grid_results = {}
    for lb in lbs:
        print(f"  {lb:>3}m", end="")
        for n in top_ns:
            ret = run_s08_params(n, lb, cf_on)
            i_sr = sr(ret, IS_START, IS_END)
            o_sr = sr(ret, OOS_START, OOS_END)
            grid_results[(n, lb)] = (i_sr, o_sr, ret)
            mark = " <<" if (o_sr == o_sr and o_sr > best_oos) else "   "
            if o_sr == o_sr and o_sr > best_oos:
                best_oos = o_sr; best_params = (n, lb); best_ret = ret
            print(f"  {i_sr:+.2f}/{o_sr:+.2f}{mark}", end="")
        print()

    print(f"\n  Best: top_n={best_params[0]}, lookback={best_params[1]}m -> OOS SR {best_oos:+.3f}")

    if cf_on:
        best_ret_s08_cf = best_ret
        best_p_cf = best_params

# Per-year: current (top3/3m/ON) vs best vs SPY
ret_current_s08 = run_s08_params(3, 3, True)
spy_raw = load_price_series("SPY", "2000-01-01", "2024-12-31", ADJ_TOTALRETURN, cache)
spy_ret = spy_raw["Close"].pct_change().reindex(pd.bdate_range("2000-01-01","2024-12-31")).fillna(0)

print()
n_b, lb_b = best_p_cf
print(f"  Per-year: Current (top3/3m) vs Best (top{n_b}/{lb_b}m) vs SPY\n")
print(f"  {'Year':>5}  {'SPY':>7}  {'S08 cur':>8}  {'S08 best':>9}  {'Excess cur':>11}")
print(f"  {'-'*5}  {'-'*7}  {'-'*8}  {'-'*9}  {'-'*11}")
ar_spy  = annual_rets(spy_ret)
ar_cur  = annual_rets(ret_current_s08)
ar_bes  = annual_rets(best_ret_s08_cf)
for yr in sorted(ar_spy):
    spy_y = ar_spy.get(yr, float("nan"))
    cur_y = ar_cur.get(yr, float("nan"))
    bes_y = ar_bes.get(yr, float("nan"))
    exc   = cur_y - spy_y if (cur_y==cur_y and spy_y==spy_y) else float("nan")
    flag  = " **" if exc==exc and exc > 0.10 else (" !!" if exc==exc and exc < -0.10 else "   ")
    print(f"  {yr:>5}  {spy_y:>+6.1%}  {cur_y:>+7.1%}   {bes_y:>+8.1%}   {exc:>+10.1%}{flag}")

i_c = sr(ret_current_s08, IS_START, IS_END); o_c = sr(ret_current_s08, OOS_START, OOS_END)
i_b = sr(best_ret_s08_cf, IS_START, IS_END); o_b = sr(best_ret_s08_cf, OOS_START, OOS_END)
print(f"\n  Current IS/OOS: {i_c:+.3f} / {o_c:+.3f}")
print(f"  Best    IS/OOS: {i_b:+.3f} / {o_b:+.3f}")


# ==============================================================================
# S35 -- SELL IN MAY: VARIANTS & PER-YEAR
# ==============================================================================
print()
print()
print(SEP)
print("  S35 -- SELL IN MAY: VARIANTS & PER-YEAR")
print(SEP)

spy_cls = load_price_series("SPY","1998-01-01","2024-12-31", ADJ_TOTALRETURN, cache)["Close"]
tlt_df  = load_price_series("TLT","2002-07-01","2024-12-31", ADJ_TOTALRETURN, cache)
agg_df  = load_price_series("AGG","2003-09-01","2024-12-31", ADJ_TOTALRETURN, cache)

idx35 = pd.bdate_range("2000-01-01","2024-12-31")
spy_r = spy_cls.reindex(idx35, method="ffill").pct_change().fillna(0)
tlt_r = tlt_df["Close"].reindex(idx35, method="ffill").pct_change().fillna(0)
agg_r = agg_df["Close"].reindex(idx35, method="ffill").pct_change().fillna(0)

def sell_may_variant(winter_months, summer_ret=None):
    mask = pd.Series(idx35.month, index=idx35).isin(winter_months)
    if summer_ret is not None:
        port = mask * spy_r + (~mask) * summer_ret.reindex(idx35).fillna(0)
    else:
        port = mask * spy_r
    to = mask.astype(float).diff().abs().fillna(0)
    return apply_costs(port, to, cost_bps, slip_bps)

WINTER = {11,12,1,2,3,4}
variants_35 = [
    ("Current: Nov-Apr flat",        sell_may_variant(WINTER)),
    ("Nov-Apr + TLT in summer",      sell_may_variant(WINTER, tlt_r)),
    ("Nov-Apr + AGG in summer",      sell_may_variant(WINTER, agg_r)),
    ("Oct-Apr flat",                 sell_may_variant({10,11,12,1,2,3,4})),
    ("Oct-Apr + TLT in summer",      sell_may_variant({10,11,12,1,2,3,4}, tlt_r)),
    ("Sep-Apr flat",                 sell_may_variant({9,10,11,12,1,2,3,4})),
    ("Nov-Mar flat (narrow)",        sell_may_variant({11,12,1,2,3})),
    ("Buy-and-hold SPY",             spy_r),
]

print()
print(f"  {'Variant':40s}  {'IS SR':>6}  {'OOS SR':>6}  {'IS CAGR':>8}  {'OOS CAGR':>9}  {'OOS MDD':>8}")
print(f"  {'-'*40}  {'-'*6}  {'-'*6}  {'-'*8}  {'-'*9}  {'-'*8}")
for name, ret in variants_35:
    i_sr = sr(ret, IS_START, IS_END)
    o_sr = sr(ret, OOS_START, OOS_END)
    i_cg = cagr_f(ret, IS_START, IS_END)
    o_cg = cagr_f(ret, OOS_START, OOS_END)
    o_md = mdd_f(ret, OOS_START, OOS_END)
    mark = " <<" if "Current" in name else "   "
    print(f"  {name:40s}  {i_sr:+.3f}   {o_sr:+.3f}  {i_cg:>+7.1%}   {o_cg:>+8.1%}   {o_md:>+7.1%}{mark}")

# Per-year
print()
ret_s35_cur = variants_35[0][1]
ret_s35_tlt = variants_35[1][1]
ar_s35 = annual_rets(ret_s35_cur)
ar_tlt = annual_rets(ret_s35_tlt)
print(f"  Per-year: Sell-in-May flat vs + TLT vs SPY\n")
print(f"  {'Year':>5}  {'SPY':>7}  {'S35 flat':>9}  {'S35+TLT':>8}  {'Exc flat':>9}  {'Beat?':>6}")
print(f"  {'-'*5}  {'-'*7}  {'-'*9}  {'-'*8}  {'-'*9}  {'-'*6}")
win_w = win_s = 0
for yr in sorted(ar_spy):
    spy_y = ar_spy.get(yr, float("nan"))
    s35_y = ar_s35.get(yr, float("nan"))
    tlt_y = ar_tlt.get(yr, float("nan"))
    exc   = s35_y - spy_y if (s35_y==s35_y and spy_y==spy_y) else float("nan")
    beat  = "YES" if exc==exc and exc > 0 else "no"
    if exc == exc:
        if exc > 0: win_w += 1
        else: win_s += 1
    print(f"  {yr:>5}  {spy_y:>+6.1%}  {s35_y:>+8.1%}   {tlt_y:>+7.1%}   {exc:>+8.1%}   {beat}")
print(f"\n  Winter beat SPY: {win_w}/{win_w+win_s} years ({win_w/(win_w+win_s):.0%})")

# Monthly seasonality
print()
print("  Average SPY return by calendar month (annualised, 2000-2024):")
spy_monthly = spy_r.groupby(spy_r.index.month).mean() * 252 * 100
months = ["Jan","Feb","Mar","Apr","May","Jun","Jul","Aug","Sep","Oct","Nov","Dec"]
for i, (m, avg) in enumerate(spy_monthly.items()):
    bar  = "#" * max(0, int(abs(avg) * 2))
    side = "+" if avg >= 0 else "-"
    mark = " << WINTER" if (i+1) in WINTER else ""
    print(f"  {months[i]:3s}  {side}{bar:<20s} {avg:+.1f}%{mark}")


# ==============================================================================
# S02 -- TS MOMENTUM: LOOKBACK SWEEP + ENSEMBLE + LONG-SHORT
# ==============================================================================
print()
print()
print(SEP)
print("  S02 -- TS MOMENTUM: LOOKBACK SWEEP + ENSEMBLE + LONG-SHORT")
print(SEP)

def run_s02(lookbacks, long_short=False):
    cfg = copy.deepcopy(BASE)
    cfg["strategies"]["s02"] = {"lookbacks": lookbacks, "long_short": long_short, "vol_target": 0.10}
    mod = importlib.import_module("strategies.s02_ts_momentum")
    return mod.run(cfg)["returns"]

s02_variants = [
    ("1m  long-flat",                 [21],        False),
    ("3m  long-flat",                 [63],        False),
    ("6m  long-flat",                 [126],       False),
    ("12m long-flat  (current)",      [252],       False),
    ("12m long-short",                [252],       True),
    ("3m  long-short",                [63],        True),
    ("6m  long-short",                [126],       True),
    ("Ensemble 3+6+12m long-flat",    [63,126,252],False),
    ("Ensemble 3+6+12m long-short",   [63,126,252],True),
]

print()
print(f"  {'Variant':38s}  {'IS SR':>6}  {'OOS SR':>6}  {'IS CAGR':>8}  {'OOS CAGR':>9}  {'OOS MDD':>8}")
print(f"  {'-'*38}  {'-'*6}  {'-'*6}  {'-'*8}  {'-'*9}  {'-'*8}")

s02_rets = {}
for name, lbs, ls in s02_variants:
    try:
        ret = run_s02(lbs, ls)
        i_sr = sr(ret, IS_START, IS_END)
        o_sr = sr(ret, OOS_START, OOS_END)
        i_cg = cagr_f(ret, IS_START, IS_END)
        o_cg = cagr_f(ret, OOS_START, OOS_END)
        o_md = mdd_f(ret, OOS_START, OOS_END)
        s02_rets[name] = ret
        print(f"  {name:38s}  {i_sr:+.3f}   {o_sr:+.3f}  {i_cg:>+7.1%}   {o_cg:>+8.1%}   {o_md:>+7.1%}")
    except Exception as e:
        print(f"  {name:38s}  ERROR: {e}")

# Per-year: 12m vs ensemble vs SPY
ret_12m = s02_rets.get("12m long-flat  (current)")
ret_ens = s02_rets.get("Ensemble 3+6+12m long-flat")
if ret_12m is not None and ret_ens is not None:
    print()
    print(f"  Per-year: 12m long-flat vs Ensemble vs SPY\n")
    ar_12m = annual_rets(ret_12m)
    ar_ens = annual_rets(ret_ens)
    print(f"  {'Year':>5}  {'SPY':>7}  {'S02 12m':>8}  {'Ensemble':>9}  {'Exc 12m':>8}")
    print(f"  {'-'*5}  {'-'*7}  {'-'*8}  {'-'*9}  {'-'*8}")
    for yr in sorted(ar_spy):
        spy_y = ar_spy.get(yr, float("nan"))
        m12_y = ar_12m.get(yr, float("nan"))
        ens_y = ar_ens.get(yr, float("nan"))
        exc   = m12_y - spy_y if (m12_y==m12_y and spy_y==spy_y) else float("nan")
        print(f"  {yr:>5}  {spy_y:>+6.1%}  {m12_y:>+7.1%}   {ens_y:>+8.1%}   {exc:>+7.1%}")

print("\nDone.")
