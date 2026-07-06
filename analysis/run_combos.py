"""
analysis/run_combos.py
Run s128-s131 combination strategies, then test each vs 8-strategy book.
Also tests whether s128/s129 could REPLACE s08/s02 in the book.
"""
import sys, os, time
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import yaml, logging, importlib
import numpy as np
import pandas as pd
from pathlib import Path

logging.basicConfig(level=logging.WARNING)

REPO = Path(__file__).parent.parent
with open(REPO / "config.yaml") as f:
    config = yaml.safe_load(f)
config["paths"]["cache_dir"] = str(REPO / "cache/parquet")
config["backtest"]["start_date"] = "2000-01-01"
config["backtest"]["end_date"]   = "2024-12-31"

IS_START, IS_END   = "2000-01-03", "2017-06-30"
OOS_START, OOS_END = "2017-07-03", "2024-12-31"

BOOK_WEIGHTS = {
    "s08": 0.0667, "s30": 0.0667, "s31": 0.0667, "s35": 0.0667, "s49": 0.0667,
    "s02": 0.1650, "s46": 0.1650,
    "s90": 0.3300,
}
BOOK_IDS = list(BOOK_WEIGHTS.keys())

STEMS = {
    "s02": "ts_momentum",       "s08": "sector_rotation",   "s30": "low_volatility",
    "s31": "vol_targeting",     "s35": "sell_in_may",       "s46": "risk_parity",
    "s49": "dollar_regime",     "s90": "credit_regime",
    "s128": "sector_rotation_9m",  "s129": "ts_momentum_1m",
    "s130": "halloween_combo",     "s131": "regime_gated",
}

CAND_NAMES = {
    "s128": "Sector Rotation 9m (A)",
    "s129": "TS Momentum 1m (A)",
    "s130": "Halloween Combo (B)",
    "s131": "Regime-Gated (C)",
}
CAND_IDS = ["s128", "s129", "s130", "s131"]


def sr(ret, s=None, e=None):
    r = ret.loc[s:e].dropna() if s else ret.dropna()
    if len(r) < 10 or r.std() < 1e-10:
        return float("nan")
    return float(r.mean() / r.std() * np.sqrt(252))


def cagr(ret, s, e):
    r = ret.loc[s:e].dropna()
    if len(r) < 10:
        return float("nan")
    return float((1 + r).prod() ** (252 / len(r)) - 1)


def mdd(ret, s, e):
    r = ret.loc[s:e].dropna()
    if len(r) < 2:
        return float("nan")
    cum = (1 + r).cumprod()
    return float((cum / cum.cummax() - 1).min())


def neff(corr_matrix):
    evals = np.linalg.eigvalsh(corr_matrix.values)
    evals = np.maximum(evals, 1e-12)
    return float((evals.sum() ** 2) / (evals ** 2).sum())


def run_strat(sid):
    mod = importlib.import_module(f"strategies.{sid}_{STEMS[sid]}")
    return mod.run(config)


SEP = "=" * 80

# ==============================================================================
# Step 1: Run book strategies
# ==============================================================================
print()
print(SEP)
print("  STEP 1: BOOK STRATEGIES (loading returns)")
print(SEP)
print()

all_rets = {}
for sid in BOOK_IDS:
    t0 = time.time()
    try:
        ret = run_strat(sid)["returns"]
        if ret is None or ret.empty:
            print(f"  {sid}: EMPTY")
            continue
        all_rets[sid] = ret
        elapsed = time.time() - t0
        print(f"  {sid:5s}  IS {sr(ret, IS_START, IS_END):+.3f}  OOS {sr(ret, OOS_START, OOS_END):+.3f}  ({elapsed:.0f}s)")
    except Exception as e:
        print(f"  {sid}: ERROR - {e}")

# ==============================================================================
# Step 2: Run combination candidates
# ==============================================================================
print()
print(SEP)
print("  STEP 2: COMBINATION CANDIDATES (s128-s131)")
print(SEP)
print()
print(f"  {'ID':5s}  {'Name':28s}  {'IS SR':>7}  {'OOS SR':>7}  {'IS CAGR':>8}  {'OOS CAGR':>9}  {'OOS MDD':>8}  {'TO/yr':>6}  {'Time':>5s}")
print(f"  {'-'*100}")

for sid in CAND_IDS:
    t0 = time.time()
    try:
        result  = run_strat(sid)
        ret     = result.get("returns", pd.Series(dtype=float))
        if ret is None or ret.empty:
            print(f"  {sid:5s}  {'EMPTY':28s}")
            continue
        all_rets[sid] = ret
        elapsed = time.time() - t0
        name    = CAND_NAMES[sid]
        is_sr   = sr(ret, IS_START, IS_END)
        oos_sr  = sr(ret, OOS_START, OOS_END)
        is_cg   = cagr(ret, IS_START, IS_END)
        oos_cg  = cagr(ret, OOS_START, OOS_END)
        oos_md  = mdd(ret, OOS_START, OOS_END)
        ann_to  = result.get("turnover_annual", float("nan"))

        def fs(v): return f"{v:+.3f}" if v == v else "  nan"
        def fp(v): return f"{v*100:+.1f}%" if v == v else "   nan"

        print(f"  {sid:5s}  {name:28s}  {fs(is_sr):>7}  {fs(oos_sr):>7}  "
              f"{fp(is_cg):>8}  {fp(oos_cg):>9}  {fp(oos_md):>8}  "
              f"{ann_to if ann_to == ann_to else 0:>6.1f}x  {elapsed:>4.0f}s")
    except Exception as e:
        elapsed = time.time() - t0
        print(f"  {sid:5s}  {CAND_NAMES[sid]:28s}  ERROR: {e}  ({elapsed:.0f}s)")

# ==============================================================================
# Step 3: Build return matrix, align on common dates
# ==============================================================================
print()
print(SEP)
print("  STEP 3: REDUNDANCY TEST -- each candidate vs 8-strategy book")
print(SEP)

book_avail = [s for s in BOOK_IDS if s in all_rets]
cand_avail = [s for s in CAND_IDS  if s in all_rets]

if len(book_avail) < len(BOOK_IDS):
    print(f"  WARNING: only {len(book_avail)}/{len(BOOK_IDS)} book strategies loaded")

common_idx = None
for sid in book_avail:
    idx = all_rets[sid].dropna().index
    common_idx = idx if common_idx is None else common_idx.intersection(idx)

if common_idx is None or len(common_idx) < 100:
    print("  ERROR: cannot establish common index")
    sys.exit(1)

ret_df = pd.DataFrame({sid: all_rets[sid].reindex(common_idx).fillna(0.0)
                       for sid in book_avail + cand_avail})

book_blend = sum(ret_df[s] * BOOK_WEIGHTS[s] for s in book_avail)
book_ret_df = ret_df[book_avail]
corr_book   = book_ret_df.corr()
neff_book   = neff(corr_book)
sr_book     = sr(book_blend)

print()
print(f"  Book baseline:  N_eff = {neff_book:.3f}   Blended SR = {sr_book:+.3f}")
print(f"  Book OOS SR  = {sr(book_blend.loc[OOS_START:OOS_END]):+.3f}")
print()
print(f"  {'ID':5s}  {'Name':26s}  {'vs s08':>7}  {'vs s02':>7}  {'max book':>8}  "
      f"{'N_eff+':>7}  {'dN_eff':>7}  {'dSR':>7}  {'Verdict':22s}")
print(f"  {'-'*115}")

survivors = []
individual = {}
for sid in cand_avail:
    cand  = ret_df[sid]
    cs08  = float(cand.corr(ret_df["s08"])) if "s08" in ret_df.columns else float("nan")
    cs02  = float(cand.corr(ret_df["s02"])) if "s02" in ret_df.columns else float("nan")

    book_corrs  = {b: float(cand.corr(ret_df[b])) for b in book_avail}
    max_corr    = max(book_corrs.values()) if book_corrs else float("nan")
    max_peer    = max(book_corrs, key=book_corrs.get) if book_corrs else "?"

    n_book   = len(book_avail)
    new_blend = (book_blend * n_book + cand) / (n_book + 1)
    delta_sr  = sr(new_blend) - sr_book

    augmented  = pd.concat([book_ret_df, cand.rename(sid)], axis=1).dropna()
    neff_aug   = neff(augmented.corr())
    delta_neff = neff_aug - neff_book

    individual[sid] = dict(max_corr=max_corr, max_peer=max_peer,
                            neff_aug=neff_aug, dneff=delta_neff, dsr=delta_sr)

    if delta_neff <= 0:
        verdict = "REDUNDANT (dN_eff<=0)"
    elif delta_neff < 0.05:
        verdict = "BORDERLINE (<0.05)"
    elif delta_sr <= 0:
        verdict = "NON-ADDITIVE (dSR<=0)"
    else:
        verdict = "SURVIVOR"
        survivors.append(sid)

    print(f"  {sid:5s}  {CAND_NAMES[sid]:26s}  {cs08:+7.3f}  {cs02:+7.3f}  {max_corr:+8.3f}  "
          f"{neff_aug:7.3f}  {delta_neff:+7.3f}  {delta_sr:+7.3f}  {verdict:22s}")

# ==============================================================================
# Step 4: Replacement test for s128 vs s08, s129 vs s02
# ==============================================================================
print()
print(SEP)
print("  STEP 4: REPLACEMENT TEST -- can s128 replace s08? can s129 replace s02?")
print(SEP)
print()

def replacement_test(replace_id, new_id, label):
    if replace_id not in ret_df.columns or new_id not in ret_df.columns:
        print(f"  {label}: data missing")
        return

    # Book without the replaced strategy, add new one at same weight
    replaced_w = BOOK_WEIGHTS[replace_id]
    book_no_old = sum(ret_df[s] * BOOK_WEIGHTS[s] for s in book_avail if s != replace_id)
    book_with_new = book_no_old + ret_df[new_id] * replaced_w

    # Also recompute N_eff for the modified book
    mod_book = [s for s in book_avail if s != replace_id] + [new_id]
    mod_ret_df = pd.concat([ret_df[s] for s in mod_book], axis=1)
    mod_ret_df.columns = mod_book
    neff_mod   = neff(mod_ret_df.corr())
    sr_mod     = sr(book_with_new)
    sr_oos_mod = sr(book_with_new.loc[OOS_START:OOS_END])

    sr_orig     = sr(book_blend)
    sr_oos_orig = sr(book_blend.loc[OOS_START:OOS_END])

    print(f"  {label}:")
    print(f"    Original book:   SR={sr_orig:+.3f}  OOS SR={sr_oos_orig:+.3f}  N_eff={neff_book:.3f}")
    print(f"    After replacement: SR={sr_mod:+.3f}  OOS SR={sr_oos_mod:+.3f}  N_eff={neff_mod:.3f}")
    dn = neff_mod - neff_book
    ds = sr_mod - sr_orig
    verdict = "UPGRADE" if ds > 0 and dn >= -0.05 else ("DEGRADED" if ds < 0 else "NEUTRAL")
    print(f"    dSR={ds:+.3f}  dN_eff={dn:+.3f}  --> {verdict}")
    print()

if "s128" in ret_df.columns:
    replacement_test("s08", "s128", "Replace s08 (3m/top3) with s128 (9m/top4)")
if "s129" in ret_df.columns:
    replacement_test("s02", "s129", "Replace s02 (12m) with s129 (1m)")

# ==============================================================================
# Step 5: Best combined portfolio (if survivors exist)
# ==============================================================================
print(SEP)
print("  STEP 5: SUMMARY")
print(SEP)
print()
print(f"  8-strategy book:  N_eff={neff_book:.3f}  SR={sr(book_blend):+.3f}  OOS SR={sr(book_blend.loc[OOS_START:OOS_END]):+.3f}")
print()
if survivors:
    print(f"  Survivors (add to book): {survivors}")
    for sid in survivors:
        r = individual[sid]
        print(f"    {sid} {CAND_NAMES[sid]}: dN_eff={r['dneff']:+.3f}  dSR={r['dsr']:+.3f}  max_book_corr={r['max_corr']:.3f}")
else:
    print("  No new survivors from redundancy test.")

# Per-year OOS comparison for all candidates
print()
print(SEP)
print("  STEP 6: OOS PER-YEAR TABLE (2017-2024)")
print(SEP)
print()

from data import load_price_series, ADJ_TOTALRETURN
spy = load_price_series("SPY", start=OOS_START, end=OOS_END,
                         adjustment=ADJ_TOTALRETURN,
                         cache_dir=config["paths"]["cache_dir"])
spy_yr = (1 + spy["Close"].pct_change(fill_method=None)).resample("YE").prod() - 1

def yr(ret):
    return (1 + ret).resample("YE").prod() - 1

cand_yr = {sid: yr(ret_df[sid].loc[OOS_START:OOS_END]) for sid in cand_avail}
book_yr = yr(book_blend.loc[OOS_START:OOS_END])

years = sorted({int(d.year) for d in book_yr.index})
header = f"  {'Year':>5}  {'SPY':>7}  {'Book':>7}" + \
         "".join(f"  {CAND_NAMES[s].split('(')[1].rstrip(')'):>5}" for s in cand_avail)
print(header)
print(f"  {'-'*70}")

for yr_i in years:
    spy_v  = next((float(v) for d, v in spy_yr.items() if d.year == yr_i), float("nan"))
    book_v = next((float(v) for d, v in book_yr.items() if d.year == yr_i), float("nan"))
    row = f"  {yr_i:>5}  {spy_v*100:>+6.1f}%  {book_v*100:>+6.1f}%"
    for sid in cand_avail:
        v = next((float(v) for d, v in cand_yr[sid].items() if d.year == yr_i), float("nan"))
        row += f"  {v*100:>+5.1f}%"
    print(row)

print()
print("Done.")
