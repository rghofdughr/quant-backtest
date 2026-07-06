"""
analysis/redundancy_new_batch.py
Redundancy test for 7 new strategy candidates vs 8-strategy validated book.
s119 excluded (IS SR -0.189 = MIRAGE).
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

BOOK_WEIGHTS = {
    "s08": 0.0667, "s30": 0.0667, "s31": 0.0667,
    "s35": 0.0667, "s49": 0.0667,
    "s02": 0.1650, "s46": 0.1650,
    "s90": 0.3300,
}
BOOK_STEMS = {
    "s08": "sector_rotation", "s30": "low_volatility", "s31": "vol_targeting",
    "s35": "sell_in_may",     "s49": "dollar_regime",  "s02": "ts_momentum",
    "s46": "risk_parity",     "s90": "credit_regime",
}
CAND_STEMS = {
    "s104": "gold_regime",          "s105": "oil_regime",
    "s113": "january_barometer",    "s115": "year_end_reversal",
    "s118": "low_vol_rotation",     "s121": "breadth_thrust",
    "s122": "52wk_high_proximity",
}
CAND_NAMES = {
    "s104": "Gold Regime",       "s105": "Oil/Energy Regime",
    "s113": "January Barometer", "s115": "Year-End Reversal",
    "s118": "Low-Vol Rotation",  "s121": "Breadth Thrust",
    "s122": "52wk High Prox",
}

def sharpe(ret):
    r = ret.dropna()
    if len(r) < 10 or r.std() < 1e-10: return float("nan")
    return float(r.mean() / r.std() * np.sqrt(252))

def neff(corr):
    ev = np.linalg.eigvalsh(corr.values)
    ev = np.maximum(ev, 1e-12)
    return float(ev.sum()**2 / (ev**2).sum())

def get_returns(sid, stem):
    mod = importlib.import_module(f"strategies.{sid}_{stem}")
    r = mod.run(config)
    return r["returns"]

# ── Step 1: Run everything ──────────────────────────────────────────────────
print("Running strategies...")
all_rets = {}
for sid, stem in {**BOOK_STEMS, **CAND_STEMS}.items():
    t0 = time.time()
    try:
        all_rets[sid] = get_returns(sid, stem)
        sr_is  = sharpe(all_rets[sid].loc[IS_START:IS_END])
        sr_oos = sharpe(all_rets[sid].loc[OOS_START:OOS_END])
        print(f"  {sid}: IS {sr_is:+.3f}  OOS {sr_oos:+.3f}  ({time.time()-t0:.0f}s)")
    except Exception as e:
        print(f"  {sid}: ERROR {e}")

# ── Step 2: Align ───────────────────────────────────────────────────────────
common_idx = None
for sid in BOOK_STEMS:
    if sid in all_rets:
        idx = all_rets[sid].dropna().index
        common_idx = idx if common_idx is None else common_idx.intersection(idx)

ret_df = pd.DataFrame({sid: all_rets[sid].reindex(common_idx).fillna(0.0)
                       for sid in list(BOOK_STEMS) + list(CAND_STEMS) if sid in all_rets})
ret_df = ret_df.dropna(how="all")

book_cols  = [s for s in BOOK_STEMS if s in ret_df.columns]
book_blend = sum(ret_df[s] * BOOK_WEIGHTS[s] for s in book_cols)
book_ret_df = ret_df[book_cols]

# ── Task 1: Correlation table ────────────────────────────────────────────────
print()
print("=" * 90)
print("  TASK 1: CORRELATION VS BOOK")
print("=" * 90)
print()
print(f"  {'ID':5s}  {'Name':20s}  {'vs s08':>7}  {'vs s02':>7}  {'vs s46':>7}  {'vs s90':>7}  {'vs blend':>8}  {'max book':>8}")
print(f"  {'-'*80}")

cand_corrs = {}
for sid in CAND_STEMS:
    if sid not in ret_df.columns:
        continue
    cand = ret_df[sid]
    corr_s08   = float(cand.corr(ret_df.get("s08", pd.Series(dtype=float))))
    corr_s02   = float(cand.corr(ret_df.get("s02", pd.Series(dtype=float))))
    corr_s46   = float(cand.corr(ret_df.get("s46", pd.Series(dtype=float))))
    corr_s90   = float(cand.corr(ret_df.get("s90", pd.Series(dtype=float))))
    corr_blend = float(cand.corr(book_blend))
    book_corrs = {b: float(cand.corr(ret_df[b])) for b in book_cols if b in ret_df.columns}
    max_corr   = max(book_corrs.values()) if book_corrs else float("nan")
    max_peer   = max(book_corrs, key=book_corrs.get) if book_corrs else "?"
    cand_corrs[sid] = dict(s08=corr_s08, s02=corr_s02, blend=corr_blend,
                           max_corr=max_corr, max_peer=max_peer)
    flag = "LIKELY-REDUNDANT" if max_corr >= 0.70 else ""
    name = CAND_NAMES[sid]
    print(f"  {sid:5s}  {name:20s}  {corr_s08:+7.3f}  {corr_s02:+7.3f}  {corr_s46:+7.3f}  {corr_s90:+7.3f}  {corr_blend:+8.3f}  {max_corr:+8.3f}  {flag}")

# ── Task 2: ΔN_eff and ΔSharpe ──────────────────────────────────────────────
print()
print("=" * 90)
print("  TASK 2: dN_eff AND dSharpe PER CANDIDATE")
print("=" * 90)
print()
corr_book  = book_ret_df.corr()
neff_book  = neff(corr_book)
sr_book    = sharpe(book_blend)
n_book     = len(book_cols)
print(f"  Book baseline: N_eff = {neff_book:.3f}   Blended SR = {sr_book:.3f}")
print()
print(f"  {'ID':5s}  {'Name':20s}  {'N_eff+c':>8}  {'dN_eff':>7}  {'dSR':>7}  {'Verdict':25s}")
print(f"  {'-'*80}")

ind_res = {}
for sid in CAND_STEMS:
    if sid not in ret_df.columns:
        continue
    cand = ret_df[sid]
    new_blend  = (book_blend * n_book + cand) / (n_book + 1)
    delta_sr   = sharpe(new_blend) - sr_book
    aug        = pd.concat([book_ret_df, cand.rename(sid)], axis=1).dropna()
    neff_aug   = neff(aug.corr())
    delta_neff = neff_aug - neff_book
    ind_res[sid] = dict(neff_aug=neff_aug, dneff=delta_neff, dsr=delta_sr)

    max_corr = cand_corrs.get(sid, {}).get("max_corr", 1.0)
    if sid in ["s119"]:
        verdict = "MIRAGE (IS SR < 0)"
    elif delta_neff <= 0:
        verdict = "REDUNDANT (dN_eff<=0)"
    elif delta_neff < 0.05:
        verdict = "BORDERLINE (<0.05)"
    elif delta_sr <= 0:
        verdict = "NON-ADDITIVE (dSR<=0)"
    elif max_corr >= 0.70:
        verdict = "HIGH-CORR (>=0.70)"
    else:
        verdict = "*** SURVIVOR ***"

    print(f"  {sid:5s}  {CAND_NAMES[sid]:20s}  {neff_aug:8.3f}  {delta_neff:+7.3f}  {delta_sr:+7.3f}  {verdict:25s}")

# ── Task 3: Group N_eff ──────────────────────────────────────────────────────
print()
print("=" * 90)
print("  GROUP N_eff: book + all 7 candidates")
print("=" * 90)
print()
avail = [s for s in CAND_STEMS if s in ret_df.columns]
all_tog = pd.concat([book_ret_df, ret_df[avail]], axis=1).dropna()
neff_all = neff(all_tog.corr())
print(f"  N_eff book alone:          {neff_book:.3f}")
print(f"  N_eff book + 7 candidates: {neff_all:.3f}  (delta {neff_all-neff_book:+.3f})")

# ── Task 4: Survivors ────────────────────────────────────────────────────────
print()
print("=" * 90)
print("  TASK 3: SURVIVORS")
print("=" * 90)
print()
survivors = [
    sid for sid, r in ind_res.items()
    if r["dneff"] > 0.05 and r["dsr"] > 0
    and cand_corrs.get(sid, {}).get("max_corr", 1.0) < 0.70
]
if survivors:
    print(f"  SURVIVORS: {survivors}")
    for sid in survivors:
        r = ind_res[sid]; c = cand_corrs[sid]
        print(f"    {sid} {CAND_NAMES[sid]}: dN_eff={r['dneff']:+.3f}  dSR={r['dsr']:+.3f}  max_corr={c['max_corr']:.3f}")
else:
    print("  ZERO SURVIVORS. Book unchanged.")

print()
print("  Book: s08 s46 s30 s02 s31 s35 s49 s90")
