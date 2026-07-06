"""
analysis/run_advanced.py
Advanced combination analysis:
  1. Book update: s08 -> s128, add s115 (passed audit)
  2. Run s132 (sector ensemble), s133 (credit-gated), s135 (blended TS mom)
  3. Redundancy test vs updated book
  4. ERC weights for updated book
  5. Vol-targeted book overlay (show impact)
  6. Seasonal basket: s115 + s113 combined sleeve
  7. Per-year OOS table
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

# Updated book: s08 -> s128, s115 added (same small weight)
# Rebalance weights slightly: s115 at 0.05, s08/s128 slot unchanged
BOOK_WEIGHTS_OLD = {
    "s08": 0.0667, "s30": 0.0667, "s31": 0.0667, "s35": 0.0667, "s49": 0.0667,
    "s02": 0.1650, "s46": 0.1650, "s90": 0.3300,
}
BOOK_WEIGHTS_NEW = {
    "s128": 0.0667, "s30": 0.0667, "s31": 0.0667, "s35": 0.0667, "s49": 0.0667,
    "s115": 0.0500,
    "s02":  0.1483, "s46": 0.1483, "s90": 0.2967,  # scale down proportionally
}

STEMS = {
    "s02":  "ts_momentum",        "s08":  "sector_rotation",   "s30":  "low_volatility",
    "s31":  "vol_targeting",      "s35":  "sell_in_may",        "s46":  "risk_parity",
    "s49":  "dollar_regime",      "s90":  "credit_regime",      "s115": "year_end_reversal",
    "s113": "january_barometer",  "s128": "sector_rotation_9m",
    "s132": "sector_ensemble",    "s133": "credit_gated_sectors",
    "s135": "ts_momentum_blended",
}

CAND_NAMES = {
    "s132": "Sector Ensemble",
    "s133": "Credit-Gated Sectors",
    "s135": "TS Mom Blended (1m+12m)",
}
CAND_IDS = ["s132", "s133", "s135"]

SEP = "=" * 80


def sr(ret, s=None, e=None):
    r = ret.loc[s:e].dropna() if s else ret.dropna()
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

def neff(corr_matrix):
    evals = np.linalg.eigvalsh(corr_matrix.values)
    evals = np.maximum(evals, 1e-12)
    return float((evals.sum() ** 2) / (evals ** 2).sum())

def run_strat(sid):
    mod = importlib.import_module(f"strategies.{sid}_{STEMS[sid]}")
    return mod.run(config)

def blend(ret_df, weights):
    avail = {s: w for s, w in weights.items() if s in ret_df.columns}
    total_w = sum(avail.values())
    return sum(ret_df[s] * (w / total_w) for s, w in avail.items())

def erc_weights(ret_df, sids):
    """Compute equal-risk-contribution weights via iterative algorithm."""
    valid = [s for s in sids if s in ret_df.columns]
    if len(valid) < 2:
        return {s: 1.0/len(valid) for s in valid}
    cov = ret_df[valid].cov().values * 252
    n = len(valid)
    w = np.ones(n) / n
    for _ in range(500):
        sigma = np.sqrt(w @ cov @ w)
        if sigma < 1e-10: break
        rc = (cov @ w) * w / sigma
        target = sigma / n
        grad = rc - target
        w = w - 0.05 * grad
        w = np.maximum(w, 1e-4)
        w /= w.sum()
    return {valid[i]: float(w[i]) for i in range(n)}


# ==============================================================================
# Step 1: Run all book + candidate strategies
# ==============================================================================
print()
print(SEP)
print("  STEP 1: RUN ALL STRATEGIES (old book + new candidates + s115 + s113)")
print(SEP)
print()

all_rets = {}
book_sids  = list(BOOK_WEIGHTS_OLD.keys()) + ["s128", "s115", "s113"]
extra_sids = CAND_IDS

all_to_run = list(dict.fromkeys(book_sids + extra_sids))

for sid in all_to_run:
    if sid not in STEMS:
        print(f"  {sid}: no stem, skip")
        continue
    t0 = time.time()
    try:
        result = run_strat(sid)
        ret = result.get("returns", pd.Series(dtype=float))
        if ret is None or ret.empty:
            print(f"  {sid}: EMPTY")
            continue
        all_rets[sid] = ret
        elapsed = time.time() - t0
        is_sr_  = sr(ret, IS_START, IS_END)
        oos_sr_ = sr(ret, OOS_START, OOS_END)
        to_     = result.get("turnover_annual", float("nan"))
        print(f"  {sid:5s}  IS {is_sr_:+.3f}  OOS {oos_sr_:+.3f}  TO {to_:.1f}x  ({elapsed:.0f}s)")
    except Exception as e:
        elapsed = time.time() - t0
        print(f"  {sid}: ERROR - {e}  ({elapsed:.0f}s)")

# ==============================================================================
# Step 2: Compare old book vs new book
# ==============================================================================
print()
print(SEP)
print("  STEP 2: OLD BOOK vs NEW BOOK (s08 -> s128, + s115)")
print(SEP)
print()

common_idx = None
for sid in ["s08", "s128"] + [s for s in BOOK_WEIGHTS_OLD if s != "s08"]:
    if sid in all_rets:
        idx = all_rets[sid].dropna().index
        common_idx = idx if common_idx is None else common_idx.intersection(idx)

if common_idx is not None and len(common_idx) > 100:
    ret_df = pd.DataFrame({sid: all_rets[sid].reindex(common_idx).fillna(0.0)
                           for sid in all_rets})

    book_old = blend(ret_df, BOOK_WEIGHTS_OLD)
    book_new = blend(ret_df, BOOK_WEIGHTS_NEW)

    def show_book(label, blend_ret, weights):
        sids_avail = [s for s in weights if s in ret_df.columns]
        corr_mat = ret_df[sids_avail].corr()
        neff_val = neff(corr_mat)
        print(f"  {label}")
        print(f"    IS SR={sr(blend_ret, IS_START, IS_END):+.3f}  OOS SR={sr(blend_ret, OOS_START, OOS_END):+.3f}  "
              f"OOS CAGR={cagr(blend_ret, OOS_START, OOS_END)*100:+.1f}%  "
              f"OOS MDD={mdd(blend_ret, OOS_START, OOS_END)*100:+.1f}%  N_eff={neff_val:.3f}")

    show_book("Old book (s08 3m/top3, no s115):", book_old, BOOK_WEIGHTS_OLD)
    show_book("New book (s128 9m/top4, + s115):", book_new, BOOK_WEIGHTS_NEW)
else:
    print("  ERROR: could not align book strategies")
    book_old = book_new = None
    ret_df = pd.DataFrame()

# ==============================================================================
# Step 3: ERC weights for new book
# ==============================================================================
print()
print(SEP)
print("  STEP 3: ERC WEIGHTS FOR NEW BOOK")
print(SEP)
print()

new_book_sids = list(BOOK_WEIGHTS_NEW.keys())
if len(ret_df) > 0:
    erc = erc_weights(ret_df, new_book_sids)
    print(f"  {'ID':5s}  {'Cluster-Parity':>16}  {'ERC':>8}  {'Vol (OOS)':>10}")
    print(f"  {'-'*50}")
    for sid in new_book_sids:
        if sid not in ret_df.columns: continue
        vol_oos = ret_df[sid].loc[OOS_START:OOS_END].std() * np.sqrt(252) if len(ret_df[sid].dropna()) > 10 else float("nan")
        cp_w = BOOK_WEIGHTS_NEW.get(sid, 0.0)
        erc_w = erc.get(sid, float("nan"))
        print(f"  {sid:5s}  {cp_w:>16.4f}  {erc_w:>8.4f}  {vol_oos*100:>9.1f}%")

    # Show ERC book performance
    book_erc = blend(ret_df, erc)
    erc_sids_avail = [s for s in erc if s in ret_df.columns]
    corr_erc = ret_df[erc_sids_avail].corr()
    print()
    print(f"  ERC book:  IS SR={sr(book_erc, IS_START, IS_END):+.3f}  "
          f"OOS SR={sr(book_erc, OOS_START, OOS_END):+.3f}  "
          f"OOS MDD={mdd(book_erc, OOS_START, OOS_END)*100:+.1f}%  "
          f"N_eff={neff(corr_erc):.3f}")

# ==============================================================================
# Step 4: Vol-targeted book overlay
# ==============================================================================
print()
print(SEP)
print("  STEP 4: VOL-TARGETED BOOK OVERLAY")
print(SEP)
print()

if book_new is not None and len(book_new.dropna()) > 100:
    vol_target = 0.10   # 10% annualised vol target
    vol_look   = 63     # 63d vol estimate

    book_vol   = book_new.rolling(vol_look).std() * np.sqrt(252)
    scale      = (vol_target / book_vol).shift(1).fillna(1.0).clip(0.5, 2.0)
    book_vt    = book_new * scale

    print(f"  Raw new book:       IS SR={sr(book_new, IS_START, IS_END):+.3f}  "
          f"OOS SR={sr(book_new, OOS_START, OOS_END):+.3f}  "
          f"OOS MDD={mdd(book_new, OOS_START, OOS_END)*100:+.1f}%")
    print(f"  Vol-targeted (10%): IS SR={sr(book_vt, IS_START, IS_END):+.3f}  "
          f"OOS SR={sr(book_vt, OOS_START, OOS_END):+.3f}  "
          f"OOS MDD={mdd(book_vt, OOS_START, OOS_END)*100:+.1f}%")
    avg_scale_oos = scale.loc[OOS_START:OOS_END].mean()
    print(f"  (Avg scale factor OOS: {avg_scale_oos:.2f}x)")

# ==============================================================================
# Step 5: Seasonal basket (s115 + s113)
# ==============================================================================
print()
print(SEP)
print("  STEP 5: SEASONAL BASKET (s115 + s113)")
print(SEP)
print()

if "s115" in all_rets and "s113" in all_rets and len(ret_df) > 0:
    r115 = ret_df["s115"] if "s115" in ret_df.columns else pd.Series(dtype=float)
    r113 = ret_df["s113"] if "s113" in ret_df.columns else pd.Series(dtype=float)

    if not r115.empty and not r113.empty:
        cross_corr = float(r115.corr(r113))
        basket     = 0.5 * r115 + 0.5 * r113

        print(f"  s115 (Year-End Rev):   IS SR={sr(r115, IS_START, IS_END):+.3f}  "
              f"OOS SR={sr(r115, OOS_START, OOS_END):+.3f}  "
              f"OOS MDD={mdd(r115, OOS_START, OOS_END)*100:+.1f}%")
        print(f"  s113 (Jan Barometer):  IS SR={sr(r113, IS_START, IS_END):+.3f}  "
              f"OOS SR={sr(r113, OOS_START, OOS_END):+.3f}  "
              f"OOS MDD={mdd(r113, OOS_START, OOS_END)*100:+.1f}%")
        print(f"  Cross-correlation s115 vs s113: {cross_corr:+.3f}")
        print(f"  Equal-weight basket:   IS SR={sr(basket, IS_START, IS_END):+.3f}  "
              f"OOS SR={sr(basket, OOS_START, OOS_END):+.3f}  "
              f"OOS MDD={mdd(basket, OOS_START, OOS_END)*100:+.1f}%")

        # Test basket vs new book
        if book_new is not None:
            n_book   = len([s for s in BOOK_WEIGHTS_NEW if s in ret_df.columns])
            new_blend = (book_new * n_book + basket) / (n_book + 1)
            ds = sr(new_blend) - sr(book_new)
            # N_eff with basket
            book_cols = [s for s in BOOK_WEIGHTS_NEW if s in ret_df.columns]
            augmented = pd.concat([ret_df[book_cols], basket.rename("seasonal")], axis=1).dropna()
            neff_aug  = neff(augmented.corr())
            neff_base = neff(ret_df[book_cols].corr())
            print(f"  Basket vs new book:    dN_eff={neff_aug - neff_base:+.3f}  dSR={ds:+.3f}")

# ==============================================================================
# Step 6: Redundancy test for s132, s133, s135 vs NEW book
# ==============================================================================
print()
print(SEP)
print("  STEP 6: REDUNDANCY TEST -- s132/s133/s135 vs NEW BOOK")
print(SEP)

if len(ret_df) > 0:
    book_cols = [s for s in BOOK_WEIGHTS_NEW if s in ret_df.columns]
    book_ret_df = ret_df[book_cols]
    corr_book   = book_ret_df.corr()
    neff_book   = neff(corr_book)
    book_blend_ = blend(ret_df, BOOK_WEIGHTS_NEW)
    sr_book     = sr(book_blend_)

    print()
    print(f"  New book baseline:  N_eff={neff_book:.3f}  SR={sr_book:+.3f}  "
          f"OOS SR={sr(book_blend_, OOS_START, OOS_END):+.3f}")
    print()
    print(f"  {'ID':5s}  {'Name':28s}  {'IS SR':>7}  {'OOS SR':>7}  {'OOS CAGR':>9}  "
          f"{'OOS MDD':>8}  {'TO':>6}  {'max_corr':>9}  {'dN_eff':>7}  {'dSR':>7}  {'Verdict':22s}")
    print(f"  {'-'*130}")

    for sid in CAND_IDS:
        if sid not in ret_df.columns:
            print(f"  {sid}: no data")
            continue

        cand = ret_df[sid]
        is_sr_  = sr(cand, IS_START, IS_END)
        oos_sr_ = sr(cand, OOS_START, OOS_END)
        oos_cg_ = cagr(cand, OOS_START, OOS_END)
        oos_md_ = mdd(cand, OOS_START, OOS_END)

        book_corrs = {b: float(cand.corr(ret_df[b])) for b in book_cols}
        max_corr   = max(book_corrs.values()) if book_corrs else float("nan")
        max_peer   = max(book_corrs, key=book_corrs.get) if book_corrs else "?"

        n_book    = len(book_cols)
        new_blend = (book_blend_ * n_book + cand) / (n_book + 1)
        delta_sr  = sr(new_blend) - sr_book

        augmented  = pd.concat([book_ret_df, cand.rename(sid)], axis=1).dropna()
        neff_aug   = neff(augmented.corr())
        delta_neff = neff_aug - neff_book

        if delta_neff <= 0:       verdict = "REDUNDANT"
        elif delta_neff < 0.05:   verdict = "BORDERLINE"
        elif delta_sr <= 0:       verdict = "NON-ADDITIVE"
        else:                     verdict = "SURVIVOR"

        def fs(v): return f"{v:+.3f}" if v == v else "  nan"
        def fp(v): return f"{v*100:+.1f}%" if v == v else "   nan"

        print(f"  {sid:5s}  {CAND_NAMES[sid]:28s}  {fs(is_sr_):>7}  {fs(oos_sr_):>7}  "
              f"{fp(oos_cg_):>9}  {fp(oos_md_):>8}  "
              f"  N/A  {max_corr:+9.3f}  {delta_neff:+7.3f}  {delta_sr:+7.3f}  {verdict:22s}")

# ==============================================================================
# Step 7: Per-year OOS comparison
# ==============================================================================
print()
print(SEP)
print("  STEP 7: OOS PER-YEAR (2017-2024)")
print(SEP)
print()

from data import load_price_series, ADJ_TOTALRETURN
spy = load_price_series("SPY", start=OOS_START, end=OOS_END,
                         adjustment=ADJ_TOTALRETURN,
                         cache_dir=config["paths"]["cache_dir"])
spy_yr = (1 + spy["Close"].pct_change(fill_method=None)).resample("YE").prod() - 1

def yr(ret):
    return (1 + ret).resample("YE").prod() - 1

show_rets = {}
if book_old is not None:
    show_rets["OldBook"] = book_old.loc[OOS_START:OOS_END]
if book_new is not None:
    show_rets["NewBook"] = book_new.loc[OOS_START:OOS_END]
for sid in CAND_IDS:
    if sid in ret_df.columns:
        show_rets[sid] = ret_df[sid].loc[OOS_START:OOS_END]

yr_data = {k: yr(v) for k, v in show_rets.items()}

years = sorted({int(d.year) for d in yr(book_old.loc[OOS_START:OOS_END]).index}) if book_old is not None else list(range(2017, 2025))
header = f"  {'Year':>5}  {'SPY':>7}" + "".join(f"  {k[:8]:>8}" for k in show_rets)
print(header)
print(f"  {'-'*80}")

for yr_i in years:
    spy_v = next((float(v) for d, v in spy_yr.items() if d.year == yr_i), float("nan"))
    row = f"  {yr_i:>5}  {spy_v*100:>+6.1f}%"
    for k, yr_s in yr_data.items():
        v = next((float(v) for d, v in yr_s.items() if d.year == yr_i), float("nan"))
        row += f"  {v*100:>+7.1f}%"
    print(row)

print()
print("Done.")
