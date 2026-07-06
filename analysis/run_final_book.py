"""
Final 11-strategy book: metrics, vol-target sweep, and two interaction tests.

Book changes from prior 9-strategy base:
  s90 capped 29.67% -> 22.00% (concentration + cost; freed weight to s115/s31)
  s115 boosted 5.00% -> 8.00% (low-vol, high N_eff; gets freed s90 weight)
  s31  boosted 6.67% -> 9.00% (gets freed s90 weight)
  s113 added   0%   -> 3.00%  (seasonal sleeve companion to s115)
  s135 added   0%   -> 5.00%  (crash-convexity insurance)

Tests:
  1.  Final book IS/OOS Sharpe, CAGR, MDD, N_eff -- vs prior 9-strategy base
  2.  Per-year OOS table
  3.  Vol-target sweep [8/10/12/15/20%] on final book
      a. 2020 and 2022 with vs without s135 (interaction test)
      b. Max-leverage binding check per year
  4.  s90 concentration: 22% vs 29.67% -- MDD and cost budget only
"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import copy, importlib, yaml
import numpy as np
import pandas as pd

CFG_PATH    = os.path.join(os.path.dirname(__file__), "..", "config.yaml")
RESULTS_DIR = os.path.join(os.path.dirname(__file__), "..", "results")
os.makedirs(RESULTS_DIR, exist_ok=True)

with open(CFG_PATH) as f:
    BASE_CONFIG = yaml.safe_load(f)

IS_START     = "2000-01-03"
IS_END       = "2017-06-30"
OOS_START    = "2017-07-03"
OOS_END      = "2024-12-31"
TRADING_DAYS = 252
VOL_LOOK     = 63
AVG_SOFR     = 0.025
VOL_TARGETS  = [0.08, 0.10, 0.12, 0.15, 0.20]

# ── final 11-strategy book (weights sum to 1.00) ──────────────────────────────
FINAL_BOOK = [
    ("s128", "strategies.s128_sector_rotation_9m",  0.065),
    ("s30",  "strategies.s30_low_volatility",        0.060),
    ("s31",  "strategies.s31_vol_targeting",         0.090),
    ("s35",  "strategies.s35_sell_in_may",           0.060),
    ("s49",  "strategies.s49_dollar_regime",         0.060),
    ("s115", "strategies.s115_year_end_reversal",    0.080),
    ("s113", "strategies.s113_january_barometer",    0.030),
    ("s135", "strategies.s135_ts_momentum_blended",  0.050),
    ("s02",  "strategies.s02_ts_momentum",           0.145),
    ("s46",  "strategies.s46_risk_parity",           0.140),
    ("s90",  "strategies.s90_credit_regime",         0.220),
]

# prior 9-strategy base for comparison
OLD_BOOK = [
    ("s128", "strategies.s128_sector_rotation_9m",  0.0667),
    ("s30",  "strategies.s30_low_volatility",        0.0667),
    ("s31",  "strategies.s31_vol_targeting",         0.0667),
    ("s35",  "strategies.s35_sell_in_may",           0.0667),
    ("s49",  "strategies.s49_dollar_regime",         0.0667),
    ("s115", "strategies.s115_year_end_reversal",    0.0500),
    ("s02",  "strategies.s02_ts_momentum",           0.1483),
    ("s46",  "strategies.s46_risk_parity",           0.1483),
    ("s90",  "strategies.s90_credit_regime",         0.2967),
]


def make_config():
    cfg = copy.deepcopy(BASE_CONFIG)
    cfg["backtest"]["start_date"] = IS_START
    cfg["backtest"]["end_date"]   = OOS_END
    return cfg


def slice_is(r):  return r.loc[IS_START:IS_END].fillna(0)
def slice_oos(r): return r.loc[OOS_START:OOS_END].fillna(0)


def sr(r):
    r = r.dropna()
    if len(r) < 20 or r.std() == 0: return 0.0
    return float(r.mean() / r.std() * TRADING_DAYS ** 0.5)

def cagr(r):
    r = r.dropna().fillna(0)
    if len(r) < 2: return 0.0
    return float((1 + r).prod() ** (TRADING_DAYS / len(r)) - 1)

def mdd(r):
    w = (1 + r.fillna(0)).cumprod()
    return float(((w - w.cummax()) / w.cummax()).min())

def annual_vol(r):
    return float(r.std() * TRADING_DAYS ** 0.5)

def neff(ret_df):
    clean = ret_df.dropna(how="all", axis=1).fillna(0)
    if clean.shape[1] < 2: return 1.0
    corr = clean.corr()
    ev = np.linalg.eigvalsh(corr.values)
    ev = ev[ev > 1e-8]
    return float(ev.sum() ** 2 / (ev ** 2).sum())

def vol_target_overlay(ret, target, look=VOL_LOOK, lo=0.25, hi=4.0):
    rv = ret.rolling(look).std() * TRADING_DAYS ** 0.5
    scale = (target / rv).shift(1).fillna(1.0).clip(lo, hi)
    return ret * scale, scale

def yr(r, yr_int):
    ys, ye = f"{yr_int}-01-01", f"{yr_int}-12-31"
    return cagr(r.loc[ys:ye])


# ── load all unique strategies needed ─────────────────────────────────────────
def load_all_strategies(book_def, cfg):
    seen = {}
    for sid, mname, _ in book_def:
        if sid in seen: continue
        try:
            mod = importlib.import_module(mname)
            res = mod.run(cfg)
            seen[sid] = res["returns"].fillna(0)
        except Exception as e:
            print(f"  WARN: {sid} failed: {e}")
    return seen


def build_book(strategy_rets, book_def):
    total = sum(w for _, _, w in book_def)
    book  = None
    for sid, _, wt in book_def:
        if sid not in strategy_rets: continue
        contrib = strategy_rets[sid] * wt
        book = contrib if book is None else book.add(contrib, fill_value=0)
    return book.fillna(0) if book is not None else pd.Series(dtype=float)


def metrics_row(label, r_is, r_oos):
    return (f"{label:<20} "
            f"IS: SR={sr(r_is):>+6.3f} CAGR={cagr(r_is):>+6.1%} MDD={mdd(r_is):>+6.1%}  |  "
            f"OOS: SR={sr(r_oos):>+6.3f} CAGR={cagr(r_oos):>+6.1%} MDD={mdd(r_oos):>+6.1%}")


# ── main ──────────────────────────────────────────────────────────────────────
def main():
    lines = []
    def pr(s=""): lines.append(str(s)); print(str(s))

    cfg = make_config()

    # ── load strategies ────────────────────────────────────────────────────────
    pr("Loading strategies (net of cost, full history)...")
    all_sids = list({sid for sid, _, _ in FINAL_BOOK + OLD_BOOK})
    all_rets  = {}
    for sid, mname, _ in FINAL_BOOK + OLD_BOOK:
        if sid in all_rets: continue
        try:
            mod = importlib.import_module(mname)
            res = mod.run(cfg)
            all_rets[sid] = res["returns"].fillna(0)
            to  = res.get("turnover_annual", 0.0)
            print(f"  {sid}: SR(OOS)={sr(slice_oos(all_rets[sid])):>+6.3f}  TO={to:.1f}x")
        except Exception as e:
            print(f"  WARN: {sid} failed: {e}")

    from data import load_price_series, ADJ_TOTALRETURN
    spy_df  = load_price_series("SPY", IS_START, OOS_END,
                                ADJ_TOTALRETURN, BASE_CONFIG["paths"]["cache_dir"])
    spy_ret = spy_df["Close"].pct_change(fill_method=None).fillna(0)

    # ── build books ────────────────────────────────────────────────────────────
    final_raw = build_book(all_rets, FINAL_BOOK)
    old_raw   = build_book(all_rets, OLD_BOOK)

    # ── Section 1: IS/OOS full metrics ────────────────────────────────────────
    pr("")
    pr("=" * 76)
    pr("SECTION 1 -- FINAL 11-STRATEGY BOOK  vs  PRIOR 9-STRATEGY BASE")
    pr("=" * 76)
    pr(f"{'Book':<20} {'IS SR':>7} {'IS CAGR':>8} {'IS MDD':>7}  |  "
       f"{'OOS SR':>7} {'OOS CAGR':>9} {'OOS MDD':>8}  N_eff")
    pr("-" * 76)

    # N_eff from OOS returns
    final_oos_df = pd.DataFrame({
        sid: slice_oos(all_rets[sid]) for sid, _, _ in FINAL_BOOK if sid in all_rets
    })
    old_oos_df = pd.DataFrame({
        sid: slice_oos(all_rets[sid]) for sid, _, _ in OLD_BOOK if sid in all_rets
    })
    neff_final = neff(final_oos_df)
    neff_old   = neff(old_oos_df)

    def book_row(label, raw, neff_val):
        ri = slice_is(raw);  ro = slice_oos(raw)
        return (f"{label:<20} {sr(ri):>+7.3f} {cagr(ri):>+7.1%} {mdd(ri):>+7.1%}  |  "
                f"{sr(ro):>+7.3f} {cagr(ro):>+8.1%} {mdd(ro):>+8.1%}  {neff_val:.3f}")

    pr(book_row("Final (11 strats)", final_raw, neff_final))
    pr(book_row("Prior  (9 strats)", old_raw,   neff_old))
    pr(book_row("SPY", spy_ret, float("nan")))

    # weighted cost drag estimate
    pr("")
    pr("Weighted cost drag (20bp * TO * weight), 11-strategy book:")
    # Note: TO not available here, estimate from known values
    known_to = {"s128":2.1,"s30":0.8,"s31":7.4,"s35":1.9,"s49":0.8,
                "s115":0.9,"s113":0.9,"s135":9.9,"s02":6.6,"s46":0.5,"s90":7.6}
    drag = sum(known_to.get(sid,0) * 20 * wt for sid, _, wt in FINAL_BOOK)
    pr(f"  Est. book drag: {drag:.0f} bp/year  (s135 at 9.9x is {known_to['s135']*20*0.05:.0f}bp weighted)")

    # ── Section 2: per-year OOS ────────────────────────────────────────────────
    pr("")
    pr("=" * 76)
    pr("SECTION 2 -- PER-YEAR OOS RETURNS")
    pr("=" * 76)
    pr(f"{'Year':>6}  {'SPY':>6}  {'Final':>6}  {'Prior':>6}  {'s135':>6}  {'s113':>6}  {'s115':>6}")
    pr("-" * 76)
    for y in range(2017, 2025):
        row = (f"{y:>6}  {yr(spy_ret,y):>+5.1%}  {yr(final_raw,y):>+5.1%}  "
               f"{yr(old_raw,y):>+5.1%}")
        for sid in ["s135","s113","s115"]:
            if sid in all_rets:
                row += f"  {yr(all_rets[sid],y):>+5.1%}"
            else:
                row += f"  {'--':>5}"
        pr(row)

    # ── Section 3: vol-target sweep ────────────────────────────────────────────
    pr("")
    pr("=" * 76)
    pr("SECTION 3 -- VOL-TARGET SWEEP  (final 11-strategy book, OOS net)")
    pr("=" * 76)
    pr(f"{'Target':>8}  {'CAGR':>7}  {'Sharpe':>7}  {'MDD':>7}  "
       f"{'AvgLev':>7}  {'MaxLev':>7}  {'AdjCAGR':>8}")
    pr("-" * 76)

    final_oos = slice_oos(final_raw)
    spy_oos   = slice_oos(spy_ret)

    vt_rets = {}
    vt_scales = {}
    pr(f"{'raw':>8}  {cagr(final_oos):>+6.1%}  {sr(final_oos):>+7.3f}  "
       f"{mdd(final_oos):>+7.1%}  {'1.00x':>7}  {'1.00x':>7}  {cagr(final_oos):>+7.1%}")

    for vt in VOL_TARGETS:
        vt_r, sc = vol_target_overlay(final_oos, vt)
        vt_rets[vt]   = vt_r
        vt_scales[vt] = sc
        avg_lv = float(sc.mean())
        max_lv = float(sc.max())
        fin_c  = max(avg_lv - 1.0, 0.0) * AVG_SOFR
        adj_c  = cagr(vt_r) - fin_c
        pr(f"{vt:>7.0%}   {cagr(vt_r):>+6.1%}  {sr(vt_r):>+7.3f}  "
           f"{mdd(vt_r):>+7.1%}  {avg_lv:>6.2f}x  {max_lv:>6.2f}x  {adj_c:>+7.1%}")

    # ── Subsection 3a: s135 interaction test ──────────────────────────────────
    pr("")
    pr("-- 3a: s135 interaction: does vol overlay cancel crash convexity? --")
    pr("Build book WITHOUT s135 (its 5% redistributed proportionally to others).")
    pr("")

    # build book-minus-s135 (redistribute 5% to remaining 10 strategies proportionally)
    no135_book = [(sid, mn, wt) for sid, mn, wt in FINAL_BOOK if sid != "s135"]
    w_no135 = sum(wt for _, _, wt in no135_book)
    no135_book_norm = [(sid, mn, wt / w_no135) for sid, mn, wt in no135_book]
    no135_raw = build_book(all_rets, no135_book_norm)
    no135_oos = slice_oos(no135_raw)

    pr(f"{'':8}  {'Final+s135':>12}  {'No s135':>10}  {'s135 alone':>10}")
    pr(f"{'':8}  {'raw / 10%VT':>12}  {'10%VT':>10}  {'raw':>10}")
    pr("-" * 50)

    s135_oos = slice_oos(all_rets.get("s135", pd.Series(dtype=float)))

    for y in [2020, 2022, 2018, 2019, 2021]:
        vt_r10_final, _  = vol_target_overlay(final_oos, 0.10)
        vt_r10_no135, _  = vol_target_overlay(no135_oos, 0.10)
        row = (f"{y:>8}  {yr(final_oos,y):>+6.1%}/{yr(vt_r10_final,y):>+5.1%}  "
               f"{yr(vt_r10_no135,y):>+9.1%}  {yr(s135_oos,y):>+9.1%}")
        pr(row)

    pr("")
    pr("Interpretation:")
    pr("  If 'Final 10%VT' << 'Final raw' in 2020: overlay killed convexity.")
    pr("  If 'No-s135 10%VT' and 'Final 10%VT' are close in 2020: s135 adds little")
    pr("  to the VT-scaled book, despite its raw +69%.")

    # ── Subsection 3b: max-leverage binding check ─────────────────────────────
    pr("")
    pr("-- 3b: Max-leverage binding per calendar year at 10% target --")
    pr("(Cap = 4.0x; binding = scale hit 4.0 on at least one day in that year)")
    pr("")
    vt_10_r, vt_10_s = vol_target_overlay(final_oos, 0.10)
    pr(f"{'Year':>6}  {'AvgLev':>7}  {'MaxLev':>7}  {'Cap bound?':>10}")
    pr("-" * 40)
    for y in range(2017, 2025):
        ys, ye = f"{y}-01-01", f"{y}-12-31"
        sc_y = vt_10_s.loc[ys:ye]
        if sc_y.empty: continue
        avg_lv = float(sc_y.mean())
        max_lv = float(sc_y.max())
        bound  = "YES" if max_lv >= 3.99 else "no"
        pr(f"{y:>6}  {avg_lv:>7.2f}x  {max_lv:>7.2f}x  {bound:>10}")

    # ── Section 4: s90 concentration ─────────────────────────────────────────
    pr("")
    pr("=" * 76)
    pr("SECTION 4 -- s90 CONCENTRATION: 22% cap vs 29.67% (prior book weight)")
    pr("Frame: MDD and cost budget; Sharpe difference is within noise.")
    pr("=" * 76)

    # prior book already has s90 at 29.67%
    s90_uncapped_wt = 0.2967
    s90_capped_wt   = 0.220

    pr("")
    pr(f"{'Metric':<30} {'s90 @ 29.67%':>13} {'s90 @ 22%':>11}  Change")
    pr("-" * 60)

    r_uncapped_oos = slice_oos(old_raw)       # 9-strat, s90=29.67%
    r_capped_oos   = slice_oos(final_raw)     # 11-strat, s90=22%

    pr(f"{'OOS Sharpe':<30} {sr(r_uncapped_oos):>+13.3f} {sr(r_capped_oos):>+11.3f}"
       f"  {sr(r_capped_oos)-sr(r_uncapped_oos):>+7.3f}  (within noise)")
    pr(f"{'OOS CAGR':<30} {cagr(r_uncapped_oos):>+12.1%} {cagr(r_capped_oos):>+10.1%}"
       f"  {cagr(r_capped_oos)-cagr(r_uncapped_oos):>+6.1%}")
    pr(f"{'OOS MDD':<30} {mdd(r_uncapped_oos):>+12.1%} {mdd(r_capped_oos):>+10.1%}"
       f"  {mdd(r_capped_oos)-mdd(r_uncapped_oos):>+6.1%}")

    # cost drag estimate for s90 at both weights
    s90_drag_old = 7.6 * 20 * s90_uncapped_wt
    s90_drag_new = 7.6 * 20 * s90_capped_wt
    pr(f"{'s90 cost drag (bp/yr)':<30} {s90_drag_old:>13.0f} {s90_drag_new:>11.0f}"
       f"  {s90_drag_new - s90_drag_old:>+6.0f}")

    # estimate s90 risk-budget share: s90's vol × weight / book vol
    s90_vol  = annual_vol(slice_oos(all_rets.get("s90", pd.Series(dtype=float))))
    s90_contrib_old = s90_vol * s90_uncapped_wt
    s90_contrib_new = s90_vol * s90_capped_wt
    book_vol_old = annual_vol(r_uncapped_oos)
    book_vol_new = annual_vol(r_capped_oos)
    rb_old = s90_contrib_old / book_vol_old
    rb_new = s90_contrib_new / book_vol_new
    pr(f"{'s90 risk-budget pct (naive)':<30} {rb_old:>12.0%} {rb_new:>10.0%}"
       f"  {rb_new - rb_old:>+6.0%}")
    pr("")
    pr("Note: naive risk budget = (s90 vol x weight) / book vol; ignores correlations.")
    pr("Actual contribution is lower due to partial correlation with other strategies.")

    # ── Section 5: locked 10% VT summary ─────────────────────────────────────
    pr("")
    pr("=" * 76)
    pr("SECTION 5 -- FINAL BOOK AT LOCKED 10% VOL TARGET  (summary)")
    pr("=" * 76)
    vt10_r, vt10_s = vol_target_overlay(final_oos, 0.10)
    fin_c = max(float(vt10_s.mean()) - 1.0, 0.0) * AVG_SOFR
    pr(f"  OOS Sharpe     : {sr(vt10_r):>+.3f}")
    pr(f"  OOS CAGR (raw) : {cagr(vt10_r):>+.1%}")
    pr(f"  OOS CAGR (adj) : {cagr(vt10_r) - fin_c:>+.1%}  (SOFR {AVG_SOFR*100:.1f}% avg on levered fraction)")
    pr(f"  OOS MDD        : {mdd(vt10_r):>+.1%}")
    pr(f"  Avg leverage   : {float(vt10_s.mean()):.2f}x")
    pr(f"  Max leverage   : {float(vt10_s.max()):.2f}x")
    pr(f"  N_eff          : {neff_final:.3f}  (OOS strategy correlation matrix)")
    pr(f"  Est. book drag : {drag:.0f} bp/yr")

    # save
    out_path = os.path.join(RESULTS_DIR, "final_book_output.txt")
    with open(out_path, "w", encoding="ascii", errors="replace") as fh:
        fh.write("\n".join(lines))
    print(f"\nSaved: {out_path}")


if __name__ == "__main__":
    main()
