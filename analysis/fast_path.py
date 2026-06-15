# -*- coding: utf-8 -*-
"""
analysis/fast_path.py -- OOS stats + multiple-testing correction from saved JSONs only.
No strategy re-execution. Skew/kurtosis unavailable -> SE = 1/sqrt(T_years); flagged.

Usage:
    cd C:\\Users\\Owner\\quant50
    python analysis/fast_path.py
    python analysis/fast_path.py --n-eff 15
    python analysis/fast_path.py --n-eff 65 25 10
"""
from __future__ import annotations
import argparse
import json
import math
import sys
from pathlib import Path

from scipy.stats import norm

RESULTS_DIR = Path("results")
ALPHA = 0.05
TRADING_DAYS = 252
EULER_GAMMA = 0.5772156649

DEFAULT_N_EFF = [65, 40, 20, 10]
ALL_SIDS = [f"s{i:02d}" for i in range(1, 103)]


# ---------------------------------------------------------------------------
# load all 65 JSONs
# ---------------------------------------------------------------------------

def load_stats() -> list[dict]:
    rows = []
    missing = []

    for sid in ALL_SIDS:
        path = RESULTS_DIR / f"{sid}_metrics.json"
        if not path.exists():
            missing.append(sid)
            continue

        with open(path) as f:
            d = json.load(f)

        if d.get("status") != "ok":
            missing.append(f"{sid}[{d.get('status')}]")
            continue

        oos = d.get("oos", {})
        iss = d.get("is",  {})

        sharpe  = oos.get("sharpe")
        vol_ann = oos.get("vol")
        n_days  = oos.get("n_days")
        cagr    = oos.get("cagr")
        mdd     = oos.get("max_dd")
        is_sh   = iss.get("sharpe")
        to      = d.get("turnover_annual")
        desc    = d.get("description", "")[:55]

        if any(v is None for v in [sharpe, vol_ann, n_days]):
            missing.append(f"{sid}[partial]")
            continue

        t_years = n_days / TRADING_DAYS
        se      = 1.0 / math.sqrt(t_years)
        t_stat  = sharpe / se if se > 0 else float("nan")

        rows.append({
            "sid":        sid,
            "desc":       desc,
            "n_days":     int(n_days),
            "t_years":    round(t_years, 2),
            "is_sr":      round(is_sh,  4) if is_sh is not None else None,
            "oos_sr":     round(sharpe, 4),
            "oos_vol":    round(vol_ann,4),
            "oos_cagr":   round(cagr,   4) if cagr  is not None else None,
            "oos_mdd":    round(mdd,    4) if mdd   is not None else None,
            "to":         round(to,     2) if to    is not None else None,
            "se":         round(se,     4),
            "t_stat":     round(t_stat, 4),
        })

    if missing:
        print(f"[WARN] Skipped / not-ok: {missing}")

    return sorted(rows, key=lambda r: r["oos_sr"], reverse=True)


# ---------------------------------------------------------------------------
# threshold computation
# ---------------------------------------------------------------------------

def _blp_t(n_eff: int) -> float:
    """Bailey-de Prado (2014) expected-maximum t-stat for N_eff i.i.d. tests."""
    if n_eff <= 1:
        return float(norm.ppf(1 - ALPHA))
    z1 = float(norm.ppf(max(1 - 1.0 / n_eff,                    1e-15)))
    z2 = float(norm.ppf(max(1 - 1.0 / (n_eff * math.e),         1e-15)))
    return (1 - EULER_GAMMA) * z1 + EULER_GAMMA * z2


def compute_thresholds(n_eff_list: list[int], median_t: float) -> dict:
    """Return {n_eff: {bonf_t, bonf_sr, blp_t, blp_sr}}."""
    out = {}
    for n in n_eff_list:
        bonf_t  = float(norm.ppf(1 - ALPHA / n))
        out[n] = dict(
            bonf_t  = bonf_t,
            bonf_sr = bonf_t / math.sqrt(median_t),
            blp_t   = _blp_t(n),
            blp_sr  = _blp_t(n) / math.sqrt(median_t),
        )
    return out


# ---------------------------------------------------------------------------
# display helpers
# ---------------------------------------------------------------------------

def _hr(w=110):
    print("=" * w)

def _sep(w=110):
    print("-" * w)

def _pf(t_stat: float, t_crit: float) -> str:
    if not math.isfinite(t_stat):
        return "  n/a"
    return " PASS" if t_stat >= t_crit else " fail"


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--n-eff", nargs="+", type=int, default=DEFAULT_N_EFF,
                        help="N_eff assumption(s) for multiple-testing (default: 65 40 20 10)")
    parser.add_argument("--sid-range", nargs=2, type=int, metavar=("FROM", "TO"),
                        help="Only include strategy numbers in [FROM, TO] inclusive")
    args = parser.parse_args()
    n_eff_list = sorted(set(args.n_eff), reverse=True)

    rows = load_stats()
    if args.sid_range:
        lo, hi = args.sid_range
        rows = [r for r in rows if lo <= int(r["sid"][1:]) <= hi]
    if not rows:
        print("ERROR: no valid strategy metrics found in results/")
        sys.exit(1)

    median_t = sorted(r["t_years"] for r in rows)[len(rows) // 2]
    thr = compute_thresholds(n_eff_list, median_t)

    # -----------------------------------------------------------------------
    # STEP 5: threshold table
    # -----------------------------------------------------------------------
    _hr()
    print("STEP 5 -- MULTIPLE-TESTING THRESHOLDS")
    _hr()
    print(f"  SE method : 1/sqrt(T_years)  [Lo-2002 unavailable -- no skew/kurt in JSONs]")
    print(f"  alpha     : {ALPHA:.0%}  (one-tailed: H1 = OOS SR > 0)")
    print(f"  Median T  : {median_t:.2f} years  ({int(median_t * TRADING_DAYS)} trading days)")
    print()
    print(f"  {'N_eff':>5s}  |  {'Bonf t_crit':>11s}  {'Bonf SR_crit':>13s}"
          f"  |  {'BLP t_crit':>10s}  {'BLP SR_crit':>12s}")
    _sep(70)
    for n in n_eff_list:
        t = thr[n]
        print(f"  {n:>5d}  |  {t['bonf_t']:>11.4f}  {t['bonf_sr']:>13.4f}"
              f"  |  {t['blp_t']:>10.4f}  {t['blp_sr']:>12.4f}")

    # -----------------------------------------------------------------------
    # STEP 4+6: per-strategy table with PASS/FAIL
    # -----------------------------------------------------------------------
    print()
    _hr()
    print("STEP 4+6 -- OOS STATS + PASS/FAIL  (sorted by OOS Sharpe, descending)")
    _hr()
    print(f"  SE = 1/sqrt(T);  t-stat = OOS_SR x sqrt(T_years);  PASS if t >= t_crit")
    print(f"  Column codes:  B<N> = Bonferroni N_eff=<N>,  P<N> = BLP-EMax N_eff=<N>")
    print()

    # header
    pf_hdr = "  ".join(f"{'B'+str(n):>5s}  {'P'+str(n):>5s}" for n in n_eff_list)
    print(f"  {'Strat':6s}  {'IS SR':>7s}  {'OOS SR':>7s}  {'t-stat':>7s}"
          f"  {'T(yr)':>6s}  {'CAGR':>7s}  {'MDD':>7s}  {'TO':>5s}  {pf_hdr}")
    _sep()

    for r in rows:
        ts   = r["t_stat"]
        is_s = f"{r['is_sr']:+7.4f}" if r["is_sr"] is not None else "    n/a"
        cagr_s = f"{r['oos_cagr']:+6.1%}" if r["oos_cagr"] is not None else "   n/a"
        mdd_s  = f"{r['oos_mdd']:+6.1%}"  if r["oos_mdd"]  is not None else "   n/a"
        to_s   = f"{r['to']:5.1f}"        if r["to"]       is not None else "  n/a"

        pf_cells = "  ".join(
            f"{_pf(ts, thr[n]['bonf_t']):>5s}  {_pf(ts, thr[n]['blp_t']):>5s}"
            for n in n_eff_list
        )

        print(f"  {r['sid']:6s}  {is_s}  {r['oos_sr']:+7.4f}  {ts:>7.3f}"
              f"  {r['t_years']:>6.2f}  {cagr_s}  {mdd_s}  {to_s}  {pf_cells}")

    # -----------------------------------------------------------------------
    # summary
    # -----------------------------------------------------------------------
    print()
    _sep()
    print()
    print("  Column key:")
    for n in n_eff_list:
        t = thr[n]
        print(f"    B{n:<3d}  Bonferroni N={n:<3d}  t_crit={t['bonf_t']:.3f}  SR_crit={t['bonf_sr']:.3f}")
        print(f"    P{n:<3d}  BLP EMax   N={n:<3d}  t_crit={t['blp_t']:.3f}  SR_crit={t['blp_sr']:.3f}")

    print()
    strictest_bonf_t = max(thr[n]["bonf_t"] for n in n_eff_list)
    strictest_blp_t  = max(thr[n]["blp_t"]  for n in n_eff_list)

    print(f"  Passes ALL Bonferroni columns (t >= {strictest_bonf_t:.3f}):")
    any_bonf = False
    for r in rows:
        if r["t_stat"] >= strictest_bonf_t:
            print(f"    {r['sid']}  OOS SR={r['oos_sr']:+.4f}  t={r['t_stat']:.3f}")
            any_bonf = True
    if not any_bonf:
        print("    (none)")

    print()
    print(f"  Passes ALL BLP columns (t >= {strictest_blp_t:.3f}):")
    any_blp = False
    for r in rows:
        if r["t_stat"] >= strictest_blp_t:
            print(f"    {r['sid']}  OOS SR={r['oos_sr']:+.4f}  t={r['t_stat']:.3f}")
            any_blp = True
    if not any_blp:
        print("    (none)")

    # -----------------------------------------------------------------------
    # save reference CSV
    # -----------------------------------------------------------------------
    import csv
    save_path = RESULTS_DIR / "original_run_sharpes.csv"
    fieldnames = ["sid","desc","is_sr","oos_sr","oos_vol","oos_cagr","oos_mdd",
                  "n_days","t_years","se","t_stat","to"]
    with open(save_path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        w.writeheader()
        w.writerows(rows)

    print()
    print(f"  Saved -> results/original_run_sharpes.csv")
    _hr()


if __name__ == "__main__":
    main()
