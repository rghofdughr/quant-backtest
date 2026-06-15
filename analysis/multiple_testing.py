"""
analysis/multiple_testing.py — OOS multiple-testing correction for 65 strategies.

First run (or --extract): re-runs all strategy modules to extract OOS daily return
series, saves to results/oos_returns.parquet.  Subsequent runs load from cache.

Steps
-----
  1. Load / extract 65×N daily OOS return series, aligned on common dates.
     Report and exclude any strategy with fewer than --min-days observations.
  2. Compute 65×65 Pearson correlation matrix (pairwise complete observations).
     Save to results/oos_corr.csv.  Print mean / std / top pairs.
  3. Hierarchical clustering (average linkage, distance = 1 - r) at thresholds
     0.3, 0.5, 0.7.  Report cluster count (= N_eff) at every threshold.
  4. Per-strategy OOS Sharpe, SE (Lo 2002 accounting for skew/kurtosis), t-stat.
  5. Multiple-testing thresholds for each N_eff:
       Bonferroni  — t_crit = Phi^{-1}(1 - alpha/N_eff)
       BLP EMax    — E[max SR] under H0 per Bailey & de Prado (2014)
  6. One table: strategy | OOS Sharpe | t-stat | PASS/FAIL × every (N_eff × method)

No allocation recommendations.  Print everything; decide yourself.

Usage
-----
    cd C:\\Users\\Owner\\quant50
    python analysis/multiple_testing.py                    # load cache if exists
    python analysis/multiple_testing.py --extract          # rebuild cache (slow, ~hours)
    python analysis/multiple_testing.py --min-days 500     # stricter overlap cut
"""

from __future__ import annotations

import argparse
import importlib
import logging
import os
import sys
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import yaml
from scipy.cluster.hierarchy import fcluster, linkage
from scipy.spatial.distance import squareform
from scipy.stats import norm

warnings.filterwarnings("ignore")
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(name)s: %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)

RESULTS_DIR = Path("results")
CACHE_FILE  = RESULTS_DIR / "oos_returns.parquet"
OOS_FRAC    = 0.30          # must match runner.py
ALPHA       = 0.05
EULER_GAMMA = 0.5772156649  # Euler-Mascheroni constant
TRADING_DAYS = 252

CLUSTER_THRESHOLDS = [0.3, 0.5, 0.7]

ALL_SIDS = [f"s{i:02d}" for i in range(1, 66)]


# ─────────────────────────────────────────────────────────────────────────────
# helpers
# ─────────────────────────────────────────────────────────────────────────────

def _load_config(path: str) -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


def _hr(char="─", width=72):
    print(char * width)


# ─────────────────────────────────────────────────────────────────────────────
# 1.  EXTRACT OOS RETURNS
# ─────────────────────────────────────────────────────────────────────────────

def _extract_oos_returns(config: dict) -> pd.DataFrame:
    """
    Import and run every strategy module, extract OOS slice, collect into wide
    DataFrame (index = trading dates, columns = strategy IDs).
    Stubs and runtime errors are skipped with a warning.
    """
    from engine import is_oos_split
    from runner import MODULE_MAP

    results: dict[str, pd.Series] = {}
    total = len(ALL_SIDS)

    for i, sid in enumerate(ALL_SIDS, 1):
        module_name = MODULE_MAP.get(sid)
        if module_name is None:
            log.warning("[%d/%d] %s: not in MODULE_MAP — skip", i, total, sid)
            continue

        try:
            mod = importlib.import_module(module_name)
        except ModuleNotFoundError:
            log.warning("[%d/%d] %s: module not found — skip", i, total, sid)
            continue

        t0 = time.time()
        log.info("[%d/%d] %s ...", i, total, sid)

        try:
            result = mod.run(config)
        except NotImplementedError as exc:
            log.warning("[%d/%d] %s: stub (%s) — skip", i, total, sid, exc)
            continue
        except Exception as exc:
            log.error("[%d/%d] %s: error — %s", i, total, sid, exc)
            continue

        ret = result.get("returns")
        if ret is None or ret.empty:
            log.warning("[%d/%d] %s: empty returns — skip", i, total, sid)
            continue

        _, oos = is_oos_split(ret, OOS_FRAC)
        results[sid] = oos
        log.info("  → %d OOS days in %.1fs", len(oos), time.time() - t0)

    if not results:
        raise RuntimeError("No strategy returns extracted.")

    return pd.DataFrame(results)


# ─────────────────────────────────────────────────────────────────────────────
# 2.  DATA QUALITY CHECK AND EXCLUSION
# ─────────────────────────────────────────────────────────────────────────────

def _check_and_filter(oos_df: pd.DataFrame, min_days: int) -> pd.DataFrame:
    n_obs = oos_df.notna().sum().sort_values()

    _hr("=")
    print(f"STEP 1 — DATA QUALITY  (min_days = {min_days})")
    _hr("=")

    excluded = n_obs[n_obs < min_days].index.tolist()
    included = n_obs[n_obs >= min_days].index.tolist()

    print(f"\n  Total strategies loaded : {oos_df.shape[1]}")
    print(f"  Total date range        : {oos_df.index[0].date()} → {oos_df.index[-1].date()}")
    print(f"  Included (>= {min_days:4d} days) : {len(included)}")
    print(f"  Excluded (<  {min_days:4d} days) : {len(excluded)}")

    if excluded:
        print(f"\n  EXCLUDED strategies:")
        for sid in excluded:
            print(f"    {sid:6s}  {n_obs[sid]:5d} OOS days")
    else:
        print(f"\n  No strategies excluded.")

    clean = oos_df[included]
    obs_range = clean.notna().sum()
    print(f"\n  OOS observation counts for included strategies:")
    print(f"    min {obs_range.min():.0f} d | median {obs_range.median():.0f} d | "
          f"max {obs_range.max():.0f} d | mean {obs_range.mean():.0f} d")

    first = clean.apply(lambda c: c.first_valid_index())
    last  = clean.apply(lambda c: c.last_valid_index())
    print(f"    OOS start range : {first.min().date()} — {first.max().date()}")
    print(f"    OOS end range   : {last.min().date()} — {last.max().date()}")

    return clean


# ─────────────────────────────────────────────────────────────────────────────
# 3.  CORRELATION MATRIX
# ─────────────────────────────────────────────────────────────────────────────

def _compute_correlation(oos_df: pd.DataFrame) -> pd.DataFrame:
    _hr("=")
    print(f"STEP 2 — CORRELATION MATRIX  ({oos_df.shape[1]}×{oos_df.shape[1]})")
    _hr("=")

    corr = oos_df.corr(method="pearson", min_periods=50)
    corr.to_csv(RESULTS_DIR / "oos_corr.csv")

    n = corr.shape[0]
    mask = ~np.eye(n, dtype=bool)
    vals = corr.values[mask]
    vals = vals[np.isfinite(vals)]

    print(f"\n  Mean pairwise corr   : {vals.mean():.4f}")
    print(f"  Median pairwise corr : {np.median(vals):.4f}")
    print(f"  Std pairwise corr    : {vals.std():.4f}")
    print(f"  Min pairwise corr    : {vals.min():.4f}")
    print(f"  Max pairwise corr    : {vals.max():.4f}")

    # Fraction of pairs above common thresholds
    for cut in [0.3, 0.5, 0.7]:
        frac = (vals > cut).mean()
        print(f"  Pairs with r > {cut:.1f}   : {frac:.1%}")

    # Top / bottom pairs
    strats = corr.columns.tolist()
    pairs = [
        (strats[i], strats[j], corr.iloc[i, j])
        for i in range(n)
        for j in range(i + 1, n)
        if np.isfinite(corr.iloc[i, j])
    ]
    pairs_sorted = sorted(pairs, key=lambda x: x[2], reverse=True)

    print(f"\n  Top 20 most correlated pairs:")
    print(f"    {'Pair':<20s}  {'r':>7s}")
    for a, b, v in pairs_sorted[:20]:
        print(f"    {a}–{b:<16s}  {v:+7.4f}")

    print(f"\n  Bottom 20 least correlated (most independent) pairs:")
    print(f"    {'Pair':<20s}  {'r':>7s}")
    for a, b, v in sorted(pairs, key=lambda x: x[2])[:20]:
        print(f"    {a}–{b:<16s}  {v:+7.4f}")

    print(f"\n  Saved → results/oos_corr.csv")
    return corr


# ─────────────────────────────────────────────────────────────────────────────
# 4.  HIERARCHICAL CLUSTERING → N_eff
# ─────────────────────────────────────────────────────────────────────────────

def _estimate_effective_n(
    corr: pd.DataFrame,
    thresholds: list[float] = CLUSTER_THRESHOLDS,
) -> dict[float, dict]:
    """
    Returns {threshold: {'n_eff': int, 'labels': pd.Series}} for each threshold.
    distance = 1 - r  (so threshold = 0.3 means same cluster only if r > 0.7)
    """
    _hr("=")
    print("STEP 3 — EFFECTIVE NUMBER OF INDEPENDENT TESTS  (hierarchical clustering)")
    _hr("=")
    print()
    print("  Linkage      : average")
    print("  Distance     : d_ij = 1 − r_ij  (NaN pairs treated as r=0)")
    print()
    print(f"  {'Threshold':>9s}  {'Corr cutoff':>12s}  {'N_eff':>6s}  {'Note':s}")
    print(f"  {'-'*52}")

    # Build distance matrix; NaN → r=0 (independent)
    c = np.where(np.isfinite(corr.values), corr.values, 0.0)
    np.clip(c, -1.0, 1.0, out=c)
    np.fill_diagonal(c, 1.0)

    dist = 1.0 - c
    dist = (dist + dist.T) / 2          # enforce symmetry
    np.clip(dist, 0.0, None, out=dist)
    np.fill_diagonal(dist, 0.0)

    condensed = squareform(dist, checks=False)
    Z = linkage(condensed, method="average")

    result: dict[float, dict] = {}

    for t in sorted(thresholds):
        labels = fcluster(Z, t=t, criterion="distance")
        n_eff  = int(len(set(labels)))
        lseries = pd.Series(labels, index=corr.columns, name=f"cluster_t{t}")

        corr_equiv = f"r > {1-t:.1f}"
        note = "tight (most strategies independent)" if t <= 0.3 \
               else ("loose (most strategies one cluster)" if t >= 0.7 else "")
        print(f"  {t:>9.1f}  {corr_equiv:>12s}  {n_eff:>6d}  {note}")
        result[t] = {"n_eff": n_eff, "labels": lseries}

    # Detail: cluster membership at middle threshold
    mid_t = sorted(thresholds)[len(thresholds) // 2]
    n_mid = result[mid_t]["n_eff"]
    labels_mid = result[mid_t]["labels"]
    groups: dict[int, list[str]] = {}
    for sid, cl in labels_mid.items():
        groups.setdefault(int(cl), []).append(sid)

    print(f"\n  Cluster membership at threshold={mid_t}  (N_eff={n_mid}):")
    for cl_id, members in sorted(groups.items(), key=lambda kv: -len(kv[1])):
        print(f"    Cluster {cl_id:3d} ({len(members):2d}): {', '.join(sorted(members))}")

    return result


# ─────────────────────────────────────────────────────────────────────────────
# 5.  PER-STRATEGY SHARPE STATS
# ─────────────────────────────────────────────────────────────────────────────

def _compute_sharpe_stats(oos_df: pd.DataFrame) -> pd.DataFrame:
    """
    For each strategy column in oos_df, compute:
      n_days, t_years, oos_sharpe, se (Lo 2002), se_simple (1/sqrt(T)), t_stat.
    """
    rows = []
    for sid in oos_df.columns:
        r = oos_df[sid].dropna()
        n = len(r)
        if n < 30:
            continue

        t_years = n / TRADING_DAYS
        vol_d   = r.std(ddof=1)
        if vol_d == 0 or not np.isfinite(vol_d):
            continue

        sharpe = float(r.mean() / vol_d * np.sqrt(TRADING_DAYS))

        # Lo (2002) SE: sqrt((1 + SR^2/2 - skew*SR + (kurt/4)*SR^2) / T)
        # kurt here is excess kurtosis (pandas .kurt() returns excess)
        try:
            skew = float(r.skew())
            ekurt = float(r.kurt())   # excess kurtosis
        except Exception:
            skew, ekurt = 0.0, 0.0

        se_lo     = float(np.sqrt(max((1 + 0.5*sharpe**2 - skew*sharpe + (ekurt/4)*sharpe**2) / t_years, 1e-9)))
        se_simple = float(1.0 / np.sqrt(t_years))
        t_stat    = float(sharpe / se_lo) if se_lo > 0 else np.nan

        rows.append({
            "strategy": sid,
            "n_days":     n,
            "t_years":    round(t_years, 2),
            "oos_sharpe": round(sharpe, 4),
            "se_lo2002":  round(se_lo, 4),
            "se_simple":  round(se_simple, 4),
            "t_stat":     round(t_stat, 4),
        })

    return pd.DataFrame(rows).set_index("strategy")


# ─────────────────────────────────────────────────────────────────────────────
# 6.  MULTIPLE-TESTING THRESHOLDS
# ─────────────────────────────────────────────────────────────────────────────

def _blp_emax_t(n_eff: int) -> float:
    """
    Bailey-de Prado (2014) expected maximum Sharpe t-statistic for N_eff
    independent strategies under H0:

        E[max t] = (1-γ) * Φ^{-1}(1 - 1/N) + γ * Φ^{-1}(1 - 1/(N·e))

    where γ = Euler-Mascheroni ≈ 0.5772.
    For N=1 falls back to the standard single-test 5% critical value.
    """
    if n_eff <= 1:
        return float(norm.ppf(1 - ALPHA))
    z1 = float(norm.ppf(max(1 - 1.0 / n_eff,                   1e-15)))
    z2 = float(norm.ppf(max(1 - 1.0 / (n_eff * np.e),          1e-15)))
    return (1 - EULER_GAMMA) * z1 + EULER_GAMMA * z2


def _compute_thresholds(
    n_eff_map: dict[float, dict],
    median_t: float,
) -> dict[float, dict]:
    """
    For each clustering threshold → N_eff, compute:
      bonf_t   : Bonferroni critical t-stat  (one-tailed)
      bonf_sr  : Bonferroni critical SR       (= bonf_t / sqrt(median_T))
      blp_t    : BLP EMax critical t-stat
      blp_sr   : BLP EMax critical SR

    Returns {linkage_threshold: {...}} dict; also prints the table.
    """
    _hr("=")
    print("STEP 5 — MULTIPLE-TESTING THRESHOLDS")
    _hr("=")
    print(f"\n  alpha             = {ALPHA:.0%}  (one-tailed: testing SR > 0)")
    print(f"  Median OOS T      = {median_t:.2f} years")
    print()
    print("  Methods:")
    print("    Bonferroni — conservative upper bound; ignores cross-strategy correlation.")
    print("    BLP EMax   — Bailey & de Prado (2014) expected maximum of N_eff i.i.d. tests;")
    print("                 slightly less conservative; still assumes independence within cluster.")
    print()
    print(f"  {'Thresh':>7s}  {'N_eff':>5s}  │  "
          f"{'Bonf t_crit':>11s}  {'Bonf SR_crit':>13s}  │  "
          f"{'BLP t_crit':>10s}  {'BLP SR_crit':>12s}")
    print(f"  {'-'*80}")

    out: dict[float, dict] = {}
    for thresh in sorted(n_eff_map):
        n_eff = n_eff_map[thresh]["n_eff"]

        bonf_t  = float(norm.ppf(1 - ALPHA / n_eff))
        bonf_sr = bonf_t / np.sqrt(median_t)

        blp_t   = _blp_emax_t(n_eff)
        blp_sr  = blp_t / np.sqrt(median_t)

        out[thresh] = {
            "n_eff":    n_eff,
            "bonf_t":   round(bonf_t,  4),
            "bonf_sr":  round(bonf_sr, 4),
            "blp_t":    round(blp_t,   4),
            "blp_sr":   round(blp_sr,  4),
        }

        print(f"  {thresh:>7.1f}  {n_eff:>5d}  │  "
              f"{bonf_t:>11.4f}  {bonf_sr:>13.4f}  │  "
              f"{blp_t:>10.4f}  {blp_sr:>12.4f}")

    return out


# ─────────────────────────────────────────────────────────────────────────────
# 7.  PASS/FAIL TABLE
# ─────────────────────────────────────────────────────────────────────────────

def _passfall_table(
    sharpe_df: pd.DataFrame,
    thresh_info: dict[float, dict],
) -> pd.DataFrame:
    _hr("=")
    print("STEP 6 — PASS/FAIL TABLE  (PASS means t_stat > critical t at alpha=5%)")
    _hr("=")
    print()

    thresholds = sorted(thresh_info)

    # Column labels: e.g. N12_Bonf, N12_BLP
    col_labels: list[str] = []
    for t in thresholds:
        n = thresh_info[t]["n_eff"]
        col_labels.append(f"N{n}_Bonf(t={t})")
        col_labels.append(f"N{n}_BLP(t={t})")

    rows = []
    for sid, row in sharpe_df.iterrows():
        ts = row["t_stat"]
        r: dict = {
            "OOS_Sharpe": row["oos_sharpe"],
            "t_stat":     row["t_stat"],
            "T_years":    row["t_years"],
        }
        n_pass_bonf = 0
        n_pass_blp  = 0
        for t in thresholds:
            n_eff   = thresh_info[t]["n_eff"]
            b_t     = thresh_info[t]["bonf_t"]
            blp_t   = thresh_info[t]["blp_t"]
            r[f"N{n_eff}_Bonf(t={t})"] = "PASS" if ts > b_t   else "fail"
            r[f"N{n_eff}_BLP(t={t})"]  = "PASS" if ts > blp_t else "fail"
            if ts > b_t:
                n_pass_bonf += 1
            if ts > blp_t:
                n_pass_blp += 1

        # Robustness summary: how many threshold assumptions does it pass?
        r["pass_bonf"] = f"{n_pass_bonf}/{len(thresholds)}"
        r["pass_blp"]  = f"{n_pass_blp}/{len(thresholds)}"
        rows.append({"strategy": sid, **r})

    df = pd.DataFrame(rows).set_index("strategy")
    df = df.sort_values("OOS_Sharpe", ascending=False)

    # Print with wide display
    pd.set_option("display.width", 200)
    pd.set_option("display.max_rows", 200)
    pd.set_option("display.max_columns", 30)
    print(df.to_string())

    df.to_csv(RESULTS_DIR / "multiple_testing_results.csv")
    print(f"\n  Saved → results/multiple_testing_results.csv")

    # Summary counts
    print()
    _hr("-")
    print("PASS COUNT SUMMARY  (strategies passing ALL threshold assumptions)")
    _hr("-")
    for method, col_suffix in [("Bonferroni", "_Bonf"), ("BLP EMax", "_BLP")]:
        all_pass = df[[c for c in df.columns if col_suffix in c]].apply(
            lambda col: (col == "PASS").all(), axis=1
        )
        print(f"\n  {method}  — passes ALL {len(thresholds)} threshold assumptions:")
        for sid in df[all_pass].index:
            row = df.loc[sid]
            print(f"    {sid:6s}  Sharpe={row['OOS_Sharpe']:+.3f}  t={row['t_stat']:.3f}")
        if not all_pass.any():
            print(f"    (none)")

    return df


# ─────────────────────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Multiple-testing corrected OOS analysis for 65 strategies"
    )
    parser.add_argument(
        "--min-days", type=int, default=252,
        help="Exclude strategies with fewer OOS days than this (default: 252)"
    )
    parser.add_argument(
        "--extract", action="store_true",
        help="Re-run all strategy modules to rebuild OOS return cache "
             "(overwrites results/oos_returns.parquet; takes 2-6 hours)"
    )
    parser.add_argument("--config", default="config.yaml")
    args = parser.parse_args()

    config = _load_config(args.config)

    _hr("=")
    print("OOS MULTIPLE-TESTING ANALYSIS — quant50")
    _hr("=")

    # ── 1. Load / extract OOS returns ────────────────────────────────────────
    if args.extract or not CACHE_FILE.exists():
        if not args.extract:
            print(f"\n[WARNING] Cache not found at {CACHE_FILE}.")
            print("  This script must re-run all strategy modules to extract OOS return")
            print("  series.  That takes approximately 2–6 hours on a full run.")
            print("  Results will be cached; subsequent runs load instantly from parquet.")
            print(f"  To skip this and use a different path, run with --extract.\n")

        print(f"\n[EXTRACTING] Running {len(ALL_SIDS)} strategy modules ...")
        oos_df = _extract_oos_returns(config)
        RESULTS_DIR.mkdir(parents=True, exist_ok=True)
        oos_df.to_parquet(CACHE_FILE)
        print(f"[SAVED] OOS returns → {CACHE_FILE}  ({oos_df.shape})")
    else:
        print(f"\n[LOADING] {CACHE_FILE}")
        oos_df = pd.read_parquet(CACHE_FILE)
        print(f"  Loaded: {oos_df.shape[1]} strategies × {oos_df.shape[0]} dates")

    # ── 2. Data quality ───────────────────────────────────────────────────────
    print()
    oos_df = _check_and_filter(oos_df, args.min_days)

    # ── 3. Correlation matrix ─────────────────────────────────────────────────
    print()
    corr = _compute_correlation(oos_df)

    # ── 4. Effective N ────────────────────────────────────────────────────────
    print()
    n_eff_map = _estimate_effective_n(corr, CLUSTER_THRESHOLDS)

    # ── 5. Sharpe stats ───────────────────────────────────────────────────────
    sharpe_df = _compute_sharpe_stats(oos_df)
    median_t  = float(sharpe_df["t_years"].median())

    _hr("=")
    print("STEP 4 — PER-STRATEGY OOS SHARPE STATISTICS")
    _hr("=")
    print(f"\n  SE method: Lo (2002) — accounts for skewness and excess kurtosis of daily returns.")
    print(f"  Simple SE (1/sqrt(T)) shown for comparison.  t_stat uses Lo SE.\n")
    print(f"  {'Strategy':8s}  {'N days':>7s}  {'T (yrs)':>8s}  "
          f"{'OOS SR':>8s}  {'SE (Lo)':>8s}  {'SE(1/√T)':>9s}  {'t-stat':>8s}")
    print(f"  {'-'*70}")
    for sid, row in sharpe_df.sort_values("oos_sharpe", ascending=False).iterrows():
        print(f"  {sid:8s}  {row['n_days']:>7.0f}  {row['t_years']:>8.2f}  "
              f"  {row['oos_sharpe']:>+7.4f}  {row['se_lo2002']:>8.4f}  "
              f"{row['se_simple']:>9.4f}  {row['t_stat']:>8.4f}")
    print(f"\n  Median OOS T : {median_t:.2f} years")

    # ── 6. Multiple-testing thresholds ────────────────────────────────────────
    print()
    thresh_info = _compute_thresholds(n_eff_map, median_t)

    # ── 7. PASS/FAIL table ────────────────────────────────────────────────────
    print()
    _passfall_table(sharpe_df, thresh_info)

    print()
    _hr("=")
    print("Done.  Output files:")
    print(f"  {RESULTS_DIR / 'oos_returns.parquet'}       — daily OOS return series (cache)")
    print(f"  {RESULTS_DIR / 'oos_corr.csv'}              — correlation matrix")
    print(f"  {RESULTS_DIR / 'multiple_testing_results.csv'}  — PASS/FAIL table")
    _hr("=")


if __name__ == "__main__":
    main()
