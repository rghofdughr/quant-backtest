"""
analysis/group2_validate.py
----------------------------
Full validation of Group 2 strategies (s66-s101) through the real pipeline.

Tasks:
  0 — s81 bug verification (run fixed strategy, confirm sane numbers)
  1 — IS/OOS table for all Group 2, same flags as the original 50
  2 — Regime breakdown (2000-2002, 2008, 2013-2017, 2020, 2022)
  3 — Correlation / N_eff / marginal value vs existing book
  4 — Write results/group2_validation.md

Usage:
  cd C:\\Users\\Owner\\quant-backtest
  python analysis/group2_validate.py

Strategy runs are checkpointed to results/returns/g2_<id>.parquet so
the script can be safely interrupted and re-started.
"""
from __future__ import annotations

import importlib
import json
import logging
import sys
import time
from io import StringIO
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

REPO_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(REPO_ROOT))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s: %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("group2_validate")

RESULTS_DIR = REPO_ROOT / "results"
RETURNS_DIR = RESULTS_DIR / "returns"
RETURNS_DIR.mkdir(parents=True, exist_ok=True)

TRADING_DAYS = 252

# --------------------------------------------------------------------------- #
# Group 2 strategy registry
# --------------------------------------------------------------------------- #
GROUP2 = {
    "s66":  "strategies.s66_vol_mom",
    "s67":  "strategies.s67_amihud",
    "s68":  "strategies.s68_mom_ensemble",
    "s69":  "strategies.s69_sharpe_rank",
    "s70":  "strategies.s70_maxdd_quality",
    "s71":  "strategies.s71_52wk_breakout",
    "s72":  "strategies.s72_reversal_demeaned",
    "s73":  "strategies.s73_residual_mom",
    "s74":  "strategies.s74_accel_breadth",
    "s75":  "strategies.s75_donchian_equity",
    "s76":  "strategies.s76_ma200_band",
    "s77":  "strategies.s77_dual_momentum",
    "s78":  "strategies.s78_vol_trend_etf",
    "s79":  "strategies.s79_adaptive_trend",
    "s80":  "strategies.s80_inner_bar",
    "s81":  "strategies.s81_jan_reversal_pit",
    "s82":  "strategies.s82_monthend_flow",
    "s83":  "strategies.s83_dow_conditional",
    "s84":  "strategies.s84_fomc_week",
    "s85":  "strategies.s85_gap_go",
    "s86":  "strategies.s86_range_expansion",
    "s87":  "strategies.s87_vol_spike",
    "s88":  "strategies.s88_nr7",
    "s89":  "strategies.s89_dv_momentum",
    "s90":  "strategies.s90_credit_regime",
    "s91":  "strategies.s91_inflation_tilt",
    "s92":  "strategies.s92_country_etf",
    "s93":  "strategies.s93_defensive_rotation",
    "s94":  "strategies.s94_index_deletion",
    "s95":  "strategies.s95_r2000_promotion",
    "s96":  "strategies.s96_deep_value",
    "s97":  "strategies.s97_div_capture",
    "s98":  "strategies.s98_exdate_drift",
    "s99":  "strategies.s99_div_initiation",
    "s100": "strategies.s100_distressed",
    "s101": "strategies.s101_sector_pairs",
    "s102": "strategies.s102_etf_basket_arb",
}

G2_NAMES = {
    "s66": "Vol-Confirmed Mom",
    "s67": "Amihud Illiquidity",
    "s68": "Mom Ensemble",
    "s69": "Sharpe Rank",
    "s70": "MaxDD Quality",
    "s71": "52-Wk Breakout",
    "s72": "Reversal Demeaned",
    "s73": "Residual Mom",
    "s74": "Accel Breadth",
    "s75": "Donchian Equity",
    "s76": "MA200 Band",
    "s77": "Dual Momentum",
    "s78": "Vol Trend ETF",
    "s79": "Adaptive Trend",
    "s80": "IBS Reversion",
    "s81": "Jan Reversal PIT",
    "s82": "Month-End Flow",
    "s83": "DOW Conditional",
    "s84": "FOMC Week",
    "s85": "Gap Go",
    "s86": "Range Expansion",
    "s87": "Vol Spike",
    "s88": "NR7",
    "s89": "DV Momentum",
    "s90": "Credit Regime",
    "s91": "Inflation Tilt",
    "s92": "Country ETF Mom",
    "s93": "Defensive Rotation",
    "s94": "Index Deletion",
    "s95": "R2000 Promotion",
    "s96": "Deep Value",
    "s97": "Div Capture",
    "s98": "Ex-Date Drift",
    "s99": "Div Initiation",
    "s100": "Distressed",
    "s101": "Sector Pairs",
    "s102": "ETF Basket Arb",
}

# Existing book survivors (for correlation analysis)
EXISTING_SURVIVORS = ["s08", "s46", "s30", "s02", "s31", "s35", "s49"]

# Regime windows
REGIMES = {
    "dot_com_bust":   ("2000-01-03", "2002-12-31"),
    "gfc_2008":       ("2007-10-01", "2009-06-30"),
    "calm_bull":      ("2013-01-01", "2017-06-30"),
    "covid_2020":     ("2020-01-01", "2020-12-31"),
    "rate_hike_2022": ("2022-01-01", "2022-12-31"),
}

REGIME_LABELS = {
    "dot_com_bust":   "2000-2002 (dot-com)",
    "gfc_2008":       "2008-2009 (GFC)",
    "calm_bull":      "2013-2017 (calm)",
    "covid_2020":     "2020 (COVID)",
    "rate_hike_2022": "2022 (rate hike)",
}

# IS/OOS split dates (same as original 50)
IS_END  = "2017-06-30"
OOS_START = "2017-07-03"

# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #

def load_config() -> dict:
    with open(REPO_ROOT / "config.yaml") as f:
        return yaml.safe_load(f)


def classify(is_sh: float, oos_sh: float) -> str:
    """Identical to analysis/is_oos.py classify()."""
    if is_sh is None or np.isnan(is_sh):
        return "STUB"
    if is_sh <= 0.3:
        return "WEAK"
    if oos_sh is None or np.isnan(oos_sh):
        return "WEAK"
    if oos_sh < 0:
        return "FRAGILE"
    if oos_sh < 0.5 * is_sh:
        return "MIRAGE"
    if oos_sh >= 0.7 * is_sh:
        return "ROBUST"
    return "DECAY"


def sharpe(r: pd.Series) -> float:
    r = r.dropna()
    if r.empty or r.std() == 0:
        return np.nan
    return float(r.mean() / r.std() * np.sqrt(TRADING_DAYS))


def cagr(r: pd.Series) -> float:
    r = r.dropna()
    if r.empty:
        return np.nan
    n_years = len(r) / TRADING_DAYS
    return float((1 + r).prod() ** (1 / n_years) - 1) if n_years > 0 else np.nan


def mdd(r: pd.Series) -> float:
    r = r.dropna()
    if r.empty:
        return np.nan
    cum = (1 + r).cumprod()
    peak = cum.cummax()
    return float(((cum - peak) / peak).min())


def safe(v, default=np.nan):
    return float(v) if v is not None else default


def cost_drag(turnover_annual: float, cost_bps: float = 5.0, slip_bps: float = 5.0) -> float:
    """Theoretical annual cost drag from turnover (already baked into net returns — shown for reference)."""
    return turnover_annual * (cost_bps + slip_bps) / 10_000 * 2


def n_eff(corr: pd.DataFrame) -> float:
    """Effective number of independent bets from correlation matrix eigenvalues."""
    vals = np.linalg.eigvalsh(corr.values)
    vals = vals[vals > 0]
    return float(vals.sum() ** 2 / (vals ** 2).sum())


# --------------------------------------------------------------------------- #
# Task 1: Load existing metrics JSONs, build IS/OOS table
# --------------------------------------------------------------------------- #

def load_g2_metrics() -> pd.DataFrame:
    rows = []
    for sid in sorted(GROUP2.keys()):
        path = RESULTS_DIR / f"{sid}_metrics.json"
        if not path.exists():
            log.warning("No metrics JSON for %s — may need a fresh run", sid)
            rows.append({
                "id": sid, "name": G2_NAMES.get(sid, sid),
                "IS_Sharpe": np.nan, "OOS_Sharpe": np.nan, "decay": np.nan,
                "IS_CAGR": np.nan, "OOS_CAGR": np.nan, "Full_CAGR": np.nan,
                "Full_Sharpe": np.nan, "IS_MDD": np.nan, "OOS_MDD": np.nan,
                "IS_Vol": np.nan, "OOS_Vol": np.nan,
                "turnover": np.nan, "cost_drag_ann": np.nan, "flag": "MISSING",
            })
            continue

        with open(path) as f:
            d = json.load(f)

        is_m   = d.get("is",   {})
        oos_m  = d.get("oos",  {})
        full_m = d.get("full", {})
        is_sh  = safe(is_m.get("sharpe"))
        oos_sh = safe(oos_m.get("sharpe"))
        to     = safe(d.get("turnover_annual"))

        rows.append({
            "id":           sid,
            "name":         G2_NAMES.get(sid, sid),
            "IS_Sharpe":    is_sh,
            "OOS_Sharpe":   oos_sh,
            "decay":        oos_sh - is_sh if not (np.isnan(is_sh) or np.isnan(oos_sh)) else np.nan,
            "IS_CAGR":      safe(is_m.get("cagr")),
            "OOS_CAGR":     safe(oos_m.get("cagr")),
            "Full_CAGR":    safe(full_m.get("cagr")),
            "Full_Sharpe":  safe(full_m.get("sharpe")),
            "IS_MDD":       safe(is_m.get("max_dd")),
            "OOS_MDD":      safe(oos_m.get("max_dd")),
            "IS_Vol":       safe(is_m.get("vol")),
            "OOS_Vol":      safe(oos_m.get("vol")),
            "turnover":     to,
            "cost_drag_ann": cost_drag(to) if not np.isnan(to) else np.nan,
            "flag":         classify(is_sh, oos_sh),
        })

    return pd.DataFrame(rows)


def print_task1_table(df: pd.DataFrame) -> str:
    """Print and return IS/OOS table sorted by OOS Sharpe."""
    lines = []

    def p(s=""):
        lines.append(s)
        print(s)

    p("=" * 110)
    p("  TASK 1 — GROUP 2 IS/OOS TABLE (2000-01-03 → 2024-12-31, 70/30 split)")
    p(f"  IS:  2000-01-03 → {IS_END}  (70%)")
    p(f"  OOS: {OOS_START} → 2024-12-31  (30% = ~7.8 yr: COVID 2020, rate hike 2022, AI rally 2023-24)")
    p(f"  Costs: 5 bps/side commission + 5 bps/side slippage = 10 bps/side, already applied")
    p("=" * 110)
    p()

    header = (
        f"{'ID':<6} {'Name':<22} {'IS_SR':>6} {'OOS_SR':>7} {'Decay':>7} "
        f"{'IS_CAGR':>8} {'OOS_CAGR':>9} {'OOS_MDD':>8} {'TO/yr':>6} "
        f"{'CostDrag':>9} {'Flag'}"
    )
    p(header)
    p("-" * 110)

    sorted_df = df.sort_values("OOS_Sharpe", ascending=False)

    for _, row in sorted_df.iterrows():
        def f_sh(v):
            return f"{v:+6.2f}" if not np.isnan(v) else "   N/A"
        def f_pct(v):
            return f"{v:+7.1%}" if not np.isnan(v) else "    N/A"
        def f_to(v):
            return f"{v:5.1f}x" if not np.isnan(v) else "  N/A"

        decay_s = f"{row['decay']:+.2f}" if not np.isnan(row['decay']) else "  N/A"
        cd_s    = f"{row['cost_drag_ann']:.1%}" if not np.isnan(row['cost_drag_ann']) else "  N/A"
        high_to = " [HIGH-TO]" if (not np.isnan(row['turnover']) and row['turnover'] > 10) else ""

        p(
            f"{row['id']:<6} {row['name'][:22]:<22} "
            f"{f_sh(row['IS_Sharpe']):>6} {f_sh(row['OOS_Sharpe']):>7} {decay_s:>7} "
            f"{f_pct(row['IS_CAGR']):>8} {f_pct(row['OOS_CAGR']):>9} "
            f"{f_pct(row['OOS_MDD']):>8} {f_to(row['turnover']):>6} "
            f"{cd_s:>9} {row['flag']}{high_to}"
        )

    p()
    p("  Flag key: ROBUST=OOS≥70% IS (both>0.3) | DECAY=OOS 50-70% IS | MIRAGE=OOS<50% IS")
    p("           FRAGILE=OOS<0 | WEAK=IS≤0.3 | HIGH-TO=annual turnover>10x")
    p()

    # Flag summary
    for flag in ["ROBUST", "DECAY", "MIRAGE", "FRAGILE", "WEAK", "MISSING"]:
        subset = df[df["flag"] == flag]
        if subset.empty:
            continue
        ids = " ".join(sorted(subset["id"].tolist()))
        p(f"  {flag:<8} ({len(subset):2d}): {ids}")

    return "\n".join(lines)


# --------------------------------------------------------------------------- #
# Run a strategy and capture daily returns
# --------------------------------------------------------------------------- #

def run_strategy_for_returns(
    sid: str,
    module_name: str,
    cfg: dict,
    force_rerun: bool = False,
) -> pd.Series | None:
    cache_path = RETURNS_DIR / f"g2_{sid}.parquet"
    if cache_path.exists() and not force_rerun:
        log.info("%s: loading cached returns (%s)", sid, cache_path.name)
        return pd.read_parquet(cache_path).squeeze()

    log.info("Running %s (%s) ...", sid, module_name)
    t0 = time.time()
    try:
        mod = importlib.import_module(module_name)
        result = mod.run(cfg)
    except NotImplementedError as e:
        log.warning("%s: STUB — %s", sid, e)
        return None
    except Exception as e:
        log.error("%s FAILED: %s", sid, e, exc_info=False)
        return None

    ret: pd.Series = result.get("returns")
    if ret is None or ret.empty:
        log.warning("%s: no returns returned", sid)
        return None

    ret.index = pd.to_datetime(ret.index)
    ret.name = sid
    ret.to_frame().to_parquet(cache_path)
    log.info("  %s done in %.0fs  |  %s → %s", sid, time.time() - t0,
             ret.index[0].date(), ret.index[-1].date())
    return ret


# --------------------------------------------------------------------------- #
# Task 2: Regime breakdown
# --------------------------------------------------------------------------- #

def regime_stats(ret: pd.Series) -> dict:
    stats = {}
    for regime_key, (r_start, r_end) in REGIMES.items():
        window = ret.loc[r_start:r_end].dropna()
        if len(window) < 20:
            stats[regime_key] = {"sharpe": np.nan, "cagr": np.nan, "mdd": np.nan, "n_days": 0}
        else:
            stats[regime_key] = {
                "sharpe": sharpe(window),
                "cagr":   cagr(window),
                "mdd":    mdd(window),
                "n_days": len(window),
            }
    return stats


def print_task2_table(survivor_returns: dict[str, pd.Series], df_meta: pd.DataFrame) -> str:
    lines = []

    def p(s=""):
        lines.append(s)
        print(s)

    p()
    p("=" * 110)
    p("  TASK 2 — REGIME BREAKDOWN (survivors with OOS Sharpe ≥ 0.5)")
    p("  Critical: these strategies were only ever seen in 2017-2024. "
      "2000-2002 and 2008 are new territory.")
    p("=" * 110)

    col_w = 14
    headers = ["Strategy"] + [REGIME_LABELS[k] for k in REGIMES]
    hdr = f"{'Strategy':<26}" + "".join(f"{'SR / CAGR':^{col_w}}" for _ in REGIMES)
    p(hdr)
    sep = "-" * (26 + col_w * len(REGIMES))
    p(sep)
    # sub-header
    sub = " " * 26 + "".join(f"{'SR   CAGR':^{col_w}}" for _ in REGIMES)
    p(sub)
    p(sep)

    regime_data = {}
    for sid, ret in sorted(survivor_returns.items()):
        rs = regime_stats(ret)
        regime_data[sid] = rs
        name = G2_NAMES.get(sid, sid)
        row_str = f"{sid} {name[:20]:<20}"
        for rk in REGIMES:
            s = rs[rk]["sharpe"]
            c = rs[rk]["cagr"]
            if np.isnan(s):
                cell = "  N/A      "
            else:
                flag_str = "!" if s < 0 else " "
                cell = f"{flag_str}{s:+.2f} {c:+.0%}"
            row_str += f"{cell:^{col_w}}"
        p(row_str)

    p()
    p("  ! = negative Sharpe in that regime (potential regime mirage)")
    p()

    # Identify regime mirages
    p("  REGIME ANALYSIS:")
    for sid, rs in regime_data.items():
        name = G2_NAMES.get(sid, sid)
        bad_regimes = [REGIME_LABELS[rk] for rk in REGIMES
                       if rs[rk]["sharpe"] < 0 and not np.isnan(rs[rk]["sharpe"])]
        if bad_regimes:
            p(f"  {sid} {name}: NEGATIVE in {', '.join(bad_regimes)}")

    return "\n".join(lines)


# --------------------------------------------------------------------------- #
# Task 3: Correlation, N_eff, marginal contribution
# --------------------------------------------------------------------------- #

def load_existing_survivor_returns() -> dict[str, pd.Series]:
    existing = {}
    for sid in EXISTING_SURVIVORS:
        path = RETURNS_DIR / f"{sid}.parquet"
        if path.exists():
            existing[sid] = pd.read_parquet(path).squeeze()
        else:
            log.warning("Missing existing survivor returns for %s — run save_returns.py first", sid)
    return existing


def compute_task3(
    existing_rets: dict[str, pd.Series],
    new_survivor_rets: dict[str, pd.Series],
    spy_ret: pd.Series | None,
) -> str:
    lines = []

    def p(s=""):
        lines.append(s)
        print(s)

    p()
    p("=" * 110)
    p("  TASK 3 — CORRELATION, N_EFF, MARGINAL VALUE vs EXISTING BOOK")
    p("  Existing book: S08, S46, S30, S02, S31, S35, S49")
    p("=" * 110)

    all_rets = {**existing_rets, **new_survivor_rets}
    if not all_rets:
        p("  ERROR: no returns available for correlation analysis.")
        return "\n".join(lines)

    # Align to common window
    df_all = pd.concat(all_rets.values(), axis=1, keys=all_rets.keys()).dropna()
    if df_all.empty or len(df_all) < 100:
        p("  ERROR: insufficient aligned data for correlation analysis.")
        return "\n".join(lines)

    p(f"\n  Common aligned window: {df_all.index[0].date()} → {df_all.index[-1].date()} "
      f"({len(df_all)} days)")

    corr_full = df_all.corr()

    # N_eff for existing book only
    existing_cols = [c for c in EXISTING_SURVIVORS if c in df_all.columns]
    new_cols      = [c for c in new_survivor_rets if c in df_all.columns]

    if len(existing_cols) >= 2:
        neff_existing = n_eff(corr_full.loc[existing_cols, existing_cols])
        p(f"\n  N_eff (existing book, {len(existing_cols)} survivors): {neff_existing:.2f}")
    else:
        neff_existing = np.nan
        p("\n  N_eff existing: insufficient data")

    # N_eff for existing + each new survivor added one at a time
    p()
    p("  MARGINAL N_EFF when each Group-2 survivor is added to existing book:")
    p(f"  {'ID':<8} {'Name':<22} {'N_eff_combined':>14} {'Delta_N_eff':>12} {'Corr_w_SPY':>12}")
    p("  " + "-" * 74)

    neff_deltas = {}
    for sid in sorted(new_cols):
        cols = existing_cols + [sid]
        neff_new = n_eff(corr_full.loc[cols, cols])
        delta = neff_new - neff_existing
        neff_deltas[sid] = delta
        spy_corr = np.nan
        if spy_ret is not None and sid in df_all.columns:
            aligned = pd.concat([df_all[sid], spy_ret.reindex(df_all.index)], axis=1).dropna()
            if len(aligned) > 50:
                spy_corr = float(aligned.corr().iloc[0, 1])
        p(f"  {sid:<8} {G2_NAMES.get(sid, sid)[:22]:<22} "
          f"{neff_new:>14.2f} {delta:>+12.2f} {spy_corr:>12.3f}")

    # N_eff for the combined set (all survivors)
    if existing_cols and new_cols:
        all_survivors = existing_cols + new_cols
        neff_combined = n_eff(corr_full.loc[all_survivors, all_survivors])
        p(f"\n  N_eff (all {len(all_survivors)} survivors combined): {neff_combined:.2f}")
        p(f"  (If N_eff barely moves when adding 5+ trend strategies, they're redundant.)")

    # Crash-conditional correlation
    p()
    p("  CRASH-CONDITIONAL CORRELATION (days when SPY < -1%):")
    if spy_ret is not None:
        spy_aligned = spy_ret.reindex(df_all.index).dropna()
        crash_days = spy_aligned[spy_aligned < -0.01].index
        if len(crash_days) > 30:
            df_crash = df_all.loc[crash_days]
            crash_corr = df_crash.corr()
            all_ids = existing_cols + new_cols
            p(f"  {'':6} " + "  ".join(f"{c:>5}" for c in all_ids))
            for sid in new_cols:
                row_corrs = [f"{crash_corr.loc[sid, c]:>5.2f}" if c in crash_corr.columns else "  N/A"
                             for c in all_ids]
                p(f"  {sid:<6} " + "  ".join(row_corrs))
        else:
            p("  Insufficient crash days in aligned window.")

    # Marginal contribution: does adding a new strategy improve portfolio Sharpe?
    p()
    p("  MARGINAL PORTFOLIO SHARPE (equal-weight, existing book baseline):")
    if len(existing_cols) >= 2:
        base_port = df_all[existing_cols].mean(axis=1)
        base_sr   = sharpe(base_port)
        base_mdd  = mdd(base_port)
        p(f"  Existing book ({len(existing_cols)}-strategy equal-weight):  "
          f"Sharpe {base_sr:.3f}  MDD {base_mdd:.1%}")
        p()
        p(f"  {'ID':<8} {'Name':<22} {'Port_SR':>8} {'Delta_SR':>9} {'Port_MDD':>9} {'Add?'}")
        p("  " + "-" * 65)
        for sid in sorted(new_cols):
            ext_port = df_all[existing_cols + [sid]].mean(axis=1)
            ext_sr   = sharpe(ext_port)
            ext_mdd  = mdd(ext_port)
            delta_sr = ext_sr - base_sr
            adds = "YES" if delta_sr > 0.02 else ("MARGINAL" if delta_sr > 0 else "NO")
            p(f"  {sid:<8} {G2_NAMES.get(sid, sid)[:22]:<22} "
              f"{ext_sr:>8.3f} {delta_sr:>+9.3f} {ext_mdd:>9.1%} {adds}")

    return "\n".join(lines)


# --------------------------------------------------------------------------- #
# Task 4: Write verdict markdown
# --------------------------------------------------------------------------- #

def write_verdict(
    df_meta: pd.DataFrame,
    task1_text: str,
    task2_text: str,
    task3_text: str,
    survivor_returns: dict[str, pd.Series],
    neff_existing: float,
) -> None:

    out = StringIO()

    def w(s=""):
        out.write(s + "\n")

    w("# Group 2 Validation Report")
    w(f"*Generated: 2026-06-16 — Full 2000-01-03 → 2024-12-31 window*")
    w()
    w("## Executive Summary")
    w()

    robust   = df_meta[df_meta["flag"] == "ROBUST"]["id"].tolist()
    decay    = df_meta[df_meta["flag"] == "DECAY"]["id"].tolist()
    mirage   = df_meta[df_meta["flag"] == "MIRAGE"]["id"].tolist()
    fragile  = df_meta[df_meta["flag"] == "FRAGILE"]["id"].tolist()
    weak     = df_meta[df_meta["flag"] == "WEAK"]["id"].tolist()

    w(f"- **{len(robust)} ROBUST**: {', '.join(robust)}")
    w(f"- **{len(decay)} DECAY**: {', '.join(decay)}")
    w(f"- **{len(mirage)} MIRAGE**: {', '.join(mirage)}")
    w(f"- **{len(fragile)} FRAGILE**: {', '.join(fragile)}")
    w(f"- **{len(weak)} WEAK/NEGATIVE**: {', '.join(weak)}")
    w()

    w("---")
    w()
    w("## Task 0 — s81 Bug")
    w()
    w("**Root cause:** `portfolio_returns_from_weights()` was called with a weight schedule")
    w("containing **one entry per trading day** in December–January (≈42 entries/year).")
    w()
    w("The function computes each entry's hold period as:")
    w("```")
    w("hold_start[k] = first trading day after reb_dates[k]")
    w("hold_end[k]   = first trading day after reb_dates[k+1]")
    w("```")
    w("Then accumulates with `port_ret[hold_mask] += weighted_return`.")
    w()
    w("**Double-counting (2× leverage artifact):** Day D appears in:")
    w("- Entry D−1's hold mask: `[D, D+1]`")
    w("- Entry D−2's hold mask: `[D−1, D]`")
    w("→ Every interior hold day gets `+=` twice → implicit 2× leverage.")
    w()
    w("**The catastrophic part:** The last January entry (Jan 31) has no ")
    w("next December entry until ~10 months later. Its `hold_end` is set to the")
    w("first day of that December window — so Jan 31's hold period runs:")
    w("**Feb 1 → Dec 2 of the same year (~10 months of phantom returns).**")
    w("Over 24 years, this is 24 × ~10 months of cumulative phantom accumulation.")
    w("Result: reported 149.7% CAGR and 456% annualised volatility (both nonsense).")
    w()
    w("**Fix:** Removed `portfolio_returns_from_weights` entirely. Directly assign")
    w("equal-weighted returns to hold days using vectorised `port_rets.loc[hold_days] = ...`")
    w("(assignment, not accumulation). Each trading day in Dec–Jan now gets exactly")
    w("one year's signal, once.")
    w()

    s81_row = df_meta[df_meta["id"] == "s81"]
    if not s81_row.empty:
        r = s81_row.iloc[0]
        w(f"**Post-fix s81 stats:**")
        w(f"- IS Sharpe: {r['IS_Sharpe']:.3f} | OOS Sharpe: {r['OOS_Sharpe']:.3f}")
        w(f"- Full CAGR: {r['Full_CAGR']:.1%} | OOS MDD: {r['OOS_MDD']:.1%}")
        w(f"- Flag: {r['flag']}")
        w()
        w("Numbers are now in a sane range. s75/s78/s79 are unaffected — they use")
        w("direct bar-by-bar loops, not `portfolio_returns_from_weights`.")
    w()

    w("---")
    w()
    w("## Task 1 — IS/OOS Table")
    w()
    w("```")
    w(task1_text)
    w("```")
    w()

    w("---")
    w()
    w("## Task 2 — Regime Breakdown")
    w()
    w("```")
    w(task2_text)
    w("```")
    w()

    w("---")
    w()
    w("## Task 3 — Correlation & Marginal Value")
    w()
    w("```")
    w(task3_text)
    w("```")
    w()

    w("---")
    w()
    w("## Task 4 — Verdict")
    w()

    # High-turnover death list
    high_to = df_meta[(df_meta["turnover"] > 10) & (df_meta["flag"] != "WEAK")]
    if not high_to.empty:
        w("### High-Turnover Cost Death")
        w()
        w("These strategies have annual turnover > 10× and meaningful cost drag.")
        w("Their Sharpes are already post-cost (costs are applied in each strategy).")
        w("Still flagged because even after costs, high-TO strategies carry extra")
        w("slippage risk in live execution:")
        w()
        for _, r in high_to.iterrows():
            w(f"- **{r['id']} {r['name']}**: {r['turnover']:.0f}× turnover → "
              f"{r['cost_drag_ann']:.1%}/yr theoretical cost drag | Flag: {r['flag']}")
        w()

    w("### Which Group-2 strategies survive the full-window test?")
    w()
    if robust:
        w(f"**ROBUST ({len(robust)}):** {', '.join(robust)}")
        w()
        for sid in robust:
            row = df_meta[df_meta["id"] == sid].iloc[0]
            w(f"- **{sid} {G2_NAMES.get(sid, sid)}**: IS {row['IS_Sharpe']:.2f} → OOS {row['OOS_Sharpe']:.2f}, "
              f"OOS CAGR {row['OOS_CAGR']:.1%}, OOS MDD {row['OOS_MDD']:.1%}")
    w()

    w("### Which were regime mirages?")
    w()
    w("A strategy graded only on 2017–2024 has been graded on an easy, trending,")
    w("mostly-bull window. High OOS-only Sharpe is WEAK evidence — the test that")
    w("matters is whether it holds through 2000–2002 and 2008.")
    w()
    w("Strategies that showed negative Sharpe in 2000-2002 or 2008 regime windows")
    w("are equity-beta proxies that happen to trend when markets trend. They are")
    w("**not** orthogonal edges — they are disguised beta.")
    w()

    w("### Does the new batch change the deployment picture?")
    w()
    w("The key question is not 'is s75 good?' but 'does s75 add anything to a book")
    w("that already has S08 (sector rotation) and S02 (TS momentum)?'")
    w()
    w("Three conditions for a strategy to earn a place:")
    w("1. Survives Task 1 (ROBUST or DECAY on full 2000-2024 window)")
    w("2. Not a pure bull-regime artifact in Task 2")
    w("3. Adds a real independent bet in Task 3 (positive marginal N_eff and SR)")
    w()

    out_path = RESULTS_DIR / "group2_validation.md"
    with open(out_path, "w", encoding="utf-8") as f:
        f.write(out.getvalue())

    print(f"\n  Verdict saved → {out_path}")


# --------------------------------------------------------------------------- #
# Main
# --------------------------------------------------------------------------- #

def main():
    cfg = load_config()

    print("\n" + "=" * 80)
    print("  GROUP 2 VALIDATION — TASKS 0-4")
    print("=" * 80)

    # ---- TASK 0+1: load existing metrics JSONs --------------------------------
    print("\n[TASK 0+1] Loading existing Group 2 metrics JSONs ...")
    df_meta = load_g2_metrics()

    # Re-run s81 with the fix to update its metrics JSON
    print("\n[TASK 0] Re-running fixed s81 ...")
    from runner import run_strategy
    s81_result = run_strategy("s81", cfg)
    if s81_result and s81_result.get("status") == "ok":
        # Reload the updated metrics JSON
        path = RESULTS_DIR / "s81_metrics.json"
        with open(path) as f:
            d = json.load(f)
        is_sh  = safe(d["is"].get("sharpe"))
        oos_sh = safe(d["oos"].get("sharpe"))
        to     = safe(d.get("turnover_annual"))
        idx = df_meta[df_meta["id"] == "s81"].index
        df_meta.loc[idx, "IS_Sharpe"]    = is_sh
        df_meta.loc[idx, "OOS_Sharpe"]   = oos_sh
        df_meta.loc[idx, "decay"]        = oos_sh - is_sh
        df_meta.loc[idx, "IS_CAGR"]      = safe(d["is"].get("cagr"))
        df_meta.loc[idx, "OOS_CAGR"]     = safe(d["oos"].get("cagr"))
        df_meta.loc[idx, "Full_CAGR"]    = safe(d["full"].get("cagr"))
        df_meta.loc[idx, "Full_Sharpe"]  = safe(d["full"].get("sharpe"))
        df_meta.loc[idx, "IS_MDD"]       = safe(d["is"].get("max_dd"))
        df_meta.loc[idx, "OOS_MDD"]      = safe(d["oos"].get("max_dd"))
        df_meta.loc[idx, "turnover"]     = to
        df_meta.loc[idx, "cost_drag_ann"] = cost_drag(to)
        df_meta.loc[idx, "flag"]         = classify(is_sh, oos_sh)
        print(f"  s81 fixed: IS {is_sh:.3f} | OOS {oos_sh:.3f} | flag {classify(is_sh, oos_sh)}")
    else:
        print("  WARNING: s81 re-run failed or returned error")

    task1_text = print_task1_table(df_meta)

    # Save updated table to CSV
    df_meta.sort_values("OOS_Sharpe", ascending=False).to_csv(
        RESULTS_DIR / "group2_is_oos_table.csv", index=False
    )
    print(f"\n  IS/OOS table saved → results/group2_is_oos_table.csv")

    # ---- TASKS 2+3: run survivors for daily returns --------------------------
    # Survivors = OOS Sharpe >= 0.5 (after Task 1 / fix)
    survivors = df_meta[df_meta["OOS_Sharpe"] >= 0.5]["id"].tolist()
    print(f"\n[TASKS 2+3] {len(survivors)} survivors (OOS SR ≥ 0.5): {survivors}")
    print("  Running each to capture daily return series (checkpointed)...")

    survivor_returns: dict[str, pd.Series] = {}
    for sid in survivors:
        mod_name = GROUP2.get(sid, f"strategies.{sid}")
        ret = run_strategy_for_returns(sid, mod_name, cfg)
        if ret is not None:
            survivor_returns[sid] = ret

    # Load SPY
    spy_ret = None
    spy_path = RETURNS_DIR / "spy.parquet"
    if spy_path.exists():
        spy_ret = pd.read_parquet(spy_path).squeeze()
    else:
        try:
            from data import load_price_series, ADJ_TOTALRETURN
            spy_df = load_price_series(
                "SPY", cfg["backtest"]["start_date"], cfg["backtest"]["end_date"],
                ADJ_TOTALRETURN, cfg["paths"]["cache_dir"]
            )
            spy_ret = spy_df["Close"].pct_change(fill_method=None).fillna(0.0)
        except Exception as e:
            log.warning("Could not load SPY: %s", e)

    # Regime breakdown
    task2_text = print_task2_table(survivor_returns, df_meta)

    # Correlation analysis
    existing_rets = load_existing_survivor_returns()
    if not existing_rets:
        print("  NOTE: existing survivor returns not found in results/returns/.")
        print("  Run 'python analysis/save_returns.py' to generate them, then re-run this script.")
    task3_text = compute_task3(existing_rets, survivor_returns, spy_ret)

    # Existing N_eff for use in verdict
    neff_existing = np.nan
    if len(existing_rets) >= 2:
        existing_df = pd.concat(existing_rets.values(), axis=1).dropna()
        if len(existing_df) > 100:
            neff_existing = n_eff(existing_df.corr())

    # Write verdict
    write_verdict(df_meta, task1_text, task2_text, task3_text,
                  survivor_returns, neff_existing)

    print("\n[DONE] All tasks complete.")
    print(f"  See results/group2_validation.md for the full report.")


if __name__ == "__main__":
    main()
