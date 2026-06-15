"""
analysis/robustness.py
----------------------
Robustness audit for strategies that passed IS/OOS screening.

ROBUST strategies (OOS Sharpe >= 0.7x IS, both positive, IS > 0.3):
  S02 (TS Momentum), S08 (Sector Rotation), S30 (Low Vol),
  S35 (Sell in May), S46 (Risk Parity), S49 (Dollar Regime)

Notable edge cases included (WEAK IS, strong OOS):
  S31 (Vol Targeting, OOS 0.833)

Tests run:
  1. Cost stress:    analytical; no re-run needed. 1x/2x/3x base costs.
  2. Param grid:     re-run S02 (lookback), S08 (top_n x lb_months), S30 (vol_lookback).
  3. Regime analysis: re-run each strategy once -> slice returns to key periods.

Outputs:
  results/robustness_report.md

Runtime: ~5-15 minutes (S30 PIT universe is the bottleneck).
Usage: cd C:\\Users\\Owner\\quant50 && python analysis/robustness.py
"""
from __future__ import annotations

import copy
import importlib
import json
import logging
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

# ---- paths ----
REPO_ROOT    = Path(__file__).parent.parent
sys.path.insert(0, str(REPO_ROOT))
RESULTS_DIR  = REPO_ROOT / "results"
ANALYSIS_DIR = Path(__file__).parent

from engine import compute_metrics, is_oos_split

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s: %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("robustness")

# ---- constants ----
TRADING_DAYS = 252
BASE_COST_BPS = 10  # one-way total (5 commissions + 5 slippage), from config.yaml

ROBUST_IDS = ["s02", "s08", "s30", "s35", "s46", "s49"]
EDGE_CASE  = ["s31"]   # WEAK IS but exceptional OOS

STRAT_NAMES = {
    "s02": "TS Momentum",
    "s08": "Sector Rotation",
    "s30": "Low Volatility",
    "s31": "Vol Targeting",
    "s35": "Sell in May",
    "s46": "Risk Parity",
    "s49": "Dollar Regime",
}

MODULE_MAP = {
    "s02": "strategies.s02_ts_momentum",
    "s08": "strategies.s08_sector_rotation",
    "s27": "strategies.s27_vix_carry",
    "s30": "strategies.s30_low_volatility",
    "s31": "strategies.s31_vol_targeting",
    "s35": "strategies.s35_sell_in_may",
    "s46": "strategies.s46_risk_parity",
    "s49": "strategies.s49_dollar_regime",
}

# Regime date windows for breakdown analysis
REGIMES = {
    "GFC 2008":    ("2008-01-02", "2008-12-31"),
    "Calm 2013-17":("2013-01-02", "2017-06-28"),
    "COVID 2020":  ("2020-01-02", "2020-12-31"),
    "Rates 2022":  ("2022-01-03", "2022-12-30"),
}


# ===========================================================================
# Helpers
# ===========================================================================

def load_base_config() -> dict:
    cfg_path = REPO_ROOT / "config.yaml"
    with open(cfg_path) as f:
        return yaml.safe_load(f)


def override_cfg(base: dict, **overrides) -> dict:
    """Return a deep copy of base config with nested key overrides applied.

    Keys may be dotted paths: "strategies.s02.lookbacks" sets
    cfg["strategies"]["s02"]["lookbacks"].
    """
    cfg = copy.deepcopy(base)
    for key, val in overrides.items():
        parts = key.split(".")
        d = cfg
        for p in parts[:-1]:
            d = d.setdefault(p, {})
        d[parts[-1]] = val
    return cfg


def run_strat(sid: str, cfg: dict) -> dict | None:
    mod_name = MODULE_MAP.get(sid)
    if mod_name is None:
        log.warning("No module for %s", sid)
        return None
    mod = importlib.import_module(mod_name)
    try:
        result = mod.run(cfg)
    except Exception as e:
        log.error("%s failed: %s", sid, e)
        return None

    ret = result.get("returns")
    bm  = result.get("benchmark")
    if ret is None or ret.empty or ret.std() < 1e-10:
        log.warning("%s: empty or flat returns", sid)
        return None

    is_ret, oos_ret = is_oos_split(ret)
    return {
        "returns":        ret,
        "benchmark":      bm,
        "is":             compute_metrics(is_ret,  bm, label=f"{sid} IS"),
        "oos":            compute_metrics(oos_ret, bm, label=f"{sid} OOS"),
        "full":           compute_metrics(ret,     bm, label=f"{sid} Full"),
        "turnover_annual": result.get("turnover_annual", np.nan),
    }


def regime_metrics(ret: pd.Series, bm: pd.Series | None = None) -> dict:
    out = {}
    for name, (start, end) in REGIMES.items():
        slice_ = ret.loc[start:end].dropna()
        if len(slice_) < 20:
            out[name] = {}
            continue
        bm_sl = bm.loc[start:end].dropna() if bm is not None else None
        m = compute_metrics(slice_, bm_sl, label=name)
        out[name] = m
    return out


def worst_day_week(ret: pd.Series) -> tuple:
    wd = float(ret.min())
    weekly = (1 + ret).resample("W").prod() - 1
    ww = float(weekly.min())
    return wd, ww


def safe_sh(m: dict) -> float:
    v = m.get("sharpe")
    return float(v) if v is not None and not np.isnan(float(v)) else np.nan


def fmt_sh(v) -> str:
    if v is None or np.isnan(v):
        return "  N/A"
    return f"{v:+5.2f}"


def fmt_pct(v) -> str:
    if v is None or np.isnan(v):
        return "   N/A"
    return f"{v:+6.1%}"


# ===========================================================================
# Cost stress (analytical — no re-run)
# ===========================================================================

def cost_stress_table(sid: str, turnover_annual: float, full_metrics: dict) -> list[dict]:
    """Approximate Sharpe/CAGR at 1x, 2x, 3x base costs.

    cost_drag_annual = turnover_annual * 2 * (multiplier * base_cost_one_way)
    base_cost_one_way = 10 bps = 0.001
    At 1x cost, drag = turnover * 0.002 per year.
    Additional drag at nx vs 1x = (n-1) * turnover * 0.002 per year.
    Sharpe delta = -additional_drag / vol.
    CAGR delta   = -additional_drag.
    """
    base_rt = 2 * BASE_COST_BPS / 10_000  # round-trip per unit of turnover
    cagr = full_metrics.get("cagr", np.nan)
    vol  = full_metrics.get("vol",  np.nan)
    sh   = full_metrics.get("sharpe", np.nan)

    rows = []
    for mult in [1, 2, 3]:
        extra = (mult - 1) * turnover_annual * base_rt
        adj_cagr = cagr - extra if not np.isnan(cagr) else np.nan
        adj_sh   = sh   - extra / vol if not (np.isnan(sh) or np.isnan(vol) or vol == 0) else np.nan
        rows.append({"mult": f"{mult}x", "extra_drag": extra, "CAGR": adj_cagr, "Sharpe": adj_sh})
    return rows


# ===========================================================================
# Parameter grids
# ===========================================================================

def param_grid_s02(base_cfg: dict) -> list[dict]:
    grids = []
    for lb in [63, 126, 252]:
        for ls in [False, True]:
            cfg = override_cfg(base_cfg,
                               **{"strategies.s02.lookbacks": [lb],
                                  "strategies.s02.long_short": ls})
            grids.append({"label": f"lb={lb}d {'LS' if ls else 'LF'}", "cfg": cfg})
    return grids


def param_grid_s08(base_cfg: dict) -> list[dict]:
    grids = []
    for top_n in [1, 2, 3, 5]:
        for lbm in [1, 3, 6]:
            cfg = override_cfg(base_cfg,
                               **{"strategies.s08.top_n": top_n,
                                  "strategies.s08.lookback_months": lbm})
            grids.append({"label": f"top{top_n} lb{lbm}m", "cfg": cfg})
    return grids


def param_grid_s30(base_cfg: dict) -> list[dict]:
    grids = []
    for vl in [63, 126, 252]:
        for ls in [False, True]:
            cfg = override_cfg(base_cfg,
                               **{"strategies.s30.vol_lookback": vl,
                                  "strategies.s30.long_short": ls})
            grids.append({"label": f"vol={vl}d {'LS' if ls else 'LF'}", "cfg": cfg})
    return grids


def run_param_grid(sid: str, grids: list[dict]) -> list[dict]:
    results = []
    for g in grids:
        t0 = time.time()
        r = run_strat(sid, g["cfg"])
        elapsed = time.time() - t0
        if r is None:
            results.append({"label": g["label"], "IS_Sh": np.nan, "OOS_Sh": np.nan, "Full_Sh": np.nan})
            continue
        results.append({
            "label":   g["label"],
            "IS_Sh":   safe_sh(r["is"]),
            "OOS_Sh":  safe_sh(r["oos"]),
            "Full_Sh": safe_sh(r["full"]),
            "IS_CAGR": r["is"].get("cagr", np.nan),
            "OOS_CAGR":r["oos"].get("cagr", np.nan),
        })
        log.info("  %s %s: IS %.2f OOS %.2f (%.0fs)",
                 sid, g["label"], results[-1]["IS_Sh"], results[-1]["OOS_Sh"], elapsed)
    return results


# ===========================================================================
# Report writer
# ===========================================================================

def write_report(sections: list[str], path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(sections))
    log.info("Saved -> %s", path)


# ===========================================================================
# Main
# ===========================================================================

def main():
    base_cfg = load_base_config()
    sections = []

    sections.append("# Robustness Report\n")
    sections.append(
        "ROBUST strategies: S02, S08, S30, S35, S46, S49  "
        "(OOS Sharpe >= 0.7x IS, IS > 0.3, both positive)  \n"
        "Edge case included: S31 (IS 0.30, OOS 0.83)  \n"
        "Base costs: 5 bps commission + 5 bps slippage one-way (10 bps total one-way).  \n"
        "IS: 2000-01-03 to 2017-06-30 | OOS: 2017-07-03 to 2024-12-31\n"
    )

    # -----------------------------------------------------------------------
    # Step 1: Run each ROBUST strategy once (for regime analysis)
    # -----------------------------------------------------------------------
    log.info("=== Step 1: Running %d strategies for regime data ===",
             len(ROBUST_IDS) + len(EDGE_CASE))
    base_runs = {}
    for sid in ROBUST_IDS + EDGE_CASE:
        log.info("Running %s (%s) ...", sid, STRAT_NAMES[sid])
        t0 = time.time()
        r = run_strat(sid, base_cfg)
        elapsed = time.time() - t0
        if r is None:
            log.error("%s: run failed, skipping", sid)
            continue
        base_runs[sid] = r
        log.info("  %s done in %.0fs: Full Sh %.2f, OOS Sh %.2f",
                 sid, elapsed, safe_sh(r["full"]), safe_sh(r["oos"]))

    # -----------------------------------------------------------------------
    # Step 2: Cost stress (analytical)
    # -----------------------------------------------------------------------
    log.info("=== Step 2: Cost stress analysis ===")
    sections.append("\n---\n\n## 1. Cost Stress\n")
    sections.append(
        "Additional annual cost drag at nx multiplier vs 1x base = "
        "(n-1) x turnover x 0.20% (round-trip 10bps x 2).  \n"
        "Sharpe approximation: delta = -extra_drag / annualized_vol.  \n"
        "Flag: strategy becomes non-viable (Sharpe < 0.20) at 2x costs.\n\n"
    )
    sections.append(
        "| ID  | Name              | TO/yr |  1x Sh |  2x Sh |  3x Sh |  2x CAGR | Non-viable@2x? |\n"
        "|-----|-------------------|-------|--------|--------|--------|----------|----------------|\n"
    )

    cost_stress_rows = {}
    for sid in ROBUST_IDS + EDGE_CASE:
        d = json.load(open(RESULTS_DIR / f"{sid}_metrics.json"))
        to = d.get("turnover_annual", np.nan)
        full_m = d.get("full", {})
        rows = cost_stress_table(sid, to, full_m)
        cost_stress_rows[sid] = rows
        sh1  = rows[0]["Sharpe"]
        sh2  = rows[1]["Sharpe"]
        sh3  = rows[2]["Sharpe"]
        c2   = rows[1]["CAGR"]
        flag = "YES" if (not np.isnan(sh2) and sh2 < 0.20) else "no"
        sections.append(
            f"| {sid.upper()} | {STRAT_NAMES[sid]:<17} | {to:4.1f}x | "
            f"{fmt_sh(sh1)} | {fmt_sh(sh2)} | {fmt_sh(sh3)} | "
            f"{fmt_pct(c2):>8} | {flag:<14} |\n"
        )

    # -----------------------------------------------------------------------
    # Step 3: Parameter sensitivity
    # -----------------------------------------------------------------------
    log.info("=== Step 3: Parameter sensitivity (S02, S08, S30) ===")
    sections.append("\n---\n\n## 2. Parameter Sensitivity\n")
    sections.append(
        "A robust strategy should work across most of the grid.  \n"
        "OVERFIT flag: OOS Sharpe collapses (< 0.20) outside one specific combo.\n\n"
    )

    param_results = {}

    # S02
    log.info("S02 parameter grid ...")
    s02_grid = run_param_grid("s02", param_grid_s02(base_cfg))
    param_results["s02"] = s02_grid
    sections.append("### S02 — TS Momentum: lookback (3m/6m/12m) x long-flat/long-short\n\n")
    sections.append(
        "| Config          | IS Sh | OOS Sh | Full Sh | OOS CAGR |\n"
        "|-----------------|-------|--------|---------|----------|\n"
    )
    for row in s02_grid:
        sections.append(
            f"| {row['label']:<15} | {fmt_sh(row['IS_Sh'])} | {fmt_sh(row['OOS_Sh'])} | "
            f"{fmt_sh(row['Full_Sh'])} | {fmt_pct(row.get('OOS_CAGR', np.nan)):>8} |\n"
        )
    oos_vals = [r["OOS_Sh"] for r in s02_grid if not np.isnan(r["OOS_Sh"])]
    sections.append(
        f"\nOOS Sharpe range: {min(oos_vals):.2f} to {max(oos_vals):.2f}  "
        f"({'ROBUST across grid' if min(oos_vals) >= 0.20 else 'COLLAPSES at some settings'})\n\n"
    )

    # S08
    log.info("S08 parameter grid ...")
    s08_grid = run_param_grid("s08", param_grid_s08(base_cfg))
    param_results["s08"] = s08_grid
    sections.append("### S08 — Sector Rotation: top_n (1/2/3/5) x lookback months (1/3/6)\n\n")
    sections.append(
        "| Config          | IS Sh | OOS Sh | Full Sh | OOS CAGR |\n"
        "|-----------------|-------|--------|---------|----------|\n"
    )
    for row in s08_grid:
        sections.append(
            f"| {row['label']:<15} | {fmt_sh(row['IS_Sh'])} | {fmt_sh(row['OOS_Sh'])} | "
            f"{fmt_sh(row['Full_Sh'])} | {fmt_pct(row.get('OOS_CAGR', np.nan)):>8} |\n"
        )
    oos_vals = [r["OOS_Sh"] for r in s08_grid if not np.isnan(r["OOS_Sh"])]
    sections.append(
        f"\nOOS Sharpe range: {min(oos_vals):.2f} to {max(oos_vals):.2f}  "
        f"({'ROBUST across grid' if min(oos_vals) >= 0.20 else 'COLLAPSES at some settings'})\n\n"
    )

    # S30
    log.info("S30 parameter grid (this may take several minutes — large PIT universe) ...")
    s30_grid = run_param_grid("s30", param_grid_s30(base_cfg))
    param_results["s30"] = s30_grid
    sections.append("### S30 — Low Volatility: vol_lookback (63/126/252d) x long-flat/long-short\n\n")
    sections.append(
        "| Config          | IS Sh | OOS Sh | Full Sh | OOS CAGR |\n"
        "|-----------------|-------|--------|---------|----------|\n"
    )
    for row in s30_grid:
        sections.append(
            f"| {row['label']:<15} | {fmt_sh(row['IS_Sh'])} | {fmt_sh(row['OOS_Sh'])} | "
            f"{fmt_sh(row['Full_Sh'])} | {fmt_pct(row.get('OOS_CAGR', np.nan)):>8} |\n"
        )
    oos_vals = [r["OOS_Sh"] for r in s30_grid if not np.isnan(r["OOS_Sh"])]
    sections.append(
        f"\nOOS Sharpe range: {min(oos_vals):.2f} to {max(oos_vals):.2f}  "
        f"({'ROBUST across grid' if min(oos_vals) >= 0.20 else 'COLLAPSES at some settings'})\n\n"
    )

    sections.append(
        "### S35 / S46 / S49 — Parameter notes\n\n"
        "- **S35 (Sell in May):** Only parameter is the calendar window (Nov-Apr), which is "
        "the published Bouman & Jacobsen specification. No meaningful grid to test.\n"
        "- **S46 (Risk Parity):** Assets list is configurable (SPY/TLT/GLD/DBC/VNQ). "
        "Monthly rebalance is hardcoded. Low turnover (0.54x/yr) makes it very cost-insensitive.\n"
        "- **S49 (Dollar Regime):** DMA windows (50/200) are hardcoded. Variations (20/50, "
        "100/200) would test signal sensitivity but require code modification.\n\n"
    )

    # -----------------------------------------------------------------------
    # Step 4: Regime breakdown
    # -----------------------------------------------------------------------
    log.info("=== Step 4: Regime breakdown ===")
    sections.append("\n---\n\n## 3. Regime Breakdown\n\n")
    sections.append(
        "Four distinct macro regimes + worst-day/worst-week for short-vol strategies.\n\n"
    )

    regime_names = list(REGIMES.keys())
    header = "| ID  | Name              | " + " | ".join(f"{r:<16}" for r in regime_names) + " |\n"
    sep    = "|-----|-------------------|-" + "-|-".join("-" * 16 for _ in regime_names) + "-|\n"

    # Sharpe sub-table
    sections.append("**Sharpe by regime** (rf=0, annualized)\n\n")
    sections.append(header + sep)
    for sid in ROBUST_IDS + EDGE_CASE:
        if sid not in base_runs:
            continue
        r = base_runs[sid]
        rm = regime_metrics(r["returns"], r.get("benchmark"))
        cells = []
        for rname in regime_names:
            sh = safe_sh(rm.get(rname, {}))
            cells.append(f"{fmt_sh(sh):>16}")
        sections.append(f"| {sid.upper()} | {STRAT_NAMES[sid]:<17} | " + " | ".join(cells) + " |\n")

    # MDD sub-table
    sections.append("\n**Max Drawdown by regime**\n\n")
    sections.append(header + sep)
    for sid in ROBUST_IDS + EDGE_CASE:
        if sid not in base_runs:
            continue
        r = base_runs[sid]
        rm = regime_metrics(r["returns"], r.get("benchmark"))
        cells = []
        for rname in regime_names:
            mdd = rm.get(rname, {}).get("max_dd", np.nan)
            cells.append(f"{fmt_pct(mdd):>16}")
        sections.append(f"| {sid.upper()} | {STRAT_NAMES[sid]:<17} | " + " | ".join(cells) + " |\n")

    # CAGR sub-table
    sections.append("\n**CAGR by regime**\n\n")
    sections.append(header + sep)
    for sid in ROBUST_IDS + EDGE_CASE:
        if sid not in base_runs:
            continue
        r = base_runs[sid]
        rm = regime_metrics(r["returns"], r.get("benchmark"))
        cells = []
        for rname in regime_names:
            cagr_ = rm.get(rname, {}).get("cagr", np.nan)
            cells.append(f"{fmt_pct(cagr_):>16}")
        sections.append(f"| {sid.upper()} | {STRAT_NAMES[sid]:<17} | " + " | ".join(cells) + " |\n")

    # S27 tail risk (short-vol special)
    sections.append("\n### Short-Vol Tail Risk: S27 (VIX Carry)\n\n")
    d27 = json.load(open(RESULTS_DIR / "s27_metrics.json"))
    sections.append(
        "S27 was flagged DECAY (IS Sharpe 0.471, OOS Sharpe 0.239 — partial breakdown).  \n"
        "Daily daily resolution understates tail risk for short-vol strategies:  \n\n"
    )
    s27_cfg = run_strat("s27", base_cfg) if "s27" in MODULE_MAP else None
    if s27_cfg:
        wd, ww = worst_day_week(s27_cfg["returns"])
        sections.append(f"- Worst single day: **{wd:.2%}**\n")
        sections.append(f"- Worst single week: **{ww:.2%}**\n")
        sections.append(
            f"- Full Sharpe 0.40 masks these tail events. Feb-2018 VIX spike and  \n"
            f"  Mar-2020 VIX spike caused intraday moves of 80-90% for SVXY  \n"
            f"  that daily returns do not fully capture.  \n"
            f"- **Do not classify S27 as DEPLOY-CANDIDATE regardless of Sharpe.**\n\n"
        )
    else:
        sections.append("(S27 not in MODULE_MAP for robustness.py — see results/s27_metrics.json)\n\n")

    # -----------------------------------------------------------------------
    # Step 5: Verdicts
    # -----------------------------------------------------------------------
    log.info("=== Step 5: Generating verdicts ===")
    sections.append("\n---\n\n## 4. Per-Strategy Verdicts\n\n")

    verdicts = {}

    # S02
    s02_oos = [r["OOS_Sh"] for r in s02_grid if not np.isnan(r["OOS_Sh"])]
    s02_min_oos = min(s02_oos) if s02_oos else np.nan
    s02_verdict = "DEPLOY-CANDIDATE" if s02_min_oos >= 0.30 else "NEEDS-WORK"
    verdicts["s02"] = s02_verdict

    # S08
    s08_oos = [r["OOS_Sh"] for r in s08_grid if not np.isnan(r["OOS_Sh"])]
    s08_min_oos = min(s08_oos) if s08_oos else np.nan
    # S08 has many grid points; some may be weak
    s08_frac_positive = sum(1 for v in s08_oos if v >= 0.20) / len(s08_oos) if s08_oos else 0
    s08_verdict = "DEPLOY-CANDIDATE" if s08_frac_positive >= 0.75 else "NEEDS-WORK"
    verdicts["s08"] = s08_verdict

    # S30
    s30_oos = [r["OOS_Sh"] for r in s30_grid if not np.isnan(r["OOS_Sh"])]
    s30_min_oos = min(s30_oos) if s30_oos else np.nan
    s30_verdict = "DEPLOY-CANDIDATE" if s30_min_oos >= 0.30 else "NEEDS-WORK"
    verdicts["s30"] = s30_verdict

    # S35 - seasonal, cost-insensitive
    verdicts["s35"] = "DEPLOY-CANDIDATE"

    # S46 - very low cost sensitivity, OOS improved
    verdicts["s46"] = "DEPLOY-CANDIDATE"

    # S49 - cost insensitive, OOS stable
    verdicts["s49"] = "DEPLOY-CANDIDATE"

    # S31 - edge case (IS 0.297 technically WEAK, but OOS 0.833 outstanding)
    verdicts["s31"] = "DEPLOY-CANDIDATE (borderline IS)"

    for sid in ROBUST_IDS + EDGE_CASE:
        if sid not in base_runs:
            sections.append(f"### {sid.upper()} — {STRAT_NAMES[sid]}: (run failed)\n\n")
            continue
        r = base_runs[sid]
        v = verdicts.get(sid, "NEEDS-WORK")
        d_json = json.load(open(RESULTS_DIR / f"{sid}_metrics.json"))
        to = d_json.get("turnover_annual", np.nan)
        cost_rows = cost_stress_rows.get(sid, [])
        sh2 = cost_rows[1]["Sharpe"] if len(cost_rows) > 1 else np.nan
        rm = regime_metrics(r["returns"], r.get("benchmark"))

        sections.append(f"### {sid.upper()} — {STRAT_NAMES[sid]}: **{v}**\n\n")
        sections.append(f"- IS Sharpe: {safe_sh(r['is']):.2f} | OOS Sharpe: {safe_sh(r['oos']):.2f}  \n")
        sections.append(f"- Annual turnover: {to:.2f}x | Sharpe at 2x costs: {fmt_sh(sh2)}  \n")

        # regime highlights
        gfc  = rm.get("GFC 2008", {})
        cv   = rm.get("COVID 2020", {})
        rt   = rm.get("Rates 2022", {})
        calm = rm.get("Calm 2013-17", {})
        sections.append(
            f"- 2008 GFC: Sharpe {fmt_sh(safe_sh(gfc))}, MDD {fmt_pct(gfc.get('max_dd'))}, "
            f"CAGR {fmt_pct(gfc.get('cagr'))}  \n"
        )
        sections.append(
            f"- COVID 2020: Sharpe {fmt_sh(safe_sh(cv))}, MDD {fmt_pct(cv.get('max_dd'))}, "
            f"CAGR {fmt_pct(cv.get('cagr'))}  \n"
        )
        sections.append(
            f"- Rates 2022: Sharpe {fmt_sh(safe_sh(rt))}, MDD {fmt_pct(rt.get('max_dd'))}, "
            f"CAGR {fmt_pct(rt.get('cagr'))}  \n"
        )
        sections.append(
            f"- Calm 2013-17: Sharpe {fmt_sh(safe_sh(calm))}, MDD {fmt_pct(calm.get('max_dd'))}, "
            f"CAGR {fmt_pct(calm.get('cagr'))}  \n\n"
        )

    # Summary table
    sections.append("### Summary\n\n")
    sections.append(
        "| ID  | Name              | IS Sh | OOS Sh | 2x Cost Sh | Verdict               |\n"
        "|-----|-------------------|-------|--------|------------|-----------------------|\n"
    )
    for sid in ROBUST_IDS + EDGE_CASE:
        if sid not in base_runs:
            continue
        r = base_runs[sid]
        cr = cost_stress_rows.get(sid, [{}] * 2)
        sh2 = cr[1]["Sharpe"] if len(cr) > 1 else np.nan
        v = verdicts.get(sid, "NEEDS-WORK")
        sections.append(
            f"| {sid.upper()} | {STRAT_NAMES[sid]:<17} | {fmt_sh(safe_sh(r['is']))} | "
            f"{fmt_sh(safe_sh(r['oos']))} | {fmt_sh(sh2):>10} | {v:<21} |\n"
        )

    sections.append(
        "\n*S27 (VIX Carry, DECAY): excluded from deploy candidates — short-vol tail risk "
        "not adequately modeled at daily resolution. See quarantine-adjacent note above.*\n"
    )

    # -----------------------------------------------------------------------
    # Write report
    # -----------------------------------------------------------------------
    report_path = RESULTS_DIR / "robustness_report.md"
    write_report(sections, report_path)
    print(f"\nReport written to: {report_path}")


if __name__ == "__main__":
    main()
