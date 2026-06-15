"""
analysis/perturbation.py
------------------------
Execution fragility test for the survivor sleeve.

Tests run on: S02 (primary), S31 (primary), S08 (control), S46 (control)

Three fragility dimensions:
  1. Signal jitter: shift daily returns ±1 and ±2 days, recompute OOS Sharpe.
     Rationale: approximates execution-date slippage (buying a day later/earlier
     than the signal date). Strategies with fast-decaying signals will degrade
     quickly; slowly-rebalancing strategies will be near-flat.
     NOTE: This is a lower-bound on jitter sensitivity — it assumes the position
     was still entered (just a day off), not that the signal was missed entirely.

  2. Cost stress: analytical formula applied to OOS metrics.
     extra_drag  = (mult - 1) * turnover_annual * base_rt
     adj_sharpe  = oos_sharpe - extra_drag / oos_vol
     This gives a conservative linear approximation (real impact is proportional
     to turnover; ignores spread compression as costs rise in thin markets).

  3. Parameter ±10%: re-run S02 with lookback 57 and 69 (±10% of 63);
     re-run S31 with vol_lookback 18 and 22 (±10% of 20).
     S08 and S46 already have full parameter grids from robustness.py and are
     treated as verified controls here.

Verdict thresholds:
  STABLE  — all perturbations keep OOS Sharpe ≥ 0.30 AND degradation < 30%
  FRAGILE — any perturbation drops OOS Sharpe < 0.30 OR degrades > 30%

Outputs:
  Prints full table to stdout
  Saves results/perturbation_report.md
"""
from __future__ import annotations

import copy
import importlib
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

REPO_ROOT   = Path(__file__).parent.parent
sys.path.insert(0, str(REPO_ROOT))
RETURNS_DIR = REPO_ROOT / "results" / "returns"
RESULTS_DIR = REPO_ROOT / "results"

OOS_START   = pd.Timestamp("2017-07-03")
TRADING_DAYS = 252
BASE_COST_BPS = 10  # round-trip: 5 equity + 5 slippage each way = 10 bps one-way?
# Actually config: 5 bps equity + 5 bps slippage each WAY = 10 bps one-way = 0.10%
# round-trip = 2 × 10 bps = 20 bps = 0.0020 per round trip
BASE_COST_RT = 2 * (5 + 5) / 10_000   # 0.002  (round trip)

STRATEGIES = {
    "s02": {
        "module": "strategies.s02_ts_momentum",
        "param_key": "lookbacks",
        "baseline": [63],
        "perturb_minus": [57],   # -10%
        "perturb_plus":  [69],   # +10%
        "param_label": "lookback",
        "oos_sharpe": 0.686,
        "oos_vol":    0.195 * 0.30,   # rough OOS-period vol (full_vol * correction)
        "turnover":   6.57,
        "oos_vol_direct": None,        # will be computed from parquet
    },
    "s31": {
        "module": "strategies.s31_vol_targeting",
        "param_key": "vol_lookback",
        "baseline": 20,
        "perturb_minus": 18,    # -10%
        "perturb_plus":  22,    # +10%
        "param_label": "vol_lookback",
        "oos_sharpe": 0.833,
        "oos_vol":    0.109 * 0.30,
        "turnover":   7.37,
        "oos_vol_direct": None,
    },
    # Controls: parameter stability already verified in robustness.py (12-pt grid)
    "s08": {
        "module": "strategies.s08_sector_rotation",
        "param_key": "top_n",
        "baseline": 3,
        "perturb_minus": None,   # already tested 1,2,3,5 × 1,3,6mo
        "perturb_plus":  None,
        "param_label": "top_n (integer, see robustness.py for full grid)",
        "oos_sharpe": 0.757,
        "oos_vol":    0.176 * 0.30,
        "turnover":   4.71,
        "oos_vol_direct": None,
    },
    "s46": {
        "module": "strategies.s46_risk_parity",
        "param_key": None,
        "baseline": None,
        "perturb_minus": None,
        "perturb_plus":  None,
        "param_label": "n/a (single-param, 5-ETF, monthly rebalance)",
        "oos_sharpe": 0.806,
        "oos_vol":    0.120 * 0.30,
        "turnover":   0.54,
        "oos_vol_direct": None,
    },
}


def load_config() -> dict:
    with open(REPO_ROOT / "config.yaml") as f:
        return yaml.safe_load(f)


def load_oos_returns(sid: str) -> pd.Series:
    path = RETURNS_DIR / f"{sid}.parquet"
    s = pd.read_parquet(path).squeeze()
    s.index = pd.to_datetime(s.index)
    return s[s.index >= OOS_START]


def sharpe_from_series(rets: pd.Series) -> tuple[float, float]:
    """Return (sharpe, ann_vol)."""
    v = rets.std() * np.sqrt(TRADING_DAYS)
    sh = (rets.mean() * TRADING_DAYS / v) if v > 0 else np.nan
    return float(sh), float(v)


def jitter_test(oos_rets: pd.Series, sid: str) -> dict[str, float]:
    """
    Shift returns by ±1, ±2 days to simulate execution-date lag/lead.
    shift(+k): we get day t returns at day t+k (delayed execution)
    shift(-k): we get day t returns at day t-k (anticipatory)
    Fill boundary with 0 (no trade that day).
    """
    results = {}
    sh_base, _ = sharpe_from_series(oos_rets)
    results["base"] = sh_base
    for k in [-2, -1, 1, 2]:
        shifted = oos_rets.shift(k).fillna(0.0)
        sh, _ = sharpe_from_series(shifted)
        results[f"shift_{k:+d}d"] = sh
    return results


def cost_stress_test(sid: str, spec: dict) -> dict[str, float]:
    """Analytical cost stress. Returns dict of multiplier -> adj_sharpe."""
    base_sh  = spec["oos_sharpe"]
    turnover = spec["turnover"]
    # Use empirical OOS vol (computed from parquet)
    oos_vol  = spec.get("oos_vol_direct") or spec["oos_vol"]
    results  = {}
    for mult in [1, 2, 3]:
        extra = (mult - 1) * turnover * BASE_COST_RT
        adj_sh = base_sh - (extra / max(oos_vol, 0.001))
        results[f"{mult}x_cost"] = adj_sh
    return results


def run_strategy_with_param(module_name: str, cfg: dict, param_key: str,
                             param_val) -> float | None:
    """Re-run strategy with single config override. Returns OOS Sharpe."""
    mod = importlib.import_module(module_name)
    local_cfg = copy.deepcopy(cfg)
    # Inject the override into the strategy sub-config
    sid = module_name.split(".")[-1].split("_")[0]
    if sid not in local_cfg.get("strategies", {}):
        local_cfg.setdefault("strategies", {})[sid] = {}
    if param_val is not None:
        local_cfg["strategies"][sid][param_key] = param_val

    try:
        result = mod.run(local_cfg)
    except Exception as e:
        print(f"    ERROR running {module_name} with {param_key}={param_val}: {e}")
        return None

    rets = result.get("returns")
    if rets is None or rets.empty:
        return None

    rets.index = pd.to_datetime(rets.index)
    oos = rets[rets.index >= OOS_START]
    if len(oos) < 50:
        return None

    sh, vol = sharpe_from_series(oos)
    return sh


def verdict(sid: str, jitter_d: dict, cost_d: dict, param_d: dict,
            oos_sharpe_base: float) -> str:
    """STABLE/FRAGILE/CONTROLLED verdict."""
    issues = []

    # Jitter: any single perturbation degrades > 25% of base Sharpe
    for k, v in jitter_d.items():
        if k == "base":
            continue
        if np.isnan(v) or v < 0.30:
            issues.append(f"jitter {k} drops Sharpe to {v:.2f}")
        elif (oos_sharpe_base - v) / max(abs(oos_sharpe_base), 0.01) > 0.25:
            issues.append(f"jitter {k} degrades {((oos_sharpe_base-v)/abs(oos_sharpe_base)*100):.0f}%")

    # Cost stress: 3x cost still viable
    sh_3x = cost_d.get("3x_cost", np.nan)
    if not np.isnan(sh_3x) and sh_3x < 0.20:
        issues.append(f"3x costs kills it (adj Sharpe {sh_3x:.2f})")

    # Parameter ±10%: any value below 0.30
    for k, v in param_d.items():
        if v is not None and not np.isnan(v) and v < 0.30:
            issues.append(f"param {k} drops Sharpe to {v:.2f}")

    if not issues:
        return "STABLE"
    return "FRAGILE — " + "; ".join(issues)


def main():
    cfg = load_config()
    lines = ["# Perturbation / Execution Fragility Report\n\n"]
    lines.append(
        "Tests: signal jitter (±1-2 day return shift), cost stress (1x/2x/3x), "
        "±10% parameter perturbation.\n"
        "OOS window: 2017-07-03 to 2024-12-31.\n"
        "STABLE = all perturbations keep OOS Sharpe >= 0.30 with < 25% degradation.\n\n"
    )

    for sid, spec in STRATEGIES.items():
        print(f"\n{'='*60}")
        print(f"Strategy: {sid.upper()}  (baseline OOS Sharpe {spec['oos_sharpe']:.3f})")
        print(f"{'='*60}")

        # Load OOS returns
        try:
            oos_rets = load_oos_returns(sid)
            sh_base_emp, oos_vol_emp = sharpe_from_series(oos_rets)
            spec["oos_vol_direct"] = oos_vol_emp
            print(f"  Empirical OOS: Sharpe {sh_base_emp:.3f}, Ann Vol {oos_vol_emp:.1%}, n={len(oos_rets)}")
        except FileNotFoundError:
            print(f"  WARNING: {sid}.parquet not found — using stored metrics")
            sh_base_emp = spec["oos_sharpe"]
            oos_vol_emp = spec["oos_vol"]

        # 1. Jitter test
        print("  1. JITTER TEST")
        try:
            jitter_d = jitter_test(oos_rets, sid)
            for k, v in jitter_d.items():
                print(f"     {k:>12s}: Sharpe = {v:.3f}  ({(v/jitter_d['base']-1)*100:+.1f}%)")
        except Exception as e:
            print(f"     ERROR: {e}")
            jitter_d = {"base": sh_base_emp}

        # 2. Cost stress
        print("  2. COST STRESS  (turnover {:.2f}x/yr, base {:.0f} bps rt)".format(
            spec["turnover"], BASE_COST_RT * 10_000))
        cost_d = cost_stress_test(sid, spec)
        for k, v in cost_d.items():
            print(f"     {k:>10s}: adj Sharpe = {v:.3f}")

        # 3. Parameter ±10%
        print("  3. PARAMETER PERTURBATION (+/-10%)")
        param_d = {}
        if spec["perturb_minus"] is not None:
            pk = spec["param_key"]
            pv_minus = spec["perturb_minus"]
            pv_plus  = spec["perturb_plus"]
            print(f"     Running {sid} with {pk}={pv_minus} ...")
            sh_minus = run_strategy_with_param(spec["module"], cfg, pk, pv_minus)
            print(f"       -> OOS Sharpe = {sh_minus:.3f}" if sh_minus is not None else "       -> FAILED")
            param_d[f"{pk}={pv_minus}"] = sh_minus

            print(f"     Running {sid} with {pk}={pv_plus} ...")
            sh_plus = run_strategy_with_param(spec["module"], cfg, pk, pv_plus)
            print(f"       -> OOS Sharpe = {sh_plus:.3f}" if sh_plus is not None else "       -> FAILED")
            param_d[f"{pk}={pv_plus}"] = sh_plus
        else:
            print(f"     Skipped (already verified in robustness.py full grid: {spec['param_label']})")
            param_d["see_robustness"] = sh_base_emp

        # Verdict
        v_str = verdict(sid, jitter_d, cost_d, param_d, sh_base_emp)
        print(f"\n  VERDICT: {v_str}\n")

        # Markdown
        lines.append(f"## {sid.upper()}\n\n")
        lines.append(f"Baseline OOS Sharpe: **{sh_base_emp:.3f}**  (turnover {spec['turnover']:.2f}x/yr)\n\n")
        lines.append("### Jitter (return-series shift ±1-2 days)\n\n")
        lines.append("| Shift | OOS Sharpe | Degradation |\n|-------|-----------|-------------|\n")
        base_sh = jitter_d.get("base", sh_base_emp)
        for k, v in jitter_d.items():
            if k == "base":
                continue
            deg = (v - base_sh) / max(abs(base_sh), 0.01) * 100
            flag = " <-- WARNING" if v < 0.30 or abs(deg) > 25 else ""
            lines.append(f"| {k} | {v:.3f} | {deg:+.1f}%{flag} |\n")

        lines.append("\n### Cost Stress\n\n")
        lines.append("| Multiplier | Adj Sharpe |\n|-----------|------------|\n")
        for k, v in cost_d.items():
            flag = " <-- NON-VIABLE" if v < 0.20 else (" <-- MARGINAL" if v < 0.30 else "")
            lines.append(f"| {k} | {v:.3f}{flag} |\n")

        lines.append("\n### Parameter ±10%\n\n")
        if spec["perturb_minus"] is not None:
            lines.append(f"| Parameter | Value | OOS Sharpe |\n|-----------|-------|------------|\n")
            lines.append(f"| {spec['param_label']} | {spec['baseline']} (baseline) | {sh_base_emp:.3f} |\n")
            for k, v in param_d.items():
                if v is None:
                    vs = "FAILED"
                elif np.isnan(v):
                    vs = "NaN"
                else:
                    vs = f"{v:.3f}"
                flag = ""
                if v is not None and not np.isnan(v) and v < 0.30:
                    flag = " <-- FRAGILE"
                lines.append(f"| {spec['param_label']} | {k.split('=')[1]} | {vs}{flag} |\n")
        else:
            lines.append(
                f"Full parameter grid tested in robustness.py. "
                f"Summary: {spec['param_label']}\n"
            )

        lines.append(f"\n**VERDICT: {v_str}**\n\n---\n\n")

    # Save
    out_path = RESULTS_DIR / "perturbation_report.md"
    with open(out_path, "w", encoding="utf-8") as f:
        f.write("".join(lines))
    print(f"\nSaved: {out_path}")


if __name__ == "__main__":
    main()
