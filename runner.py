"""
runner.py — execute any subset of the 50 strategies.

Usage:
    python runner.py --strategies s01 s02 s30        # run specific strategies
    python runner.py --group A                        # run a whole group (A-H)
    python runner.py --all                            # run all implemented strategies
    python runner.py --smoke                          # quick smoke test (1yr, 50 symbols)
"""

from __future__ import annotations
import argparse
import importlib
import json
import logging
import os
import sys
import time
from pathlib import Path
from typing import List, Optional

import pandas as pd
import yaml

from engine import build_master_table, compute_metrics, is_oos_split, plot_tearsheet

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("runner")

# Strategy-to-group mapping
GROUPS = {
    "A": ["s01","s02","s03","s04","s05","s06","s07","s08"],
    "B": ["s09","s10","s11","s12","s13","s14","s15"],
    "C": ["s16","s17","s18","s19","s20","s21","s22"],
    "D": ["s23","s24","s25","s26","s27"],
    "E": ["s28","s29","s30","s31","s32","s33"],
    "F": ["s34","s35","s36","s37","s38","s39"],
    "G": ["s40","s41","s42","s43","s44","s45"],
    "H": ["s46","s47","s48","s49","s50"],
    # Batch 2 — Norgate US Equities novel strategies
    "I": ["s66","s67","s68","s69","s70","s71","s72","s73","s74","s75"],
    "J": ["s76","s77","s78","s79","s80","s81","s82","s83","s84"],
    "K": ["s85","s86","s87","s88","s89","s90","s91","s92","s93"],
    "L": ["s94","s95","s96","s97","s98","s99","s100","s101","s102"],
}
ALL_STRATEGIES = [s for grp in GROUPS.values() for s in grp]

# Map short strategy ID → full module name inside strategies/
MODULE_MAP = {
    "s01": "strategies.s01_cs_momentum",
    "s02": "strategies.s02_ts_momentum",
    "s03": "strategies.s03_dma_crossover",
    "s04": "strategies.s04_52wk_high",
    "s05": "strategies.s05_residual_momentum",
    "s06": "strategies.s06_intraday_momentum",
    "s07": "strategies.s07_donchian",
    "s08": "strategies.s08_sector_rotation",
    "s09": "strategies.s09_st_reversal",
    "s10": "strategies.s10_bollinger_reversion",
    "s11": "strategies.s11_pairs_cointegration",
    "s12": "strategies.s12_gap_fade",
    "s13": "strategies.s13_rsi2_bounce",
    "s14": "strategies.s14_vwap_reversion",
    "s15": "strategies.s15_ou_spread",
    "s16": "strategies.s16_book_to_market",
    "s17": "strategies.s17_earnings_yield",
    "s18": "strategies.s18_ev_ebitda",
    "s19": "strategies.s19_piotroski",
    "s20": "strategies.s20_gross_profitability",
    "s21": "strategies.s21_net_issuance",
    "s22": "strategies.s22_accruals",
    "s23": "strategies.s23_fx_carry",
    "s24": "strategies.s24_commodity_carry",
    "s25": "strategies.s25_bond_carry",
    "s26": "strategies.s26_dividend_yield",
    "s27": "strategies.s27_vix_carry",
    "s28": "strategies.s28_short_straddle",
    "s29": "strategies.s29_variance_risk_premium",
    "s30": "strategies.s30_low_volatility",
    "s31": "strategies.s31_vol_targeting",
    "s32": "strategies.s32_dispersion",
    "s33": "strategies.s33_earnings_iv_crush",
    "s34": "strategies.s34_turn_of_month",
    "s35": "strategies.s35_sell_in_may",
    "s36": "strategies.s36_day_of_week",
    "s37": "strategies.s37_fomc_drift",
    "s38": "strategies.s38_pre_holiday",
    "s39": "strategies.s39_jan_reversal",
    "s40": "strategies.s40_pead",
    "s41": "strategies.s41_index_addition",
    "s42": "strategies.s42_insider_buying",
    "s43": "strategies.s43_analyst_revisions",
    "s44": "strategies.s44_merger_arb",
    "s45": "strategies.s45_short_squeeze",
    "s46": "strategies.s46_risk_parity",
    "s47": "strategies.s47_yield_curve",
    "s48": "strategies.s48_gold_copper",
    "s49": "strategies.s49_dollar_regime",
    "s50": "strategies.s50_managed_futures",
    "s51": "strategies.s51_skewness",
    "s52": "strategies.s52_idiovol",
    "s53": "strategies.s53_bab",
    "s54": "strategies.s54_adx_momentum",
    "s55": "strategies.s55_lt_reversal",
    "s56": "strategies.s56_vol_momentum",
    "s57": "strategies.s57_52wk_low",
    "s58": "strategies.s58_cross_asset_mom",
    "s59": "strategies.s59_volvol",
    "s60": "strategies.s60_corr_regime",
    "s61": "strategies.s61_sortino_mom",
    "s62": "strategies.s62_accel_mom",
    "s63": "strategies.s63_etf_breakout",
    "s64": "strategies.s64_overnight",
    "s65": "strategies.s65_dispersion_s08",
    # Batch 2
    "s66": "strategies.s66_vol_mom",
    "s67": "strategies.s67_amihud",
    "s68": "strategies.s68_mom_ensemble",
    "s69": "strategies.s69_sharpe_rank",
    "s70": "strategies.s70_maxdd_quality",
    "s71": "strategies.s71_52wk_breakout",
    "s72": "strategies.s72_reversal_demeaned",
    "s73": "strategies.s73_residual_mom",
    "s74": "strategies.s74_accel_breadth",
    "s75": "strategies.s75_donchian_equity",
    "s76": "strategies.s76_ma200_band",
    "s77": "strategies.s77_dual_momentum",
    "s78": "strategies.s78_vol_trend_etf",
    "s79": "strategies.s79_adaptive_trend",
    "s80": "strategies.s80_inner_bar",
    "s81": "strategies.s81_jan_reversal_pit",
    "s82": "strategies.s82_monthend_flow",
    "s83": "strategies.s83_dow_conditional",
    "s84": "strategies.s84_fomc_week",
    "s85": "strategies.s85_gap_go",
    "s86": "strategies.s86_range_expansion",
    "s87": "strategies.s87_vol_spike",
    "s88": "strategies.s88_nr7",
    "s89": "strategies.s89_dv_momentum",
    "s90": "strategies.s90_credit_regime",
    "s91": "strategies.s91_inflation_tilt",
    "s92": "strategies.s92_country_etf",
    "s93": "strategies.s93_defensive_rotation",
    "s94": "strategies.s94_index_deletion",
    "s95": "strategies.s95_r2000_promotion",
    "s96": "strategies.s96_deep_value",
    "s97": "strategies.s97_div_capture",
    "s98": "strategies.s98_exdate_drift",
    "s99": "strategies.s99_div_initiation",
    "s100": "strategies.s100_distressed",
    "s101": "strategies.s101_sector_pairs",
    "s102": "strategies.s102_etf_basket_arb",
}


def load_config(path: str = "config.yaml") -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


def run_strategy(name: str, config: dict, smoke: bool = False) -> Optional[dict]:
    module_name = MODULE_MAP.get(name, f"strategies.{name}")
    try:
        mod = importlib.import_module(module_name)
    except ModuleNotFoundError:
        log.warning("Strategy %s not yet implemented — skipping.", name)
        return None

    if smoke:
        cfg = dict(config)
        cfg["backtest"] = dict(cfg.get("backtest", {}))
        cfg["backtest"]["start_date"] = "2018-01-01"
        cfg["backtest"]["end_date"]   = "2020-12-31"
        cfg["smoke"] = True
    else:
        cfg = config

    t0 = time.time()
    log.info("Running %s ...", name)
    try:
        result = mod.run(cfg)
    except NotImplementedError as e:
        log.warning("%s: requires external data — %s", name, e)
        return {"name": name, "status": "stub", "reason": str(e)}
    except Exception as e:
        log.error("%s FAILED: %s", name, e, exc_info=True)
        return {"name": name, "status": "error", "reason": str(e)}

    elapsed = time.time() - t0
    log.info("%s done in %.1fs", name, elapsed)

    returns: pd.Series = result.get("returns")
    if returns is None or returns.empty:
        log.warning("%s returned no data.", name)
        return {"name": name, "status": "no_data"}

    oos_frac = config.get("backtest", {}).get("oos_fraction", 0.30)
    spy_returns: pd.Series = result.get("benchmark")

    is_ret, oos_ret = is_oos_split(returns, oos_frac)

    metrics_is  = compute_metrics(is_ret,  spy_returns, label=f"{name} IS")
    metrics_oos = compute_metrics(oos_ret, spy_returns, label=f"{name} OOS")
    metrics_full = compute_metrics(returns, spy_returns, label=name)

    results_dir = Path(config.get("paths", {}).get("results_dir", "results"))
    results_dir.mkdir(parents=True, exist_ok=True)

    # Tearsheet — full period
    try:
        plot_tearsheet(
            returns, metrics_full, spy_returns,
            title=f"{name.upper()} — {result.get('description', '')}",
            save_path=str(results_dir / f"{name}_tearsheet.png"),
        )
    except Exception as e:
        log.warning("%s tearsheet failed: %s", name, e)

    # Save metrics JSON
    out = {
        "name": name,
        "status": "ok",
        "description": result.get("description", ""),
        "turnover_annual": result.get("turnover_annual"),
        "is":   metrics_is,
        "oos":  metrics_oos,
        "full": metrics_full,
    }
    with open(results_dir / f"{name}_metrics.json", "w") as f:
        json.dump(out, f, indent=2, default=str)

    return out


def main():
    parser = argparse.ArgumentParser(description="quant50 strategy runner")
    parser.add_argument("--strategies", nargs="+", help="e.g. s01 s02 s30")
    parser.add_argument("--group",      help="Group letter A-H")
    parser.add_argument("--all",        action="store_true")
    parser.add_argument("--smoke",      action="store_true", help="Quick smoke test (2yr window)")
    parser.add_argument("--config",     default="config.yaml")
    args = parser.parse_args()

    config = load_config(args.config)

    if args.all:
        targets = ALL_STRATEGIES
    elif args.group:
        targets = GROUPS.get(args.group.upper(), [])
    elif args.strategies:
        targets = [s.lower() for s in args.strategies]
    else:
        parser.print_help()
        sys.exit(1)

    log.info("Running %d strategies: %s", len(targets), targets)
    all_results = []
    for name in targets:
        r = run_strategy(name, config, smoke=args.smoke)
        if r:
            all_results.append(r)

    # Master comparison table
    ok = [r["full"] for r in all_results if r.get("status") == "ok" and "full" in r]
    if ok:
        master = build_master_table(ok)
        results_dir = Path(config.get("paths", {}).get("results_dir", "results"))
        master.to_csv(results_dir / "master_results.csv")
        log.info("Master table saved → results/master_results.csv")
        print("\n=== MASTER RESULTS (sorted by Sharpe) ===")
        print(master.sort_values("sharpe", ascending=False).to_string())

    # Summary of stubs / errors
    not_ok = [r for r in all_results if r.get("status") != "ok"]
    if not_ok:
        print("\n=== SKIPPED / STUBBED ===")
        for r in not_ok:
            print(f"  {r['name']:8s} [{r['status']}] {r.get('reason','')[:80]}")


if __name__ == "__main__":
    main()
