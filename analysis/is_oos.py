"""
analysis/is_oos.py
------------------
IS/OOS split analysis for all 50 strategies.
Reads from results/s??_metrics.json (pre-computed by runner.py).

IS:  2000-01-03 -> 2017-06-30  (70% of business days)
OOS: 2017-07-03 -> 2024-12-31  (30% - contains COVID, 2022 rate hike, 2023-2024 AI rally)

Classification rules (rf = 0):
  ROBUST  - OOS Sharpe ≥ 0.7x IS Sharpe, both positive, IS > 0.3
  DECAY   - OOS positive but 0.5x-0.7x IS Sharpe (partial breakdown)
  MIRAGE  - IS > 0.3 but OOS < 0.5x IS (worked then broke)
  FRAGILE - IS positive, OOS negative (sign flip)
  WEAK    - IS Sharpe <= 0.3 (never strong enough to judge)
  STUB    - All-zero returns; needs external data to activate
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

REPO_ROOT    = Path(__file__).parent.parent
RESULTS_DIR  = REPO_ROOT / "results"
ANALYSIS_DIR = Path(__file__).parent

STRAT_NAMES = {
    "s01": "CS Momentum",
    "s02": "TS Momentum",
    "s03": "DMA Crossover",
    "s04": "52-Week High",
    "s05": "Residual Momentum",
    "s06": "Intraday (proxy)",
    "s07": "Donchian",
    "s08": "Sector Rotation",
    "s09": "ST Reversal",
    "s10": "Bollinger Rev.",
    "s11": "Pairs Coint.",
    "s12": "Gap Fade",
    "s13": "RSI2 Bounce",
    "s14": "VWAP Rev. (proxy)",
    "s15": "OU Spread",
    "s16": "Book/Market",
    "s17": "Earnings Yield",
    "s18": "EV/EBITDA",
    "s19": "Piotroski",
    "s20": "Gross Profitability",
    "s21": "Net Issuance",
    "s22": "Accruals",
    "s23": "FX Carry",
    "s24": "Commodity Carry",
    "s25": "Bond Carry",
    "s26": "Dividend Yield",
    "s27": "VIX Carry",
    "s28": "Short Straddle",
    "s29": "Variance Risk Prem.",
    "s30": "Low Volatility",
    "s31": "Vol Targeting",
    "s32": "Dispersion",
    "s33": "Earnings IV Crush",
    "s34": "Turn of Month",
    "s35": "Sell in May",
    "s36": "Day of Week",
    "s37": "FOMC Drift",
    "s38": "Pre-Holiday",
    "s39": "Jan Reversal",
    "s40": "PEAD",
    "s41": "Index Addition",
    "s42": "Insider Buying",
    "s43": "Analyst Revisions",
    "s44": "Merger Arb",
    "s45": "Short Squeeze",
    "s46": "Risk Parity",
    "s47": "Yield Curve",
    "s48": "Gold/Copper Ratio",
    "s49": "Dollar Regime",
    "s50": "Managed Futures",
}

QUARANTINED = {"s07", "s26", "s45"}
STUBS       = {"s16", "s17", "s18", "s19", "s20", "s22",
               "s28", "s32", "s33", "s40", "s42", "s43", "s44"}
CONVICTION  = ["s02", "s30", "s46"]

# S27 short-vol caveat
SHORT_VOL = {"s27"}


def classify(is_sh: float, oos_sh: float) -> str:
    if is_sh is None or np.isnan(is_sh):
        return "STUB"
    if is_sh <= 0.3:
        return "WEAK"
    # IS > 0.3
    if oos_sh is None or np.isnan(oos_sh):
        return "WEAK"
    if oos_sh < 0:
        return "FRAGILE"
    if oos_sh < 0.5 * is_sh:
        return "MIRAGE"
    if oos_sh >= 0.7 * is_sh:
        return "ROBUST"
    return "DECAY"


def safe(v, default=np.nan):
    if v is None:
        return default
    return float(v)


def fmt_sh(v):
    if v is None or np.isnan(v):
        return "  N/A"
    return f"{v:+5.2f}"


def fmt_pct(v):
    if v is None or np.isnan(v):
        return "   N/A"
    return f"{v:+6.1%}"


def load_all() -> pd.DataFrame:
    rows = []
    for path in sorted(RESULTS_DIR.glob("s??_metrics.json")):
        sid = path.stem[:3]  # "s01"
        with open(path) as f:
            d = json.load(f)

        is_m   = d.get("is",   {})
        oos_m  = d.get("oos",  {})
        full_m = d.get("full", {})

        is_sh  = safe(is_m.get("sharpe"))
        oos_sh = safe(oos_m.get("sharpe"))

        flag = "STUB" if sid in STUBS else classify(is_sh, oos_sh)

        rows.append({
            "id":           sid,
            "name":         STRAT_NAMES.get(sid, sid),
            "IS_Sharpe":    is_sh,
            "OOS_Sharpe":   oos_sh,
            "decay":        oos_sh - is_sh if not (np.isnan(is_sh) or np.isnan(oos_sh)) else np.nan,
            "IS_CAGR":      safe(is_m.get("cagr")),
            "OOS_CAGR":     safe(oos_m.get("cagr")),
            "Full_CAGR":    safe(full_m.get("cagr")),
            "IS_MDD":       safe(is_m.get("max_dd")),
            "OOS_MDD":      safe(oos_m.get("max_dd")),
            "IS_Vol":       safe(is_m.get("vol")),
            "OOS_Vol":      safe(oos_m.get("vol")),
            "Full_Sharpe":  safe(full_m.get("sharpe")),
            "IS_start":     is_m.get("start", ""),
            "IS_end":       is_m.get("end", ""),
            "OOS_start":    oos_m.get("start", ""),
            "OOS_end":      oos_m.get("end", ""),
            "turnover":     safe(d.get("turnover_annual")),
            "flag":         flag,
            "quarantined":  sid in QUARANTINED,
            "stub":         sid in STUBS,
            "short_vol":    sid in SHORT_VOL,
        })

    return pd.DataFrame(rows).sort_values("id")


def print_header(title: str):
    bar = "=" * 82
    print(f"\n{bar}")
    print(f"  {title}")
    print(bar)


def print_table_header():
    print(f"{'ID':<6} {'Name':<22} {'IS_Sh':>6} {'OOS_Sh':>7} {'Decay':>6} "
          f"{'IS_MDD':>7} {'OOS_MDD':>8} {'IS_CAGR':>8} {'OOS_CAGR':>9} {'Flag'}")
    print("-" * 82)


def print_row(row: pd.Series, marker: str = ""):
    name = row["name"][:22]
    is_sh  = row["IS_Sharpe"]
    oos_sh = row["OOS_Sharpe"]
    decay  = row["decay"]
    print(
        f"{row['id']:<6} {name:<22} "
        f"{fmt_sh(is_sh):>6} {fmt_sh(oos_sh):>7} "
        f"{(f'{decay:+.2f}' if not np.isnan(decay) else '  N/A'):>6} "
        f"{fmt_pct(row['IS_MDD']):>7} "
        f"{fmt_pct(row['OOS_MDD']):>8} "
        f"{fmt_pct(row['IS_CAGR']):>8} "
        f"{fmt_pct(row['OOS_CAGR']):>9} "
        f"{row['flag']}{marker}"
    )


def main():
    df = load_all()
    if df.empty:
        print("ERROR: No s??_metrics.json found in results/")
        sys.exit(1)

    # --- Date windows ---
    sample = df[df["IS_start"] != ""].iloc[0]
    print(f"IS window:   {sample['IS_start']} to {sample['IS_end']}  (70%, 2000-2017)")
    print(f"OOS window:  {sample['OOS_start']} to {sample['OOS_end']}  (30%, 2017-2024)")
    print(f"OOS stress:  COVID crash (2020), 2022 rate hike, 2023-24 AI rally")
    print(f"rf = 0 throughout")

    # --- Conviction picks ---
    print_header("CONVICTION PICKS: S02 (TS Momentum) / S30 (Low Vol) / S46 (Risk Parity)")
    verdicts = {"ROBUST": "ROBUST [pass]", "WEAK": "WEAK [too weak IS]", "DECAY": "DECAY [partial]", "MIRAGE": "MIRAGE [fail]", "FRAGILE": "FRAGILE [fail]"}
    for sid in CONVICTION:
        row = df[df["id"] == sid].iloc[0]
        v = verdicts.get(row["flag"], row["flag"])
        print(f"\n  {sid.upper()} - {row['name']}")
        print(f"    IS  [{row['IS_start']} -> {row['IS_end']}]:  "
              f"Sharpe {row['IS_Sharpe']:.2f}, CAGR {row['IS_CAGR']:.1%}, MDD {row['IS_MDD']:.1%}")
        print(f"    OOS [{row['OOS_start']} -> {row['OOS_end']}]:  "
              f"Sharpe {row['OOS_Sharpe']:.2f}, CAGR {row['OOS_CAGR']:.1%}, MDD {row['OOS_MDD']:.1%}")
        decay_str = f"{row['decay']:+.2f}" if not np.isnan(row["decay"]) else "N/A"
        print(f"    Sharpe decay: {decay_str}  ->  {v}")

    # --- Main table: active strategies only (no quarantined, no stubs) ---
    print_header("IS/OOS TABLE - Active Strategies (sorted by OOS Sharpe desc)")
    print("  Quarantined [S07/S26/S45] and stubs [S16-S20,S22,S28,S32-S33,S40,S42-S44] excluded.\n")

    active = df[~df["quarantined"] & ~df["stub"]].copy()
    active_sorted = active.sort_values("OOS_Sharpe", ascending=False)

    print_table_header()
    for _, row in active_sorted.iterrows():
        marker = "  [!] short-vol" if row["short_vol"] else ""
        print_row(row, marker)

    # --- Edge-case callout ---
    print_header("NOTABLE EDGE CASES")
    edge = {
        "s31": ("WEAK by rule (IS 0.297 < 0.30 threshold), but OOS 0.833 is the best of any "
                "active strategy. Not overfit - vol-targeting mechanically improves when OOS "
                "has clearer vol regimes (COVID/2022). Worth including in robustness audit."),
        "s48": ("WEAK by rule (IS 0.273), but OOS 0.515 improved. Gold/copper ratio is a real "
                "macro signal; the IS underperformed because 2000-2017 had mixed commodity cycles."),
        "s29": ("WEAK by rule (IS 0.127), but OOS 0.527 improved strongly. VRP is well-documented; "
                "IS was hurt by post-2008 option vol collapse. OOS signal is meaningful."),
        "s03": ("WEAK by rule (IS 0.064), OOS 0.385. DMA crossover barely registered IS; OOS "
                "benefited from trending 2017-2024 environment. Possible regime dependence."),
        "s10": ("IS -0.002 (essentially zero), OOS 0.475. Bollinger reversion - inactive in "
                "low-vol 2000-2017 bull; activated by regime volatility. Do not promote yet."),
    }
    for sid, note in edge.items():
        row = df[df["id"] == sid].iloc[0]
        print(f"\n  {sid.upper()} - {row['name']}: IS {row['IS_Sharpe']:.2f} -> OOS {row['OOS_Sharpe']:.2f}")
        # wrap note
        words = note.split()
        line = "    "
        for w in words:
            if len(line) + len(w) > 80:
                print(line)
                line = "    " + w + " "
            else:
                line += w + " "
        if line.strip():
            print(line)

    # --- Flag summary ---
    print_header("FLAG SUMMARY")
    for flag in ["ROBUST", "DECAY", "MIRAGE", "FRAGILE", "WEAK", "STUB"]:
        subset = active[active["flag"] == flag] if flag != "STUB" else df[df["stub"]]
        if subset.empty:
            continue
        ids = " ".join(subset["id"].str.upper().tolist())
        print(f"  {flag:<8} ({len(subset):2d})  {ids}")

    print("\n  Note: WEAK includes many strategies with IS <= 0.3 regardless of OOS direction.")
    print("        S31/S48/S29 improved OOS despite WEAK IS; include in robustness audit.")

    # --- Quarantined section ---
    print_header("QUARANTINED (see results/quarantine.md)")
    q = df[df["quarantined"]].sort_values("id")
    print_table_header()
    for _, row in q.iterrows():
        print_row(row)

    # --- Stub section ---
    print_header("STUBS - Awaiting External Data (all return 0%)")
    stub_df = df[df["stub"]].sort_values("id")
    print(f"  {'ID':<6} {'Name':<25} {'Data source needed'}")
    print("  " + "-" * 65)
    stub_sources = {
        "s16": "Sharadar Core US Fundamentals (~$40/mo)",
        "s17": "Sharadar Core US Fundamentals (~$40/mo)",
        "s18": "Sharadar Core US Fundamentals (~$40/mo)",
        "s19": "Sharadar Core US Fundamentals (~$40/mo)",
        "s20": "Sharadar Core US Fundamentals (~$40/mo)",
        "s22": "Sharadar Core US Fundamentals (~$40/mo)",
        "s28": "ORATS / CBOE options data (~$100/mo)",
        "s32": "ORATS / CBOE options data (~$100/mo)",
        "s33": "ORATS / CBOE options data (~$100/mo)",
        "s40": "Zacks / Sharadar Earnings Surprises (~$50/mo)",
        "s42": "SEC EDGAR Form 4 (free API)",
        "s43": "Zacks / Sharadar Earnings (~$50/mo)",
        "s44": "Refinitiv / Bloomberg M&A ($$$)",
    }
    for _, row in stub_df.iterrows():
        src = stub_sources.get(row["id"], "unknown")
        print(f"  {row['id']:<6} {row['name']:<25} {src}")

    # --- Save CSV ---
    out_path = RESULTS_DIR / "is_oos_table.csv"
    save_cols = ["id", "name", "IS_Sharpe", "OOS_Sharpe", "decay",
                 "IS_CAGR", "OOS_CAGR", "IS_MDD", "OOS_MDD",
                 "Full_Sharpe", "Full_CAGR", "turnover", "flag",
                 "quarantined", "stub"]
    df[save_cols].sort_values("OOS_Sharpe", ascending=False).to_csv(out_path, index=False)
    print(f"\n  Saved -> {out_path}")


if __name__ == "__main__":
    main()
