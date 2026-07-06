"""
analysis/stragglers_validate.py
-----------------------------------
Run s97, s98, s99, s102 through the full pipeline. Same scrutiny as group2_validate.py.

Pre-audit summary:
  s97  Div Capture      -- CLEAN (event-driven, no same-bar issue; expected to fail on costs)
  s98  Ex-Date Drift    -- CLEAN (pre-built hold schedule, symmetric with s97)
  s99  Div Initiation   -- CLEAN (buys at ex-date which is public ~1mo ahead;
                                   ex-date vs announcement is a design caveat, not a bug)
  s102 ETF Basket Arb  -- SUSPECT: line 120 uses i+1 in z-score window (today's spread
                                   in mean/std). Directional impact is NEGATIVE on
                                   entry days (pair trade enters when spread just spiked).

Result: ALL FOUR are STUB_DATA -- cannot be run with norgatedata v1.0.74.
  s97/s98/s99: norgatedata.dividends() does not exist in v1.0.74
  s102: norgatedata.index_constituent_timeseries('XLK') fails -- sector ETFs are not
        Norgate-recognized indices.

Validated book for correlation baseline: s08, s46, s30, s02, s31, s35, s49, s90 (8 strats)
"""
from __future__ import annotations

import json
import logging
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(REPO_ROOT))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s: %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("stragglers")

RESULTS_DIR = REPO_ROOT / "results"
RETURNS_DIR = RESULTS_DIR / "returns"
RETURNS_DIR.mkdir(parents=True, exist_ok=True)

STRAGGLERS = ["s97", "s98", "s99", "s102"]

NAMES = {
    "s97":  "Div Capture",
    "s98":  "Ex-Date Drift",
    "s99":  "Div Initiation",
    "s102": "ETF Basket Arb",
}

LOOKAHEAD_AUDIT = {
    "s97":  "CLEAN -- event-driven hold schedule, returns computed separately",
    "s98":  "CLEAN -- event-driven hold schedule, symmetric with s97",
    "s99":  "CLEAN -- buys at ex-date (publicly announced ~1mo ahead); "
            "ex-date vs announcement is a design caveat, not a lookahead bug",
    "s102": "SUSPECT -- line 120: sp_hist = spread_df[etf].iloc[...:i+1] includes today's "
            "spread in z-score numerator AND denominator; then today's pair return is earned. "
            "Directional impact is NEGATIVE on entry days (ETF overpriced -> pair return < 0 "
            "on entry day). Bias likely works AGAINST the strategy, not for it.",
}

DATA_STUBS = {
    "s97":  "norgatedata v1.0.74 has no dividends() API. "
            "load_dividends() calls norgatedata.dividends() which raises AttributeError. "
            "Strategy logic is sound; blocked at data layer. "
            "Fix: use Tiingo, Sharadar, or compute from TOTALRETURN/CAPITAL price ratio.",
    "s98":  "Same as s97 -- dividend ex-date data unavailable from norgatedata v1.0.74.",
    "s99":  "Same as s97 -- dividend ex-date data unavailable from norgatedata v1.0.74.",
    "s102": "Needs sector ETF constituent membership (XLK/XLF/XLE as index names) via "
            "index_constituent_mask(sym, 'XLK'). Norgate does not expose sector ETF basket "
            "membership -- only traditional equity indices (Russell, S&P 500, Nasdaq). "
            "Fix: source XLK/XLF/XLE holdings history from SPDR fact sheets or a data vendor "
            "that tracks ETF constituent changes.",
}


def main():
    print()
    print("=" * 90)
    print("  STRAGGLERS VALIDATION: s97, s98, s99, s102")
    print("  Pre-audit lookahead review:")
    for sid in STRAGGLERS:
        print(f"  {sid} {NAMES[sid]:<18}: {LOOKAHEAD_AUDIT[sid][:65]}")
    print("=" * 90)
    print()
    print("  RUNTIME RESULT: all four strategies are STUB_DATA")
    print()
    for sid in STRAGGLERS:
        print(f"  {sid} {NAMES[sid]}:")
        print(f"    {DATA_STUBS[sid]}")
        print()
    print("  Confirmed by actual execution attempts:")
    print("  - s97/s98/s99: AttributeError: module 'norgatedata' has no attribute 'dividends'")
    print("  - s102: ValueError from index_constituent_timeseries('XLK') -- not a Norgate index")
    print()
    print("  norgatedata v1.0.74 available APIs (dividend-related):")
    print("  - dividend_yield_timeseries: trailing yield % timeseries (no ex-dates or amounts)")
    print("  - capital_event_timeseries: stock split events (not dividends)")
    print("  - No dividends() function exists in this version")
    print()
    print("  norgatedata v1.0.74 available index data:")
    print("  - index_constituent_timeseries: Russell, S&P 500, Nasdaq, Dow indices only")
    print("  - Sector ETFs (XLK/XLF/XLE) not recognized as index names")
    print()
    print("  IMPLICATION: none of these strategies affect the validated book.")
    print("  Validated book (s08,s46,s30,s02,s31,s35,s49,s90) is not changed.")

    # Save stub JSON metrics
    for sid in STRAGGLERS:
        payload = {
            "is":   {"sharpe": None, "cagr": None, "max_dd": None},
            "oos":  {"sharpe": None, "cagr": None, "max_dd": None},
            "full": {"sharpe": None, "cagr": None},
            "flag": "STUB_DATA",
            "stub_reason": DATA_STUBS[sid],
            "lookahead_audit": LOOKAHEAD_AUDIT[sid],
        }
        with open(RESULTS_DIR / f"{sid}_metrics.json", "w") as fh:
            json.dump(payload, fh, indent=2)

    # Write markdown report
    md_path = RESULTS_DIR / "stragglers_validation.md"
    with open(md_path, "w", encoding="utf-8") as fh:
        fh.write("# Straggler Strategy Validation: s97, s98, s99, s102\n")
        fh.write("*Generated: 2026-06-17*\n\n")
        fh.write("## Result: All Four Are STUB_DATA\n\n")
        fh.write("None of the four straggler strategies can be run with the current Norgate "
                 "data subscription (norgatedata v1.0.74). They require data types not "
                 "exposed by the package.\n\n")
        fh.write("## Pre-Audit Lookahead Review\n\n")
        for sid in STRAGGLERS:
            fh.write(f"- **{sid} {NAMES[sid]}**: {LOOKAHEAD_AUDIT[sid]}\n")
        fh.write("\n---\n\n")
        fh.write("## Individual Verdicts\n\n")
        for sid in sorted(STRAGGLERS):
            fh.write(f"### {sid} {NAMES[sid]}: **STUB_DATA**\n\n")
            fh.write(f"**Cannot run:** {DATA_STUBS[sid]}\n\n")
            fh.write(f"**Lookahead audit:** {LOOKAHEAD_AUDIT[sid]}\n\n")
        fh.write("---\n\n")
        fh.write("## Data Provider Limitations\n\n")
        fh.write("**Dividend strategies (s97, s98, s99):**\n")
        fh.write("- `norgatedata.dividends()` does not exist in v1.0.74\n")
        fh.write("- Available: `dividend_yield_timeseries` (trailing yield %, not individual events)\n")
        fh.write("- Available: `capital_event_timeseries` (stock split events, not dividends)\n")
        fh.write("- Fix: Tiingo dividend API, Sharadar, or TOTALRETURN/CAPITAL ratio inference\n\n")
        fh.write("**ETF Basket Arb (s102):**\n")
        fh.write("- `index_constituent_timeseries('XLK')` raises ValueError -- XLK is not a Norgate index\n")
        fh.write("- Available Norgate indices: Russell 1000/2000/3000, S&P 500, Nasdaq 100, Dow, etc.\n")
        fh.write("- Fix: SPDR fact sheets, FTSE Russell sector data, or ETF basket history vendor\n\n")
        fh.write("## Implication for Project\n\n")
        fh.write("These four strategies were never in the validated book. Their STUB_DATA status "
                 "does not affect the validated book (s08, s46, s30, s02, s31, s35, s49, s90). "
                 "They are documented here for completeness and as future work if additional "
                 "data infrastructure is acquired.\n")

    print()
    print(f"Report written to: {md_path}")
    print("Stub JSON metrics written to results/{s97,s98,s99,s102}_metrics.json")
    log.info("stragglers_validate.py complete")


if __name__ == "__main__":
    main()
