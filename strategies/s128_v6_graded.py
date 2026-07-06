"""
S128-V6 -- Sector Rotation, 9m/top-4, graded per-holding cash filter.
Instead of all-or-nothing cash (if top sector <= 0, go 100% cash),
drops only individual holdings with negative 9m momentum and equal-weights
the survivors. Goes to cash only if no top-4 sector has positive momentum.
Parameters identical to s128 except graded_cash_filter=True.
"""
import sys, os, copy, importlib
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

DESCRIPTION = "Sector rotation, 9m/top-4, graded per-holding cash filter (V6)"


def run(config: dict) -> dict:
    cfg = copy.deepcopy(config)
    cfg.setdefault("strategies", {}).setdefault("s08", {})
    cfg["strategies"]["s08"].update({
        "top_n": 4,
        "lookback_months": 9,
        "abs_momentum_cash_filter": True,
        "graded_cash_filter": True,   # V6 modification
    })
    base = importlib.import_module("strategies.s08_sector_rotation")
    result = base.run(cfg)
    result["description"] = DESCRIPTION
    return result
