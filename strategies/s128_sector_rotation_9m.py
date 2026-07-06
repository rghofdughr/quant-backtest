"""
S128 -- Sector Rotation, improved params: 9m lookback, top-4, cash filter ON
Parameter upgrade from S08 (3m/top3). Identified as best OOS config in deep-dive grid.
"""
import sys, os, copy, importlib
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

DESCRIPTION = "Sector rotation, 9m lookback, top-4, cash filter ON (upgraded S08)"


def run(config: dict) -> dict:
    cfg = copy.deepcopy(config)
    cfg.setdefault("strategies", {}).setdefault("s08", {})
    cfg["strategies"]["s08"].update({
        "top_n": 4,
        "lookback_months": 9,
        "abs_momentum_cash_filter": True,
    })
    base = importlib.import_module("strategies.s08_sector_rotation")
    result = base.run(cfg)
    result["description"] = DESCRIPTION
    return result
