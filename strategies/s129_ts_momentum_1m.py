"""
S129 -- TS Momentum, 1m lookback, long-flat, vol-targeted (upgraded S02)
1m lookback identified as best OOS config in deep-dive sweep (OOS SR +0.887 vs +0.373 at 12m).
"""
import sys, os, copy, importlib
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

DESCRIPTION = "TS momentum, 1m lookback, long-flat, vol-targeted (upgraded S02)"


def run(config: dict) -> dict:
    cfg = copy.deepcopy(config)
    cfg.setdefault("strategies", {}).setdefault("s02", {})
    cfg["strategies"]["s02"].update({
        "lookbacks": [21],   # ~1 trading month
        "long_short": False,
    })
    base = importlib.import_module("strategies.s02_ts_momentum")
    result = base.run(cfg)
    result["description"] = DESCRIPTION
    return result
