"""
analysis/save_returns.py
------------------------
Run the 7 survivor strategies (+ SPY benchmark) and save daily net-return
series to results/returns/<id>.parquet.

Survivors:
  Core    : s08 (sector rotation), s46 (risk parity), s30 (low-vol LF)
  Maybe   : s31 (vol targeting),   s35 (sell-in-may)
  Narrow  : s02 (TS momentum 63d), s49 (dollar regime)

S30 is deliberately run with long_short=false (already set in config.yaml).
S02 uses lookbacks[0]=63 (already correct in config.yaml).

S30 (S&P 500 PIT universe ~1100 names) is the slow one — expect ~3-5 minutes.

Usage:
  cd C:\\Users\\Owner\\quant50
  python analysis/save_returns.py
"""
from __future__ import annotations

import importlib
import logging
import sys
import time
from pathlib import Path

import pandas as pd
import yaml

REPO_ROOT   = Path(__file__).parent.parent
sys.path.insert(0, str(REPO_ROOT))

from data import load_price_series, ADJ_TOTALRETURN  # noqa: E402

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s: %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("save_returns")

SURVIVORS = {
    "s08": "strategies.s08_sector_rotation",
    "s46": "strategies.s46_risk_parity",
    "s30": "strategies.s30_low_volatility",
    "s31": "strategies.s31_vol_targeting",
    "s35": "strategies.s35_sell_in_may",
    "s02": "strategies.s02_ts_momentum",
    "s49": "strategies.s49_dollar_regime",
}


def load_config() -> dict:
    with open(REPO_ROOT / "config.yaml") as f:
        return yaml.safe_load(f)


def run_and_save(sid: str, module_name: str, cfg: dict, out_dir: Path) -> pd.Series | None:
    out_path = out_dir / f"{sid}.parquet"
    if out_path.exists():
        log.info("%s: already saved (%s), skipping", sid, out_path.name)
        ser = pd.read_parquet(out_path).squeeze()
        log.info("  loaded %d rows, %s to %s", len(ser), ser.index[0].date(), ser.index[-1].date())
        return ser

    log.info("Running %s (%s) ...", sid, module_name)
    t0 = time.time()
    mod = importlib.import_module(module_name)
    try:
        result = mod.run(cfg)
    except Exception as e:
        log.error("%s failed: %s", sid, e)
        return None

    ret: pd.Series = result.get("returns")
    if ret is None or ret.empty:
        log.warning("%s: returned no data", sid)
        return None

    # ensure DatetimeIndex
    ret.index = pd.to_datetime(ret.index)
    ret.name = sid

    ret.to_frame().to_parquet(out_path)
    elapsed = time.time() - t0
    log.info("  saved %d rows, %s to %s  (%.0fs)",
             len(ret), ret.index[0].date(), ret.index[-1].date(), elapsed)
    return ret


def save_spy(cfg: dict, out_dir: Path) -> pd.Series | None:
    out_path = out_dir / "spy.parquet"
    if out_path.exists():
        log.info("spy: already saved, skipping")
        return pd.read_parquet(out_path).squeeze()

    start = cfg["backtest"]["start_date"]
    end   = cfg["backtest"]["end_date"]
    cache = cfg["paths"]["cache_dir"]
    log.info("Loading SPY total return ...")
    df = load_price_series("SPY", start, end, ADJ_TOTALRETURN, cache)
    if df.empty:
        log.error("SPY load failed")
        return None

    trading_idx = pd.bdate_range(start, end)
    spy_ret = (df["Close"]
               .reindex(trading_idx, method="ffill")
               .pct_change(fill_method=None)
               .fillna(0.0))
    spy_ret.name = "spy"
    spy_ret.index = pd.to_datetime(spy_ret.index)
    spy_ret.to_frame().to_parquet(out_path)
    log.info("  SPY saved: %d rows", len(spy_ret))
    return spy_ret


def main():
    cfg     = load_config()
    out_dir = REPO_ROOT / "results" / "returns"
    out_dir.mkdir(parents=True, exist_ok=True)

    log.info("Saving return series for %d survivors + SPY ...", len(SURVIVORS))
    log.info("Output: %s", out_dir)

    total_t0 = time.time()
    saved = {}
    for sid, mod_name in SURVIVORS.items():
        ret = run_and_save(sid, mod_name, cfg, out_dir)
        if ret is not None:
            saved[sid] = ret

    spy = save_spy(cfg, out_dir)

    log.info("")
    log.info("=== Done in %.0fs ===", time.time() - total_t0)
    log.info("Saved %d/%d strategies + SPY", len(saved), len(SURVIVORS))

    # Quick alignment check
    if len(saved) >= 2:
        common = pd.concat(saved.values(), axis=1).dropna()
        log.info("Common aligned window: %s to %s  (%d trading days)",
                 common.index[0].date(), common.index[-1].date(), len(common))

    missing = [sid for sid in SURVIVORS if sid not in saved]
    if missing:
        log.warning("MISSING: %s — check strategy errors above", missing)


if __name__ == "__main__":
    main()
