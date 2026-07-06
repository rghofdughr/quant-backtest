"""
Two analyses following the research critique:

PART 1: Gross-vs-net breakdown per strategy.
  Confirms every reported Sharpe is net of 5+5bp one-way costs (20bp round-trip),
  shows actual drag vs expected from turnover, so high-TO strategies are transparent.

PART 2: Vol-target sweep across [8, 10, 12, 15, 20]% targets.
  Makes the risk/leverage choice explicit. Shows CAGR, Sharpe, MDD, avg leverage,
  max leverage, and financing-adjusted CAGR (SOFR cost on borrowed fraction).

PART 3: Per-year table at each vol target vs SPY.

Runs full-date strategies (2000-2024), extracts OOS window for all metrics.
"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import copy, importlib, yaml
import numpy as np
import pandas as pd

CFG_PATH    = os.path.join(os.path.dirname(__file__), "..", "config.yaml")
RESULTS_DIR = os.path.join(os.path.dirname(__file__), "..", "results")
os.makedirs(RESULTS_DIR, exist_ok=True)

with open(CFG_PATH) as f:
    BASE_CONFIG = yaml.safe_load(f)

IS_START     = "2000-01-03"
FULL_END     = "2024-12-31"
OOS_START    = "2017-07-03"
OOS_END      = "2024-12-31"
TRADING_DAYS = 252
VOL_LOOK     = 63
AVG_SOFR     = 0.025   # rough avg SOFR over 2017-2024 OOS window

BOOK = [
    ("s128", "strategies.s128_sector_rotation_9m",  0.0667),
    ("s30",  "strategies.s30_low_volatility",        0.0667),
    ("s31",  "strategies.s31_vol_targeting",         0.0667),
    ("s35",  "strategies.s35_sell_in_may",           0.0667),
    ("s49",  "strategies.s49_dollar_regime",         0.0667),
    ("s115", "strategies.s115_year_end_reversal",    0.0500),
    ("s02",  "strategies.s02_ts_momentum",           0.1483),
    ("s46",  "strategies.s46_risk_parity",           0.1483),
    ("s90",  "strategies.s90_credit_regime",         0.2967),
]

VOL_TARGETS = [0.08, 0.10, 0.12, 0.15, 0.20]


def make_config(zero_cost=False):
    cfg = copy.deepcopy(BASE_CONFIG)
    cfg["backtest"]["start_date"] = IS_START
    cfg["backtest"]["end_date"]   = FULL_END
    if zero_cost:
        cfg["costs"]["equity_cost_bps"]      = 0
        cfg["costs"]["equity_slippage_bps"]  = 0
        cfg["costs"]["futures_cost_ticks"]   = 0
        cfg["costs"]["futures_slippage_bps"] = 0
    return cfg


def oos(r):
    """Trim a full-history return series to the OOS window."""
    return r.loc[OOS_START:OOS_END].fillna(0.0)


def sr(r):
    r = r.dropna()
    if len(r) < 20 or r.std() == 0:
        return 0.0
    return float(r.mean() / r.std() * TRADING_DAYS ** 0.5)


def cagr(r):
    r = r.dropna().fillna(0)
    if len(r) < 2:
        return 0.0
    return float((1 + r).prod() ** (TRADING_DAYS / len(r)) - 1)


def mdd(r):
    w = (1 + r.fillna(0)).cumprod()
    d = (w - w.cummax()) / w.cummax()
    return float(d.min())


def vol_target_overlay(ret, target, look=VOL_LOOK, lo=0.25, hi=4.0):
    rolling_vol = ret.rolling(look).std() * TRADING_DAYS ** 0.5
    scale = (target / rolling_vol).shift(1).fillna(1.0).clip(lo, hi)
    return ret * scale, scale


def beat_spy_pct(strat_ret, spy_ret):
    """Fraction of rolling 1-year windows where strat CAGR > SPY CAGR."""
    n = len(strat_ret)
    wins = sum(
        1 for i in range(n - TRADING_DAYS)
        if cagr(strat_ret.iloc[i:i + TRADING_DAYS]) > cagr(spy_ret.iloc[i:i + TRADING_DAYS])
    )
    total = n - TRADING_DAYS
    return wins / total if total > 0 else 0.0


# ─────────────────────────────────────────────────────────────────────────────
def main():
    lines = []
    def pr(s=""):
        lines.append(str(s))
        print(str(s))

    cfg_gross  = make_config(zero_cost=True)
    cfg_net    = make_config(zero_cost=False)

    # ─── PART 1: gross vs net ────────────────────────────────────────────────
    pr("=" * 76)
    pr("PART 1 -- GROSS vs NET PER STRATEGY  (OOS: 2017-07-03 to 2024-12-31)")
    pr("=" * 76)
    pr("  Note: apply_costs uses 20bp round-trip per unit turnover (5+5 one-way x2).")
    pr("")
    pr(f"{'ID':<7} {'Gross SR':>9} {'Net SR':>9}  {'dSR':>6}  {'TO(x)':>6}  "
       f"{'Actual bp':>10} {'Expected bp':>12}  Match?")
    pr("-" * 76)

    oos_net   = {}
    oos_gross = {}
    all_to    = {}

    for sid, mname, wt in BOOK:
        try:
            mod = importlib.import_module(mname)
            res_g = mod.run(cfg_gross)
            res_n = mod.run(cfg_net)
        except Exception as e:
            pr(f"{sid:<7}  ERROR: {e}")
            continue

        g_full = res_g["returns"].fillna(0)
        n_full = res_n["returns"].fillna(0)
        to_val = res_n.get("turnover_annual", 0.0)

        # Align indices
        idx = g_full.index.union(n_full.index)
        g_full = g_full.reindex(idx, fill_value=0)
        n_full = n_full.reindex(idx, fill_value=0)

        g = oos(g_full)
        n = oos(n_full)

        sr_g = sr(g)
        sr_n = sr(n)
        dsr  = sr_g - sr_n

        # Actual drag in bps/year from daily difference
        actual_bp = float((g - n).mean() * TRADING_DAYS * 10_000)
        # Expected from TO * round-trip (20bp)
        exp_bp    = to_val * 20.0
        match     = "OK" if abs(actual_bp - exp_bp) < 10 else "CHECK"

        pr(f"{sid:<7} {sr_g:>+9.3f} {sr_n:>+9.3f}  {dsr:>+6.3f}  {to_val:>6.1f}x  "
           f"{actual_bp:>9.1f}bp {exp_bp:>11.1f}bp  {match}")

        oos_net[sid]   = n
        oos_gross[sid] = g
        all_to[sid]    = to_val

    # Book-level cost summary
    total_wt_drag = sum(
        all_to.get(sid, 0) * 20.0 * wt for sid, _, wt in BOOK if sid in all_to
    )
    pr("")
    pr(f"Weighted book drag (20bp * TO * weight, annualized): {total_wt_drag:.1f} bp/year")
    pr("A strategy IS correctly costed if Actual bp ~ Expected bp (+/- 5bp).")
    pr("Large 'CHECK' flags indicate TO accounting difference (e.g. one-way vs round-trip).")

    # ─── Build weighted book (net returns, OOS) ──────────────────────────────
    book_net = sum(
        oos_net[sid] * wt for sid, _, wt in BOOK if sid in oos_net
    )
    book_net = book_net.fillna(0)

    book_gross = sum(
        oos_gross[sid] * wt for sid, _, wt in BOOK if sid in oos_gross
    )
    book_gross = book_gross.fillna(0)

    # SPY benchmark
    from data import load_price_series, ADJ_TOTALRETURN
    spy_df  = load_price_series("SPY", IS_START, FULL_END,
                                ADJ_TOTALRETURN, BASE_CONFIG["paths"]["cache_dir"])
    spy_ret = oos(spy_df["Close"].pct_change(fill_method=None))

    # ─── PART 2: vol-target sweep ────────────────────────────────────────────
    pr("")
    pr("=" * 76)
    pr("PART 2 -- VOL-TARGET SWEEP  (new 9-strategy book, OOS net returns)")
    pr("=" * 76)
    pr("  Overlay: scale = (target / 63d-rolling-vol), lag 1d, clipped [0.25, 4.0x].")
    pr("  Clip means max 4x gross leverage; floor means never less than 25% of target.")
    pr("")
    pr(f"{'Target':>8}  {'CAGR':>7}  {'Sharpe':>7}  {'MDD':>8}  "
       f"{'AvgLev':>7}  {'MaxLev':>7}  {'Beat SPY':>9}")
    pr("-" * 76)
    pr(f"{'SPY':>8}  {cagr(spy_ret):>+6.1%}  {'--':>7}  {mdd(spy_ret):>+8.1%}  "
       f"{'--':>7}  {'--':>7}  {'--':>9}")
    pr(f"{'raw book':>8}  {cagr(book_net):>+6.1%}  {sr(book_net):>+7.3f}  "
       f"{mdd(book_net):>+8.1%}  {'1.00x':>7}  {'1.00x':>7}  {'--':>9}")
    pr(f"{'raw gross':>8}  {cagr(book_gross):>+6.1%}  {sr(book_gross):>+7.3f}  "
       f"{mdd(book_gross):>+8.1%}  {'1.00x':>7}  {'1.00x':>7}  {'--':>9}")
    pr("")

    vt_rets  = {}
    vt_scale = {}
    for vt in VOL_TARGETS:
        vt_r, sc = vol_target_overlay(book_net, vt)
        vt_rets[vt]  = vt_r
        vt_scale[vt] = sc

        beat = beat_spy_pct(vt_r, spy_ret)
        pr(f"{vt:>7.0%}   {cagr(vt_r):>+6.1%}  {sr(vt_r):>+7.3f}  "
           f"{mdd(vt_r):>+8.1%}  {sc.mean():>6.2f}x  {sc.max():>6.2f}x  {beat:>8.0%}")

    pr("")
    pr("AvgLev = mean daily scale factor over OOS; MaxLev = single-day peak.")
    pr("Beat SPY = % of rolling 1-year windows where scaled book CAGR > SPY CAGR.")

    # ─── PART 3: per-year table ───────────────────────────────────────────────
    pr("")
    pr("=" * 76)
    pr("PART 3 -- PER-YEAR RETURNS AT EACH VOL TARGET  (vs SPY)")
    pr("=" * 76)
    hdr = f"{'Year':>6}  {'SPY':>6}  {'raw':>6}"
    for vt in VOL_TARGETS:
        hdr += f"  {int(vt * 100):>3}%"
    pr(hdr)
    pr("-" * 76)

    for yr in range(2017, 2025):
        ys, ye = f"{yr}-01-01", f"{yr}-12-31"
        spy_y  = cagr(spy_ret.loc[ys:ye])
        raw_y  = cagr(book_net.loc[ys:ye])
        row    = f"{yr:>6}  {spy_y:>+5.1%}  {raw_y:>+5.1%}"
        for vt in VOL_TARGETS:
            y_r = cagr(vt_rets[vt].loc[ys:ye])
            row += f"  {y_r:>+4.1%}"
        pr(row)

    # ─── PART 4: financing-adjusted summary ───────────────────────────────────
    pr("")
    pr("=" * 76)
    pr("PART 4 -- FINANCING-ADJUSTED CAGR  (deduct SOFR on borrowed fraction)")
    pr(f"  Assumed avg SOFR over OOS 2017-2024: {AVG_SOFR*100:.1f}%")
    pr("  Financing cost = max(AvgLev - 1, 0) x SOFR  (pay only on levered portion)")
    pr("=" * 76)
    pr(f"{'Target':>8}  {'CAGR':>8}  {'FinCost':>8}  {'AdjCAGR':>8}  {'AdjSharpe':>10}")
    pr("-" * 76)
    pr(f"{'raw book':>8}  {cagr(book_net):>+7.1%}  {'0.00%':>8}  {cagr(book_net):>+7.1%}  "
       f"{sr(book_net):>+9.3f}")

    for vt in VOL_TARGETS:
        vt_r   = vt_rets[vt]
        sc     = vt_scale[vt]
        avg_lv = float(sc.mean())
        fin_c  = max(avg_lv - 1.0, 0.0) * AVG_SOFR
        adj_r  = vt_r - fin_c / TRADING_DAYS
        pr(f"{vt:>7.0%}   {cagr(vt_r):>+7.1%}  {fin_c*100:>7.2f}%  "
           f"{cagr(adj_r):>+7.1%}  {sr(adj_r):>+9.3f}")

    pr("")
    pr("Financing cost is a rough lower bound; actual cost depends on broker margin")
    pr("rate, which typically runs SOFR + 50-200bp for retail or SOFR + 10-50bp prime.")
    pr("Gap risk (fast gap past stop on levered fraction) is not modeled here.")

    # ─── save ────────────────────────────────────────────────────────────────
    out_path = os.path.join(RESULTS_DIR, "vol_sweep_output.txt")
    with open(out_path, "w", encoding="ascii", errors="replace") as fh:
        fh.write("\n".join(lines))
    print(f"\nSaved: {out_path}")


if __name__ == "__main__":
    main()
