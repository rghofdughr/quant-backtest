"""
run_defensive_variants.py
Defensive-asset (gold / bond / commodity) rotation strategies.

Step 0: Correlation diagnostic -- which defensive assets are least redundant
        with the existing 11-strategy book (especially s90, s46, s31).

Mechanism R: Reactive (equity-stress driven -- control, expected to mirror s90/s46).
Mechanism O: Own-dynamics (asset's own trend/momentum -- most likely orthogonal).
Mechanism X: Cross-asset momentum (GTAA/dual-momentum style -- expected to mirror s02).

OOS 2017-07-03 to 2024-12-31. Costs: 20bp round-trip (apply_costs defaults).
"""
import sys, os, importlib, warnings, yaml
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
from pathlib import Path

from data import load_price_series, ADJ_TOTALRETURN
from engine import portfolio_returns_from_weights, apply_costs

PROJECT_ROOT = Path(__file__).resolve().parent.parent
CACHE_DIR    = PROJECT_ROOT / "cache" / "parquet"
RESULTS_DIR  = PROJECT_ROOT / "results"
RESULTS_DIR.mkdir(exist_ok=True)

PRICE_START = "2000-01-01"
OOS_START   = "2017-07-03"
OOS_END     = "2024-12-31"
IS_START    = "2000-01-01"

# All ETFs we may need for prices or signals
ALL_ASSETS = ["SPY", "GLD", "TLT", "IEF", "SHY", "TIP", "DBC", "HYG", "LQD", "VNQ"]

FINAL_BOOK = [
    ("s128", "strategies.s128_sector_rotation_9m",  0.065),
    ("s30",  "strategies.s30_low_volatility",        0.060),
    ("s31",  "strategies.s31_vol_targeting",         0.090),
    ("s35",  "strategies.s35_sell_in_may",           0.060),
    ("s49",  "strategies.s49_dollar_regime",         0.060),
    ("s115", "strategies.s115_year_end_reversal",    0.080),
    ("s113", "strategies.s113_january_barometer",    0.030),
    ("s135", "strategies.s135_ts_momentum_blended",  0.050),
    ("s02",  "strategies.s02_ts_momentum",           0.145),
    ("s46",  "strategies.s46_risk_parity",           0.140),
    ("s90",  "strategies.s90_credit_regime",         0.220),
]

GATE_CORR  = 0.70
GATE_DNEFF = 0.05
GATE_DSR   = 0.0
TEST_WT    = 0.05

# ──────────────────────────────────────────────────────────────────────────────
# Output helpers
# ──────────────────────────────────────────────────────────────────────────────

_outlines = []

def pr(s=""):
    _outlines.append(str(s))
    try:
        print(s)
    except UnicodeEncodeError:
        print(str(s).encode("ascii", "replace").decode("ascii"))

def save_output():
    out_path = RESULTS_DIR / "defensive_variants_output.txt"
    with open(out_path, "w", encoding="utf-8") as f:
        f.write("\n".join(_outlines) + "\n")
    pr(f"Output saved to {out_path}")

# ──────────────────────────────────────────────────────────────────────────────
# Metric helpers
# ──────────────────────────────────────────────────────────────────────────────

def sharpe(ret):
    r = ret.dropna()
    if len(r) == 0 or r.std() < 1e-9:
        return 0.0
    return float(r.mean() / r.std() * np.sqrt(252))

def neff(corr_mat):
    eigs = np.linalg.eigvalsh(corr_mat)
    eigs = np.maximum(eigs, 0.0)
    s, s2 = eigs.sum(), (eigs ** 2).sum()
    return float(s * s / s2) if s2 > 1e-12 else 1.0

def perf_metrics(ret):
    """Returns (sharpe, cagr, mdd) for a return series."""
    r = ret.dropna()
    if len(r) == 0:
        return 0.0, 0.0, 0.0
    sr = sharpe(r)
    cum = (1.0 + r).cumprod()
    years = len(r) / 252.0
    cagr = float(cum.iloc[-1] ** (1.0 / max(years, 0.01)) - 1.0)
    roll_max = cum.cummax()
    mdd = float(((cum - roll_max) / roll_max).min())
    return sr, cagr, mdd

def gate_metrics(variant_net, book_oos_ret, strat_rets):
    """Redundancy gate analysis."""
    oos_idx = pd.bdate_range(OOS_START, OOS_END)
    v  = variant_net.reindex(oos_idx).fillna(0.0)
    bk = book_oos_ret.reindex(oos_idx).fillna(0.0)

    strat_mat = pd.DataFrame({
        sid: r.reindex(oos_idx).fillna(0.0) for sid, r in strat_rets.items()
    })

    corr_book = float(v.corr(bk))

    strat_corrs = {sid: float(v.corr(strat_mat[sid])) for sid in strat_mat.columns}
    max_strat_corr = max(strat_corrs.values())
    best_match     = max(strat_corrs, key=strat_corrs.get)

    # dN_eff
    n11 = neff(strat_mat.corr().values)
    strat_plus = strat_mat.copy(); strat_plus["__v__"] = v
    n12 = neff(strat_plus.corr().values)
    dneff = n12 - n11

    # dSR at 5% test weight
    blended = bk * (1.0 - TEST_WT) + v * TEST_WT
    dsr = sharpe(blended) - sharpe(bk)

    fails = []
    if max_strat_corr >= GATE_CORR:  fails.append("corr")
    if dneff <= GATE_DNEFF:          fails.append("dNeff")
    if dsr    <  GATE_DSR:           fails.append("dSR")

    return dict(
        corr_book=corr_book,
        max_strat_corr=max_strat_corr,
        best_match=best_match,
        dneff=dneff, dsr=dsr,
        pass_all=(len(fails) == 0),
        gates_str=("/".join(fails) if fails else "PASS"),
    )

# ──────────────────────────────────────────────────────────────────────────────
# Price loading
# ──────────────────────────────────────────────────────────────────────────────

def load_prices():
    full_idx = pd.bdate_range(PRICE_START, OOS_END)
    prices = {}
    for sym in ALL_ASSETS:
        df = load_price_series(sym, PRICE_START, OOS_END, ADJ_TOTALRETURN, str(CACHE_DIR))
        if not df.empty and "Close" in df.columns:
            s = df["Close"].reindex(full_idx, method="ffill")
            if s.dropna().empty:
                pr(f"  WARNING: {sym} has no data")
            else:
                prices[sym] = s
                pr(f"  {sym}: {s.first_valid_index().date()} to {s.last_valid_index().date()}")
        else:
            pr(f"  WARNING: {sym} not in Norgate cache")
    return prices

def load_book_returns():
    cfg_path = PROJECT_ROOT / "config.yaml"
    with open(cfg_path) as f:
        cfg = yaml.safe_load(f)
    cfg["backtest"]["start_date"] = IS_START
    cfg["backtest"]["end_date"]   = OOS_END
    cfg["paths"]["cache_dir"]     = str(CACHE_DIR)
    oos_idx = pd.bdate_range(OOS_START, OOS_END)
    strat_rets = {}
    for sid, mname, _ in FINAL_BOOK:
        mod = importlib.import_module(mname)
        res = mod.run(cfg)
        strat_rets[sid] = res["returns"].fillna(0.0)
        pr(f"  {sid}: OOS SR={sharpe(strat_rets[sid].reindex(oos_idx).fillna(0)):+.3f}")

    book_oos = sum(
        strat_rets[sid].reindex(oos_idx).fillna(0.0) * wt
        for sid, _, wt in FINAL_BOOK
    )
    return book_oos, strat_rets

# ──────────────────────────────────────────────────────────────────────────────
# Signal helpers (all lookbacks in trading days)
# ──────────────────────────────────────────────────────────────────────────────

def ma_signal(prices, sym, date, window):
    """1 if price > window-day SMA; 1 (invested) if insufficient history."""
    if sym not in prices:
        return 1
    s = prices[sym].loc[:date].dropna()
    if len(s) < window + 1:
        return 1
    return 1 if float(s.iloc[-1]) > float(s.iloc[-window:].mean()) else 0

def vol_ratio_signal(prices, sym, date, st=21, lt=252):
    """Short-term vol / long-term vol. Returns 1.0 if insufficient history."""
    if sym not in prices:
        return 1.0
    ret = prices[sym].loc[:date].dropna().pct_change(fill_method=None).dropna()
    if len(ret) < lt:
        return 1.0
    v_st = float(ret.iloc[-st:].std()) * 252 ** 0.5
    v_lt = float(ret.iloc[-lt:].std()) * 252 ** 0.5
    return v_st / v_lt if v_lt > 1e-9 else 1.0

def mom_score(prices, sym, date, lb):
    """Lookback-day price momentum; None if insufficient history."""
    if sym not in prices:
        return None
    s = prices[sym].loc[:date].dropna()
    if len(s) < lb + 2:
        return None
    p1, p0 = float(s.iloc[-1]), float(s.iloc[-lb])
    return (p1 / p0 - 1.0) if p0 > 1e-9 else None

def hyg_lqd_signal(prices, date, ma_days=63):
    """1 (risk-on) if HYG/LQD ratio > ma_days MA; 1 if data unavailable."""
    if "HYG" not in prices or "LQD" not in prices:
        return 1
    h = prices["HYG"].loc[:date].dropna()
    l = prices["LQD"].loc[:date].dropna()
    idx = h.index.intersection(l.index)
    if len(idx) < ma_days + 1:
        return 1
    ratio = h.loc[idx] / l.loc[idx]
    cur = float(ratio.iloc[-1])
    ma  = float(ratio.iloc[-ma_days:].mean())
    return 1 if cur > ma else 0

# ──────────────────────────────────────────────────────────────────────────────
# Weight-schedule builders
# ──────────────────────────────────────────────────────────────────────────────

def _reb_dates():
    return pd.date_range(PRICE_START, OOS_END, freq="BME")

# -- Mechanism R: equity-stress reactive (control) --

def build_R_ma200(prices):
    """SPY above 200d MA -> 100% SPY; else 50/50 GLD+TLT."""
    ws = {}
    for d in _reb_dates():
        if ma_signal(prices, "SPY", d, 200):
            ws[d] = {"SPY": 1.0}
        else:
            ws[d] = {"GLD": 0.5, "TLT": 0.5}
    return ws

def build_R_vol(prices):
    """21d/252d SPY vol ratio > 1.5 -> defensive (GLD+TLT); else SPY."""
    ws = {}
    for d in _reb_dates():
        vr = vol_ratio_signal(prices, "SPY", d, st=21, lt=252)
        ws[d] = {"SPY": 1.0} if vr < 1.5 else {"GLD": 0.5, "TLT": 0.5}
    return ws

def build_R_credit(prices):
    """HYG/LQD ratio > 63d MA (identical to s90 signal) -> SPY; else GLD+TLT.
    Control: does the same credit signal + different defensive asset help?"""
    ws = {}
    for d in _reb_dates():
        if hyg_lqd_signal(prices, d, ma_days=63):
            ws[d] = {"SPY": 1.0}
        else:
            ws[d] = {"GLD": 0.5, "TLT": 0.5}
    return ws

# -- Mechanism O: own-dynamics, asset-driven --

def build_O_goldtrend(prices):
    """Gold's own 6m (126d) MA trend drives allocation.
    GLD above MA -> 100% GLD; else -> IEF if IEF > 10m MA; else cash.
    No equity exposure. Gold drives the switch, not equities."""
    ws = {}
    for d in _reb_dates():
        if ma_signal(prices, "GLD", d, 126):
            ws[d] = {"GLD": 1.0}
        elif ma_signal(prices, "IEF", d, 210):
            ws[d] = {"IEF": 1.0}
        else:
            ws[d] = {}   # cash
    return ws

def build_O_bondtrend(prices):
    """TLT own 10m (210d) MA drives allocation.
    TLT above MA -> 100% TLT; else -> GLD if GLD > 6m MA; else cash.
    No equity exposure. Bond trend drives the switch."""
    ws = {}
    for d in _reb_dates():
        if ma_signal(prices, "TLT", d, 210):
            ws[d] = {"TLT": 1.0}
        elif ma_signal(prices, "GLD", d, 126):
            ws[d] = {"GLD": 1.0}
        else:
            ws[d] = {}
    return ws

def build_O_multi(prices):
    """Hold each of {GLD, TLT, IEF, TIP, DBC} that beats SHY on 6m abs momentum.
    Equal-weight winners; cash when none qualify.
    No equity. Driven entirely by each asset's own trend vs the cash hurdle.
    TIP is the key asset not in s46's universe."""
    DEFS = ["GLD", "TLT", "IEF", "TIP", "DBC"]
    ws = {}
    for d in _reb_dates():
        shy_m = mom_score(prices, "SHY", d, 126)
        hurdle = shy_m if shy_m is not None else 0.0
        winners = [sym for sym in DEFS
                   if sym in prices
                   and mom_score(prices, sym, d, 126) is not None
                   and mom_score(prices, sym, d, 126) > hurdle]
        if winners:
            w = 1.0 / len(winners)
            ws[d] = {s: w for s in winners}
        else:
            ws[d] = {}
    return ws

# -- Mechanism X: cross-asset momentum --

def build_X_gtaa5(prices):
    """Faber-style GTAA: rank {SPY, TLT, IEF, GLD, DBC} by 12m momentum.
    Hold top-3 with positive momentum, equal-weight. Cash if none positive."""
    UNIV = ["SPY", "TLT", "IEF", "GLD", "DBC"]
    ws = {}
    for d in _reb_dates():
        scores = {s: m for s in UNIV
                  if (m := mom_score(prices, s, d, 252)) is not None}
        ranked = sorted(scores.items(), key=lambda x: -x[1])
        top = [(k, v) for k, v in ranked[:3] if v > 0.0]
        ws[d] = {k: 1.0 / len(top) for k, _ in top} if top else {}
    return ws

def build_X_gtaa6(prices):
    """Same as X-gtaa5 but 10m lookback and expanded universe including TIP."""
    UNIV = ["SPY", "TLT", "IEF", "GLD", "DBC", "TIP"]
    ws = {}
    for d in _reb_dates():
        scores = {s: m for s in UNIV
                  if (m := mom_score(prices, s, d, 210)) is not None}
        ranked = sorted(scores.items(), key=lambda x: -x[1])
        top = [(k, v) for k, v in ranked[:3] if v > 0.0]
        ws[d] = {k: 1.0 / len(top) for k, _ in top} if top else {}
    return ws

def build_X_dual(prices):
    """Dual momentum: rank {SPY, GLD, TLT} by 6m absolute momentum.
    Hold top-1 if it beats SHY (positive absolute momentum); else cash.
    This is closest to s77 (GEM) -- serves as a cross-asset control."""
    UNIV = ["SPY", "GLD", "TLT"]
    ws = {}
    for d in _reb_dates():
        shy_m = mom_score(prices, "SHY", d, 126)
        hurdle = shy_m if shy_m is not None else 0.0
        scores = {s: m for s in UNIV
                  if (m := mom_score(prices, s, d, 126)) is not None}
        if not scores:
            ws[d] = {}
            continue
        best = max(scores, key=scores.get)
        ws[d] = {best: 1.0} if scores[best] > hurdle else {}
    return ws

# ──────────────────────────────────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────────────────────────────────

VARIANTS = [
    # (label, mechanism, description, builder)
    ("R-ma200",     "R-Reactive",   "SPY>200dMA->SPY; else 50/50 GLD+TLT",              build_R_ma200),
    ("R-vol",       "R-Reactive",   "21d/252d vol>1.5->GLD+TLT; else SPY",              build_R_vol),
    ("R-credit",    "R-Reactive",   "HYG/LQD>63dMA->SPY; else GLD+TLT (=s90 signal)",  build_R_credit),
    ("O-goldtrend", "O-Own-dyn",    "GLD>126dMA->GLD; elif IEF>210dMA->IEF; else cash", build_O_goldtrend),
    ("O-bondtrend", "O-Own-dyn",    "TLT>210dMA->TLT; elif GLD>126dMA->GLD; else cash", build_O_bondtrend),
    ("O-multi",     "O-Own-dyn",    "Hold GLD/TLT/IEF/TIP/DBC if 6m mom > SHY",        build_O_multi),
    ("X-gtaa5",     "X-Cross-mom",  "Top-3 of SPY/TLT/IEF/GLD/DBC by 12m; cash filter", build_X_gtaa5),
    ("X-gtaa6",     "X-Cross-mom",  "Top-3 of +TIP universe by 10m; cash filter",       build_X_gtaa6),
    ("X-dual",      "X-Cross-mom",  "Dual-mom SPY/GLD/TLT 6m vs SHY (s77 control)",    build_X_dual),
]


def main():
    pr("Loading asset prices (full history 2000-2024)...")
    prices = load_prices()

    pr("")
    pr("Loading existing 11-strategy book returns...")
    book_oos, strat_rets = load_book_returns()

    oos_idx = pd.bdate_range(OOS_START, OOS_END)
    pr(f"  Book OOS SR={sharpe(book_oos):+.3f}  N_eff={neff(pd.DataFrame({sid: r.reindex(oos_idx).fillna(0) for sid, r in strat_rets.items()}).corr().values):.3f}")

    # Build close_df for the engine (full history, all loaded assets)
    full_idx = pd.bdate_range(PRICE_START, OOS_END)
    close_df = pd.DataFrame({
        sym: prices[sym].reindex(full_idx, method="ffill")
        for sym in prices
    })

    # ──────────────────────────────────────────────────────────────────────────
    # Step 0: Diagnostic -- asset return vs strategy correlation
    # ──────────────────────────────────────────────────────────────────────────
    DIAG_ASSETS = [s for s in ["GLD", "TLT", "IEF", "SHY", "TIP", "DBC"] if s in prices]
    KEY_STRATS  = ["s90", "s46", "s31", "s02", "s128", "s30"]

    asset_rets  = pd.DataFrame({
        sym: prices[sym].reindex(oos_idx, method="ffill").pct_change(fill_method=None).fillna(0)
        for sym in DIAG_ASSETS
    })
    strat_ret_df = pd.DataFrame({
        sid: strat_rets[sid].reindex(oos_idx).fillna(0) for sid in strat_rets
    })

    diag_corr = {}
    for sym in DIAG_ASSETS:
        diag_corr[sym] = {sid: float(asset_rets[sym].corr(strat_ret_df[sid]))
                          for sid in strat_ret_df.columns}

    pr("")
    pr("=" * 105)
    pr("STEP 0 -- DIAGNOSTIC: DEFENSIVE ASSET vs EXISTING STRATEGY CORRELATIONS (OOS 2017-2024)")
    pr("=" * 105)
    pr(f"  s46 universe: SPY+TLT+GLD+DBC+VNQ (inv-vol weighted; always invested)")
    pr(f"  s90 universe: SPY or FLAT (HYG/LQD credit regime; never holds bonds/gold)")
    pr(f"  s31 universe: SPY only (vol-target; never holds bonds/gold)")
    pr(f"  TIP/IEF/SHY: not in any existing strategy's universe")
    pr("")
    hdr = f"  {'Asset':<6}  " + "  ".join(f"{s:>6}" for s in KEY_STRATS) + "  {'Book':>6}  Least-redundant-with"
    pr(hdr)
    pr("  " + "-" * 80)

    for sym in DIAG_ASSETS:
        vals = [diag_corr[sym].get(s, float("nan")) for s in KEY_STRATS]
        bk   = float(asset_rets[sym].corr(book_oos))
        # The strategy it overlaps LEAST (for building on)
        min_key_corr = min((v for v in vals if not np.isnan(v)), default=0.0)
        min_strat    = KEY_STRATS[int(np.argmin([abs(v) for v in vals]))]
        pr(f"  {sym:<6}  " + "  ".join(f"{v:+.3f}" for v in vals) + f"  {bk:+.3f}  ({min_strat}, {min_key_corr:+.3f})")

    pr("")
    pr("  Implications for strategy design:")
    pr("  - High corr to s46 -> asset already in risk-parity sleeve; marginal value low")
    pr("  - High corr to s90 -> asset moves with credit regime (equity stress proxy)")
    pr("  - Low corr to both -> potentially new information for the book")

    # ──────────────────────────────────────────────────────────────────────────
    # Run all 9 variants
    # ──────────────────────────────────────────────────────────────────────────
    pr("")
    pr("Running 9 variants...")
    results = []

    for label, mech, desc, builder_fn in VARIANTS:
        pr(f"  Running {label}...")
        ws = builder_fn(prices)

        gross_ret, to_ser = portfolio_returns_from_weights(
            ws, close_df, OOS_START, OOS_END, execution_lag=1
        )
        net_ret  = apply_costs(gross_ret, to_ser)
        ann_to   = float(to_ser.sum() / max(len(to_ser) / 252.0, 1.0))

        gr_sr, gr_cagr, gr_mdd = perf_metrics(gross_ret)
        nt_sr, nt_cagr, nt_mdd = perf_metrics(net_ret)

        g = gate_metrics(net_ret, book_oos, strat_rets)

        yr_rets = {}
        for yr in range(2017, 2025):
            r_yr = net_ret[net_ret.index.year == yr]
            yr_rets[yr] = float((1.0 + r_yr).prod() - 1.0) if not r_yr.empty else float("nan")

        results.append(dict(
            label=label, mech=mech, desc=desc,
            gr_sr=gr_sr, nt_sr=nt_sr, nt_cagr=nt_cagr, nt_mdd=nt_mdd,
            ann_to=ann_to, yr_rets=yr_rets,
            **g,
        ))

    # ──────────────────────────────────────────────────────────────────────────
    # Main results table
    # ──────────────────────────────────────────────────────────────────────────
    pr("")
    pr("=" * 115)
    pr("DEFENSIVE-ASSET ROTATION  OOS 2017-07-03 to 2024-12-31")
    pr("=" * 115)
    pr(f"Book baseline:  OOS SR={sharpe(book_oos):+.3f}  CAGR=+11.7%  MDD=-15.5%")
    pr("SE(Sharpe,8yr) ~= +/-0.35  --  gaps < 0.10 are statistical ties")
    pr("")
    pr(f"{'Label':<14} {'GrSR':>6} {'NetSR':>6} {'CAGR':>7} {'MDD':>7} {'TO':>5}  {'CorrBk':>7} {'MaxStr':>7} {'Match':>5} {'dNeff':>7} {'dSR':>6}  Gates")
    pr("-" * 115)

    cur_mech = None
    for r in results:
        if r["mech"] != cur_mech:
            cur_mech = r["mech"]
            pr(f"-- {cur_mech} --")
        pr(
            f"  {r['label']:<12} {r['gr_sr']:+.3f} {r['nt_sr']:+.3f} "
            f"{r['nt_cagr']:+.1%} {r['nt_mdd']:+.1%} {r['ann_to']:>4.1f}x "
            f" {r['corr_book']:+.3f}  {r['max_strat_corr']:+.3f} "
            f"{r['best_match']:>5} {r['dneff']:+.4f} {r['dsr']:+.4f}  {r['gates_str']}"
        )

    # ──────────────────────────────────────────────────────────────────────────
    # Per-year returns
    # ──────────────────────────────────────────────────────────────────────────
    pr("")
    pr("=" * 115)
    pr("PER-YEAR NET RETURNS (cap: watch for single-year drivers)")
    pr("=" * 115)
    years = list(range(2017, 2025))

    book_yr = {}
    for yr in years:
        r_yr = book_oos[book_oos.index.year == yr]
        book_yr[yr] = float((1.0 + r_yr).prod() - 1.0) if not r_yr.empty else float("nan")

    labels_short = [r["label"] for r in results]
    hdr2 = f"  {'Year':<5}" + f"  {'Book':>8}" + "".join(f"  {lbl:>12}" for lbl in labels_short)
    pr(hdr2)
    pr("  " + "-" * (len(hdr2) - 2))

    for yr in years:
        row = f"  {yr:<5}  {book_yr[yr]:>+7.1%}"
        for r in results:
            v = r["yr_rets"].get(yr, float("nan"))
            row += f"  {v:>+11.1%}" if not np.isnan(v) else f"  {'N/A':>11}"
        pr(row)

    # Flag years driving Sharpe (any year > 2x its proportional share)
    pr("")
    pr("  Single-year dependency check (flag if one year > ~40% of total CAGR):")
    for r in results:
        yr_vals = {yr: r["yr_rets"].get(yr, 0.0) for yr in years if not np.isnan(r["yr_rets"].get(yr, float("nan")))}
        total_pos = sum(v for v in yr_vals.values() if v > 0)
        if total_pos > 0.01:
            max_yr = max(yr_vals, key=yr_vals.get)
            frac   = yr_vals[max_yr] / total_pos
            if frac > 0.40:
                pr(f"  FLAG {r['label']}: {max_yr} = {yr_vals[max_yr]:+.1%}  ({frac:.0%} of total positive return)")

    # ──────────────────────────────────────────────────────────────────────────
    # Gate summary
    # ──────────────────────────────────────────────────────────────────────────
    pr("")
    pr("=" * 115)
    pr("REDUNDANCY GATE SUMMARY")
    pr(f"Gates: max_strat_corr < {GATE_CORR}  |  dN_eff > +{GATE_DNEFF}  |  dSR >= {GATE_DSR}  (test weight: {TEST_WT:.0%})")
    pr("=" * 115)

    passing = [r for r in results if r["pass_all"]]
    pr(f"Variants passing ALL gates: {len(passing)} of {len(results)}")
    if passing:
        for r in passing:
            pr(f"  PASS: {r['label']:<14} NetSR={r['nt_sr']:+.3f}  corr_book={r['corr_book']:+.3f}  "
               f"max_strat={r['max_strat_corr']:+.3f}  dNeff={r['dneff']:+.4f}  dSR={r['dsr']:+.4f}")
    else:
        pr("  None passed all gates.")

    pr("")
    pr("PRIOR HYPOTHESIS CHECK:")
    for mech_key, mech_label in [("R-Reactive", "R (expected: correlated with s90/s46)"),
                                   ("O-Own-dyn",  "O (expected: most orthogonal)"),
                                   ("X-Cross-mom","X (expected: correlated with s02/s77)")]:
        pr(f"  {mech_label}:")
        for r in results:
            if r["mech"] == mech_key:
                pr(f"    {r['label']:<14}: corr_book={r['corr_book']:+.3f}  "
                   f"best_match={r['best_match']}({r['max_strat_corr']:+.3f})  "
                   f"dNeff={r['dneff']:+.4f}  gates={r['gates_str']}")

    pr("")
    pr("STRUCTURAL NOTE:")
    pr("  s46 holds TLT+GLD+DBC continuously (inv-vol weighted, 14% book weight).")
    pr("  Any strategy holding those assets unconditionally will correlate with s46.")
    pr("  Only conditional exposure (hold-when-trending) can break this correlation.")
    pr("  2022 is the key year: bonds -26%, gold flat, DBC +25%. O variants must")
    pr("  show they exited bonds in 2022 to claim genuine non-redundancy vs s46.")

    pr("")
    pr("Done.")
    save_output()


if __name__ == "__main__":
    main()
