"""
run_s128_variants.py
Test 10 variants of the s128 sector rotation strategy.
Each variant is a self-contained modification tested independently.
OOS 2017-07-03 to 2024-12-31. Costs: 20bp round-trip.

Variants:
  V0  Baseline: s128 re-implemented (9m, top-4, equal-wt, cash filter)
  V1  Inverse-vol weighting of selected sectors
  V2  Portfolio-level vol targeting (10% target, 1.5x max leverage)
  V3  Blend 3/6/9/12m lookbacks by average rank
  V4  Skip-month gap (9m signal ending 1m ago, i.e. 210d-21d window)
  V5  No-trade band / hysteresis (buy top-4, hold until top-6)
  V6  Graded per-holding cash filter (drop negative-9m holdings individually)
  V7  Defensive sleeve (IEF/GLD/SHY by momentum when all sectors negative)
  V8  Broader universe (11 SPDR + EFA/EEM/EWJ international ETFs)
  V9  Diversification-aware greedy selection (skip corr > 0.85 with selected)
  V10 Cash yield fix (hold SHY when in cash instead of earning 0%)

Analysis:
  - Standalone: GrSR, NetSR, CAGR, MDD, turnover
  - vs s128 baseline: corr_s128, delta_SR_book (swap s128 for variant in full book)
  - vs other 10 book strategies: max_other_corr (redundancy check for replacement)
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

PRICE_START = "1998-01-01"
OOS_START   = "2017-07-03"
OOS_END     = "2024-12-31"
IS_START    = "2000-01-01"

# 11 SPDR sector ETFs (XLC from Jun 2018; missing data handled gracefully)
SECTORS = ["XLK", "XLF", "XLE", "XLV", "XLI", "XLY", "XLP", "XLRE", "XLB", "XLU", "XLC"]
# Extra ETFs for V8 (international)
EXTRA_V8 = ["EFA", "EEM", "EWJ"]
# Defensive ETFs for V7 / V10
DEF_ETFS = ["IEF", "GLD", "SHY"]

# s128 baseline parameters (frozen)
TOP_N     = 4
LB_DAYS   = 189    # 9m * 21 trading-days/month
VOL_LB    = 63     # trailing window for realized-vol computations
TARGET_VOL = 0.10  # V2 vol target
MAX_LEV   = 1.5    # V2 max leverage
BAND      = 6      # V5 hysteresis band

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
S128_WT = 0.065

_outlines = []

def pr(s=""):
    _outlines.append(str(s))
    try:
        print(s)
    except UnicodeEncodeError:
        print(str(s).encode("ascii", "replace").decode("ascii"))

def save_output():
    p = RESULTS_DIR / "s128_variants_output.txt"
    with open(p, "w", encoding="utf-8") as f:
        f.write("\n".join(_outlines) + "\n")

# ── Metric helpers ───────────────────────────────────────────────────────────

def sharpe(ret):
    r = ret.dropna()
    if len(r) == 0 or r.std() < 1e-10:
        return 0.0
    return float(r.mean() / r.std() * np.sqrt(252))

def perf(ret):
    r = ret.dropna()
    if len(r) == 0:
        return 0.0, 0.0, 0.0
    sr = sharpe(r)
    yr = len(r) / 252.0
    cum = (1.0 + r).cumprod()
    cagr = float(cum.iloc[-1] ** (1.0 / max(yr, 0.01)) - 1.0)
    mdd  = float(((cum - cum.cummax()) / cum.cummax()).min())
    return sr, cagr, mdd

# ── Core signal helpers ──────────────────────────────────────────────────────

def mom_scores(close_df, snap, lb, skip=0):
    """
    Momentum scores for all columns of close_df.
    snap: integer row index (most recent date).
    lb: lookback in trading days.
    skip: days to skip at the recent end (skip-month variant).
    """
    now  = max(0, snap - skip)
    past = max(0, snap - lb - skip)
    p1   = close_df.iloc[now]
    p0   = close_df.iloc[past]
    return ((p1 / p0) - 1.0).dropna().sort_values(ascending=False)

def inv_vol_weights(syms, ret_df, snap, vol_lb=VOL_LB):
    """Inverse-vol weights for given symbols using trailing vol_lb days."""
    window = ret_df.iloc[max(0, snap - vol_lb):snap][syms]
    vols = window.std() * np.sqrt(252)
    vols = vols.replace(0.0, np.nan).dropna()
    valid = [s for s in syms if s in vols.index]
    if not valid:
        n = len(syms)
        return {s: 1.0/n for s in syms}
    inv = 1.0 / vols[valid]
    total = inv.sum()
    base = {s: float(inv[s] / total) for s in valid}
    # Fill any missing with equal share
    missing = [s for s in syms if s not in base]
    if missing:
        n_miss = len(missing)
        fill_w = (1.0 - sum(base.values())) / n_miss if n_miss else 0.0
        for s in missing:
            base[s] = max(fill_w, 0.0)
    return base

def port_realized_vol(syms, weights, ret_df, snap, vol_lb=VOL_LB):
    """
    Annualised realised vol of a portfolio using the trailing vol_lb-day window.
    weights: dict {sym: float}
    """
    if not syms:
        return TARGET_VOL
    window = ret_df.iloc[max(0, snap - vol_lb):snap][[s for s in syms if s in ret_df.columns]]
    if window.empty or window.shape[0] < 10:
        return TARGET_VOL
    w_arr = np.array([weights.get(s, 0.0) for s in window.columns])
    cov   = window.cov().values * 252
    port_var = float(w_arr @ cov @ w_arr)
    return max(np.sqrt(max(port_var, 1e-8)), 1e-4)

def pairwise_corr(syms, ret_df, snap, corr_lb=VOL_LB):
    """Correlation matrix for syms over trailing corr_lb days."""
    window = ret_df.iloc[max(0, snap - corr_lb):snap][[s for s in syms if s in ret_df.columns]]
    if window.shape[0] < 10:
        return pd.DataFrame(np.eye(len(syms)), index=syms, columns=syms)
    return window.corr()

# ── Weight schedule builders ─────────────────────────────────────────────────

def _reb_dates():
    return pd.date_range(PRICE_START, OOS_END, freq="BME")

def _snap(close_df, d):
    """Row index of the last available date on or before d."""
    avail = close_df.index[close_df.index <= d]
    return len(avail) - 1

# V0: Baseline (re-implements s128 exactly)
def build_V0_baseline(close_df):
    ws = {}
    for d in _reb_dates():
        s = _snap(close_df, d)
        if s < LB_DAYS + 2:
            ws[d] = {}
            continue
        m = mom_scores(close_df, s, LB_DAYS)
        if m.empty or float(m.iloc[0]) <= 0:
            ws[d] = {}
        else:
            top = m.index[:TOP_N].tolist()
            ws[d] = {sym: 1.0 / len(top) for sym in top}
    return ws

# V1: Inverse-vol weighting
def build_V1_invvol(close_df, ret_df):
    ws = {}
    for d in _reb_dates():
        s = _snap(close_df, d)
        if s < LB_DAYS + 2:
            ws[d] = {}
            continue
        m = mom_scores(close_df, s, LB_DAYS)
        if m.empty or float(m.iloc[0]) <= 0:
            ws[d] = {}
        else:
            top = m.index[:TOP_N].tolist()
            ws[d] = inv_vol_weights(top, ret_df, s)
    return ws

# V2: Portfolio-level vol targeting (10%, max 1.5x leverage)
def build_V2_voltarget(close_df, ret_df):
    ws = {}
    for d in _reb_dates():
        s = _snap(close_df, d)
        if s < LB_DAYS + 2:
            ws[d] = {}
            continue
        m = mom_scores(close_df, s, LB_DAYS)
        if m.empty or float(m.iloc[0]) <= 0:
            ws[d] = {}
            continue
        top = m.index[:TOP_N].tolist()
        ew  = {sym: 1.0 / len(top) for sym in top}
        pv  = port_realized_vol(top, ew, ret_df, s)
        scale = min(TARGET_VOL / pv, MAX_LEV)
        ws[d] = {sym: w * scale for sym, w in ew.items()}
    return ws

# V3: Blend 3/6/9/12m lookbacks by average rank
def build_V3_blend(close_df):
    LOOKBACKS = [63, 126, 189, 252]
    ws = {}
    for d in _reb_dates():
        s = _snap(close_df, d)
        if s < max(LOOKBACKS) + 2:
            ws[d] = {}
            continue
        all_syms = set()
        rank_sum  = {}
        for lb in LOOKBACKS:
            m = mom_scores(close_df, s, lb)
            if m.empty:
                continue
            all_syms.update(m.index)
            for rank, sym in enumerate(m.index, 1):
                rank_sum[sym] = rank_sum.get(sym, 0) + rank
        if not rank_sum:
            ws[d] = {}
            continue
        avg_rank = pd.Series(rank_sum).sort_values()  # lower = better
        top = avg_rank.index[:TOP_N].tolist()
        # Cash filter: use 9m return of top-ranked sector
        m9 = mom_scores(close_df, s, LB_DAYS)
        top_9m = float(m9[top[0]]) if top[0] in m9.index else float(m9.iloc[0]) if not m9.empty else -1
        if top_9m <= 0:
            ws[d] = {}
        else:
            ws[d] = {sym: 1.0 / len(top) for sym in top}
    return ws

# V4: Skip-month gap (signal from 210d ago to 21d ago)
def build_V4_skipmonth(close_df):
    SKIP = 21
    ws = {}
    for d in _reb_dates():
        s = _snap(close_df, d)
        if s < LB_DAYS + SKIP + 2:
            ws[d] = {}
            continue
        m = mom_scores(close_df, s, LB_DAYS, skip=SKIP)
        if m.empty or float(m.iloc[0]) <= 0:
            ws[d] = {}
        else:
            top = m.index[:TOP_N].tolist()
            ws[d] = {sym: 1.0 / len(top) for sym in top}
    return ws

# V5: No-trade band / hysteresis (buy top-4, hold until top-6)
def build_V5_hysteresis(close_df):
    ws = {}
    current = set()
    for d in _reb_dates():
        s = _snap(close_df, d)
        if s < LB_DAYS + 2:
            ws[d] = {}
            current = set()
            continue
        m = mom_scores(close_df, s, LB_DAYS)
        if m.empty or float(m.iloc[0]) <= 0:
            ws[d] = {}
            current = set()
            continue
        top4 = set(m.index[:TOP_N].tolist())
        top6 = set(m.index[:BAND].tolist())
        # Keep existing holdings still in top-6
        held = {h for h in current if h in top6}
        # Buy new entries from top-4
        for sym in m.index[:TOP_N]:
            if len(held) >= TOP_N:
                break
            held.add(sym)
        current = held
        n = len(current)
        ws[d] = {sym: 1.0 / n for sym in current} if n > 0 else {}
    return ws

# V6: Graded per-holding cash filter (drop sectors with negative 9m return)
def build_V6_graded(close_df):
    ws = {}
    for d in _reb_dates():
        s = _snap(close_df, d)
        if s < LB_DAYS + 2:
            ws[d] = {}
            continue
        m = mom_scores(close_df, s, LB_DAYS)
        if m.empty:
            ws[d] = {}
            continue
        top = m.index[:TOP_N].tolist()
        pos = [sym for sym in top if float(m[sym]) > 0]
        if not pos:
            ws[d] = {}
        else:
            ws[d] = {sym: 1.0 / len(pos) for sym in pos}
    return ws

# V7: Defensive sleeve when all sectors negative
def build_V7_defensive(close_df, def_close):
    """When sectors all negative: hold best of {IEF, GLD, SHY} by 9m momentum."""
    DEF = [s for s in ["IEF", "GLD", "SHY"] if s in def_close.columns]
    # Merge for combined price DataFrame
    combined = pd.concat([close_df, def_close[DEF]], axis=1).sort_index()
    ws = {}
    for d in _reb_dates():
        s = _snap(combined, d)
        if s < LB_DAYS + 2:
            ws[d] = {}
            continue
        m = mom_scores(combined[SECTORS], s, LB_DAYS)
        if not m.empty and float(m.iloc[0]) > 0:
            top = m.index[:TOP_N].tolist()
            ws[d] = {sym: 1.0 / len(top) for sym in top}
        else:
            # Defensive sleeve: pick best momentum defensive asset
            if DEF:
                m_def = mom_scores(combined[DEF], s, LB_DAYS)
                m_def = m_def[m_def > 0]
                if not m_def.empty:
                    ws[d] = {m_def.index[0]: 1.0}
                else:
                    ws[d] = {}
            else:
                ws[d] = {}
    return ws, combined

# V8: Broader universe (11 SPDR + EFA/EEM/EWJ)
def build_V8_broad(all_close, ret_all):
    ws = {}
    for d in _reb_dates():
        s = _snap(all_close, d)
        if s < LB_DAYS + 2:
            ws[d] = {}
            continue
        m = mom_scores(all_close, s, LB_DAYS)
        if m.empty or float(m.iloc[0]) <= 0:
            ws[d] = {}
        else:
            top = m.index[:TOP_N].tolist()
            ws[d] = {sym: 1.0 / len(top) for sym in top}
    return ws

# V9: Diversification-aware greedy selection (skip corr > 0.85 with already selected)
def build_V9_divaware(close_df, ret_df, corr_thresh=0.85):
    ws = {}
    for d in _reb_dates():
        s = _snap(close_df, d)
        if s < LB_DAYS + 2:
            ws[d] = {}
            continue
        m = mom_scores(close_df, s, LB_DAYS)
        if m.empty or float(m.iloc[0]) <= 0:
            ws[d] = {}
            continue
        corr_mat = pairwise_corr(list(m.index), ret_df, s)
        selected = []
        for sym in m.index:
            if len(selected) >= TOP_N:
                break
            if not selected:
                selected.append(sym)
            else:
                max_c = max(
                    float(corr_mat.loc[sym, sel]) if (sym in corr_mat.index and sel in corr_mat.columns)
                    else 0.0
                    for sel in selected
                )
                if max_c < corr_thresh:
                    selected.append(sym)
                # if correlated with all existing, skip (try next ranked sector)
        if not selected:
            ws[d] = {}
        else:
            # Fill to top_n if we couldn't find enough orthogonal (fall back to momentum rank)
            if len(selected) < TOP_N:
                for sym in m.index:
                    if len(selected) >= TOP_N:
                        break
                    if sym not in selected:
                        selected.append(sym)
            ws[d] = {sym: 1.0 / len(selected) for sym in selected}
    return ws

# V10: Cash yield fix (hold SHY when in cash instead of earning 0%)
def build_V10_cashyield(close_df, shy_close):
    """Identical to V0 baseline but routes cash periods to SHY total return."""
    ws = {}
    for d in _reb_dates():
        s = _snap(close_df, d)
        if s < LB_DAYS + 2:
            if not shy_close.empty:
                ws[d] = {"SHY": 1.0}
            else:
                ws[d] = {}
            continue
        m = mom_scores(close_df, s, LB_DAYS)
        if m.empty or float(m.iloc[0]) <= 0:
            ws[d] = {"SHY": 1.0} if not shy_close.empty else {}
        else:
            top = m.index[:TOP_N].tolist()
            ws[d] = {sym: 1.0 / len(top) for sym in top}
    return ws

# ── Data loading ─────────────────────────────────────────────────────────────

def load_all_prices():
    full_idx = pd.bdate_range(PRICE_START, OOS_END)
    prices = {}
    for sym in SECTORS + EXTRA_V8 + DEF_ETFS:
        df = load_price_series(sym, PRICE_START, OOS_END, ADJ_TOTALRETURN, str(CACHE_DIR))
        if not df.empty and "Close" in df.columns:
            prices[sym] = df["Close"].reindex(full_idx, method="ffill")
    return prices, full_idx

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

    book_full = sum(
        strat_rets[sid].reindex(oos_idx).fillna(0.0) * wt
        for sid, _, wt in FINAL_BOOK
    )
    return book_full, strat_rets

# ── Main ─────────────────────────────────────────────────────────────────────

def main():
    pr("Loading prices...")
    prices, full_idx = load_all_prices()
    for sym in SECTORS:
        if sym not in prices:
            pr(f"  WARNING: {sym} missing")
    pr(f"  Sectors: {sum(1 for s in SECTORS if s in prices)}/{len(SECTORS)}")
    pr(f"  V8 extra: {[s for s in EXTRA_V8 if s in prices]}")
    pr(f"  Defensive: {[s for s in DEF_ETFS if s in prices]}")

    # Build DataFrames
    sect_close = pd.DataFrame(
        {s: prices[s] for s in SECTORS if s in prices}
    ).reindex(full_idx, method="ffill")
    sect_ret = sect_close.pct_change(fill_method=None).fillna(0.0)

    all_close = pd.DataFrame(
        {s: prices[s] for s in SECTORS + EXTRA_V8 if s in prices}
    ).reindex(full_idx, method="ffill")
    all_ret = all_close.pct_change(fill_method=None).fillna(0.0)

    def_close = pd.DataFrame(
        {s: prices[s] for s in DEF_ETFS if s in prices}
    ).reindex(full_idx, method="ffill")

    shy_ser = prices.get("SHY", pd.Series(dtype=float))

    pr("")
    pr("Loading book returns...")
    book_oos, strat_rets = load_book_returns()
    oos_idx = pd.bdate_range(OOS_START, OOS_END)
    pr(f"  Book OOS SR={sharpe(book_oos):+.3f}")
    pr(f"  s128 OOS SR={sharpe(strat_rets['s128'].reindex(oos_idx).fillna(0)):+.3f}")

    # Book without s128 (for replacement analysis)
    book_no_s128 = sum(
        strat_rets[sid].reindex(oos_idx).fillna(0.0) * wt
        for sid, _, wt in FINAL_BOOK if sid != "s128"
    )

    # ── Build all weight schedules ──────────────────────────────────────────
    pr("")
    pr("Building weight schedules...")

    pr("  V0 baseline...")
    ws_v0 = build_V0_baseline(sect_close)

    pr("  V1 inv-vol...")
    ws_v1 = build_V1_invvol(sect_close, sect_ret)

    pr("  V2 vol-target...")
    ws_v2 = build_V2_voltarget(sect_close, sect_ret)

    pr("  V3 blend lookbacks...")
    ws_v3 = build_V3_blend(sect_close)

    pr("  V4 skip-month...")
    ws_v4 = build_V4_skipmonth(sect_close)

    pr("  V5 hysteresis...")
    ws_v5 = build_V5_hysteresis(sect_close)

    pr("  V6 graded filter...")
    ws_v6 = build_V6_graded(sect_close)

    pr("  V7 defensive sleeve...")
    ws_v7, combined_v7 = build_V7_defensive(sect_close, def_close)

    pr("  V8 broad universe...")
    ws_v8 = build_V8_broad(all_close, all_ret)

    pr("  V9 diversity-aware...")
    ws_v9 = build_V9_divaware(sect_close, sect_ret, corr_thresh=0.85)

    pr("  V10 cash yield fix...")
    ws_v10 = build_V10_cashyield(sect_close, shy_ser)

    VARIANTS = [
        ("V0-base",     "Baseline: 9m/top-4/EW/cash-filter",              ws_v0,  sect_close),
        ("V1-invvol",   "Inverse-vol weights on selected sectors",         ws_v1,  sect_close),
        ("V2-voltgt",   "EW + 10% portfolio vol target (max 1.5x lev)",   ws_v2,  sect_close),
        ("V3-blend",    "Blended 3/6/9/12m momentum (avg rank)",          ws_v3,  sect_close),
        ("V4-skipm",    "Skip-month gap (9m ending 1m ago)",              ws_v4,  sect_close),
        ("V5-hyst",     "No-trade band: buy top-4, hold until top-6",     ws_v5,  sect_close),
        ("V6-graded",   "Per-holding abs-mom filter (partial cash)",       ws_v6,  sect_close),
        ("V7-def",      "Defensive sleeve (best of IEF/GLD/SHY in cash)", ws_v7,  combined_v7),
        ("V8-broad",    "Expanded universe (+EFA/EEM/EWJ)",               ws_v8,  all_close),
        ("V9-divaware", "Diversity-aware greedy selection (corr<0.85)",   ws_v9,  sect_close),
        ("V10-cshyld",  "Cash yield fix (hold SHY when out of sectors)",  ws_v10, pd.concat([sect_close, def_close[["SHY"]]], axis=1) if "SHY" in def_close.columns else sect_close),
    ]

    # ── Run all variants ────────────────────────────────────────────────────
    pr("")
    pr("Running engine for all variants...")
    results = []
    s128_oos = strat_rets["s128"].reindex(oos_idx).fillna(0.0)

    other_10_sids = [sid for sid, _, _ in FINAL_BOOK if sid != "s128"]

    for label, desc, ws, price_df in VARIANTS:
        pr(f"  {label}...")
        gross, to = portfolio_returns_from_weights(ws, price_df, OOS_START, OOS_END, execution_lag=1)
        net = apply_costs(gross, to)
        ann_to = float(to.sum() / max(len(to) / 252.0, 1.0))

        gr_sr, gr_cagr, gr_mdd = perf(gross)
        nt_sr, nt_cagr, nt_mdd = perf(net)

        net_oos = net.reindex(oos_idx).fillna(0.0)

        # Correlation to s128 baseline
        corr_s128 = float(net_oos.corr(s128_oos))

        # Book delta SR (swap s128 for this variant at s128's weight)
        book_with_v = book_no_s128 + net_oos * S128_WT
        delta_sr = sharpe(book_with_v) - sharpe(book_oos)

        # Max correlation to OTHER 10 book strategies (not s128)
        other_corrs = {
            sid: float(net_oos.corr(strat_rets[sid].reindex(oos_idx).fillna(0.0)))
            for sid in other_10_sids
        }
        max_other = max(other_corrs.values()) if other_corrs else 0.0
        best_other = max(other_corrs, key=other_corrs.get) if other_corrs else "n/a"

        # Per-year returns
        yr_rets = {}
        for yr in range(2017, 2025):
            r_yr = net_oos[net_oos.index.year == yr]
            yr_rets[yr] = float((1.0 + r_yr).prod() - 1.0) if not r_yr.empty else float("nan")

        results.append(dict(
            label=label, desc=desc,
            gr_sr=gr_sr, nt_sr=nt_sr, nt_cagr=nt_cagr, nt_mdd=nt_mdd, ann_to=ann_to,
            corr_s128=corr_s128, delta_sr=delta_sr,
            max_other=max_other, best_other=best_other,
            yr_rets=yr_rets,
        ))

    # ── Output ──────────────────────────────────────────────────────────────
    pr("")
    pr("=" * 120)
    pr("S128 VARIANTS  OOS 2017-07-03 to 2024-12-31")
    pr("=" * 120)
    pr(f"Book baseline: OOS SR=+1.054  CAGR=+11.7%  MDD=-15.5%")
    pr(f"s128 in book at 6.5% weight (OOS SR={sharpe(s128_oos):+.3f})")
    pr(f"SE(Sharpe,8yr)~=+/-0.35  |  gaps < 0.10 are ties")
    pr(f"'delta_SR_book' = book Sharpe change if we swap s128 for this variant at 6.5% weight")
    pr(f"'max_other_corr' = max corr to the OTHER 10 book strategies (not s128)")
    pr("")
    pr(f"{'Label':<14} {'GrSR':>6} {'NetSR':>6} {'CAGR':>7} {'MDD':>7} {'TO':>5}  {'corrS128':>8} {'dSR_bk':>7} {'maxOth':>7} {'bestOth':>6}")
    pr("-" * 120)

    for r in results:
        pr(
            f"  {r['label']:<12} {r['gr_sr']:+.3f} {r['nt_sr']:+.3f} "
            f"{r['nt_cagr']:+.1%} {r['nt_mdd']:+.1%} {r['ann_to']:>4.1f}x  "
            f"{r['corr_s128']:>+7.3f}  {r['delta_sr']:>+6.4f}  {r['max_other']:>+6.3f}  {r['best_other']:>6}"
        )

    pr("")
    pr("=" * 120)
    pr("PER-YEAR NET RETURNS")
    pr("=" * 120)
    years = list(range(2017, 2025))

    # Book and s128 per year
    book_yr = {}
    s128_yr = {}
    for yr in years:
        r_yr = book_oos[book_oos.index.year == yr]
        book_yr[yr] = float((1.0 + r_yr).prod() - 1.0) if not r_yr.empty else float("nan")
        r_s = s128_oos[s128_oos.index.year == yr]
        s128_yr[yr] = float((1.0 + r_s).prod() - 1.0) if not r_s.empty else float("nan")

    hdr = f"  {'Year':<5}" + f"  {'Book':>8}" + f"  {'s128-base':>10}" + \
          "".join(f"  {r['label']:>12}" for r in results[1:])  # skip V0 (it IS the baseline)
    pr(hdr)
    pr("  " + "-" * (len(hdr) - 2))
    for yr in years:
        row = f"  {yr:<5}  {book_yr[yr]:>+7.1%}  {s128_yr[yr]:>+9.1%}"
        for r in results[1:]:
            v = r["yr_rets"].get(yr, float("nan"))
            row += f"  {v:>+11.1%}" if not np.isnan(v) else f"  {'N/A':>11}"
        pr(row)

    pr("")
    pr("=" * 120)
    pr("SUMMARY AND RECOMMENDATIONS")
    pr("=" * 120)
    baseline_sr = results[0]["nt_sr"]
    pr(f"  Baseline (V0) net SR: {baseline_sr:+.3f}")
    pr("")

    # Sort by delta_sr (improvement to book)
    ranked = sorted(results[1:], key=lambda x: -x["delta_sr"])
    pr("  Ranked by book SR improvement (delta_SR_book, most improved first):")
    for r in ranked:
        beats = "BEATS" if r["nt_sr"] > baseline_sr + 0.05 else ("TIES" if r["nt_sr"] > baseline_sr - 0.05 else "LAGS")
        pr(f"    {r['label']:<14} NetSR={r['nt_sr']:+.3f} ({beats} V0)  "
           f"corrS128={r['corr_s128']:+.3f}  dSR_book={r['delta_sr']:+.4f}  "
           f"maxOth={r['max_other']:+.3f}({r['best_other']})")

    pr("")
    # Stackable combinations: which variants are independent enough to combine?
    pr("  Orthogonality to baseline (corr_s128 < 0.95 suggests genuine modification):")
    for r in results[1:]:
        note = "genuinely different" if r["corr_s128"] < 0.95 else "close variant"
        pr(f"    {r['label']:<14} corr_s128={r['corr_s128']:+.3f}  ({note})")

    pr("")
    pr("Done.")
    save_output()


if __name__ == "__main__":
    main()
