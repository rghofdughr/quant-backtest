"""
run_s128_variants_v2.py  — Four follow-up tasks on s128 variants

Task 1: V6-graded on official runner
  Compare official s128 (NetSR +1.016) vs official V6 (same schedule, graded filter).

Task 2: Block-bootstrap SE on SR deltas
  Block-bootstrap monthly returns (block=12m, N=2000) for all 11 variants from run_s128_variants.py.
  Report distribution of each (V_i - V0) SR delta: mean, SE, 5/95pct, p(delta>0).

Task 3: Decompose V6's edge ex-2022
  V6 vs V0 with 2022 excluded. Per-year contribution table.

Task 4: V5+V6 hybrid
  Hysteresis band (buy top-4, hold until top-6) combined with graded per-holding filter.
  Standalone metrics + vs individual V5 and V6.
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

SECTORS = ["XLK", "XLF", "XLE", "XLV", "XLI", "XLY", "XLP", "XLRE", "XLB", "XLU", "XLC"]
TOP_N   = 4
LB_DAYS = 189   # 9 months
VOL_LB  = 63
BAND    = 6     # V5 hysteresis hold-band

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

N_BOOT     = 2000
BLOCK_MONTHS = 12   # 1-year blocks to capture annual seasonality
RNG = np.random.default_rng(42)

_outlines = []

def pr(s=""):
    _outlines.append(str(s))
    try:
        print(s)
    except UnicodeEncodeError:
        print(str(s).encode("ascii", "replace").decode("ascii"))

def save_output():
    p = RESULTS_DIR / "s128_variants_v2_output.txt"
    with open(p, "w", encoding="utf-8") as f:
        f.write("\n".join(_outlines) + "\n")

# ── Metric helpers ────────────────────────────────────────────────────────────

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

def to_monthly(daily_ret):
    """Convert daily returns to monthly (month-end resample)."""
    return (1 + daily_ret).resample("ME").prod() - 1

# ── Signal helpers ─────────────────────────────────────────────────────────────

def _reb_dates():
    return pd.date_range(PRICE_START, OOS_END, freq="BME")

def _snap(close_df, d):
    avail = close_df.index[close_df.index <= d]
    return len(avail) - 1

def mom_scores(close_df, snap, lb):
    now  = snap
    past = max(0, snap - lb)
    p1   = close_df.iloc[now]
    p0   = close_df.iloc[past]
    return ((p1 / p0) - 1.0).dropna().sort_values(ascending=False)

# ── Weight schedule builders ───────────────────────────────────────────────────

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
        pos = [sym for sym in m.index[:TOP_N] if float(m[sym]) > 0]
        ws[d] = {sym: 1.0 / len(pos) for sym in pos} if pos else {}
    return ws

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
        top6 = set(m.index[:BAND].tolist())
        held = {h for h in current if h in top6}
        for sym in m.index[:TOP_N]:
            if len(held) >= TOP_N:
                break
            held.add(sym)
        current = held
        n = len(current)
        ws[d] = {sym: 1.0 / n for sym in current} if n > 0 else {}
    return ws

def build_V5V6_hybrid(close_df):
    """
    Hysteresis (hold until top-6) AND graded filter (drop negative momentum).
    Retention rule: must be in top-6 AND have positive 9m return.
    Buy rule: new additions from top-4, positive momentum only.
    Cash rule: if no positive-momentum sectors available.
    """
    ws = {}
    current = set()
    for d in _reb_dates():
        s = _snap(close_df, d)
        if s < LB_DAYS + 2:
            ws[d] = {}
            current = set()
            continue
        m = mom_scores(close_df, s, LB_DAYS)
        if m.empty:
            ws[d] = {}
            current = set()
            continue

        # Global cash check: if the best sector (by momentum rank) is negative,
        # no point running the band logic — go to cash and reset.
        if float(m.iloc[0]) <= 0:
            ws[d] = {}
            current = set()
            continue

        top6 = set(m.index[:BAND].tolist())
        pos_set = {sym for sym in m.index if float(m[sym]) > 0}

        # Retain: must be in top-6 AND have positive 9m return
        held = {h for h in current if h in top6 and h in pos_set}

        # Add new: from top-4 positive momentum sectors only
        for sym in m.index[:TOP_N]:
            if len(held) >= TOP_N:
                break
            if sym in pos_set:
                held.add(sym)

        current = held
        if current:
            n = len(current)
            ws[d] = {sym: 1.0 / n for sym in current}
        else:
            ws[d] = {}
    return ws

# ── Block bootstrap ────────────────────────────────────────────────────────────

def block_bootstrap_sr(monthly_ret, n_boot=N_BOOT, block_size=BLOCK_MONTHS):
    """
    Circular block bootstrap on monthly returns.
    Returns array of N_boot Sharpe estimates (annualised from monthly).
    """
    m = monthly_ret.values
    n = len(m)
    n_blocks = int(np.ceil(n / block_size))
    srs = np.empty(n_boot)
    for i in range(n_boot):
        starts = RNG.integers(0, n, size=n_blocks)
        blocks = [m[s: s + block_size] if s + block_size <= n
                  else np.concatenate([m[s:], m[:s + block_size - n]])
                  for s in starts]
        sample = np.concatenate(blocks)[:n]
        mu  = np.mean(sample)
        std = np.std(sample, ddof=1)
        if std < 1e-10:
            srs[i] = 0.0
        else:
            srs[i] = mu / std * np.sqrt(12)   # monthly -> annualised
    return srs

# ── Data and book loading ──────────────────────────────────────────────────────

def load_sector_prices():
    full_idx = pd.bdate_range(PRICE_START, OOS_END)
    prices = {}
    for sym in SECTORS:
        df = load_price_series(sym, PRICE_START, OOS_END, ADJ_TOTALRETURN, str(CACHE_DIR))
        if not df.empty and "Close" in df.columns:
            prices[sym] = df["Close"].reindex(full_idx, method="ffill")
    close_df = pd.DataFrame(prices).reindex(full_idx, method="ffill")
    return close_df

def load_official_strategy(mname, cfg):
    mod = importlib.import_module(mname)
    return mod.run(cfg)

def make_cfg(start=IS_START, end=OOS_END):
    cfg_path = PROJECT_ROOT / "config.yaml"
    with open(cfg_path) as f:
        cfg = yaml.safe_load(f)
    cfg["backtest"]["start_date"] = start
    cfg["backtest"]["end_date"]   = end
    cfg["paths"]["cache_dir"]     = str(CACHE_DIR)
    return cfg

# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    oos_idx = pd.bdate_range(OOS_START, OOS_END)

    # ── Load prices and build BME weight schedules ────────────────────────────
    pr("Loading sector prices...")
    sect_close = load_sector_prices()
    pr(f"  {sect_close.shape[1]} sectors loaded")

    pr("Building BME weight schedules...")
    ws_v0 = build_V0_baseline(sect_close)
    ws_v6 = build_V6_graded(sect_close)
    ws_v5 = build_V5_hysteresis(sect_close)
    ws_h  = build_V5V6_hybrid(sect_close)
    pr("  Done.")

    pr("Running BME engine for V0, V5, V6, hybrid...")
    gr0, to0 = portfolio_returns_from_weights(ws_v0, sect_close, OOS_START, OOS_END, execution_lag=1)
    gr6, to6 = portfolio_returns_from_weights(ws_v6, sect_close, OOS_START, OOS_END, execution_lag=1)
    gr5, to5 = portfolio_returns_from_weights(ws_v5, sect_close, OOS_START, OOS_END, execution_lag=1)
    grh, toh = portfolio_returns_from_weights(ws_h,  sect_close, OOS_START, OOS_END, execution_lag=1)

    net0 = apply_costs(gr0, to0).reindex(oos_idx).fillna(0.0)
    net6 = apply_costs(gr6, to6).reindex(oos_idx).fillna(0.0)
    net5 = apply_costs(gr5, to5).reindex(oos_idx).fillna(0.0)
    neth = apply_costs(grh, toh).reindex(oos_idx).fillna(0.0)

    ann_to0 = float(to0.sum() / max(len(to0)/252, 1))
    ann_to6 = float(to6.sum() / max(len(to6)/252, 1))
    ann_to5 = float(to5.sum() / max(len(to5)/252, 1))
    ann_toh = float(toh.sum() / max(len(toh)/252, 1))

    # ── Task 1: Official runner ────────────────────────────────────────────────
    pr("")
    pr("=" * 90)
    pr("TASK 1 — V6-graded on official runner (identical schedule to s128)")
    pr("=" * 90)
    pr("Running official s128...")
    cfg = make_cfg()
    res_s128 = load_official_strategy("strategies.s128_sector_rotation_9m", cfg)
    ret_s128 = res_s128["returns"].reindex(oos_idx).fillna(0.0)

    pr("Running official s128-V6...")
    res_v6 = load_official_strategy("strategies.s128_v6_graded", cfg)
    ret_v6_off = res_v6["returns"].reindex(oos_idx).fillna(0.0)

    sr_s128     = sharpe(ret_s128)
    cagr_s128   = perf(ret_s128)[1]
    mdd_s128    = perf(ret_s128)[2]

    sr_v6_off   = sharpe(ret_v6_off)
    cagr_v6_off = perf(ret_v6_off)[1]
    mdd_v6_off  = perf(ret_v6_off)[2]

    corr_official = float(ret_s128.corr(ret_v6_off))

    pr(f"  {'Strategy':<20} {'NetSR':>7} {'CAGR':>8} {'MDD':>8} {'Corr':>8}")
    pr(f"  {'-'*55}")
    pr(f"  {'s128 (official)':20} {sr_s128:>+7.3f} {cagr_s128:>+7.1%} {mdd_s128:>+7.1%}")
    pr(f"  {'s128-V6 (official)':20} {sr_v6_off:>+7.3f} {cagr_v6_off:>+7.1%} {mdd_v6_off:>+7.1%} {corr_official:>+7.3f}")
    pr(f"  {'V6 - s128 delta':20} {sr_v6_off - sr_s128:>+7.3f} {cagr_v6_off - cagr_s128:>+7.1%}")
    pr("")

    # Per-year for the official pair
    pr("  Per-year net returns (official runner):")
    pr(f"  {'Year':<6} {'s128':>9} {'V6-off':>9} {'delta':>9}")
    pr(f"  {'-'*36}")
    for yr in range(2017, 2025):
        s_yr = ret_s128[ret_s128.index.year == yr]
        v_yr = ret_v6_off[ret_v6_off.index.year == yr]
        s_r  = float((1+s_yr).prod()-1) if not s_yr.empty else float("nan")
        v_r  = float((1+v_yr).prod()-1) if not v_yr.empty else float("nan")
        d_r  = v_r - s_r if not (np.isnan(s_r) or np.isnan(v_r)) else float("nan")
        pr(f"  {yr:<6} {s_r:>+8.1%} {v_r:>+8.1%} {d_r:>+8.1%}")

    delta_sr_off = sr_v6_off - sr_s128
    verdict = "BEATS s128" if delta_sr_off > 0.05 else ("TIES s128" if delta_sr_off > -0.05 else "LAGS s128")
    pr(f"")
    pr(f"  VERDICT: V6 {verdict}  (delta SR = {delta_sr_off:+.3f}, SE~+/-0.35)")

    # ── Task 2: Block-bootstrap SE on SR deltas ────────────────────────────────
    pr("")
    pr("=" * 90)
    pr("TASK 2 — Block-bootstrap SE on SR deltas (BME variants)")
    pr(f"         N_boot={N_BOOT}, block={BLOCK_MONTHS}m circular, seed=42")
    pr("=" * 90)

    # Load all 11 variant net returns from Task 1 results (re-run them)
    pr("Re-running all 11 BME variants for bootstrap...")
    from analysis.run_s128_variants import (
        build_V1_invvol, build_V2_voltarget, build_V3_blend, build_V4_skipmonth,
        build_V7_defensive, build_V8_broad, build_V9_divaware, build_V10_cashyield,
        load_all_prices, DEF_ETFS, EXTRA_V8, SECTORS as SECT2,
    )
    # reload price data with the full universe
    prices_all, full_idx = load_all_prices()
    from analysis.run_s128_variants import PRICE_START as PS2
    full_idx2 = pd.bdate_range(PS2, OOS_END)

    def mk_df(syms):
        return pd.DataFrame(
            {s: prices_all[s] for s in syms if s in prices_all}
        ).reindex(full_idx2, method="ffill")

    sect_c2  = mk_df(SECT2)
    sect_r2  = sect_c2.pct_change(fill_method=None).fillna(0.0)
    all_c2   = mk_df(SECT2 + EXTRA_V8)
    all_r2   = all_c2.pct_change(fill_method=None).fillna(0.0)
    def_c2   = mk_df(DEF_ETFS)
    shy_ser2 = prices_all.get("SHY", pd.Series(dtype=float))

    ws_v1 = build_V1_invvol(sect_c2, sect_r2)
    ws_v2 = build_V2_voltarget(sect_c2, sect_r2)
    ws_v3 = build_V3_blend(sect_c2)
    ws_v4 = build_V4_skipmonth(sect_c2)
    ws_v7, comb_v7 = build_V7_defensive(sect_c2, def_c2)
    ws_v8 = build_V8_broad(all_c2, all_r2)
    ws_v9 = build_V9_divaware(sect_c2, sect_r2, corr_thresh=0.85)
    ws_v10 = build_V10_cashyield(sect_c2, shy_ser2)
    sh_col = pd.concat([sect_c2, def_c2[["SHY"]]], axis=1) if "SHY" in def_c2.columns else sect_c2

    def run_variant(ws, price_df, label):
        g, t = portfolio_returns_from_weights(ws, price_df, OOS_START, OOS_END, execution_lag=1)
        n = apply_costs(g, t).reindex(oos_idx).fillna(0.0)
        pr(f"    {label}...")
        return n

    pr("  Running engine for variants 1-10...")
    v_net = {
        "V0-base":     net0,
        "V1-invvol":   run_variant(ws_v1,  sect_c2,  "V1"),
        "V2-voltgt":   run_variant(ws_v2,  sect_c2,  "V2"),
        "V3-blend":    run_variant(ws_v3,  sect_c2,  "V3"),
        "V4-skipm":    run_variant(ws_v4,  sect_c2,  "V4"),
        "V5-hyst":     net5,
        "V6-graded":   net6,
        "V7-def":      run_variant(ws_v7,  comb_v7,  "V7"),
        "V8-broad":    run_variant(ws_v8,  all_c2,   "V8"),
        "V9-divaware": run_variant(ws_v9,  sect_c2,  "V9"),
        "V10-cshyld":  run_variant(ws_v10, sh_col,   "V10"),
        "V5V6-hybrid": neth,
    }

    # Monthly returns for bootstrap
    mthly = {k: to_monthly(v) for k, v in v_net.items()}
    sr_v0_boot = block_bootstrap_sr(mthly["V0-base"])

    pr("")
    pr(f"  {'Variant':<14} {'NetSR':>7} {'Boot_mean':>10} {'Boot_SE':>8} {'P95lo':>8} {'P95hi':>8} {'P(V>V0)':>8}")
    pr(f"  {'-'*68}")
    boot_results = {}
    for label, net_r in v_net.items():
        sr_vi = sharpe(net_r)
        sr_vi_boot = block_bootstrap_sr(mthly[label])
        delta_boot = sr_vi_boot - sr_v0_boot
        mean_d  = float(np.mean(delta_boot))
        se_d    = float(np.std(delta_boot, ddof=1))
        lo_d    = float(np.percentile(delta_boot, 5))
        hi_d    = float(np.percentile(delta_boot, 95))
        p_pos   = float(np.mean(delta_boot > 0))
        boot_results[label] = dict(sr=sr_vi, mean_d=mean_d, se=se_d, lo=lo_d, hi=hi_d, p_pos=p_pos)
        marker = " **" if p_pos > 0.70 else ("  *" if p_pos > 0.55 else "")
        pr(f"  {label:<14} {sr_vi:>+7.3f} {mean_d:>+9.3f}  {se_d:>7.3f}  {lo_d:>+7.3f}  {hi_d:>+7.3f}  {p_pos:>7.2%}{marker}")

    pr("")
    pr("  ** = P(V>V0) > 70%  |  * = P(V>V0) > 55%")
    pr("  Delta is V_i minus V0 bootstrap distribution (positive = outperforms V0).")
    pr("  V0-base shows all zeros by construction.")

    # ── Task 3: V6 edge decomposition ex-2022 ────────────────────────────────
    pr("")
    pr("=" * 90)
    pr("TASK 3 — V6 edge decomposition: per-year and ex-2022")
    pr("=" * 90)

    years = list(range(2017, 2025))
    yr_v0 = {}
    yr_v6 = {}
    for yr in years:
        r0 = net0[net0.index.year == yr]
        r6 = net6[net6.index.year == yr]
        yr_v0[yr] = float((1+r0).prod()-1) if not r0.empty else float("nan")
        yr_v6[yr] = float((1+r6).prod()-1) if not r6.empty else float("nan")

    pr(f"  {'Year':<6} {'V0 ret':>9} {'V6 ret':>9} {'delta':>9}  Note")
    pr(f"  {'-'*60}")
    for yr in years:
        v0r = yr_v0[yr]
        v6r = yr_v6[yr]
        d   = v6r - v0r if not (np.isnan(v0r) or np.isnan(v6r)) else float("nan")
        note = " <-- 2022 energy divergence" if yr == 2022 else ""
        pr(f"  {yr:<6} {v0r:>+8.1%} {v6r:>+8.1%} {d:>+8.1%}{note}")

    # Ex-2022 SR comparison
    mask_ex22 = (net0.index.year != 2022)
    sr0_full   = sharpe(net0)
    sr6_full   = sharpe(net6)
    sr0_ex22   = sharpe(net0[mask_ex22])
    sr6_ex22   = sharpe(net6[mask_ex22])
    sr0_22only = sharpe(net0[net0.index.year == 2022])
    sr6_22only = sharpe(net6[net6.index.year == 2022])

    pr("")
    pr(f"  {'Window':<25} {'V0 SR':>8} {'V6 SR':>8} {'V6-V0':>8}")
    pr(f"  {'-'*53}")
    pr(f"  {'Full 2017-2024':25} {sr0_full:>+8.3f} {sr6_full:>+8.3f} {sr6_full-sr0_full:>+8.3f}")
    pr(f"  {'Ex-2022 (2017-2021,2023-24)':25} {sr0_ex22:>+8.3f} {sr6_ex22:>+8.3f} {sr6_ex22-sr0_ex22:>+8.3f}")
    pr(f"  {'2022 only':25} {sr0_22only:>+8.3f} {sr6_22only:>+8.3f} {sr6_22only-sr0_22only:>+8.3f}")

    # Bootstrap ex-2022 delta
    mask_ex22_m = to_monthly(net0[mask_ex22]).index
    m0_ex22 = to_monthly(net0[mask_ex22])
    m6_ex22 = to_monthly(net6[mask_ex22])
    boot0_ex = block_bootstrap_sr(m0_ex22)
    boot6_ex = block_bootstrap_sr(m6_ex22)
    d_boot_ex = boot6_ex - boot0_ex
    pr(f"")
    pr(f"  Bootstrap (ex-2022) delta SR:  mean={np.mean(d_boot_ex):+.3f}  SE={np.std(d_boot_ex, ddof=1):.3f}  "
       f"P(V6>V0)={np.mean(d_boot_ex>0):.2%}")
    pr(f"  Interpretation: {'V6 edge is GENERAL (not 2022-only)' if np.mean(d_boot_ex>0) > 0.55 else 'V6 edge is CONCENTRATED in 2022'}")

    # ── Task 4: V5+V6 hybrid standalone ─────────────────────────────────────
    pr("")
    pr("=" * 90)
    pr("TASK 4 — V5+V6 hybrid (hysteresis band + graded per-holding filter)")
    pr("=" * 90)

    # How often does V5 band conflict with V6 filter?
    # Count rebalances where V5 would keep a sector that V6 would drop
    n_v5_keeps = 0
    n_v5_v6_conflict = 0
    current_hyster = set()
    for d in _reb_dates():
        s = _snap(sect_close, d)
        if s < LB_DAYS + 2:
            current_hyster = set()
            continue
        m = mom_scores(sect_close, s, LB_DAYS)
        if m.empty or float(m.iloc[0]) <= 0:
            current_hyster = set()
            continue
        top6 = set(m.index[:BAND].tolist())
        neg_mom = {sym for sym in m.index if float(m[sym]) <= 0}
        v5_kept = {h for h in current_hyster if h in top6}
        conflict = v5_kept & neg_mom  # sectors V5 keeps but V6 would drop
        if conflict:
            n_v5_v6_conflict += 1
        held = set(v5_kept)
        for sym in m.index[:TOP_N]:
            if len(held) >= TOP_N:
                break
            held.add(sym)
        current_hyster = held

    pr(f"  V5/V6 interaction: {n_v5_v6_conflict} rebalance months where V5 band keeps a sector V6 filter would drop")

    sr0, cagr0, mdd0 = perf(net0)
    sr5, cagr5, mdd5 = perf(net5)
    sr6, cagr6, mdd6 = perf(net6)
    srh, cagrh, mddh = perf(neth)

    pr("")
    pr(f"  {'Strategy':<16} {'NetSR':>7} {'CAGR':>8} {'MDD':>8} {'TO':>5}  Notes")
    pr(f"  {'-'*65}")
    pr(f"  {'V0-baseline':16} {sr0:>+7.3f} {cagr0:>+7.1%} {mdd0:>+7.1%} {ann_to0:>4.1f}x  Reference")
    pr(f"  {'V5-hyst':16} {sr5:>+7.3f} {cagr5:>+7.1%} {mdd5:>+7.1%} {ann_to5:>4.1f}x  Hysteresis only")
    pr(f"  {'V6-graded':16} {sr6:>+7.3f} {cagr6:>+7.1%} {mdd6:>+7.1%} {ann_to6:>4.1f}x  Graded filter only")
    pr(f"  {'V5V6-hybrid':16} {srh:>+7.3f} {cagrh:>+7.1%} {mddh:>+7.1%} {ann_toh:>4.1f}x  Both combined")

    # Correlation matrix for the four
    rets_mat = pd.DataFrame({"V0": net0, "V5": net5, "V6": net6, "V5V6": neth})
    corr_mat = rets_mat.corr()
    pr("")
    pr("  Correlation matrix:")
    pr(f"  {'':8}" + "".join(f"  {c:>7}" for c in corr_mat.columns))
    for row in corr_mat.index:
        pr(f"  {row:<8}" + "".join(f"  {corr_mat.loc[row,c]:>+7.3f}" for c in corr_mat.columns))

    # Per-year for hybrid
    pr("")
    pr(f"  Per-year net returns:")
    pr(f"  {'Year':<6} {'V0':>9} {'V5':>9} {'V6':>9} {'V5V6':>9}")
    pr(f"  {'-'*46}")
    for yr in years:
        def yr_ret(r):
            s = r[r.index.year == yr]
            return float((1+s).prod()-1) if not s.empty else float("nan")
        pr(f"  {yr:<6} {yr_ret(net0):>+8.1%} {yr_ret(net5):>+8.1%} {yr_ret(net6):>+8.1%} {yr_ret(neth):>+8.1%}")

    # Bootstrap delta for hybrid vs V0 and vs V6
    m0  = to_monthly(net0)
    m5  = to_monthly(net5)
    m6  = to_monthly(net6)
    mh  = to_monthly(neth)
    b0  = block_bootstrap_sr(m0)
    b5  = block_bootstrap_sr(m5)
    b6  = block_bootstrap_sr(m6)
    bh  = block_bootstrap_sr(mh)
    dh0 = bh - b0
    dh6 = bh - b6
    pr("")
    pr(f"  Bootstrap delta V5V6 vs V0: mean={np.mean(dh0):+.3f}  SE={np.std(dh0,ddof=1):.3f}  P(hybrid>V0)={np.mean(dh0>0):.2%}")
    pr(f"  Bootstrap delta V5V6 vs V6: mean={np.mean(dh6):+.3f}  SE={np.std(dh6,ddof=1):.3f}  P(hybrid>V6)={np.mean(dh6>0):.2%}")
    pr(f"  Bootstrap delta V5   vs V0: mean={np.mean(b5-b0):+.3f}  SE={np.std(b5-b0,ddof=1):.3f}  P(V5>V0)={np.mean(b5>b0):.2%}")
    pr(f"  Bootstrap delta V6   vs V0: mean={np.mean(b6-b0):+.3f}  SE={np.std(b6-b0,ddof=1):.3f}  P(V6>V0)={np.mean(b6>b0):.2%}")

    pr("")
    pr("=" * 90)
    pr("OVERALL CONCLUSIONS")
    pr("=" * 90)
    pr(f"  1. Official V6 vs s128: delta SR = {delta_sr_off:+.3f}  ({verdict})")
    v6_boot_full = boot_results.get("V6-graded", {})
    pr(f"  2. Bootstrap P(V6>V0 BME): {v6_boot_full.get('p_pos', float('nan')):.2%}  mean delta = {v6_boot_full.get('mean_d', float('nan')):+.3f}  SE = {v6_boot_full.get('se', float('nan')):.3f}")
    pr(f"  3. V6 ex-2022: delta SR = {sr6_ex22-sr0_ex22:+.3f}  P(V6>V0 ex-2022) = {np.mean(d_boot_ex>0):.2%}")
    pr(f"  4. Hybrid vs V0: P(hybrid>V0) = {np.mean(dh0>0):.2%}  vs V6: P(hybrid>V6) = {np.mean(dh6>0):.2%}")
    pr("")
    pr("  FLAGGED for future work: universe breadth (industry groups / international sectors)")
    pr("  Rationale: book corr 0.89 caps s128 weight at 6.5%; V8 only weakly probed this.")
    pr("  Next pass: GICs industry-group ETFs (XLK sub-sectors: SOXX, HACK, CIBR etc.) or")
    pr("  regional international (EFA, EM, AAXJ). Not part of this pass.")
    pr("")
    pr("Done.")
    save_output()


if __name__ == "__main__":
    main()
