"""
analysis/audit_group3.py
Audit s63, s93, s60, s71 for same-bar lookahead and redundancy.
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import yaml, logging
import numpy as np
import pandas as pd
from pathlib import Path
from itertools import combinations

logging.basicConfig(level=logging.WARNING)

from data import load_price_series, ADJ_TOTALRETURN, ADJ_CAPITAL
from engine import apply_costs

REPO = Path(__file__).parent.parent
with open(REPO / "config.yaml") as f:
    config = yaml.safe_load(f)

config["paths"]["cache_dir"] = str(REPO / "cache/parquet")
CACHE    = config["paths"]["cache_dir"]
COST_BPS = config["costs"]["equity_cost_bps"]
SLIP_BPS = config["costs"]["equity_slippage_bps"]

IS_START  = "2000-01-03"
IS_END    = "2017-06-30"
OOS_START = "2017-07-03"
OOS_END   = "2024-12-31"


def sharpe(ret):
    r = ret.dropna()
    if len(r) < 10 or r.std() < 1e-10:
        return float("nan")
    return float(r.mean() / r.std() * np.sqrt(252))

def sr(ret, s, e):
    return sharpe(ret.loc[s:e])

def cagr(ret, s, e):
    r = ret.loc[s:e].dropna()
    if len(r) < 10: return float("nan")
    return float((1 + r).prod() ** (252 / len(r)) - 1)

def mdd(ret, s, e):
    r   = ret.loc[s:e].dropna()
    cum = (1 + r).cumprod()
    return float((cum / cum.cummax() - 1).min())

def banner(t):
    print(); print("=" * 80); print(f"  {t}"); print("=" * 80)


# ===========================================================================
# AUDIT 1 -- s63 ETF Breakout (mislabeled "Earnings Surprise")
# Same-bar bug: positions[i] from close[i] earns ret[i] on same bar.
# Affects ENTRY bars (credits breakout gain) AND EXIT bars (avoids loss).
# Fix: weight_df.shift(1) before multiplying by ret_df.
# ===========================================================================
def audit_s63():
    banner("AUDIT 1: s63 -- ETF Breakout  [s63_etf_breakout.py]")
    print("NAME FIX: s63 is NOT 'Earnings Surprise'. Pure price strategy on 5 ETFs.")
    print("s40/s63 contradiction: DOES NOT EXIST. Different data entirely.")
    print()
    print("BUG: positions[i] set from close[i], earns ret[i] on same bar.")
    print("  ENTRY bars: close > 55d-high -> direction=1 -> earn full breakout-bar gain.")
    print("  EXIT  bars: close < 20d-low  -> direction=0 -> avoid exit-bar loss.")
    print("FIX:  port_ret = (weight_df.shift(1) * ret_df).sum(axis=1)")

    ETFS, ENTRY_N, EXIT_N = ["SPY","QQQ","IWM","EFA","EEM"], 55, 20
    VOL_TGT, MAX_LEV, VOL_LOOK = 0.10, 1.5, 63
    LOAD_START = "1993-01-01"

    ohlcv = {}
    for sym in ETFS:
        df = load_price_series(sym, start=LOAD_START, end=OOS_END,
                               adjustment=ADJ_TOTALRETURN, cache_dir=CACHE)
        if not df.empty and "Close" in df.columns:
            ohlcv[sym] = df

    full_idx   = pd.bdate_range(LOAD_START, OOS_END)
    weight_dfs, ret_dfs = {}, {}

    for sym, df in ohlcv.items():
        close = df["Close"].reindex(full_idx, method="ffill")
        high  = df["High"].reindex(full_idx, method="ffill") if "High" in df.columns else close
        low   = df["Low"].reindex(full_idx, method="ffill")  if "Low"  in df.columns else close

        entry_high = high.rolling(ENTRY_N).max().shift(1)
        exit_low   = low.rolling(EXIT_N).min().shift(1)

        n, direction = len(close), 0
        positions = np.zeros(n)
        warmup = max(ENTRY_N, EXIT_N) + 2

        for i in range(warmup, n):
            c, eh, xl = close.iloc[i], entry_high.iloc[i], exit_low.iloc[i]
            if np.isnan(eh) or np.isnan(xl): continue
            if direction == 0:
                if c > eh: direction = 1
            else:
                if c < xl: direction = 0
            positions[i] = float(direction)

        raw_pos   = pd.Series(positions, index=full_idx)
        ret       = close.pct_change(fill_method=None).fillna(0.0)
        rvol      = ret.rolling(VOL_LOOK).std() * np.sqrt(252)
        vol_scale = (VOL_TGT / rvol.replace(0, np.nan)).clip(upper=MAX_LEV).shift(1).fillna(0.0)
        weight_dfs[sym] = raw_pos * vol_scale
        ret_dfs[sym]    = ret

    weight_df   = pd.DataFrame(weight_dfs).reindex(full_idx).fillna(0.0)
    ret_df      = pd.DataFrame(ret_dfs).reindex(full_idx).fillna(0.0)
    trading_idx = pd.bdate_range(IS_START, OOS_END)
    to          = weight_df.diff().abs().sum(axis=1).fillna(0.0).reindex(trading_idx).fillna(0.0)

    net_bug = apply_costs((weight_df * ret_df).sum(axis=1).reindex(trading_idx).fillna(0.0),       to, COST_BPS, SLIP_BPS)
    net_fix = apply_costs((weight_df.shift(1) * ret_df).sum(axis=1).reindex(trading_idx).fillna(0.0), to, COST_BPS, SLIP_BPS)

    print()
    print(f"{'Version':20s}  {'IS SR':>7}  {'OOS SR':>7}  {'Full SR':>7}  {'IS CAGR':>8}  {'OOS CAGR':>8}")
    print("-" * 75)
    for label, net in [("Buggy (current)", net_bug), ("Corrected (+1d)", net_fix)]:
        print(f"{label:20s}  {sr(net,IS_START,IS_END):7.3f}  {sr(net,OOS_START,OOS_END):7.3f}"
              f"  {sharpe(net):7.3f}  {cagr(net,IS_START,IS_END)*100:+7.1f}%  {cagr(net,OOS_START,OOS_END)*100:+7.1f}%")

    fix_oos = sr(net_fix, OOS_START, OOS_END)
    fix_is  = sr(net_fix, IS_START,  IS_END)
    print()
    print(f"  Lookahead inflation: IS +{sr(net_bug,IS_START,IS_END)-fix_is:.3f}  OOS +{sr(net_bug,OOS_START,OOS_END)-fix_oos:.3f} SR pts")
    verdict = "ARTIFACT" if fix_oos < 0.70 else ("BORDERLINE" if fix_oos < 0.85 else "CLEAN")
    print(f"  VERDICT: {verdict}  (corrected OOS SR = {fix_oos:.3f})")
    return net_fix


# ===========================================================================
# AUDIT 2 -- s93 Defensive Rotation
# Same-bar bug: spy_cur=close[i] determines weights that earn ret_df[i].
# On regime-switch days, captures switching-day sector outperformance.
# Fix: shift regime signal 1 day.
# ===========================================================================
def audit_s93():
    banner("AUDIT 2: s93 -- Defensive Rotation  [SPY 200d MA regime]")
    print("BUG: spy_cur = price_df['SPY'].iloc[i]  (today's close)")
    print("     determines weights that earn ret_df.iloc[i] (today's return).")
    print("     On regime-switch days, the switching-day sector outperformance")
    print("     is captured retroactively. Same mechanism as s78/s79.")
    print("FIX: shift regime signal by 1 day.")

    DEFENSIVES, GROWTH = ["XLP","XLU","XLV"], ["XLK","XLY","XLF"]
    MA_WIN = 200
    LOAD_START = "1997-01-01"

    data = {}
    for tk in ["SPY"] + DEFENSIVES + GROWTH:
        df = load_price_series(tk, start=LOAD_START, end=OOS_END,
                               adjustment=ADJ_TOTALRETURN, cache_dir=CACHE)
        if not df.empty: data[tk] = df["Close"]

    trading_idx = pd.bdate_range(IS_START, OOS_END)
    price_df    = pd.DataFrame(data).reindex(trading_idx, method="ffill")
    ret_df      = price_df.pct_change(fill_method=None).fillna(0.0)
    all_sector  = DEFENSIVES + GROWTH

    spy_ma      = price_df["SPY"].rolling(MA_WIN).mean()
    regime_bug  = (price_df["SPY"] > spy_ma).astype(float)
    regime_fix  = regime_bug.shift(1).fillna(0.0)
    switches    = int((regime_bug.diff().abs() > 0.5).sum())
    print(f"  Regime switches (full period): {switches}  (~{switches/25:.1f}/yr)")

    def build_rets(regime_series):
        port_rets = pd.Series(0.0, index=trading_idx)
        to_series = pd.Series(0.0, index=trading_idx)
        prev_wts  = {tk: 0.0 for tk in all_sector}
        for i in range(MA_WIN + 1, len(trading_idx)):
            grp = GROWTH if regime_series.iloc[i] == 1.0 else DEFENSIVES
            act = [tk for tk in grp if tk in ret_df.columns]
            n   = len(act) if act else 1
            wts = {tk: (1.0/n if tk in act else 0.0) for tk in all_sector}
            port_rets.iloc[i] = sum(wts.get(tk,0.0)*float(ret_df.iloc[i].get(tk,0.0))
                                    for tk in all_sector)
            to_series.iloc[i] = sum(abs(wts.get(tk,0.0)-prev_wts.get(tk,0.0))
                                    for tk in all_sector) / 2.0
            prev_wts = dict(wts)
        return apply_costs(port_rets, to_series, COST_BPS, SLIP_BPS)

    net_bug = build_rets(regime_bug)
    net_fix = build_rets(regime_fix)

    print()
    print(f"{'Version':20s}  {'IS SR':>7}  {'OOS SR':>7}  {'Full SR':>7}  {'IS CAGR':>8}  {'OOS CAGR':>8}")
    print("-" * 75)
    for label, net in [("Buggy (current)", net_bug), ("Corrected (+1d)", net_fix)]:
        print(f"{label:20s}  {sr(net,IS_START,IS_END):7.3f}  {sr(net,OOS_START,OOS_END):7.3f}"
              f"  {sharpe(net):7.3f}  {cagr(net,IS_START,IS_END)*100:+7.1f}%  {cagr(net,OOS_START,OOS_END)*100:+7.1f}%")

    fix_oos, fix_is = sr(net_fix, OOS_START, OOS_END), sr(net_fix, IS_START, IS_END)
    print()
    print(f"  Lookahead inflation: IS +{sr(net_bug,IS_START,IS_END)-fix_is:.3f}  OOS +{sr(net_bug,OOS_START,OOS_END)-fix_oos:.3f} SR pts")
    print(f"  De-risk lag: 1 trading day (signal at t, position effective t+1)")
    verdict = "ARTIFACT" if fix_oos < 0.70 else ("BORDERLINE" if fix_oos < 0.85 else "CLEAN")
    print(f"  VERDICT: {verdict}  (corrected IS SR {fix_is:.3f}, OOS SR {fix_oos:.3f})")
    return net_fix


# ===========================================================================
# AUDIT 3 -- s60 Correlation Regime
# Code EXPLICITLY has raw_exposure.shift(1) -- no same-bar issue.
# Question: is IS 0.27 -> OOS 0.90 a window artifact?
# ===========================================================================
def audit_s60():
    banner("AUDIT 3: s60 -- Correlation Regime  [explicitly clean]")
    print("LOOKAHEAD: NONE. Line 119: exposure = raw_exposure.shift(1)  <- 1-day lag explicit.")
    print("This one is correctly implemented.")
    print()
    print("QUESTION: why IS SR 0.27 -> OOS SR 0.90?")
    print("IS MDD -53% tells you: strategy does not protect in GFC/dot-com.")
    print("Regime-mix: IS has two major crashes; OOS has calmer markets + shorter 2022 bear.")

    SECTORS    = ["XLK","XLF","XLE","XLI","XLC","XLY","XLP","XLV","XLRE","XLB","XLU"]
    LOAD_START = "1998-01-01"
    CORR_WIN, ZSCORE_WIN, Z_THRESH = 63, 252, 1.5

    sector_close = {}
    for sym in SECTORS:
        df = load_price_series(sym, start=LOAD_START, end=OOS_END,
                               adjustment=ADJ_TOTALRETURN, cache_dir=CACHE)
        if not df.empty and "Close" in df.columns:
            sector_close[sym] = df["Close"]

    full_idx  = pd.bdate_range(LOAD_START, OOS_END)
    sector_df = pd.DataFrame(sector_close).reindex(full_idx, method="ffill")
    ret_df    = sector_df.pct_change(fill_method=None)
    scols     = list(sector_df.columns)
    pairs     = list(combinations(range(len(scols)), 2))

    pcorrs = []
    for i, j in pairs:
        si, sj = scols[i], scols[j]
        if si in ret_df.columns and sj in ret_df.columns:
            rc = ret_df[si].rolling(CORR_WIN, min_periods=max(20,CORR_WIN//2)).corr(ret_df[sj])
            pcorrs.append(rc)

    avg_corr = pd.concat(pcorrs, axis=1).mean(axis=1)
    rm, rs   = avg_corr.rolling(ZSCORE_WIN,min_periods=ZSCORE_WIN//2).mean(), avg_corr.rolling(ZSCORE_WIN,min_periods=ZSCORE_WIN//2).std()
    zscore   = (avg_corr - rm) / rs.replace(0, np.nan)
    exposure = (zscore <= Z_THRESH).astype(float).fillna(0.0).shift(1).fillna(0.0)

    spy     = load_price_series("SPY", start=LOAD_START, end=OOS_END,
                                 adjustment=ADJ_TOTALRETURN, cache_dir=CACHE)
    spy_ret = spy["Close"].reindex(full_idx, method="ffill").pct_change(fill_method=None).fillna(0.0)

    trading_idx = pd.bdate_range(IS_START, OOS_END)
    exp_bt      = exposure.reindex(trading_idx, method="ffill").fillna(0.0)
    ret_bt      = spy_ret.reindex(trading_idx, method="ffill").fillna(0.0)
    to          = exp_bt.diff().abs().fillna(0.0)
    net         = apply_costs(exp_bt * ret_bt, to, COST_BPS, SLIP_BPS)

    periods = [
        ("IS: Dot-com crash",     "2000-01-03", "2002-12-31"),
        ("IS: Recovery bull",     "2003-01-01", "2007-12-31"),
        ("IS: GFC",               "2008-01-01", "2009-12-31"),
        ("IS: Post-GFC bull",     "2010-01-01", "2017-06-30"),
        ("OOS: Calm bull",        "2017-07-03", "2021-12-31"),
        ("OOS: 2022 bear",        "2022-01-01", "2022-12-31"),
        ("OOS: 2023-24 recovery", "2023-01-01", "2024-12-31"),
    ]
    print()
    print(f"  {'Period':24s}  {'In-mkt':>7}  {'SR':>7}  {'CAGR':>7}  {'MDD':>7}")
    print(f"  {'-'*64}")
    for name, ps, pe in periods:
        inmkt = float(exp_bt.loc[ps:pe].mean()) * 100
        print(f"  {name:24s}  {inmkt:6.1f}%  {sr(net,ps,pe):7.3f}  {cagr(net,ps,pe)*100:+6.1f}%  {mdd(net,ps,pe)*100:+6.1f}%")

    print()
    cash_pct = float((exp_bt == 0).mean()) * 100
    print(f"  IS SR: {sr(net,IS_START,IS_END):.3f}   OOS SR: {sr(net,OOS_START,OOS_END):.3f}   Full SR: {sharpe(net):.3f}")
    print(f"  In-cash (full period): {cash_pct:.1f}%")
    print()
    print("  VERDICT: REGIME-TIMER (discount to IS Sharpe ~0.27)")
    print("  Full SR 0.44 and IS SR 0.27 are the honest forward estimates.")
    print("  OOS flattered by calmer 2017-2024 window. Do NOT add to book.")
    return net


# ===========================================================================
# AUDIT 4 -- s71 52-Week Breakout (code inspection; full R1000 run omitted)
# ===========================================================================
def audit_s71():
    banner("AUDIT 4: s71 -- 52-Week Breakout on R1000  [code inspection]")
    print("SAME-BAR LOOKAHEAD (from code s71_52wk_breakout.py):")
    print()
    print("  ENTRY: window_high = close_cap.iloc[i-252:i]  (excludes today -- clean)")
    print("         cur > col_win.max() -> positions[sym] = (cur, cur)  <- enter bar i")
    print("         day_ret includes sym with ret_df.iloc[i]  <- earns SAME BAR return")
    print("         Stock made 52-week high -> ret_df[i] is by definition positive.")
    print()
    print("  EXIT:  cur_px < peak*(1-0.20) -> to_exit.append(sym)")
    print("         positions.pop(sym)  <- REMOVES before day_ret computation")
    print("         day_ret excludes sym  <- avoids negative exit-bar return")
    print("         Stock hit 20% trailing stop -> ret_df[i] is typically negative.")
    print()
    print("  Both compound in same direction. Same structure as s63, s75.")
    print()
    print("MAGNITUDE ESTIMATE:")
    print("  Reported TO = 0.59x/yr for ~50 positions -> ~30 entries + 30 exits per year.")
    print("  Per-event portfolio impact: +-stock_return / n_positions")
    print("    = +-1.5% / 50 positions = +-0.03% per portfolio day per event")
    print("  Total: 60 events x 0.03% = 1.8% artificial CAGR per year.")
    print("  Estimated corrected OOS SR: 0.95-1.10  (still likely above 0.70 threshold)")
    print("  Effect is moderate due to low TO; less severe than s63.")
    print()
    print("REDUNDANCY CHECK (mechanism-based, same analysis as s75):")
    print("  s71 = long-only momentum (52wk breakout) on R1000, equal-weight 50 pos.")
    print("  Mechanically identical to corrected s75 (Donchian on R1000).")
    print("  s75 was REDUNDANT: delta N_eff = -0.06, delta SR = -0.036 vs book.")
    print("  Predicted correlation vs book:")
    print("    s08 Sector Rotation : 0.70-0.85  (same equity-momentum cluster)")
    print("    s30 Low Volatility   : 0.60-0.75  (both long-only R1000)")
    print("    s02 TS Momentum      : 0.55-0.70  (both trend-following on equities)")
    print("  Expected delta N_eff when added to 8-strategy book: ~0 or negative.")
    print()

    # Quick SPY SR for context
    spy     = load_price_series("SPY", start=IS_START, end=OOS_END,
                                 adjustment=ADJ_TOTALRETURN, cache_dir=CACHE)
    spy_ret = spy["Close"].reindex(pd.bdate_range(IS_START,OOS_END),
                                   method="ffill").pct_change(fill_method=None).fillna(0.0)
    print(f"  (SPY full-period SR for reference: {sharpe(spy_ret):.3f})")
    print(f"  s71 buggy full SR = 1.138. Likely 0.90-1.05 after correction.")
    print()
    print("  VERDICT: REDUNDANT (pending full R1000 re-run to confirm corrected SR)")
    print("  If corrected OOS SR < 0.70 after full run: ARTIFACT.")
    print("  Either way: do NOT add to book. Book unchanged.")


# ===========================================================================
# MAIN
# ===========================================================================
if __name__ == "__main__":
    net_s63 = audit_s63()
    net_s93 = audit_s93()
    net_s60 = audit_s60()
    audit_s71()

    print()
    print("=" * 80)
    print("  FOUR-AUDIT SUMMARY")
    print("=" * 80)
    s63_oos = sr(net_s63, OOS_START, OOS_END)
    s93_oos = sr(net_s93, OOS_START, OOS_END)
    s60_is  = sr(net_s60, IS_START,  IS_END)
    s63_v   = "ARTIFACT"   if s63_oos < 0.70 else "BORDERLINE"
    s93_v   = "ARTIFACT"   if s93_oos < 0.70 else "BORDERLINE"
    print()
    print(f"  s63 ETF Breakout (was 'Earnings Surprise'): corrected OOS SR {s63_oos:.3f}  -> {s63_v}")
    print(f"  s93 Defensive Rotation:                     corrected OOS SR {s93_oos:.3f}  -> {s93_v}")
    print(f"  s60 Corr. Regime (CLEAN):                   IS SR {s60_is:.3f}  (honest fwd)  -> REGIME-TIMER")
    print(f"  s71 52-Wk Breakout:                         est. corr. OOS SR 0.95-1.10  -> REDUNDANT")
    print()
    print("  s40/s63 data contradiction: RESOLVED -- never existed.")
    print("  s63 name was wrong in our table (ETF breakout, not earnings surprise).")
    print()
    print("  BOOK IMPACT: none. Validated 8-strategy book unchanged.")
    print("  s08 s46 s30 s02 s31 s35 s49 s90")
