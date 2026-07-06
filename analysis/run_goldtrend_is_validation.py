"""
run_goldtrend_is_validation.py

IS validation for O-goldtrend: 2000-01-01 to 2017-06-30.
FROZEN parameters from OOS run. No retuning.

Signal (identical specification to OOS):
  if GLD price > 126d SMA:  hold 100% GLD
  elif IEF price > 210d SMA: hold 100% IEF
  else:                       cash (0% invested)
  Monthly rebalance (BME). 20bp round-trip. Next-session execution lag = 1.

Implementation note on pre-data period:
  OOS run used ma_signal() which returns 1 (hold) for insufficient history.
  This NEVER fired OOS: GLD had 12.6yr of history by 2017-07-03.
  IS uses ma_signal_strict() which returns 0 (skip) for insufficient history.
  The two functions are IDENTICAL whenever data >= window+1 days.
  Ambiguity: use a gold spot/futures proxy pre-2004? Decision: no proxy.
  Reason: introducing a non-GLD price series would change the signal asset,
  not just the signal lookback. The strategy is defined on GLD. Pre-inception,
  the strategy falls through to IEF check (or cash). This is the conservative
  choice and avoids a backhanded splice that could flatter IS performance.

Data coverage:
  GLD: 2004-11-18 onward  (126d warmup done ~2005-05-26)
  IEF: 2002-07-26 onward  (210d warmup done ~2003-06-24)
  2000-2002-07: both unavailable -> cash
  2002-07 to 2004-11: IEF available only (or cash)
  2004-11 onwards: full strategy logic active
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

# ── Date windows ─────────────────────────────────────────────────────────────
PRICE_START = "2000-01-01"
IS_START    = "2000-01-01"
IS_END      = "2017-06-30"
OOS_START   = "2017-07-03"
OOS_END     = "2024-12-31"

# ── FROZEN parameters from OOS run ──────────────────────────────────────────
GLD_MA  = 126   # 6-month SMA for gold trend signal
IEF_MA  = 210   # 10-month SMA for IEF fallback signal
FREQ    = "BME" # monthly rebalance frequency

# Known data inception dates
GLD_INCEP = pd.Timestamp("2004-11-18")
IEF_INCEP = pd.Timestamp("2002-07-26")

# ── Book weights (same as OOS run) ──────────────────────────────────────────
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

# ── OOS reference numbers (from prior run — for side-by-side comparison) ────
OOS_REF = dict(
    gr_sr=0.689, nt_sr=0.586, cagr=0.070, mdd=-0.251, ann_to=6.1,
    corr_book=0.275, corr_s46=0.481, dneff=0.3825, dsr=0.0162,
    single_yr_flag="2024 = +33.5% (42% of positive return)",
)

_outlines = []

def pr(s=""):
    _outlines.append(str(s))
    try:
        print(s)
    except UnicodeEncodeError:
        print(str(s).encode("ascii", "replace").decode("ascii"))

def save_output():
    path = RESULTS_DIR / "goldtrend_is_validation.txt"
    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(_outlines) + "\n")
    pr(f"  Saved to {path}")

# ── Metric helpers ───────────────────────────────────────────────────────────

def sharpe(ret):
    r = ret.dropna()
    if len(r) == 0 or r.std() < 1e-10:
        return 0.0
    return float(r.mean() / r.std() * np.sqrt(252))

def perf(ret):
    """Returns (sharpe, cagr, mdd)."""
    r = ret.dropna()
    if len(r) == 0:
        return 0.0, 0.0, 0.0
    sr = sharpe(r)
    years = len(r) / 252.0
    cum  = (1.0 + r).cumprod()
    cagr = float(cum.iloc[-1] ** (1.0 / max(years, 0.01)) - 1.0)
    mdd  = float(((cum - cum.cummax()) / cum.cummax()).min())
    return sr, cagr, mdd

def neff(corr_mat):
    eigs = np.maximum(np.linalg.eigvalsh(corr_mat), 0.0)
    s, s2 = eigs.sum(), (eigs**2).sum()
    return float(s*s/s2) if s2 > 1e-12 else 1.0

# ── Signal helper (strict: 0 for insufficient history) ──────────────────────

def ma_signal_strict(prices, sym, date, window):
    """
    Returns 1 if price > window-day SMA AND data is sufficient.
    Returns 0 if data is insufficient or price is below MA.
    Contrast with OOS ma_signal() which returned 1 for insufficient data.
    The two are identical whenever len(data) >= window+1.
    """
    if sym not in prices:
        return 0
    s = prices[sym].loc[:date].dropna()
    if len(s) < window + 1:
        return 0   # <-- key difference vs OOS: don't default to holding
    return 1 if float(s.iloc[-1]) > float(s.iloc[-window:].mean()) else 0

# ── Weight-schedule builder (IS, frozen parameters) ─────────────────────────

def build_goldtrend_is(prices):
    """
    O-goldtrend weight schedule for the IS window.
    FROZEN: GLD_MA=126, IEF_MA=210, FREQ=BME, monthly rebalance.
    Uses ma_signal_strict to handle pre-GLD-inception period correctly.
    """
    ws = {}
    for d in pd.date_range(PRICE_START, IS_END, freq=FREQ):
        if ma_signal_strict(prices, "GLD", d, GLD_MA):
            ws[d] = {"GLD": 1.0}
        elif ma_signal_strict(prices, "IEF", d, IEF_MA):
            ws[d] = {"IEF": 1.0}
        else:
            ws[d] = {}  # cash
    return ws

# ── Load prices ──────────────────────────────────────────────────────────────

def load_prices():
    full_idx = pd.bdate_range(PRICE_START, IS_END)
    prices = {}
    for sym in ["GLD", "IEF", "SHY"]:
        df = load_price_series(sym, PRICE_START, IS_END, ADJ_TOTALRETURN, str(CACHE_DIR))
        if not df.empty and "Close" in df.columns:
            prices[sym] = df["Close"].reindex(full_idx, method="ffill")
    return prices

# ── Load book IS returns ─────────────────────────────────────────────────────

def load_book_is():
    cfg_path = PROJECT_ROOT / "config.yaml"
    with open(cfg_path) as f:
        cfg = yaml.safe_load(f)
    cfg["backtest"]["start_date"] = IS_START
    cfg["backtest"]["end_date"]   = IS_END
    cfg["paths"]["cache_dir"]     = str(CACHE_DIR)

    is_idx = pd.bdate_range(IS_START, IS_END)
    strat_rets = {}
    for sid, mname, _ in FINAL_BOOK:
        try:
            mod = importlib.import_module(mname)
            res = mod.run(cfg)
            strat_rets[sid] = res["returns"].fillna(0.0)
            sr_is = sharpe(strat_rets[sid].reindex(is_idx).fillna(0))
            pr(f"  {sid}: IS SR={sr_is:+.3f}")
        except Exception as e:
            pr(f"  {sid}: FAILED ({e})")

    book_is = sum(
        strat_rets[sid].reindex(is_idx).fillna(0.0) * wt
        for sid, _, wt in FINAL_BOOK
        if sid in strat_rets
    )
    return book_is, strat_rets

# ── 2013 stress test: monthly allocation table ───────────────────────────────

def stress_test_table(prices, ws):
    """
    Monthly allocation table 2011-01 to 2015-12.
    Shows: date, GLD price, GLD 126d MA, signal, allocation, next-month GLD return.
    Key question: was the strategy out of GLD before/during the April 2013 crash?
    """
    reb_dates = pd.date_range("2011-01-01", "2015-12-31", freq=FREQ)
    rows = []

    for i, d in enumerate(reb_dates):
        # GLD info
        gld_s = prices["GLD"].loc[:d].dropna() if "GLD" in prices else pd.Series(dtype=float)
        if len(gld_s) >= GLD_MA + 1:
            gld_px  = float(gld_s.iloc[-1])
            gld_ma  = float(gld_s.iloc[-GLD_MA:].mean())
            gld_sig = "ABOVE" if gld_px > gld_ma else "BELOW"
        elif not gld_s.empty:
            gld_px  = float(gld_s.iloc[-1])
            gld_ma  = float("nan")
            gld_sig = "warmup"
        else:
            gld_px  = float("nan")
            gld_ma  = float("nan")
            gld_sig = "no-data"

        # IEF info
        ief_s = prices["IEF"].loc[:d].dropna() if "IEF" in prices else pd.Series(dtype=float)
        if len(ief_s) >= IEF_MA + 1:
            ief_px  = float(ief_s.iloc[-1])
            ief_ma  = float(ief_s.iloc[-IEF_MA:].mean())
            ief_sig = "ABOVE" if ief_px > ief_ma else "BELOW"
        else:
            ief_px  = float("nan")
            ief_ma  = float("nan")
            ief_sig = "warmup"

        # Actual allocation
        alloc_dict = ws.get(d, {})
        if "GLD" in alloc_dict:
            alloc = "GLD"
        elif "IEF" in alloc_dict:
            alloc = "IEF"
        else:
            alloc = "CASH"

        # Next-month GLD and IEF total returns
        if i < len(reb_dates) - 1:
            d_next = reb_dates[i + 1]
            def _mret(sym):
                if sym not in prices: return float("nan")
                cur  = prices[sym].loc[:d].dropna()
                nxt  = prices[sym].loc[:d_next].dropna()
                if cur.empty or nxt.empty: return float("nan")
                return float(nxt.iloc[-1] / cur.iloc[-1] - 1.0)
            gld_ret = _mret("GLD")
            ief_ret = _mret("IEF")
        else:
            gld_ret = ief_ret = float("nan")

        rows.append((d, gld_px, gld_ma, gld_sig, ief_sig, alloc, gld_ret, ief_ret))

    return rows

# ── Main ─────────────────────────────────────────────────────────────────────

def main():
    pr("=" * 90)
    pr("O-goldtrend IS VALIDATION  (2000-01-01 to 2017-06-30)")
    pr("FROZEN parameters: GLD 126d MA -> GLD; elif IEF 210d MA -> IEF; else cash")
    pr("=" * 90)

    # Data coverage
    pr("")
    pr("DATA COVERAGE NOTE:")
    pr(f"  GLD (gold ETF): inception 2004-11-18. Pre-inception -> GLD signal cannot fire.")
    pr(f"  IEF (7-10yr Treasury): inception 2002-07-26. Pre-inception -> cash.")
    pr(f"  No gold spot/futures proxy used. Decision: run on ETF data only.")
    pr(f"  Consequence: 2000-2002 = cash; 2002-2004 = IEF-or-cash; 2004+ = full logic.")
    pr(f"  126d MA warmup completes ~2005-05-26. 210d warmup completes ~2003-06-24.")
    pr(f"  The critical 2011-2015 gold bear is FULLY within GLD history. No data gap.")

    pr("")
    pr("Loading prices...")
    prices = load_prices()
    for sym, s in prices.items():
        fv = s.first_valid_index()
        lv = s.last_valid_index()
        pr(f"  {sym}: {fv.date()} to {lv.date()}  ({s.dropna().shape[0]} obs)")

    pr("")
    pr("Building O-goldtrend weight schedule (IS, frozen parameters)...")
    ws = build_goldtrend_is(prices)

    # Build close_df for engine
    full_idx = pd.bdate_range(PRICE_START, IS_END)
    close_df = pd.DataFrame({
        sym: prices[sym].reindex(full_idx, method="ffill")
        for sym in prices
    })

    gross_ret, to_ser = portfolio_returns_from_weights(
        ws, close_df, IS_START, IS_END, execution_lag=1
    )
    net_ret = apply_costs(gross_ret, to_ser)
    ann_to  = float(to_ser.sum() / max(len(to_ser) / 252.0, 1.0))

    is_idx = pd.bdate_range(IS_START, IS_END)
    gross_ret = gross_ret.reindex(is_idx).fillna(0.0)
    net_ret   = net_ret.reindex(is_idx).fillna(0.0)

    gr_sr_is, gr_cagr_is, gr_mdd_is = perf(gross_ret)
    nt_sr_is, nt_cagr_is, nt_mdd_is = perf(net_ret)

    # Per-year returns
    yr_ret_is = {}
    for yr in range(2000, 2018):
        r_yr = net_ret[net_ret.index.year == yr]
        yr_ret_is[yr] = float((1.0 + r_yr).prod() - 1.0) if not r_yr.empty else float("nan")

    # GLD standalone IS (buy-and-hold benchmark)
    gld_daily = prices["GLD"].reindex(is_idx, method="ffill").pct_change(fill_method=None).fillna(0)
    gld_bh_sr, gld_bh_cagr, gld_bh_mdd = perf(gld_daily)

    pr("")
    pr("Loading book IS returns (11-strategy book applied to IS data)...")
    book_is, strat_rets_is = load_book_is()

    nt_sr_book_is = sharpe(book_is)
    pr(f"  Book IS SR={nt_sr_book_is:+.3f}")

    # IS correlation
    strat_mat_is = pd.DataFrame({
        sid: r.reindex(is_idx).fillna(0.0)
        for sid, r in strat_rets_is.items()
    })
    corr_book_is  = float(net_ret.corr(book_is.reindex(is_idx).fillna(0.0)))
    corr_s46_is   = float(net_ret.corr(strat_mat_is.get("s46", pd.Series(0.0, index=is_idx))))

    # dN_eff IS
    n_before = neff(strat_mat_is.corr().values)
    strat_plus = strat_mat_is.copy(); strat_plus["__v__"] = net_ret
    n_after  = neff(strat_plus.corr().values)
    dneff_is = n_after - n_before

    # dSR IS
    book_is_aligned = book_is.reindex(is_idx).fillna(0.0)
    blended_is = book_is_aligned * 0.95 + net_ret * 0.05
    dsr_is = sharpe(blended_is) - nt_sr_book_is

    # ── Main metrics table ──────────────────────────────────────────────────
    pr("")
    pr("=" * 90)
    pr("IS vs OOS SIDE-BY-SIDE COMPARISON")
    pr("=" * 90)
    pr(f"{'Metric':<28} {'IS (2000-2017)':>16} {'OOS (2017-2024)':>16}")
    pr("-" * 62)
    pr(f"{'Gross Sharpe':<28} {gr_sr_is:>+15.3f} {OOS_REF['gr_sr']:>+15.3f}")
    pr(f"{'Net Sharpe':<28} {nt_sr_is:>+15.3f} {OOS_REF['nt_sr']:>+15.3f}")
    pr(f"{'Net CAGR':<28} {nt_cagr_is:>+14.1%} {OOS_REF['cagr']:>+14.1%}")
    pr(f"{'Max Drawdown':<28} {nt_mdd_is:>+14.1%} {OOS_REF['mdd']:>+14.1%}")
    pr(f"{'Turnover (ann.)':<28} {ann_to:>14.1f}x {OOS_REF['ann_to']:>14.1f}x")
    pr(f"{'corr_book':<28} {corr_book_is:>+15.3f} {OOS_REF['corr_book']:>+15.3f}")
    pr(f"{'corr_s46':<28} {corr_s46_is:>+15.3f} {OOS_REF['corr_s46']:>+15.3f}")
    pr(f"{'dN_eff':<28} {dneff_is:>+15.4f} {OOS_REF['dneff']:>+15.4f}")
    pr(f"{'dSR (5% test wt)':<28} {dsr_is:>+15.4f} {OOS_REF['dsr']:>+15.4f}")
    pr(f"{'GLD buy-and-hold SR':<28} {gld_bh_sr:>+15.3f} {'(OOS: n/a)':>16}")
    pr("")
    pr(f"  IS note: 2000-2002 = cash (no ETF data); 2002-2004 = IEF/cash only.")
    pr(f"  GLD buy-and-hold IS Sharpe = {gld_bh_sr:+.3f} (GLD from 2004 onward).")

    # ── Per-year returns ────────────────────────────────────────────────────
    pr("")
    pr("=" * 90)
    pr("PER-YEAR NET RETURNS (IS + OOS)")
    pr("=" * 90)

    # OOS per-year reference (from prior run output)
    oos_yr = {
        2017: -0.055, 2018: +0.004, 2019: +0.114, 2020: +0.226,
        2021: -0.129, 2022: +0.016, 2023: +0.108, 2024: +0.335,
    }
    # Book IS per-year
    book_yr_is = {}
    for yr in range(2000, 2018):
        r_yr = book_is[book_is.index.year == yr]
        book_yr_is[yr] = float((1.0 + r_yr).prod() - 1.0) if not r_yr.empty else float("nan")

    pr(f"  {'Year':<5} {'Strategy':>10}  {'Book IS':>10}  Regime / note")
    pr("  " + "-" * 70)
    for yr in range(2000, 2018):
        sv = yr_ret_is.get(yr, float("nan"))
        bv = book_yr_is.get(yr, float("nan"))
        sv_s = f"{sv:>+9.1%}" if not np.isnan(sv) else f"{'N/A':>9}"
        bv_s = f"{bv:>+9.1%}" if not np.isnan(bv) else f"{'N/A':>9}"
        note = ""
        if yr <= 2001:   note = "(cash: no ETF data)"
        elif yr == 2002: note = "(IEF warmup; GLD not yet)"
        elif yr == 2003: note = "(IEF or cash)"
        elif yr == 2004: note = "(GLD partial yr from Nov)"
        elif yr in (2010, 2011): note = "<-- late gold bull"
        elif yr == 2013: note = "<-- STRESS TEST: GLD -28%"
        elif yr == 2015: note = "<-- end of gold bear"
        pr(f"  {yr:<5} {sv_s}  {bv_s}  {note}")

    pr("")
    pr("  OOS per-year (reference from prior run):")
    pr(f"  {'Year':<5} {'Strategy':>10}  Note")
    pr("  " + "-" * 50)
    oos_notes = {2020: "COVID spike", 2021: "gold trend broke", 2024: "42% of positive return"}
    for yr in range(2017, 2025):
        sv = oos_yr.get(yr, float("nan"))
        sv_s = f"{sv:>+9.1%}" if not np.isnan(sv) else f"{'N/A':>9}"
        note = oos_notes.get(yr, "")
        pr(f"  {yr:<5} {sv_s}  {note}")

    # ── 2013 stress test ───────────────────────────────────────────────────
    pr("")
    pr("=" * 90)
    pr("2013 GOLD CRASH STRESS TEST -- Monthly signal table 2011-2015")
    pr("Key: Did the 126d MA signal exit gold BEFORE the April 2013 crash (-28% yr)?")
    pr("=" * 90)
    pr(f"  {'Date':<10}  {'GLD($)':>7}  {'126dMA($)':>9}  {'GLD-sig':>7}  {'IEF-sig':>7}  "
       f"{'Alloc':>5}  {'GLD m-ret':>9}  {'IEF m-ret':>9}")
    pr("  " + "-" * 90)

    rows = stress_test_table(prices, ws)
    for (d, gld_px, gld_ma, gld_sig, ief_sig, alloc, gld_ret, ief_ret) in rows:
        dt_s     = d.strftime("%Y-%m-%d")
        gld_px_s = f"{gld_px:>7.2f}" if not np.isnan(gld_px) else f"{'N/A':>7}"
        gld_ma_s = f"{gld_ma:>9.2f}" if not np.isnan(gld_ma) else f"{'N/A':>9}"
        gld_r_s  = f"{gld_ret:>+8.1%}" if not np.isnan(gld_ret) else f"{'N/A':>8}"
        ief_r_s  = f"{ief_ret:>+8.1%}" if not np.isnan(ief_ret) else f"{'N/A':>8}"
        # Flag April 2013 crash
        flag = " <-- CRASH (-15% Apr)" if (d.year == 2013 and d.month == 3) else ""
        flag = " <-- gold bear starts" if (d.year == 2011 and d.month == 8) else flag
        flag = " <-- taper tantrum" if (d.year == 2013 and d.month == 5) else flag
        pr(f"  {dt_s}  {gld_px_s}  {gld_ma_s}  {gld_sig:>7}  {ief_sig:>7}  "
           f"{alloc:>5}  {gld_r_s}  {ief_r_s}{flag}")

    # ── Return concentration analysis ──────────────────────────────────────
    pr("")
    pr("=" * 90)
    pr("RETURN CONCENTRATION ANALYSIS")
    pr("=" * 90)
    full_years = {yr: v for yr, v in yr_ret_is.items() if not np.isnan(v)}
    oos_full   = {yr: v for yr, v in oos_yr.items() if not np.isnan(v)}

    def concentration(yr_dict, label):
        pos_years = {y: v for y, v in yr_dict.items() if v > 0}
        if not pos_years:
            pr(f"  {label}: no positive years")
            return
        total_pos = sum(pos_years.values())
        max_yr = max(pos_years, key=pos_years.get)
        frac   = pos_years[max_yr] / total_pos
        pr(f"  {label}: best year = {max_yr} ({pos_years[max_yr]:+.1%}), "
           f"{frac:.0%} of total positive return across {len(pos_years)} positive years")
        top3 = sorted(pos_years.items(), key=lambda x: -x[1])[:3]
        top3_frac = sum(v for _, v in top3) / total_pos
        pr(f"           Top-3 years = {top3_frac:.0%} of total positive return: "
           f"{', '.join(f'{y}({v:+.1%})' for y, v in top3)}")

    concentration(full_years, "IS (2000-2017)")
    concentration(oos_full,   "OOS (2017-2024)")
    pr("")
    pr("  Interpretation: lumpy return in one window but different years = tail insurance behavior.")
    pr("  Lumpy only in OOS, flat/negative in IS = window artifact.")

    # ── Redundancy check ───────────────────────────────────────────────────
    pr("")
    pr("=" * 90)
    pr("REDUNDANCY vs BOOK (IS)")
    pr("=" * 90)
    pr(f"  corr_book IS  = {corr_book_is:+.3f}   (OOS was {OOS_REF['corr_book']:+.3f})")
    pr(f"  corr_s46 IS   = {corr_s46_is:+.3f}   (OOS was {OOS_REF['corr_s46']:+.3f})")
    pr(f"  dN_eff IS     = {dneff_is:+.4f}   (OOS was {OOS_REF['dneff']:+.4f})")
    pr(f"  dSR IS (5wt)  = {dsr_is:+.4f}   (OOS was {OOS_REF['dsr']:+.4f})")
    pr("")

    # ── Pass/Fail verdict ──────────────────────────────────────────────────
    pr("=" * 90)
    pr("PASS / FAIL VERDICT")
    pr("=" * 90)

    crit1_pass = nt_sr_is > 0.0
    crit2_pass = corr_book_is < 0.50   # well below 0.70 gate
    crit3_desc = "(see 2013 table above)"

    # Check 2013 specifically: was strategy NOT in GLD during March 2013 rebalance?
    mar13 = pd.Timestamp("2013-03-29")  # last BME in March 2013
    mar13_alloc = ws.get(mar13, {})
    if not mar13_alloc:
        # Try finding closest BME
        for d in pd.date_range("2013-03-01", "2013-03-31", freq=FREQ):
            if d in ws:
                mar13 = d
                mar13_alloc = ws[d]
                break
    was_out_mar13 = "GLD" not in mar13_alloc

    # Check 2013 annual: positive or negative?
    yr13_ret = yr_ret_is.get(2013, float("nan"))
    crit3_pass = was_out_mar13 and (not np.isnan(yr13_ret) and yr13_ret > -0.10)

    pr(f"")
    pr(f"  Criterion 1 -- Positive IS net Sharpe (insurance standard, not 0.7 bar):")
    pr(f"    IS net SR = {nt_sr_is:+.3f}  -> {'PASS' if crit1_pass else 'FAIL'}")
    pr(f"")
    pr(f"  Criterion 2 -- Orthogonality holds in IS (corr_book well below 0.70):")
    pr(f"    IS corr_book = {corr_book_is:+.3f}  -> {'PASS' if crit2_pass else 'CHECK'}")
    pr(f"    IS dN_eff    = {dneff_is:+.4f}  (positive = adds diversification)")
    pr(f"")
    pr(f"  Criterion 3 -- 2013 stress test (6m MA signal exited gold before crash):")
    alloc_mar13_str = "GLD" if "GLD" in mar13_alloc else ("IEF" if "IEF" in mar13_alloc else "CASH")
    pr(f"    March 2013 BME allocation: {alloc_mar13_str}  (April crash = -15% in gold)")
    yr13_s = f"{yr13_ret:+.1%}" if not np.isnan(yr13_ret) else "N/A"
    pr(f"    Full-year 2013 return: {yr13_s}  (GLD full year = -28.3%)")
    pr(f"    -> {'PASS: signal exited gold before crash' if was_out_mar13 else 'FAIL: still in GLD during crash'}")
    pr(f"")

    # Overall verdict
    if crit1_pass and crit2_pass and crit3_pass:
        verdict = "PASS"
        desc = ("Positive IS Sharpe, orthogonality replicates, signal exited gold before 2013 crash. "
                "Mechanism is durable across both gold regimes.")
    elif not crit1_pass:
        verdict = "FAIL"
        desc = "Negative or zero IS Sharpe. Mechanism does not generate alpha outside 2017-2024 window."
    elif not crit3_pass and not was_out_mar13:
        verdict = "FAIL"
        desc = "Strategy rode gold down in 2013. Signal did not protect against the regime break. OOS stress-hedge was luck."
    elif crit1_pass and crit3_pass and not crit2_pass:
        verdict = "PARTIAL"
        desc = "IS Sharpe positive and 2013 test passes, but orthogonality did not replicate."
    else:
        verdict = "PARTIAL"
        desc = ("Diversifying in both windows but weak/negative return in IS confirms: "
                "'orthogonal but not reliably additive.' Same conclusion the OOS run reached.")

    pr(f"  OVERALL VERDICT: {verdict}")
    pr(f"  {desc}")
    pr("")

    # Extra context if partial
    if nt_sr_is > 0.0 and nt_sr_is < 0.30:
        pr("  NOTE: IS SR is positive but weak. Most of the IS return likely came from")
        pr("  the 2000s gold bull (GLD above 126d MA almost continuously 2005-2011).")
        pr("  That is BETA to gold, not timing. The 2013 test is what separates timing from beta.")
        pr("  If 2013 was avoided, the MA signal has genuine timing merit despite the weak overall SR.")

    pr("")
    save_output()


if __name__ == "__main__":
    main()
