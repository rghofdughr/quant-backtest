"""
Stock-level sector-rotation variants — 3 mechanisms × 3 signals × 2 weightings.

Mechanism A — Two-stage: top-4 sectors by 9m ETF momentum (identical to s128),
  then within each winning sector pick top-5 S&P 500 stocks by stock signal.
Mechanism B — Cross-sector neutral: equal sector budgets (1/11 each), top-3
  stocks per sector by absolute signal rank. No directional sector bet.
Mechanism C — Within-sector relative strength: same as B but signal is each
  stock's excess score over its sector average, removing the sector-level factor.

Stock-selection signals (all three run under each mechanism):
  mom     — 12m-1m momentum  (most likely to overlap s128)
  lowvol  — inverse trailing 252d annualized vol
  rev     — prior-1m loser bounce (inverted 21d return)

Weighting per holding:
  cap  — capital-adj price × current shares (proportional to true hist. market cap)
  ew   — equal weight within allocation

PIT discipline: S&P 500 constituents from Norgate point-in-time; capital-adjusted
prices for market-cap ranking (fixes the 40× NVDA overstatement from unadjusted
prices × current shares); total-return prices for performance; delisted included;
signal through rebalance date; next-session execution.

OOS 2017-07-03 to 2024-12-31.  Costs: 20bp round-trip (defaults in apply_costs).
"""
import sys, os, copy, importlib, pickle, warnings, yaml
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
from pathlib import Path

from data import load_price_series, ADJ_TOTALRETURN, ADJ_CAPITAL
from engine import apply_costs, portfolio_returns_from_weights

# ── paths ────────────────────────────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).resolve().parent.parent
CACHE_DIR    = PROJECT_ROOT / "cache" / "parquet"
RESULTS_DIR  = PROJECT_ROOT / "results"
RESULTS_DIR.mkdir(exist_ok=True)

CFG_PATH = PROJECT_ROOT / "config.yaml"
with open(CFG_PATH) as _f:
    BASE_CFG = yaml.safe_load(_f)

# ── book definition ───────────────────────────────────────────────────────────
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

SECTORS_ETF = ["XLK","XLF","XLE","XLV","XLI","XLY","XLP","XLRE","XLB","XLU","XLC"]
GICS_TO_ETF = {
    "Information Technology":  "XLK",
    "Financials":              "XLF",
    "Energy":                  "XLE",
    "Health Care":             "XLV",
    "Industrials":             "XLI",
    "Consumer Discretionary":  "XLY",
    "Consumer Staples":        "XLP",
    "Real Estate":             "XLRE",
    "Materials":               "XLB",
    "Utilities":               "XLU",
    "Communication Services":  "XLC",
}
ETF_TO_GICS = {v: k for k, v in GICS_TO_ETF.items()}
ALL_GICS    = list(GICS_TO_ETF.keys())

# ── parameters ───────────────────────────────────────────────────────────────
OOS_START = "2017-07-03"
OOS_END   = "2024-12-31"
FULL_START = "2000-01-01"
TD = 252

MECH_A_TOP_SECTORS = 4
MECH_A_STOCKS      = 5     # per winning sector → 20 max positions
MECH_BC_STOCKS     = 3     # per sector (11 sectors → 33 positions)

ETF_LB   = 189             # 9m sector momentum lookback (Mech A)
MOM_LB   = 252; MOM_SKIP = 21
VOL_LB   = 252
REV_LB   = 21

# Redundancy gates
GATE_CORR  = 0.70          # max correlation to any existing strategy
GATE_DNEFF = 0.05
GATE_DSR   = 0.0
TEST_WT    = 0.05


# ═══════════════════════════════════════════════════════════════════════════════
# Universe helpers
# ═══════════════════════════════════════════════════════════════════════════════

def load_universe():
    """Return (gics_map, shares_map, pit_df, capadj_df, tr_full) from cache."""
    cache = CACHE_DIR

    with open(cache / "univ_gics.pkl",   "rb") as f: gics_map   = pickle.load(f)
    with open(cache / "univ_shares.pkl", "rb") as f: shares_map = pickle.load(f)

    pit_df    = pd.read_parquet(cache / "univ_pit_sp500.parquet")
    pit_df.index = pd.to_datetime(pit_df.index)

    capadj_df = pd.read_parquet(cache / "univ_unadjclose.parquet")
    capadj_df.index = pd.to_datetime(capadj_df.index)

    tr_full   = pd.read_parquet(cache / "univ_tr_prices.parquet")
    tr_full.index = pd.to_datetime(tr_full.index)

    return gics_map, shares_map, pit_df, capadj_df, tr_full


def pit_on(pit_df, date):
    """Set of S&P 500 members as-of date (PIT)."""
    rows = pit_df.loc[pit_df.index <= date]
    if rows.empty: return set()
    return set(rows.iloc[-1][rows.iloc[-1].astype(bool)].index)


def mcap_on(capadj_df, shares_map, date, syms):
    """Market cap proxy for a list of syms as-of date. Returns {sym: float}."""
    rows = capadj_df.loc[capadj_df.index <= date]
    if rows.empty: return {}
    row = rows.iloc[-1]
    out = {}
    for s in syms:
        if s in row.index and s in shares_map:
            v = row[s] * shares_map[s]
            if pd.notna(v) and v > 0:
                out[s] = float(v)
    return out


def sector_stocks(gics_map, pit_syms, gics, tr_cols):
    """Stocks in gics sector that are PIT members and have TR price data."""
    return [s for s, g in gics_map.items()
            if g == gics and s in pit_syms and s in tr_cols]


# ═══════════════════════════════════════════════════════════════════════════════
# Signal functions
# ═══════════════════════════════════════════════════════════════════════════════

def sig_momentum(tr, date):
    """12m-1m cross-sectional momentum. Higher score = stronger momentum."""
    avail = tr.index[tr.index <= date]
    if len(avail) < MOM_LB + MOM_SKIP + 2:
        return pd.Series(dtype=float)
    p_now  = tr.loc[avail[-1 - MOM_SKIP]]
    p_past = tr.loc[avail[-MOM_LB]]
    return ((p_now / p_past) - 1.0).replace([np.inf, -np.inf], np.nan).dropna()


def sig_lowvol(tr, date):
    """Inverse trailing 252d vol. Higher score = lower vol (buy low-vol names)."""
    avail = tr.index[tr.index <= date]
    if len(avail) < VOL_LB + 2:
        return pd.Series(dtype=float)
    rets = tr.loc[avail[-(VOL_LB + 1):]].pct_change(fill_method=None).dropna(how="all")
    vol  = rets.std() * TD**0.5
    return (-vol).replace([np.inf, -np.inf], np.nan).dropna()


def sig_reversal(tr, date):
    """Prior 1m loser bounce. Inverted 21d return; higher score = bigger prior loser."""
    avail = tr.index[tr.index <= date]
    if len(avail) < REV_LB + 2:
        return pd.Series(dtype=float)
    p_now  = tr.loc[avail[-1]]
    p_past = tr.loc[avail[-1 - REV_LB]]
    return (-(p_now / p_past - 1.0)).replace([np.inf, -np.inf], np.nan).dropna()


SIGNAL_FNS = {"mom": sig_momentum, "lowvol": sig_lowvol, "rev": sig_reversal}


# ═══════════════════════════════════════════════════════════════════════════════
# ETF sector signal (Mechanism A)
# ═══════════════════════════════════════════════════════════════════════════════

def etf_top_sectors(etf_full, date):
    """Top-4 sector ETFs by 9m momentum, or [] if cash filter triggers."""
    avail = etf_full.index[etf_full.index <= date]
    if len(avail) < ETF_LB + 2:
        return []
    p_now  = etf_full.loc[avail[-1]]
    p_past = etf_full.loc[avail[-ETF_LB]]
    rets   = ((p_now / p_past) - 1.0).dropna().sort_values(ascending=False)
    if rets.empty or rets.iloc[0] <= 0:
        return []
    return rets.iloc[:MECH_A_TOP_SECTORS].index.tolist()


# ═══════════════════════════════════════════════════════════════════════════════
# Weighting helpers
# ═══════════════════════════════════════════════════════════════════════════════

def apply_weights(syms, mcaps, mode, budget):
    """Build {sym: weight} dict for a sector allocation."""
    if not syms:
        return {}
    if mode == "cap":
        total = sum(mcaps.get(s, 0.0) for s in syms)
        if total <= 0:
            return {s: budget / len(syms) for s in syms}
        return {s: budget * mcaps.get(s, 0.0) / total for s in syms}
    else:  # equal
        return {s: budget / len(syms) for s in syms}


# ═══════════════════════════════════════════════════════════════════════════════
# Mechanism builders → {pd.Timestamp: {sym: weight}}
# ═══════════════════════════════════════════════════════════════════════════════

def build_mech_a(reb_dates, pit_df, gics_map, tr_full, capadj_df, shares_map,
                 etf_full, signal_fn, mode):
    tr_cols = set(tr_full.columns)
    ws = {}
    for reb_d in reb_dates:
        top_etfs = etf_top_sectors(etf_full, reb_d)
        if not top_etfs:
            ws[reb_d] = {}
            continue

        pit_syms = pit_on(pit_df, reb_d)
        scores   = signal_fn(tr_full, reb_d)
        budget   = 1.0 / MECH_A_TOP_SECTORS

        wt = {}
        for etf in top_etfs:
            gics = ETF_TO_GICS.get(etf)
            if gics is None:
                continue
            cands = [s for s in sector_stocks(gics_map, pit_syms, gics, tr_cols)
                     if s in scores.index]
            if not cands:
                continue
            top_syms = scores[cands].nlargest(MECH_A_STOCKS).index.tolist()
            if not top_syms:
                continue
            mc = mcap_on(capadj_df, shares_map, reb_d, top_syms)
            wt.update(apply_weights(top_syms, mc, mode, budget))

        ws[reb_d] = wt
    return ws


def build_mech_b(reb_dates, pit_df, gics_map, tr_full, capadj_df, shares_map,
                 signal_fn, mode):
    """Cross-sector neutral: equal sector budget, top stocks by ABSOLUTE signal."""
    tr_cols = set(tr_full.columns)
    budget  = 1.0 / len(ALL_GICS)
    ws = {}
    for reb_d in reb_dates:
        pit_syms = pit_on(pit_df, reb_d)
        scores   = signal_fn(tr_full, reb_d)

        wt = {}
        for gics in ALL_GICS:
            cands = [s for s in sector_stocks(gics_map, pit_syms, gics, tr_cols)
                     if s in scores.index]
            if not cands:
                continue
            top_syms = scores[cands].nlargest(MECH_BC_STOCKS).index.tolist()
            if not top_syms:
                continue
            mc = mcap_on(capadj_df, shares_map, reb_d, top_syms)
            wt.update(apply_weights(top_syms, mc, mode, budget))

        ws[reb_d] = wt
    return ws


def build_mech_c(reb_dates, pit_df, gics_map, tr_full, capadj_df, shares_map,
                 signal_fn, mode):
    """Within-sector RS: stock signal minus sector average (removes sector factor)."""
    tr_cols = set(tr_full.columns)
    budget  = 1.0 / len(ALL_GICS)
    ws = {}
    for reb_d in reb_dates:
        pit_syms = pit_on(pit_df, reb_d)
        scores   = signal_fn(tr_full, reb_d)

        wt = {}
        for gics in ALL_GICS:
            cands = [s for s in sector_stocks(gics_map, pit_syms, gics, tr_cols)
                     if s in scores.index]
            if len(cands) < 2:
                continue
            sec_scores = scores[cands]
            excess     = sec_scores - sec_scores.mean()   # subtract sector average
            top_syms   = excess.nlargest(MECH_BC_STOCKS).index.tolist()
            if not top_syms:
                continue
            mc = mcap_on(capadj_df, shares_map, reb_d, top_syms)
            wt.update(apply_weights(top_syms, mc, mode, budget))

        ws[reb_d] = wt
    return ws


# ═══════════════════════════════════════════════════════════════════════════════
# Book return loader
# ═══════════════════════════════════════════════════════════════════════════════

def load_book_returns():
    """Run 11 existing strategies; return (book_oos_ret, {sid: oos_ret})."""
    cfg = copy.deepcopy(BASE_CFG)
    cfg["backtest"]["start_date"] = FULL_START
    cfg["backtest"]["end_date"]   = OOS_END

    strat_rets = {}
    for sid, mname, _ in FINAL_BOOK:
        if sid in strat_rets:
            continue
        try:
            mod = importlib.import_module(mname)
            res = mod.run(cfg)
            full = res["returns"].fillna(0)
            strat_rets[sid] = full.loc[OOS_START:OOS_END]
        except Exception as e:
            print(f"  WARN: {sid} failed: {e}")

    book_ret = None
    for sid, _, wt in FINAL_BOOK:
        if sid not in strat_rets:
            continue
        contrib = strat_rets[sid] * wt
        book_ret = contrib if book_ret is None else book_ret.add(contrib, fill_value=0)

    return book_ret.fillna(0), strat_rets


# ═══════════════════════════════════════════════════════════════════════════════
# Metric helpers
# ═══════════════════════════════════════════════════════════════════════════════

def sr(r):
    r = r.dropna()
    return float(r.mean() / r.std() * TD**0.5) if len(r) > 20 and r.std() > 0 else 0.0

def cagr(r):
    r = r.dropna().fillna(0)
    return float((1 + r).prod() ** (TD / len(r)) - 1) if len(r) > 1 else 0.0

def mdd(r):
    w = (1 + r.fillna(0)).cumprod()
    return float(((w - w.cummax()) / w.cummax()).min())

def yr(r, y):
    sl = r.loc[f"{y}-01-01":f"{y}-12-31"]
    return cagr(sl) if not sl.empty else float("nan")

def neff(df):
    clean = df.dropna(how="all", axis=1).fillna(0)
    if clean.shape[1] < 2:
        return 1.0
    ev = np.linalg.eigvalsh(clean.corr().values)
    ev = ev[ev > 1e-8]
    return float(ev.sum()**2 / (ev**2).sum())


def gate_metrics(variant_ret, book_ret, strat_rets):
    """Compute all redundancy gate metrics for a variant."""
    vr = variant_ret.reindex(book_ret.index).fillna(0)
    br = book_ret.fillna(0)

    # Max correlation to any single existing strategy
    max_strat_corr = max(
        float(vr.corr(sr_oos.reindex(vr.index).fillna(0)))
        for sr_oos in strat_rets.values()
        if not sr_oos.empty
    ) if strat_rets else 0.0

    corr_book = float(vr.corr(br))

    # dN_eff: add variant as 12th column to strategy correlation matrix
    oos_df = pd.DataFrame({sid: s.reindex(vr.index).fillna(0)
                           for sid, s in strat_rets.items()})
    n_old = neff(oos_df)
    oos_df["_new"] = vr.values
    n_new = neff(oos_df)
    dneff = n_new - n_old

    # dSR: blend at 5% test weight
    new_book = br * (1 - TEST_WT) + vr * TEST_WT
    dsr = sr(new_book) - sr(br)

    return {
        "max_strat_corr": max_strat_corr,
        "corr_book":      corr_book,
        "dneff":          dneff,
        "dsr":            dsr,
        "pass_corr":      max_strat_corr < GATE_CORR,
        "pass_dneff":     dneff > GATE_DNEFF,
        "pass_dsr":       dsr >= GATE_DSR,
        "pass_all":       (max_strat_corr < GATE_CORR
                           and dneff > GATE_DNEFF
                           and dsr >= GATE_DSR),
    }


# ═══════════════════════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════════════════════

def main():
    lines = []
    def pr(s=""):
        s = str(s)
        lines.append(s)
        try:
            print(s)
        except UnicodeEncodeError:
            print(s.encode("ascii", errors="replace").decode("ascii"))

    # ── load universe ─────────────────────────────────────────────────────────
    pr("Loading universe...")
    gics_map, shares_map, pit_df, capadj_df, tr_full = load_universe()
    pr(f"  TR prices: {tr_full.shape}  |  GICS: {len(gics_map)}  |  Shares: {len(shares_map)}")

    # ── load ETF full-history prices for Mech A sector signal ────────────────
    pr("Loading ETF prices (full history for sector signal)...")
    etf_prices = {}
    full_idx = pd.bdate_range(FULL_START, OOS_END)
    for etf in SECTORS_ETF:
        df = load_price_series(etf, FULL_START, OOS_END, ADJ_TOTALRETURN, str(CACHE_DIR))
        if not df.empty:
            etf_prices[etf] = df["Close"]
    etf_full = pd.DataFrame(etf_prices).reindex(full_idx, method="ffill")

    # ── load existing book return stream ──────────────────────────────────────
    pr("Loading existing 11-strategy book returns...")
    book_ret, strat_rets = load_book_returns()
    book_sr = sr(book_ret)
    pr(f"  Book OOS SR={book_sr:+.3f}  N_eff={neff(pd.DataFrame(strat_rets)):.3f}")

    # ── rebalance schedule ────────────────────────────────────────────────────
    reb_dates = pd.date_range(OOS_START, OOS_END, freq="BME")

    # ── run all variants ──────────────────────────────────────────────────────
    results = {}

    for sig_name, signal_fn in SIGNAL_FNS.items():
        for mech_name, builder_fn in [
            ("A", build_mech_a),
            ("B", build_mech_b),
            ("C", build_mech_c),
        ]:
            for mode in ["cap", "ew"]:
                label = f"{mech_name}-{sig_name}-{mode}"
                pr(f"  Running {label}...")

                if mech_name == "A":
                    ws = builder_fn(reb_dates, pit_df, gics_map, tr_full, capadj_df,
                                   shares_map, etf_full, signal_fn, mode)
                else:
                    ws = builder_fn(reb_dates, pit_df, gics_map, tr_full, capadj_df,
                                   shares_map, signal_fn, mode)

                port_ret, to_ser = portfolio_returns_from_weights(
                    ws, tr_full, OOS_START, OOS_END, execution_lag=1)

                net_ret = apply_costs(port_ret, to_ser)
                ann_to  = float(to_ser.sum() / (len(port_ret) / TD))

                gates = gate_metrics(net_ret, book_ret, strat_rets)

                results[label] = {
                    "gross_ret": port_ret,
                    "net_ret":   net_ret,
                    "to":        to_ser,
                    "ann_to":    ann_to,
                    "mech":      mech_name,
                    "signal":    sig_name,
                    "mode":      mode,
                    **gates,
                }

    # ═══════════════════════════════════════════════════════════════════════════
    # Report
    # ═══════════════════════════════════════════════════════════════════════════
    pr("")
    pr("=" * 100)
    pr("STOCK-LEVEL SECTOR ROTATION  OOS 2017-07-03 to 2024-12-31")
    pr("=" * 100)
    pr(f"Book baseline:  OOS SR={book_sr:+.3f}  CAGR={cagr(book_ret):+.1%}  MDD={mdd(book_ret):+.1%}")
    pr(f"SE(Sharpe,8yr) ~= +/-0.35  --  gaps < 0.10 are statistical ties")
    pr("")

    # Mechanism descriptions
    pr("Mechanism A: Two-stage sector momentum (s128 signal) -> top-5 stocks per winning sector")
    pr("Mechanism B: Cross-sector neutral, equal sector budgets, top-3 by ABSOLUTE signal")
    pr("Mechanism C: Within-sector RS, top-3 by stock-vs-sector excess signal")
    pr("Signals: mom=12m-1m momentum  lowvol=inverse vol  rev=1m reversal (buy losers)")
    pr("Weighting: cap=capital-adj market cap  ew=equal weight")
    pr("")

    # ── main results table ────────────────────────────────────────────────────
    pr(f"{'Label':<18} {'GrSR':>6} {'NetSR':>6} {'CAGR':>7} {'MDD':>7} "
       f"{'TO(x)':>6} {'CorrBk':>7} {'MaxStr':>7} {'dNeff':>6} {'dSR':>6} "
       f"{'Gates':>8}")
    pr("-" * 100)

    for mech_name in ["A", "B", "C"]:
        pr(f"-- Mechanism {mech_name} --")
        for sig_name in ["mom", "lowvol", "rev"]:
            for mode in ["cap", "ew"]:
                label = f"{mech_name}-{sig_name}-{mode}"
                if label not in results:
                    continue
                r = results[label]
                gr = r["gross_ret"]; nr = r["net_ret"]
                gate_str = ("PASS" if r["pass_all"] else
                            "/".join(["corr" if not r["pass_corr"] else "",
                                      "dNeff" if not r["pass_dneff"] else "",
                                      "dSR"   if not r["pass_dsr"]   else ""]
                                     ).strip("/").replace("//", "/") or "FAIL")

                pr(f"{label:<18} {sr(gr):>+6.3f} {sr(nr):>+6.3f} "
                   f"{cagr(nr):>+6.1%} {mdd(nr):>+6.1%} "
                   f"{r['ann_to']:>6.1f}x "
                   f"{r['corr_book']:>7.3f} "
                   f"{r['max_strat_corr']:>7.3f} "
                   f"{r['dneff']:>+6.3f} "
                   f"{r['dsr']:>+6.3f} "
                   f"{gate_str:>8}")
        pr("")

    # ── per-year table for net returns ────────────────────────────────────────
    pr("=" * 100)
    pr("PER-YEAR NET RETURNS  (cap-weight variants only)")
    pr("=" * 100)
    cap_labels = [f"{m}-{s}-cap" for m in ["A","B","C"] for s in ["mom","lowvol","rev"]]
    present = [l for l in cap_labels if l in results]
    hdr = f"{'Year':>6}  {'Book':>7}" + "".join(f"  {l[:10]:>12}" for l in present)
    pr(hdr)
    pr("-" * (16 + 14*len(present)))
    for y in range(2017, 2025):
        row = f"{y:>6}  {yr(book_ret, y):>+6.1%}"
        for l in present:
            v = yr(results[l]["net_ret"], y)
            row += f"  {v:>+11.1%}" if not np.isnan(v) else f"  {'--':>11}"
        pr(row)

    # ── cap vs equal-weight diagnostic ────────────────────────────────────────
    pr("")
    pr("=" * 100)
    pr("CAP-WEIGHT vs EQUAL-WEIGHT DIAGNOSTIC")
    pr("(Large gap = result driven by megacap beta, not stock selection)")
    pr("=" * 100)
    pr(f"{'Label':<14} {'Cap NetSR':>10} {'EW NetSR':>10} {'Gap':>8} {'Megacap?':>10}")
    pr("-" * 60)
    for mech_name in ["A", "B", "C"]:
        for sig_name in ["mom", "lowvol", "rev"]:
            cap_l = f"{mech_name}-{sig_name}-cap"
            ew_l  = f"{mech_name}-{sig_name}-ew"
            if cap_l not in results or ew_l not in results:
                continue
            cap_sr = sr(results[cap_l]["net_ret"])
            ew_sr  = sr(results[ew_l]["net_ret"])
            gap    = cap_sr - ew_sr
            pr(f"{cap_l:<14} {cap_sr:>+10.3f} {ew_sr:>+10.3f} {gap:>+8.3f} "
               f"{'YES' if gap > 0.10 else 'no':>10}")

    # ── gate summary ──────────────────────────────────────────────────────────
    pr("")
    pr("=" * 100)
    pr("REDUNDANCY GATE SUMMARY")
    pr(f"Gates: max_strat_corr < {GATE_CORR:.2f}  |  dN_eff > +{GATE_DNEFF:.2f}  "
       f"|  dSR >= {GATE_DSR:.1f}  (test weight: {TEST_WT:.0%})")
    pr("=" * 100)
    passers = [(l, r) for l, r in results.items() if r["pass_all"]]
    pr(f"Variants passing ALL gates: {len(passers)} of {len(results)}")
    if passers:
        pr("")
        for l, r in sorted(passers, key=lambda x: -x[1]["dsr"]):
            pr(f"  PASS  {l:<20}  NetSR={sr(r['net_ret']):+.3f}  corr={r['corr_book']:.3f}  "
               f"max_strat={r['max_strat_corr']:.3f}  dNeff={r['dneff']:+.3f}  dSR={r['dsr']:+.3f}")
    else:
        pr("  None passed all gates.")

    pr("")
    pr("PRIOR HYPOTHESIS CHECK:")
    for mech in ["B", "C"]:
        for sig in ["lowvol", "rev"]:
            l_cap = f"{mech}-{sig}-cap"
            l_ew  = f"{mech}-{sig}-ew"
            if l_cap in results:
                r = results[l_cap]
                pr(f"  {l_cap}: corr_book={r['corr_book']:.3f}  max_strat={r['max_strat_corr']:.3f}  "
                   f"dNeff={r['dneff']:+.3f}  gates={'PASS' if r['pass_all'] else 'FAIL'}")

    # ── save ──────────────────────────────────────────────────────────────────
    out_path = RESULTS_DIR / "stock_variants_output.txt"
    with open(out_path, "w", encoding="ascii", errors="replace") as f:
        f.write("\n".join(lines))
    pr(f"\nSaved: {out_path}")


if __name__ == "__main__":
    main()
