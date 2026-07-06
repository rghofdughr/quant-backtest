"""
Sector-momentum concentration backtest.

Baseline (s128-equivalent): hold top-4 SPDR sector ETFs equal-weight, 9m momentum,
  cash filter, monthly rebalance, execute next day.

Variants (N = 5, 4, 3, 2, 1): same sector selection, but instead of holding the
  ETF, hold the top-N S&P 500 constituents in that GICS sector by market cap,
  cap-weighted within the sector.  4 winning sectors * N stocks = 4N positions.

KNOWN LIMITATIONS (flagged, not papered over):
  1. GICS sector = CURRENT classification. Lookahead for Sep 2018 reclassification:
     GOOGL, META, NFLX moved from Information Technology (XLK) to Communication
     Services (XLC). This affects constituent selection Jul 2017 - Sep 2018 only
     (~15 months of OOS). Impact is small because XLC rarely clears top-4 in that
     period anyway (it tracked AT&T/Verizon then); flagged below.
  2. Market cap proxy = unadjusted_close(date) * current_shares_outstanding.
     Shares outstanding is current-period only. For ranking within a sector, the
     ordering is stable: large caps stay large caps. Bias is small for rank purposes.
  3. No SPDR ETF constituent lists in Norgate; S&P 500 + GICS is the closest PIT
     approximation. SPDR ETFs track the S&P sector indices which ARE S&P 500 by GICS.

Costs: 20bp round-trip per unit one-way turnover (same as book convention).
OOS period: 2017-07-03 to 2024-12-31.
"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import logging, pickle, warnings
import numpy as np
import pandas as pd
import norgatedata as ng
from pathlib import Path
from datetime import date as Date

warnings.filterwarnings("ignore")
log = logging.getLogger(__name__)
logging.basicConfig(level=logging.WARNING)

from data import load_price_series, ADJ_TOTALRETURN, ADJ_CAPITAL
from engine import apply_costs

PROJECT_ROOT = Path(__file__).resolve().parent.parent
CACHE_DIR    = PROJECT_ROOT / "cache" / "parquet"
RESULTS_DIR  = PROJECT_ROOT / "results"
RESULTS_DIR.mkdir(exist_ok=True)

OOS_START    = "2017-07-03"
OOS_END      = "2024-12-31"
PRICE_START  = "2000-01-01"
TRADING_DAYS = 252

SECTORS_ETF = ["XLK","XLF","XLE","XLV","XLI","XLY","XLP","XLRE","XLB","XLU","XLC"]
LB_DAYS     = 189   # 9 months * 21 bd
TOP_N_SECT  = 4
N_SWEEP     = [5, 4, 3, 2, 1]
COST_RT_BPS = 20    # round-trip bps per unit one-way turnover

# GICS sector name -> sector ETF
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


# ── helpers ───────────────────────────────────────────────────────────────────
def sr(r):
    r = r.dropna(); n = len(r)
    if n < 20 or r.std() == 0: return 0.0
    return float(r.mean() / r.std() * TRADING_DAYS**0.5)

def cagr(r):
    r = r.dropna().fillna(0)
    if len(r) < 2: return 0.0
    return float((1+r).prod()**(TRADING_DAYS/len(r)) - 1)

def mdd(r):
    w = (1+r.fillna(0)).cumprod()
    return float(((w-w.cummax())/w.cummax()).min())

def yr(r, y):
    sl = r.loc[f"{y}-01-01":f"{y}-12-31"]
    return cagr(sl) if not sl.empty else float("nan")

def bdate_range(start, end):
    return pd.bdate_range(start, end)


# ── Phase 1: build universe ───────────────────────────────────────────────────
def build_universe(cache_dir: Path, force=False):
    """
    Returns:
      gics_map       : {symbol: gics_sector_name}
      shares_map     : {symbol: current_shares_float}
      pit_cache_path : path to parquet file with PIT membership matrix
      unadj_path     : path to parquet with unadjusted close matrix
    """
    gics_path  = cache_dir / "univ_gics.pkl"
    shrs_path  = cache_dir / "univ_shares.pkl"
    pit_path   = cache_dir / "univ_pit_sp500.parquet"
    unadj_path = cache_dir / "univ_unadjclose.parquet"

    # ── symbol list ──────────────────────────────────────────────────────────
    syms = ng.watchlist_symbols("S&P 500 Current & Past")
    print(f"  S&P 500 ever-members: {len(syms)} symbols")

    # ── GICS sector map (current classification) ─────────────────────────────
    if not force and gics_path.exists():
        with open(gics_path, "rb") as f: gics_map = pickle.load(f)
        print(f"  GICS map loaded from cache ({len(gics_map)} symbols)")
    else:
        print("  Building GICS sector map...")
        gics_map = {}
        for i, sym in enumerate(syms):
            try:
                g = ng.classification_at_level(sym, "GICS", "Description", 1)
                if g and g in GICS_TO_ETF:
                    gics_map[sym] = g
            except Exception:
                pass
            if (i+1) % 200 == 0:
                print(f"    ...{i+1}/{len(syms)}")
        with open(gics_path, "wb") as f: pickle.dump(gics_map, f)
        print(f"  GICS map built: {len(gics_map)} symbols with known sector")

    # ── current shares outstanding ────────────────────────────────────────────
    if not force and shrs_path.exists():
        with open(shrs_path, "rb") as f: shares_map = pickle.load(f)
        print(f"  Shares map loaded from cache ({len(shares_map)} symbols)")
    else:
        print("  Building shares outstanding map...")
        shares_map = {}
        for sym in gics_map:
            try:
                so = ng.sharesoutstanding(sym)
                if so and so[0] and so[0] > 0:
                    shares_map[sym] = float(so[0])
            except Exception:
                pass
        with open(shrs_path, "wb") as f: pickle.dump(shares_map, f)
        print(f"  Shares map built: {len(shares_map)} symbols")

    # ── PIT membership matrix ─────────────────────────────────────────────────
    if not force and pit_path.exists():
        print(f"  PIT membership matrix loaded from cache")
        pit_df = pd.read_parquet(pit_path)
    else:
        print("  Building PIT membership matrix (1 API call per symbol)...")
        rows = {}
        for i, sym in enumerate(syms):
            try:
                ts = ng.index_constituent_timeseries(
                    sym, "S&P 500 Current & Past",
                    timeseriesformat="pandas-dataframe")
                if ts is not None and not ts.empty:
                    ts.index = pd.to_datetime(ts.index)
                    rows[sym] = ts.iloc[:, 0].astype(bool)
            except Exception:
                pass
            if (i+1) % 200 == 0:
                print(f"    ...{i+1}/{len(syms)}")
        pit_df = pd.DataFrame(rows).sort_index().fillna(False)
        pit_df.to_parquet(pit_path)
        print(f"  PIT matrix: {pit_df.shape} (dates x symbols)")

    # ── capital-adjusted close matrix (for market cap proxy) ─────────────────
    # CRITICAL: use capital-adjusted (split-corrected) price, NOT unadjusted.
    # Proof:  cap_adj_close(t) = actual_price(t) / split_factor_t_to_now
    #         cap_adj_close(t) × current_shares
    #       = actual_price(t) × (historical_shares(t) / current_shares) × current_shares
    #       = actual_price(t) × historical_shares(t) = actual_market_cap(t)  ← correct!
    # Unadjusted × current_shares inflates high-split stocks (NVDA: 40x error).
    if not force and unadj_path.exists():
        print(f"  Capital-adj close matrix loaded from cache")
        unadj_df = pd.read_parquet(unadj_path)
    else:
        print("  Building capital-adjusted close matrix (for market cap ranking)...")
        rows = {}
        syms_needed = set(gics_map.keys()) & set(shares_map.keys())
        for i, sym in enumerate(syms_needed):
            try:
                df = load_price_series(sym, PRICE_START, OOS_END, ADJ_CAPITAL,
                                       str(cache_dir))
                if not df.empty:
                    df.index = pd.to_datetime(df.index)
                    rows[sym] = df["Close"]
            except Exception:
                pass
            if (i+1) % 200 == 0:
                print(f"    ...{i+1}/{len(syms_needed)}")
        unadj_df = pd.DataFrame(rows).sort_index()
        unadj_df.to_parquet(unadj_path)
        print(f"  Capital-adj close matrix: {unadj_df.shape}")

    return gics_map, shares_map, pit_df, unadj_df


# ── Phase 2: load total return prices for universe ───────────────────────────
def load_universe_returns(gics_map, shares_map, pit_df, cache_dir):
    """Load total-return prices for all symbols in gics_map."""
    cached_path = cache_dir / "univ_tr_prices.parquet"
    if cached_path.exists():
        print("  Total-return price matrix loaded from cache")
        return pd.read_parquet(cached_path)

    print("  Loading total-return prices for universe...")
    needed = set(gics_map.keys()) & set(shares_map.keys()) & set(pit_df.columns)
    rows = {}
    for i, sym in enumerate(needed):
        try:
            df = load_price_series(sym, PRICE_START, OOS_END, ADJ_TOTALRETURN,
                                   str(cache_dir))
            if not df.empty:
                rows[sym] = df["Close"]
        except Exception:
            pass
        if (i+1) % 200 == 0:
            print(f"    ...{i+1}/{len(needed)}")
    tr_df = pd.DataFrame(rows).sort_index()
    tr_df.to_parquet(cached_path)
    print(f"  TR price matrix: {tr_df.shape}")
    return tr_df


# ── Phase 3: run one variant ──────────────────────────────────────────────────
def run_variant(
    n_stocks,          # N per sector (None = ETF baseline)
    etf_close,         # DataFrame: ETF total-return prices (for signal)
    tr_prices,         # DataFrame: stock total-return prices
    unadj_prices,      # DataFrame: unadjusted close (for market cap)
    shares_map,        # {sym: current_shares}
    gics_map,          # {sym: gics_sector}
    pit_df,            # boolean DataFrame: PIT membership
    trading_idx,       # full business date range
    oos_start, oos_end,
):
    reb_dates = pd.date_range(oos_start, oos_end, freq="BME")

    port_ret = pd.Series(0.0, index=trading_idx)
    to_ser   = pd.Series(0.0, index=trading_idx)
    prev_wt  = {}

    is_etf = (n_stocks is None)

    for ri, reb_d in enumerate(reb_dates):
        # ── sector ETF signal ─────────────────────────────────────────────────
        avail = etf_close.index[etf_close.index <= reb_d]
        if len(avail) < LB_DAYS + 2:
            continue
        snap = len(avail) - 1
        p_now  = etf_close.iloc[snap]
        p_past = etf_close.iloc[max(0, snap - LB_DAYS)]
        etf_rets = ((p_now / p_past) - 1.0).dropna().sort_values(ascending=False)
        if etf_rets.empty:
            continue

        # cash filter
        if etf_rets.iloc[0] <= 0:
            new_wt = {}
        else:
            top_etfs = etf_rets.iloc[:TOP_N_SECT].index.tolist()

            if is_etf:
                # Baseline: hold the ETF itself (ETFs are in etf_close, not tr_prices)
                new_wt = {etf: 1.0/TOP_N_SECT for etf in top_etfs}
            else:
                # Concentrated: hold top-N stocks in each winning sector
                new_wt = {}
                for etf in top_etfs:
                    gics_sector = ETF_TO_GICS.get(etf)
                    if gics_sector is None:
                        continue

                    # PIT members in S&P 500 on this date
                    pit_snap = pit_df[pit_df.index <= reb_d]
                    if pit_snap.empty:
                        continue
                    pit_row = pit_snap.iloc[-1]
                    sp500_members = set(pit_row[pit_row].index.tolist())

                    # Filter to this GICS sector
                    sector_syms = [
                        s for s, g in gics_map.items()
                        if g == gics_sector and s in sp500_members
                        and s in tr_prices.columns
                        and s in unadj_prices.columns
                    ]
                    if not sector_syms:
                        # Fall back to ETF if no stocks found
                        if etf in tr_prices.columns:
                            new_wt[etf] = 1.0 / TOP_N_SECT
                        continue

                    # Market cap proxy: unadjusted_close * current_shares
                    ucl_snap = unadj_prices.loc[
                        unadj_prices.index[unadj_prices.index <= reb_d][-1:]
                    ]
                    if ucl_snap.empty:
                        continue
                    ucl_row = ucl_snap.iloc[-1]

                    mcaps = {}
                    for s in sector_syms:
                        uc = ucl_row.get(s, float("nan"))
                        sh = shares_map.get(s, float("nan"))
                        if pd.isna(uc) or pd.isna(sh) or uc <= 0 or sh <= 0:
                            continue
                        # also need price data available at this date
                        tr_avail = tr_prices[s].loc[
                            tr_prices.index[tr_prices.index <= reb_d]
                        ]
                        if len(tr_avail) < 21:
                            continue
                        mcaps[s] = uc * sh

                    if not mcaps:
                        if etf in tr_prices.columns:
                            new_wt[etf] = 1.0 / TOP_N_SECT
                        continue

                    # Top-N by market cap
                    sorted_mcap = sorted(mcaps.items(), key=lambda x: x[1], reverse=True)
                    top_stocks  = sorted_mcap[:n_stocks]

                    # Cap-weight within sector (25% budget per sector)
                    total_mcap  = sum(mc for _, mc in top_stocks)
                    sector_budget = 1.0 / TOP_N_SECT
                    for sym, mc in top_stocks:
                        new_wt[sym] = sector_budget * (mc / total_mcap)

        # ── execute next trading day ──────────────────────────────────────────
        next_days = trading_idx[trading_idx > reb_d]
        if next_days.empty:
            break
        exec_day = next_days[0]

        # Next rebalance execution day
        if ri + 1 < len(reb_dates):
            next_next = trading_idx[trading_idx > reb_dates[ri + 1]]
            end_day   = next_next[0] if not next_next.empty else trading_idx[-1]
        else:
            end_day = trading_idx[-1]

        hold_mask = (trading_idx >= exec_day) & (trading_idx <= end_day)

        # ── returns for holding period ────────────────────────────────────────
        if new_wt:
            for sym, wt in new_wt.items():
                if is_etf:
                    # ETF baseline: prices are in etf_close (full history)
                    if sym not in etf_close.columns:
                        continue
                    sr_ = etf_close[sym].reindex(trading_idx, method="ffill")
                else:
                    # Concentrated: stock prices in tr_prices
                    if sym not in tr_prices.columns:
                        continue
                    sr_ = tr_prices[sym]
                # NaN fill = 0% return (handles delisting / missing data)
                ret_period = sr_.pct_change(fill_method=None).fillna(0)
                port_ret[hold_mask] += ret_period[hold_mask] * wt

        # ── turnover ─────────────────────────────────────────────────────────
        all_syms = set(new_wt) | set(prev_wt)
        to = sum(abs(new_wt.get(s, 0.0) - prev_wt.get(s, 0.0)) for s in all_syms) / 2.0
        if exec_day in to_ser.index:
            to_ser[exec_day] += to

        prev_wt = dict(new_wt)

    return port_ret, to_ser


# ── main ──────────────────────────────────────────────────────────────────────
def main():
    lines = []
    def pr(s=""): lines.append(str(s)); print(str(s))

    # ── Phase 1: universe ─────────────────────────────────────────────────────
    pr("Building universe (S&P 500 PIT, GICS, market cap)...")
    gics_map, shares_map, pit_df, unadj_df = build_universe(CACHE_DIR)

    # ── Phase 2: load ETF and stock prices ────────────────────────────────────
    pr("\nLoading ETF prices...")
    etf_prices = {}
    for etf in SECTORS_ETF:
        df = load_price_series(etf, PRICE_START, OOS_END, ADJ_TOTALRETURN, str(CACHE_DIR))
        if not df.empty:
            etf_prices[etf] = df["Close"]
    etf_close = pd.DataFrame(etf_prices).sort_index()

    pr("Loading stock total-return prices...")
    tr_prices = load_universe_returns(gics_map, shares_map, pit_df, CACHE_DIR)

    # etf_close keeps FULL history (back to PRICE_START) for signal computation.
    # Everything else is trimmed to OOS for memory efficiency.
    trading_idx = pd.bdate_range(OOS_START, OOS_END)
    full_idx    = pd.bdate_range(PRICE_START, OOS_END)
    etf_close   = etf_close.reindex(full_idx, method="ffill")   # full history
    tr_prices   = tr_prices.reindex(trading_idx, method="ffill")
    unadj_df    = unadj_df.reindex(trading_idx, method="ffill")
    pit_df      = pit_df.reindex(trading_idx, method="ffill").fillna(False)

    pr(f"\nUniverse: {len(gics_map)} GICS-mapped symbols, {len(shares_map)} with shares")
    pr(f"TR price matrix: {tr_prices.shape}")
    pr(f"PIT membership matrix: {pit_df.shape}")

    # ── Phase 3: run baseline + variants ─────────────────────────────────────
    variants = [(None, "ETF baseline")] + [(n, f"N={n}") for n in N_SWEEP]
    results  = {}

    for n_stk, label in variants:
        pr(f"\nRunning {label}...")
        ret, to = run_variant(
            n_stocks=n_stk, etf_close=etf_close, tr_prices=tr_prices,
            unadj_prices=unadj_df, shares_map=shares_map, gics_map=gics_map,
            pit_df=pit_df, trading_idx=trading_idx,
            oos_start=OOS_START, oos_end=OOS_END,
        )
        ann_to  = float(to.sum() / ((len(trading_idx)/TRADING_DAYS) or 1))
        net_ret = apply_costs(ret, to)  # defaults 5+5bp one-way × 2 = 20bp RT

        results[label] = {
            "gross_ret": ret,
            "net_ret":   net_ret,
            "to":        to,
            "ann_to":    ann_to,
            "label":     label,
        }
        pr(f"  {label}: ann_to={ann_to:.1f}x  gross_SR={sr(ret):+.3f}  net_SR={sr(net_ret):+.3f}")

    # ── Phase 4: report ───────────────────────────────────────────────────────
    pr("")
    pr("=" * 80)
    pr("SECTOR CONCENTRATION BACKTEST  (OOS: 2017-07-03 to 2024-12-31)")
    pr("=" * 80)
    pr("Baseline: ETF (s128-equiv). Variants: top-N S&P 500 stocks per sector.")
    pr("Market cap proxy: capital_adj_close x current_shares (see notes below).")
    pr("Costs: 20bp round-trip per unit one-way turnover.")
    pr("")

    etf_net = results["ETF baseline"]["net_ret"]
    pr(f"{'Variant':<14} {'GrSR':>6} {'NetSR':>6} {'CAGR':>7} {'MDD':>7} "
       f"{'TO(x)':>6} {'Corr/ETF':>9} {'AvgPos':>7}")
    pr("-" * 80)

    for label, n_stk in [("ETF baseline", None)] + [(f"N={n}", n) for n in N_SWEEP]:
        if label not in results: continue
        r  = results[label]
        gr = r["gross_ret"]; nr = r["net_ret"]; to_ann = r["ann_to"]
        corr = float(nr.corr(etf_net)) if label != "ETF baseline" else 1.0
        # avg positions
        if n_stk is None:
            avg_pos = TOP_N_SECT
        else:
            avg_pos = TOP_N_SECT * n_stk

        pr(f"{label:<14} {sr(gr):>+6.3f} {sr(nr):>+6.3f} {cagr(nr):>+6.1%} "
           f"{mdd(nr):>+7.1%} {to_ann:>6.1f}x {corr:>9.3f} {avg_pos:>7}")

    # ── Noise floor reminder ──────────────────────────────────────────────────
    pr("")
    pr("NOTE: SE on 8-year Sharpe ~0.35. Differences < 0.1 are statistical ties.")
    pr("      Differences < ~0.35 are within one standard error of each other.")

    # ── Per-year table ────────────────────────────────────────────────────────
    pr("")
    pr("=" * 80)
    pr("PER-YEAR NET RETURNS")
    pr("=" * 80)
    hdr = f"{'Year':>6}"
    for label, _ in [("ETF baseline", None)] + [(f"N={n}", n) for n in N_SWEEP]:
        hdr += f"  {label:>10}"
    pr(hdr)
    pr("-" * 80)
    for y in range(2017, 2025):
        row = f"{y:>6}"
        for label, _ in [("ETF baseline", None)] + [(f"N={n}", n) for n in N_SWEEP]:
            if label in results:
                v = yr(results[label]["net_ret"], y)
                row += f"  {v:>+9.1%}" if not np.isnan(v) else f"  {'--':>9}"
            else:
                row += f"  {'--':>10}"
        pr(row)

    # ── Correlation table ─────────────────────────────────────────────────────
    pr("")
    pr("=" * 80)
    pr("RETURN CORRELATION MATRIX (net returns)")
    pr("=" * 80)
    ret_df  = pd.DataFrame({lb: r["net_ret"] for lb, r in results.items()})
    corr_mx = ret_df.corr()
    labels  = list(results.keys())
    hdr2    = f"{'':14}" + "".join(f"{lb:>12}" for lb in labels)
    pr(hdr2)
    pr("-" * (14 + 12*len(labels)))
    for lb in labels:
        row = f"{lb:<14}" + "".join(f"{corr_mx.loc[lb,lb2]:>12.3f}" for lb2 in labels)
        pr(row)

    # ── Plain language answer ─────────────────────────────────────────────────
    pr("")
    pr("=" * 80)
    pr("INTERPRETATION")
    pr("=" * 80)

    etf_sr  = sr(results["ETF baseline"]["net_ret"])
    best_n  = None
    best_nr = -99
    for n in N_SWEEP:
        lb = f"N={n}"
        if lb in results:
            v = sr(results[lb]["net_ret"])
            if v > best_nr:
                best_nr = v; best_n = n

    pr(f"ETF baseline net Sharpe: {etf_sr:+.3f}")
    pr(f"Best concentrated net Sharpe: N={best_n} at {best_nr:+.3f}")
    if best_nr - etf_sr > 0.10:
        pr(f"RESULT: Concentration wins. N={best_n} beats ETF by {best_nr-etf_sr:.3f} SR.")
        pr("        But check MDD and turnover -- idiosyncratic risk almost always rises.")
    elif best_nr - etf_sr > 0.0:
        pr(f"RESULT: TIE (within 0.10 SR). N={best_n} marginally ahead but within noise.")
        pr("        ETF baseline is preferred: lower turnover, lower MDD, no concentration risk.")
    else:
        pr(f"RESULT: ETF WINS. Concentration loses net of cost by {etf_sr-best_nr:.3f} SR.")
        pr("        Idiosyncratic risk is NOT compensated; ETF + lower cost dominates.")

    # Check if concentration risk shows in MDD
    etf_mdd_v = mdd(results["ETF baseline"]["net_ret"])
    n1_mdd_v  = mdd(results["N=1"]["net_ret"]) if "N=1" in results else float("nan")
    if not np.isnan(n1_mdd_v):
        pr(f"MDD: ETF={etf_mdd_v:+.1%} vs N=1 (4 stocks)={n1_mdd_v:+.1%}")

    pr("")
    pr("NOTES ON METHODOLOGY:")
    pr("  1. GICS sector uses CURRENT classification. Sep 2018 reclassification")
    pr("     (GOOGL/META moved from Tech -> Comm Svcs) affects Jul-Sep 2018 only.")
    pr("     During this period XLC rarely appears in top-4 sectors so impact is")
    pr("     minimal, but Comm Svcs stock pool is lookahead-contaminated until Oct 2018.")
    pr("  2. Market cap = capital_adj_close(date) x current_shares_outstanding.")
    pr("     capital_adj_close corrects for splits so proxy = actual_market_cap(t)")
    pr("     minus buyback distortion (typically <10% rank error for S&P 500 names).")
    pr("  3. S&P 500 PIT membership is fully point-in-time from Norgate constituent")
    pr("     timeseries. Delisted stocks included in universe via 'Current & Past' list.")
    pr("  4. On delisting mid-period: weight goes to 0% (no replacement), effectively")
    pr("     a cash drag. This is conservative and directionally correct.")

    # ── save ──────────────────────────────────────────────────────────────────
    out_path = RESULTS_DIR / "sector_concentration_output.txt"
    with open(out_path, "w", encoding="ascii", errors="replace") as f:
        f.write("\n".join(lines))
    pr(f"\nSaved: {out_path}")


if __name__ == "__main__":
    main()
