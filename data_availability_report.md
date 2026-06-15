# Data Availability Report
**Generated:** 2026-06-13  
**Norgate version:** 1.0.74  
**Purpose:** Pre-implementation audit of which of the 50 strategies can run on Norgate alone vs. what external vendors are needed.

---

## Summary counts

| Status | Count | Strategies |
|--------|-------|-----------|
| Fully Norgate-native | 23 | S02–S05, S07–S10, S12, S13, S21\*, S24\*, S26\*, S27\*, S30, S31, S34–S39, S41, S46–S50 |
| Norgate + hardcoded calendar | 3 | S37, S38, S41 |
| Requires intraday data | 3 | S06, S12 (exact), S14 |
| Requires fundamentals vendor | 7 | S16–S20, S22, S26 (payout ratio) |
| Requires options data | 3 | S28, S32, S33 |
| Requires external rates/macro | 3 | S23 (FX rates), S25 (yields), S47 (FRED) |
| Requires alternative data | 5 | S40 (EPS surprise), S42 (Form 4), S43 (estimates), S44 (M&A), S45 (short interest) |
| Options + VIX futures (partial) | 2 | S29, S33 |

\* Partially doable; see notes.

---

## Group A — Momentum & Trend

| # | Strategy | Norgate data available? | External need | Stub? |
|---|----------|------------------------|---------------|-------|
| S01 | Cross-sectional 12-1 momentum | YES — TOTALRETURN prices, R1000 C&P PIT | None | No |
| S02 | Time-series (absolute) momentum | YES — futures (`&ES`,`&CL`,`&GC`,`&ZB`, etc.) + ETFs | None | No |
| S03 | Dual MA crossover + vol filter | YES — SPY + sector ETFs + futures | None | No |
| S04 | 52-week high proximity | YES — S&P 500 C&P PIT prices | None | No |
| S05 | Residual momentum | YES — prices sufficient; FF factors proxied with ETFs (SPY/IWM/IWD as MKT/SMB/HML) | None | No |
| S06 | Intraday momentum (first→last hour) | **NO — Norgate does not provide intraday OHLC bars** | Intraday bars (Polygon, IQFeed, Norgate intraday add-on) | Yes — daily open-to-close approximation only |
| S07 | Donchian channel breakout | YES — futures + ETFs, ATR from daily OHLC | None | No |
| S08 | Sector rotation (SPDR ETFs) | YES — XLK, XLF, XLE, XLV, XLI, XLY, XLP, XLRE, XLB, XLU, XLC all in Norgate | None | No |

**S06 note:** Norgate provides daily OHLC only (no intraday bars). Will implement an approximation: first-hour proxy = (Open − prev Close), last-hour proxy = (Close − intraday mid); flag limitations clearly. For a rigorous implementation, use Polygon.io or a Norgate intraday subscription.

---

## Group B — Mean Reversion

| # | Strategy | Norgate data available? | External need | Stub? |
|---|----------|------------------------|---------------|-------|
| S09 | Short-term reversal | YES | None | No |
| S10 | Bollinger Band reversion | YES — ETFs | None | No |
| S11 | Pairs trading (cointegration) | YES — S&P 500 C&P PIT prices for spread formation | None (rolling cointegration in Python) | No |
| S12 | Overnight gap fade | **PARTIAL** — daily open/close available; Norgate lacks intraday for news filter | Earnings date flag from Norgate corporate events (partial) | No for gap; flag no-news filter |
| S13 | RSI(2) oversold bounce | YES | None | No |
| S14 | VWAP reversion (intraday) | **NO — requires intraday** | Intraday data vendor | Yes — daily analog (price vs. 20-day moving average) |
| S15 | OU-process spread trading | YES — ETF pairs prices | None (OU fitting in Python) | No |

---

## Group C — Value & Fundamentals

| # | Strategy | Norgate data available? | External need | Stub? |
|---|----------|------------------------|---------------|-------|
| S16 | Book-to-market deciles | **NO** | Sharadar (via Nasdaq Data Link) or Compustat — quarterly B/M, 4-month lag | Yes |
| S17 | Earnings yield (E/P) | **NO** | Same — trailing 12-month EPS | Yes |
| S18 | EV/EBITDA composite value | **NO** | Same + EV, EBITDA, FCF | Yes |
| S19 | Piotroski F-score | **NO** | Same — 9 binary criteria from financial statements | Yes |
| S20 | Gross profitability | **NO** | Same — gross profit, total assets | Yes |
| S21 | Net share issuance | **PARTIAL** — Norgate carries split-adjusted share counts via corporate actions; YoY change is calculable. But not a clean PIT shares-outstanding timeseries. | Flag: approximation via adjusted share count delta | Partial — implement with caveat |
| S22 | Accruals anomaly | **NO** | Fundamentals vendor — total accruals / total assets | Yes |

**Recommended fundamentals vendor:** Sharadar Core US Fundamentals (via Nasdaq Data Link) — ~$40/month, covers 14,000+ US equities, PIT via `datekey`. Alternatively, SEC EDGAR XBRL via `edgar-online` or `sec-edgar-api` (free but requires parsing).

---

## Group D — Carry & Yield

| # | Strategy | Norgate data available? | External need | Stub? |
|---|----------|------------------------|---------------|-------|
| S23 | FX carry | **PARTIAL** — Norgate has currency futures (`&6E`,`&6J`,`&6B`,`&6A`,`&6C`). Roll yield as carry proxy is doable. True interest-rate differential needs central bank rate data (FRED). | FRED for exact rate differentials; roll-yield proxy works without it | Partial |
| S24 | Commodity futures carry (roll yield) | **PARTIAL** — Norgate continuous contracts available. True term structure (near vs. next month prices) needs individual monthly contract data; check Norgate futures chain coverage | Individual contract months from Norgate (check subscription) | Partial — use roll-return from continuous price changes |
| S25 | Bond term-structure carry | **PARTIAL** — ZT/ZF/ZN/ZB futures available. Carry+roll-down calculation needs yield curve data. | FRED (free) for yield curve; futures prices give duration-adjusted return | Partial — implement with FRED stub |
| S26 | Dividend yield + sustainability | **PARTIAL** — Norgate distributions give 12-mo dividend yield. Payout ratio needs EPS (fundamentals vendor). 5-yr dividend growth is computable from Norgate distributions alone. | Fundamentals vendor for payout ratio | Partial |
| S27 | VIX term-structure carry | **PARTIAL** — Norgate may have VX (VIX futures). Verify. ETP proxies (VXX, SVXY) have post-2009 history in Norgate. | VX futures chain for early history; ETPs post-2009 sufficient | Partial |

**FRED note (S25, S47):** FRED data is free via `pandas-datareader` or the `fredapi` package. Will add a `fred_loader.py` helper. No subscription required.

---

## Group E — Volatility

| # | Strategy | Norgate data available? | External need | Stub? |
|---|----------|------------------------|---------------|-------|
| S28 | Short straddle with regime filter | **NO — requires options** | Options data vendor: ORATS (~$100/mo) or CBOE LiveVol | Yes — build logic, stub loader |
| S29 | Variance risk premium | **PARTIAL** — VIX index (^VIX) may be in Norgate as index; VIX futures partial. Realized variance from SPY prices is fully doable. | VIX index history (CBOE free download as fallback) | Partial |
| S30 | Low-volatility anomaly | YES — S&P 500 C&P PIT + daily returns for vol/beta | None | No |
| S31 | Vol-targeting overlay (SPY) | YES | None | No |
| S32 | Dispersion trade | **NO — requires options** | Same as S28 | Yes |
| S33 | Earnings IV crush | **NO — requires options + earnings dates** | Options vendor + earnings calendar (e.g., Zacks, Compustat) | Yes |

**Options note:** ORATS historical options data starts ~2007 for US equities. Norgate does NOT carry options. These three strategies are fully stubbed — the entry/exit logic and P&L attribution are built, but the options loader interface is a clean stub.

---

## Group F — Seasonality & Calendar

| # | Strategy | Norgate data available? | External need | Stub? |
|---|----------|------------------------|---------------|-------|
| S34 | Turn-of-month | YES — SPY + S&P 500 breadth | None | No |
| S35 | Sell-in-May / Halloween | YES — SPY | None | No |
| S36 | Day-of-week effects | YES — ES futures / SPY | None | No |
| S37 | FOMC drift | YES — SPY prices; FOMC dates hardcoded (Fed website historical calendar is static/free) | FOMC calendar (hardcoded 2000–2024 in `calendars.py`) | No |
| S38 | Pre-holiday drift | YES — SPY; US holiday calendar hardcoded | US market holidays (hardcoded via `pandas_market_calendars` or static list) | No |
| S39 | January tax-loss reversal | YES — prior-year return deciles from Norgate, point-in-time universe | None | No |

**Calendar note:** Will create `calendars.py` with hardcoded FOMC dates (2000–2024) and US market holidays. Both are static, publicly available.

---

## Group G — Event-Driven & Alternative

| # | Strategy | Norgate data available? | External need | Stub? |
|---|----------|------------------------|---------------|-------|
| S40 | Post-earnings announcement drift (PEAD) | **PARTIAL** — event dates may be in Norgate corporate events; EPS surprise (SUE) needs estimates | EPS estimates: Zacks (~$50/mo), Sharadar Earnings Estimates, or Refinitiv | Yes for SUE; event-date logic implemented |
| S41 | Index addition/deletion | YES — `index_constituent_timeseries` gives effective dates; announcement dates may lag (need secondary source for full pre-announcement drift) | Secondary announcement dates (optional enhancement) | No (effective-date version fully doable) |
| S42 | Insider buying clusters | **NO** | SEC EDGAR Form 4 (free via EDGAR API) or InsiderInsights vendor | Yes |
| S43 | Analyst revision momentum | **NO** | I/B/E/S, FactSet, or Zacks estimate revisions | Yes |
| S44 | Merger arbitrage | **NO** | Deal database: Refinitiv, Bloomberg, or a specialty vendor | Yes |
| S45 | Short-squeeze candidates | **PARTIAL** — momentum portion fully Norgate; short interest needs FINRA biweekly releases (free download, but require parsing) | FINRA short interest (free biweekly downloads) or SIR vendor | Partial |

**S41 note:** Norgate `index_constituent_timeseries` gives the effective change date, which is when the price impact is largest. Announcement date (typically 1–2 weeks prior) is harder to get; will be an enhancement. The effective-date effect is fully testable.

---

## Group H — Cross-Asset & Macro

| # | Strategy | Norgate data available? | External need | Stub? |
|---|----------|------------------------|---------------|-------|
| S46 | Risk parity vs 60/40 | YES — SPY, TLT, GLD, DBC, VNQ all in Norgate | None | No |
| S47 | Yield-curve slope as equity timing | **PARTIAL** — ZN/ZT spread as proxy; true 10y-2y needs FRED Treasury yields | FRED (free) for exact yields | Partial — implement with FRED loader |
| S48 | Gold/copper ratio risk filter | YES — GLD/COPX or HG futures in Norgate | None | No |
| S49 | Dollar regime filter on EM equities | YES — UUP (dollar ETF) + EEM in Norgate | None | No |
| S50 | Managed-futures trend (20+ markets) | YES — Norgate has futures for ES, NQ, RTY, YM, ZB, ZN, ZF, ZT, 6E, 6J, 6B, 6A, 6C, CL, NG, GC, SI, HG, ZC, ZW, ZS | None | No |

---

## External data summary — what to get before stubbed strategies can run

| Vendor / Source | Strategies unblocked | Cost | Notes |
|----------------|---------------------|------|-------|
| **Sharadar Core US Fundamentals** (Nasdaq Data Link) | S16, S17, S18, S19, S20, S22, S26 (payout) | ~$40/mo | PIT financials, 14K+ US equities, 2001–present |
| **FRED** (Federal Reserve) | S25, S47 | Free | `fredapi` or `pandas-datareader`; add `fred_loader.py` |
| **ORATS / CBOE LiveVol** | S28, S29, S32, S33 | ~$100/mo | Historical options prices, IV surfaces |
| **FINRA short interest** (biweekly flat file) | S45 | Free download | Parse at finra.org/investors/research/short-sale |
| **Zacks / Sharadar Earnings** | S40, S43 | ~$50/mo | EPS estimates + actuals; surprise = (actual − consensus)/|consensus| |
| **SEC EDGAR Form 4** (API) | S42 | Free | SEC EDGAR full-text search API; rate-limited |
| **Refinitiv / Bloomberg M&A** | S44 | $$$+ | Proprietary deal databases |
| **FOMC calendar** (hardcoded) | S37 | Free | Static list 2000–2024; included in `calendars.py` |
| **Norgate intraday add-on** | S06, S14 | Subscription add-on | Check NDU for intraday offering |

---

## Implementation order (recommended)

**Batch 1 (S01, S02, S03, S07, S08)** — Fully Norgate-native momentum/trend. No external deps. Good warm-up.  
**Batch 2 (S09, S10, S11, S13, S15)** — Mean reversion. Fully Norgate-native.  
**Batch 3 (S30, S31, S34, S35, S46)** — Low-vol anomaly + seasonality + risk parity. Fully Norgate-native.  
**Batch 4 (S04, S05, S21, S39, S41)** — Momentum variants + event effects. Fully Norgate-native.  
**Batch 5 (S37, S38, S47, S48, S49, S50)** — Calendar + macro + managed futures. Mostly native; S47 needs FRED.  
**Batch 6 (S23, S24, S25, S26, S27, S29)** — Carry/yield partials + VIX. Partial Norgate.  
**Batch 7+ (S16–S20, S22, S28, S32, S33, S40, S42–S45)** — Require external vendors. Implement logic + stubs; run when vendor data plugged in.

---

*This report is static. Re-run `python -c "import norgatedata; print(norgatedata.watchlist_names())"` to verify watchlist names match your subscription.*
