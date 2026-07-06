# Straggler Strategy Validation: s97, s98, s99, s102
*Generated: 2026-06-17*

## Result: All Four Are STUB_DATA

None of the four straggler strategies can be run with the current Norgate data subscription (norgatedata v1.0.74). They require data types not exposed by the package.

## Pre-Audit Lookahead Review

- **s97 Div Capture**: CLEAN -- event-driven hold schedule, returns computed separately
- **s98 Ex-Date Drift**: CLEAN -- event-driven hold schedule, symmetric with s97
- **s99 Div Initiation**: CLEAN -- buys at ex-date (publicly announced ~1mo ahead); ex-date vs announcement is a design caveat, not a lookahead bug
- **s102 ETF Basket Arb**: SUSPECT -- line 120: sp_hist = spread_df[etf].iloc[...:i+1] includes today's spread in z-score numerator AND denominator; then today's pair return is earned. Directional impact is NEGATIVE on entry days (ETF overpriced -> pair return < 0 on entry day). Bias likely works AGAINST the strategy, not for it.

---

## Individual Verdicts

### s102 ETF Basket Arb: **STUB_DATA**

**Cannot run:** Needs sector ETF constituent membership (XLK/XLF/XLE as index names) via index_constituent_mask(sym, 'XLK'). Norgate does not expose sector ETF basket membership -- only traditional equity indices (Russell, S&P 500, Nasdaq). Fix: source XLK/XLF/XLE holdings history from SPDR fact sheets or a data vendor that tracks ETF constituent changes.

**Lookahead audit:** SUSPECT -- line 120: sp_hist = spread_df[etf].iloc[...:i+1] includes today's spread in z-score numerator AND denominator; then today's pair return is earned. Directional impact is NEGATIVE on entry days (ETF overpriced -> pair return < 0 on entry day). Bias likely works AGAINST the strategy, not for it.

### s97 Div Capture: **STUB_DATA**

**Cannot run:** norgatedata v1.0.74 has no dividends() API. load_dividends() calls norgatedata.dividends() which raises AttributeError. Strategy logic is sound; blocked at data layer. Fix: use Tiingo, Sharadar, or compute from TOTALRETURN/CAPITAL price ratio.

**Lookahead audit:** CLEAN -- event-driven hold schedule, returns computed separately

### s98 Ex-Date Drift: **STUB_DATA**

**Cannot run:** Same as s97 -- dividend ex-date data unavailable from norgatedata v1.0.74.

**Lookahead audit:** CLEAN -- event-driven hold schedule, symmetric with s97

### s99 Div Initiation: **STUB_DATA**

**Cannot run:** Same as s97 -- dividend ex-date data unavailable from norgatedata v1.0.74.

**Lookahead audit:** CLEAN -- buys at ex-date (publicly announced ~1mo ahead); ex-date vs announcement is a design caveat, not a lookahead bug

---

## Data Provider Limitations

**Dividend strategies (s97, s98, s99):**
- `norgatedata.dividends()` does not exist in v1.0.74
- Available: `dividend_yield_timeseries` (trailing yield %, not individual events)
- Available: `capital_event_timeseries` (stock split events, not dividends)
- Fix: Tiingo dividend API, Sharadar, or TOTALRETURN/CAPITAL ratio inference

**ETF Basket Arb (s102):**
- `index_constituent_timeseries('XLK')` raises ValueError -- XLK is not a Norgate index
- Available Norgate indices: Russell 1000/2000/3000, S&P 500, Nasdaq 100, Dow, etc.
- Fix: SPDR fact sheets, FTSE Russell sector data, or ETF basket history vendor

## Implication for Project

These four strategies were never in the validated book. Their STUB_DATA status does not affect the validated book (s08, s46, s30, s02, s31, s35, s49, s90). They are documented here for completeness and as future work if additional data infrastructure is acquired.
