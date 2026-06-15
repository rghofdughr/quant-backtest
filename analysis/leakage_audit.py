"""
leakage_audit.py
Look-ahead / scaling / multiplicity audit for the Norgate backtest program.

Engine-agnostic: you feed it the EXACT signal series your strategy generated
plus the aligned price panel. Run this before trusting any Sharpe.

Conventions
-----------
prices : pd.DataFrame (cols = symbols) or pd.Series (single asset), daily, date index.
pos    : same shape/index as prices. The position/target weight KNOWN AT THE CLOSE
         of day t (booleans are fine; they're treated as weights). Use the series
         your backtest actually used -- do not "clean" it first.
"""
import numpy as np
import pandas as pd
from scipy.stats import norm

TRADING_DAYS = 252


def _agg(r):
    """Cross-sectional -> equal-weight portfolio return per day (pandas mean
    skips NaN, i.e. averages over active names). Single asset -> itself."""
    return r.mean(axis=1) if isinstance(r, pd.DataFrame) else r


def sharpe(r, ann=True):
    r = r.dropna()
    if len(r) < 30 or r.std() == 0:
        return np.nan
    s = r.mean() / r.std()
    return s * np.sqrt(TRADING_DAYS) if ann else s


# ======================================================================
# TEST 1 -- TIMING TRIPTYCH. The single most powerful look-ahead detector.
# The ONLY difference between the three is when the position is allowed to
# earn. If T0 >> T1, the headline number was same-bar leakage.
# ======================================================================
def timing_triptych(pos, close, open_=None):
    ret_cc = close.pct_change()                      # close-to-close, realized at close_t

    r_same = pos * ret_cc                             # T0: pos_t * ret_t  <-- THE BUG
    r_next = pos.shift(1) * ret_cc                    # T1: pos_{t-1} * ret_t  (honest minimum)

    out = {
        "T0_same_bar_SR":   sharpe(_agg(r_same)),
        "T1_next_close_SR": sharpe(_agg(r_next)),
    }
    if open_ is not None:
        ret_oc = close / open_ - 1.0                  # same-day open->close
        r_real = pos.shift(1) * ret_oc                # T2: decide close_t, fill open_{t+1}, exit close_{t+1}
        out["T2_open_fill_SR"] = sharpe(_agg(r_real)) # (1-day-hold approximation; misses overnight on multi-day holds)

    t1 = out["T1_next_close_SR"]
    out["leak_ratio_T0_over_T1"] = (out["T0_same_bar_SR"] / t1) if t1 not in (0.0, np.nan) else np.inf
    return out
# VERDICT: leak_ratio > ~2-3 (or a sign flip between T0 and T1) == same-bar leakage.
#          A clean strategy loses only a little going close->next. S86 should go ~6.9 -> ~0.


# ======================================================================
# TEST 2 -- P&L ATTRIBUTION. How much cumulative return lands on the bar the
# signal fires? >~60% on the signal day == leakage.
# ======================================================================
def signal_day_attribution(pos, close):
    ret = close.pct_change()
    on_signal = _agg(pos * ret).sum()
    next_bar  = _agg(pos.shift(1) * ret).sum()
    total = on_signal + next_bar
    return {
        "frac_on_signal_day": (on_signal / total) if total else np.nan,
        "pnl_signal_day": on_signal,
        "pnl_next_bar": next_bar,
    }


# ======================================================================
# TEST 3 -- LEAK-FREE ROLLING HELPERS. The bug in channel/vol strategies is
# almost always a window that includes the current bar. Drop these into the
# strategy; if the result MOVES, the original had a same-bar window leak.
# ======================================================================
def donchian_high(close, n):   # breakout = close > donchian_high(close, n)
    return close.rolling(n).max().shift(1)            # max of PRIOR n bars, excludes today

def donchian_low(close, n):
    return close.rolling(n).min().shift(1)

def trailing_vol(close, n):    # vol usable at close_t to size the position held into t+1
    return close.pct_change().rolling(n).std().shift(1)


# ======================================================================
# TEST 4 -- DELISTED/PENNY-STOCK SCALING (the S81 "252% vol" class of bug).
# Including delisted names is correct; equal-weighting into $0.05 names is not.
# ======================================================================
def return_sanity(close, min_price=5.0, ret_clip=0.50):
    ret = close.pct_change()
    tradable = close.shift(1) >= min_price            # filter on YESTERDAY's price (no look-ahead)
    ret_clean = ret.where(tradable).clip(-ret_clip, ret_clip)
    return {
        "daily_rets_gt_50pct": int((ret.abs() > 0.50).sum().sum()),
        "max_single_day_ret": float(ret.max().max()),
        "min_single_day_ret": float(ret.min().min()),
        "vol_raw_annualized":      float(_agg(ret).std() * np.sqrt(TRADING_DAYS)),
        "vol_filtered_annualized": float(_agg(ret_clean).std() * np.sqrt(TRADING_DAYS)),
    }
# If vol_raw is multiples of vol_filtered, your headline vol/Sharpe is penny-stock noise.


# ======================================================================
# TEST 5 -- MULTIPLICITY. Deflated Sharpe over the WHOLE program (all batches),
# not per batch. Lopez de Prado / Bailey.
# IMPORTANT UNITS: pass PER-PERIOD (non-annualized) Sharpes and T = #observations.
# Your table is annualized daily -> divide each annualized SR by sqrt(252).
# ======================================================================
def expected_max_sharpe(trial_sharpes_per_period):
    """E[max Sharpe] under the null (all true SR = 0), given how many strategies
    you ran and the dispersion across them."""
    x = np.asarray(trial_sharpes_per_period, dtype=float)
    N = len(x)
    V = np.var(x, ddof=1)
    g = 0.5772156649015329                            # Euler-Mascheroni
    z1 = norm.ppf(1.0 - 1.0 / N)
    z2 = norm.ppf(1.0 - 1.0 / (N * np.e))
    return np.sqrt(V) * ((1.0 - g) * z1 + g * z2)

def deflated_sharpe(best_sr_per_period, trial_sharpes_per_period, T_obs,
                    skew=0.0, kurt=3.0):
    """Probability the best observed Sharpe beats the null AFTER accounting for N
    trials. Returns ~prob it's real. <0.95 (or <0.975 two-sided) == not significant."""
    sr0 = expected_max_sharpe(trial_sharpes_per_period)
    num = (best_sr_per_period - sr0) * np.sqrt(T_obs - 1.0)
    den = np.sqrt(1.0 - skew * best_sr_per_period + ((kurt - 1.0) / 4.0) * best_sr_per_period**2)
    return float(norm.cdf(num / den)), float(sr0 * np.sqrt(TRADING_DAYS))  # (DSR prob, annualized null benchmark)


# ----------------------------------------------------------------------
# Example wiring
# ----------------------------------------------------------------------
if __name__ == "__main__":
    # close, open_ = your Norgate panels (pandas-dataframe format); pos = your signal
    # print(timing_triptych(pos, close, open_))          # S86/S75/S88
    # print(signal_day_attribution(pos, close))          # confirm
    # print(return_sanity(close))                        # S81
    #
    # Multiplicity over the full program:
    # all_ann_sr = [ ... every IS Sharpe from batch1 + batch2 ... ]   # ~85 values
    # per_period = np.array(all_ann_sr) / np.sqrt(252)
    # best = max(all_ann_sr) / np.sqrt(252)
    # print(deflated_sharpe(best, per_period, T_obs=len(close)))
    pass
