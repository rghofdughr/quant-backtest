"""
S15 — Ornstein-Uhlenbeck spread trading on strongly-linked ETF pairs
Universe:  ETF pairs with strong economic relationship (GLD/GDX, SPY/IVV, TLT/IEF, etc.)
Signal:    Fit OU process to spread using maximum-likelihood (Vasicek). Position size
           proportional to deviation from mean, scaled by estimated half-life.
           Trades out as spread reverts toward mean; stops at 3× estimated vol.
Execution: Daily signal; next-day execution.
Sizing:    Position inversely proportional to half-life × spread vol; dollar-neutral per pair.
"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import logging
import numpy as np
import pandas as pd
from scipy.optimize import minimize

from data import load_price_series, ADJ_TOTALRETURN
from engine import apply_costs

log = logging.getLogger(__name__)
DESCRIPTION = "OU-process mean-reversion, economically-linked ETF pairs, size by half-life"
TRADING_DAYS = 252

OU_PAIRS = [
    ("GLD",  "GDX",   252),   # gold price vs miners, 252-day estimation
    ("TLT",  "IEF",   252),   # long vs intermediate treasuries
    ("SPY",  "IVV",   126),   # S&P 500 ETF pair
    ("EEM",  "VWO",   252),   # EM ETFs
    ("XLK",  "QQQ",   252),   # tech sector vs Nasdaq
    ("XLE",  "OIH",   252),   # energy vs oil services
    ("GLD",  "SLV",   252),   # gold vs silver
    ("IWM",  "SLY",   252),   # Russell 2000 vs S&P 600
]

ENTRY_SIGMA = 1.5   # enter at 1.5σ deviation
STOP_SIGMA  = 3.0   # stop at 3σ


def _fit_ou(spread: np.ndarray) -> tuple[float, float, float]:
    """
    MLE fit of Ornstein-Uhlenbeck process: dX = κ(μ−X)dt + σdW.
    Returns (kappa, mu, sigma) — kappa = mean-reversion speed, half-life = ln(2)/kappa.
    """
    n = len(spread)
    x0 = spread[:-1]
    x1 = spread[1:]

    def neg_log_lik(params):
        kappa, mu, sigma = params
        if kappa <= 0 or sigma <= 0:
            return 1e10
        dt = 1.0 / TRADING_DAYS
        e_x  = x0 * np.exp(-kappa * dt) + mu * (1 - np.exp(-kappa * dt))
        var  = sigma ** 2 / (2 * kappa) * (1 - np.exp(-2 * kappa * dt))
        if var <= 0:
            return 1e10
        return (n * np.log(np.sqrt(var)) +
                0.5 * np.sum((x1 - e_x) ** 2) / var)

    x0_params = [1.0, float(spread.mean()), float(spread.std() + 1e-8)]
    try:
        res = minimize(neg_log_lik, x0_params, method="Nelder-Mead",
                       options={"maxiter": 2000, "xatol": 1e-5, "fatol": 1e-5})
        if res.success:
            k, m, s = res.x
            if k > 0 and s > 0:
                return float(k), float(m), float(s)
    except Exception:
        pass

    # Fallback: simple OLS AR(1) estimate
    from numpy.polynomial.polynomial import polyfit as pfit
    slope, intercept = np.polyfit(x0, x1, 1)
    kappa  = max(1e-4, -np.log(max(1e-8, slope)))
    mu     = intercept / max(1e-8, 1 - slope)
    sigma  = float(np.std(x1 - slope * x0 - intercept))
    return kappa, mu, sigma


def _backtest_ou_pair(y: pd.Series, x: pd.Series, form_days: int,
                      cost_bps: float, slip_bps: float) -> pd.Series:
    both = pd.concat([y, x], axis=1).dropna()
    if len(both) < form_days + 20:
        return pd.Series(dtype=float)

    y_r, x_r = both.iloc[:, 0], both.iloc[:, 1]
    n = len(y_r)

    from statsmodels.regression.linear_model import OLS
    from statsmodels.tools import add_constant

    # Rolling hedge ratio (OLS, updated monthly)
    hedge_arr = np.full(n, np.nan)
    kappa_arr = np.full(n, np.nan)
    mu_arr    = np.full(n, np.nan)
    sig_arr   = np.full(n, np.nan)

    for i in range(form_days, n, 21):   # update monthly
        y_f = y_r.iloc[i - form_days:i].values
        x_f = x_r.iloc[i - form_days:i].values
        try:
            res = OLS(y_f, add_constant(x_f)).fit()
            h   = float(res.params[1])
            sp  = y_f - h * x_f
            k, m, s = _fit_ou(sp)
            for j in range(i, min(i + 21, n)):
                hedge_arr[j] = h
                kappa_arr[j] = k
                mu_arr[j]    = m
                sig_arr[j]   = s
        except Exception:
            pass

    hedge_s = pd.Series(hedge_arr, index=y_r.index).ffill()
    kappa_s = pd.Series(kappa_arr, index=y_r.index).ffill()
    mu_s    = pd.Series(mu_arr,    index=y_r.index).ffill()
    sig_s   = pd.Series(sig_arr,   index=y_r.index).ffill().replace(0, np.nan)

    spread  = y_r - hedge_s * x_r
    z_score = ((spread - mu_s) / sig_s).shift(1)  # lag 1 day for execution

    half_life = (np.log(2) / kappa_s.clip(lower=1e-4)).clip(upper=252)

    # Position: size ∝ -z / (half_life × 0.1), max 1.0 (proportional to deviation)
    # Positive z → short spread (sell Y, buy X)
    raw_pos = -(z_score / (half_life / TRADING_DAYS * 10).clip(lower=0.01)).clip(-1.5, 1.5)
    raw_pos[z_score.abs() < 0.5] = 0.0   # dead-band near mean
    raw_pos[z_score.abs() > STOP_SIGMA] = 0.0  # stop

    pos_s = raw_pos.fillna(0.0)

    y_ret = y_r.pct_change(fill_method=None).fillna(0)
    x_ret = x_r.pct_change(fill_method=None).fillna(0)
    pnl   = pos_s * (0.5 * y_ret - 0.5 * x_ret)

    turnover = pos_s.diff().abs() / 2.0
    one_way  = (cost_bps + slip_bps) / 10_000
    net_pnl  = pnl - turnover * one_way * 2

    return net_pnl


def run(config: dict) -> dict:
    cfg      = config["backtest"]
    start    = cfg["start_date"]
    end      = cfg["end_date"]
    is_smoke = config.get("smoke", False)
    cache    = config["paths"]["cache_dir"]

    cost_bps = config["costs"]["equity_cost_bps"]
    slip_bps = config["costs"]["equity_slippage_bps"]

    pairs = OU_PAIRS[:3] if is_smoke else OU_PAIRS

    all_syms = list({s for p in pairs for s in p[:2]})
    prices: dict[str, pd.Series] = {}
    for sym in all_syms:
        df = load_price_series(sym, start=start, end=end,
                               adjustment=ADJ_TOTALRETURN, cache_dir=cache)
        if not df.empty and "Close" in df.columns:
            prices[sym] = df["Close"]

    log.info("S15: loaded %d symbols for %d pairs", len(prices), len(pairs))
    trading_idx = pd.bdate_range(start, end)

    pair_returns = []
    for sym_y, sym_x, form_days in pairs:
        if sym_y not in prices or sym_x not in prices:
            log.warning("S15: skipping (%s,%s) — data missing", sym_y, sym_x)
            continue
        log.debug("S15: OU fitting (%s,%s) ...", sym_y, sym_x)
        pr = _backtest_ou_pair(prices[sym_y], prices[sym_x], form_days, cost_bps, slip_bps)
        if not pr.empty:
            pair_returns.append(pr.reindex(trading_idx).fillna(0.0))

    if not pair_returns:
        return {"returns": pd.Series(dtype=float), "benchmark": pd.Series(dtype=float),
                "description": DESCRIPTION, "turnover_annual": 0.0}

    port_ret = pd.concat(pair_returns, axis=1).fillna(0.0).mean(axis=1)

    spy = load_price_series("SPY", start, end, ADJ_TOTALRETURN, cache)
    bm  = spy["Close"].pct_change(fill_method=None).reindex(port_ret.index)

    log.info("S15 done (%d active pairs)", len(pair_returns))

    return {
        "returns": port_ret, "benchmark": bm,
        "description": DESCRIPTION, "turnover_annual": None,
    }
