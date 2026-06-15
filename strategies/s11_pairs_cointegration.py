"""
S11 — Pairs trading via cointegration (Engle-Granger)
Universe:  Intra-sector pairs from S&P 500 C&P; form pairs within same GICS sector proxy.
           Pair candidates: hard-coded sector ETF pairs + sector-member ETF proxies.
Signal:    Rolling Engle-Granger cointegration on 252-day formation window.
           z-score of spread: enter at ±2σ, exit at 0, stop-loss at ±3σ.
Execution: Next open (approx next close) after signal.
Sizing:    Dollar-neutral per pair; equal weight across active pairs.
Note:      Cointegration tested only on pre-trade formation data (no look-ahead in pair selection).
"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import logging
import warnings
import numpy as np
import pandas as pd
from statsmodels.tsa.stattools import coint
from statsmodels.regression.linear_model import OLS
from statsmodels.tools import add_constant

from data import load_price_series, ADJ_TOTALRETURN
from engine import apply_costs

log = logging.getLogger(__name__)
DESCRIPTION = "Pairs cointegration trading, S&P sector ETF pairs + liquid intra-sector pairs"
TRADING_DAYS = 252

# Pre-defined economically-linked pair candidates (sector ETFs + proxies)
PAIR_CANDIDATES = [
    ("XLF", "KRE"),    # financials broad vs regional banks
    ("XLK", "SMH"),    # tech vs semis
    ("XLE", "OIH"),    # energy broad vs oil services
    ("GLD", "GDX"),    # gold price vs gold miners
    ("SPY", "IVV"),    # S&P 500 ETF pair (ultra-tight)
    ("QQQ", "ONEQ"),   # Nasdaq ETF pair
    ("XLV", "IHF"),    # health vs health facilities
    ("XLU", "IDU"),    # utilities broad vs IDU
    ("EEM", "VWO"),    # EM broad pair
    ("TLT", "IEF"),    # long vs intermediate treasuries
    ("XLY", "XLP"),    # consumer discretionary vs staples (non-coint but classic spread)
    ("IWM", "SLY"),    # Russell 2000 vs S&P 600 small-cap
]

ENTRY_Z    = 2.0
EXIT_Z     = 0.0
STOP_Z     = 3.0
FORM_DAYS  = 252
COINT_PVAL = 0.05


def _spread_zscore(y: pd.Series, x: pd.Series, hedge: float) -> pd.Series:
    """Compute z-score of the spread y - hedge*x over a rolling window."""
    spread = y - hedge * x
    mu  = spread.rolling(FORM_DAYS).mean()
    sig = spread.rolling(FORM_DAYS).std().replace(0, np.nan)
    return (spread - mu) / sig


def _backtest_pair(y: pd.Series, x: pd.Series,
                   cost_bps: float, slip_bps: float) -> pd.Series:
    """
    Backtest a single pair. Returns daily P&L series (dollar-neutral).
    Hedge ratio from OLS: y = alpha + hedge*x + epsilon.
    Pair selection and hedge ratio estimation use only data up to formation date.
    """
    # Align
    both = pd.concat([y, x], axis=1).dropna()
    if len(both) < FORM_DAYS + 20:
        return pd.Series(dtype=float)

    y_a, x_a = both.iloc[:, 0], both.iloc[:, 1]
    n = len(y_a)

    # Rolling OLS hedge ratio (estimated on formation window, applied to next day)
    hedge_arr = np.full(n, np.nan)
    coint_pass = np.zeros(n, dtype=bool)

    for i in range(FORM_DAYS, n):
        y_f = y_a.iloc[i - FORM_DAYS:i].values
        x_f = x_a.iloc[i - FORM_DAYS:i].values
        # OLS hedge ratio
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            try:
                res = OLS(y_f, add_constant(x_f)).fit()
                hedge_arr[i] = float(res.params[1])
                # Cointegration test (EG) every 21 days to save time
                if i % 21 == 0:
                    _, pval, _ = coint(y_f, x_f)
                    coint_pass[i] = pval < COINT_PVAL
                else:
                    coint_pass[i] = coint_pass[i - 1] if i > 0 else False
            except Exception:
                hedge_arr[i] = np.nan

    hedge_s = pd.Series(hedge_arr, index=y_a.index).ffill()
    coint_s = pd.Series(coint_pass, index=y_a.index).astype(bool)

    spread   = y_a - hedge_s * x_a
    mu       = spread.rolling(FORM_DAYS).mean().shift(1)
    sig      = spread.rolling(FORM_DAYS).std().shift(1).replace(0, np.nan)
    z_score  = ((spread - mu) / sig).shift(1)  # shift so signal is from yesterday close

    # State machine (path-dependent)
    pos   = 0   # +1 = long Y/short X; -1 = short Y/long X
    positions = []
    for i in range(n):
        z = z_score.iloc[i]
        cp = coint_s.iloc[i]

        if np.isnan(z) or not cp:
            positions.append(0)
            continue

        if pos == 1:
            if z >= EXIT_Z or z <= -STOP_Z:
                pos = 0
        elif pos == -1:
            if z <= EXIT_Z or z >= STOP_Z:
                pos = 0

        if pos == 0:
            if z <= -ENTRY_Z and cp:
                pos = 1    # spread too low: buy Y, sell X
            elif z >= ENTRY_Z and cp:
                pos = -1   # spread too high: sell Y, buy X

        positions.append(pos)

    pos_s = pd.Series(positions, index=y_a.index, dtype=float)

    # P&L: long Y / short X for pos=+1; weight=0.5 each side
    y_ret = y_a.pct_change(fill_method=None).fillna(0)
    x_ret = x_a.pct_change(fill_method=None).fillna(0)

    pnl      = pos_s * (0.5 * y_ret - 0.5 * x_ret)
    turnover = pos_s.diff().abs() / 2.0   # 0, 0.5, or 1.0 at transitions

    one_way = (cost_bps + slip_bps) / 10_000
    net_pnl = pnl - turnover * one_way * 2

    return net_pnl


def run(config: dict) -> dict:
    cfg      = config["backtest"]
    start    = cfg["start_date"]
    end      = cfg["end_date"]
    is_smoke = config.get("smoke", False)
    cache    = config["paths"]["cache_dir"]

    cost_bps = config["costs"]["equity_cost_bps"]
    slip_bps = config["costs"]["equity_slippage_bps"]

    pairs = PAIR_CANDIDATES[:4] if is_smoke else PAIR_CANDIDATES

    # Load all unique symbols
    all_syms = list({s for pair in pairs for s in pair})
    prices: dict[str, pd.Series] = {}
    for sym in all_syms:
        df = load_price_series(sym, start=start, end=end,
                               adjustment=ADJ_TOTALRETURN, cache_dir=cache)
        if not df.empty and "Close" in df.columns:
            prices[sym] = df["Close"]

    log.info("S11: loaded %d symbols for %d pairs", len(prices), len(pairs))
    trading_idx = pd.bdate_range(start, end)

    pair_returns = []
    for sym_y, sym_x in pairs:
        if sym_y not in prices or sym_x not in prices:
            log.warning("S11: skipping (%s,%s) — data missing", sym_y, sym_x)
            continue
        log.debug("S11: backtesting pair (%s, %s) ...", sym_y, sym_x)
        pair_ret = _backtest_pair(prices[sym_y], prices[sym_x], cost_bps, slip_bps)
        if not pair_ret.empty:
            pair_returns.append(pair_ret.reindex(trading_idx).fillna(0.0))

    if not pair_returns:
        return {"returns": pd.Series(dtype=float), "benchmark": pd.Series(dtype=float),
                "description": DESCRIPTION, "turnover_annual": 0.0}

    # Equal-weight across pairs
    port_ret = pd.concat(pair_returns, axis=1).fillna(0.0).mean(axis=1)

    spy = load_price_series("SPY", start, end, ADJ_TOTALRETURN, cache)
    bm  = spy["Close"].pct_change(fill_method=None).reindex(port_ret.index)

    log.info("S11 done (%d active pairs). No further cost adjustment (costs in per-pair calc).",
             len(pair_returns))

    return {
        "returns": port_ret, "benchmark": bm,
        "description": DESCRIPTION, "turnover_annual": None,
    }
