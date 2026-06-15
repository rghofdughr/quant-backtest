"""
S92 — Country ETF momentum.
Rank single-country ETFs by 12-1m total return; long top quartile.
FX is embedded in ETF return (feature, not bug — state it).
"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import logging
import numpy as np
import pandas as pd
from data import load_price_series, ADJ_TOTALRETURN
from engine import portfolio_returns_from_weights, apply_costs

log = logging.getLogger(__name__)
TRADING_DAYS = 252
DESCRIPTION = "Country ETF 12-1m momentum: long top-quartile single-country ETFs. FX embedded in return."

# iShares single-country ETFs. Earliest launch: EWJ/EWG/EWU/EWZ/EWY ~1996-2001;
# others 2003-2012. Include only those with reasonable history.
COUNTRY_ETFS = [
    "EWJ",  # Japan
    "EWG",  # Germany
    "EWU",  # UK
    "EWZ",  # Brazil
    "EWY",  # South Korea
    "EWA",  # Australia
    "EWC",  # Canada
    "EWP",  # Spain
    "EWI",  # Italy
    "EWQ",  # France
    "EWL",  # Switzerland
    "EWN",  # Netherlands
    "EWT",  # Taiwan
    "EWH",  # Hong Kong
    "EWS",  # Singapore
    "EWM",  # Malaysia
    "EWX",  # Emerging small-cap proxy (SPDR)
    "EEM",  # Emerging markets (broad)
    "EFA",  # Developed ex-US
]

def run(config):
    cfg = config["backtest"]
    start, end = cfg["start_date"], cfg["end_date"]
    cache = config["paths"]["cache_dir"]
    cost_bps = config["costs"]["equity_cost_bps"]
    slip_bps = config["costs"]["equity_slippage_bps"]
    start_load = "1997-01-01"

    data = {}
    for tk in COUNTRY_ETFS:
        df = load_price_series(tk, start=start_load, end=end, adjustment=ADJ_TOTALRETURN, cache_dir=cache)
        if not df.empty:
            data[tk] = df["Close"]

    if len(data) < 5:
        raise RuntimeError(f"S92: only {len(data)} country ETFs loaded — need at least 5")

    log.info("S92: %d country ETFs loaded", len(data))

    trading_idx = pd.bdate_range(start, end)
    price_df = pd.DataFrame(data).reindex(trading_idx, method="ffill")

    reb_dates = pd.date_range(start, end, freq="BME")
    MOM_WIN = 252
    SKIP = 21
    WARMUP = MOM_WIN + SKIP + 5

    weight_schedule = {}
    for rd in reb_dates:
        rd = min(rd, trading_idx[-1])
        pos = trading_idx.searchsorted(rd)
        if pos < WARMUP:
            continue

        p_skip = price_df.iloc[pos - SKIP]
        p_12m = price_df.iloc[max(pos - SKIP - MOM_WIN, 0)]

        # Only include ETFs that existed at this date
        mom = {}
        for tk in price_df.columns:
            pn = p_skip.get(tk)
            pp = p_12m.get(tk)
            if pn and pp and pp > 0 and np.isfinite(pn) and np.isfinite(pp):
                # Check that the ETF had data 12m ago (not just ffill from launch)
                early_data = price_df[tk].iloc[max(pos - SKIP - MOM_WIN, 0):max(pos - SKIP - MOM_WIN + 5, 1)].dropna()
                if len(early_data) < 3:
                    continue
                mom[tk] = pn / pp - 1.0

        if len(mom) < 4:
            continue

        sr = pd.Series(mom)
        top = sr[sr >= sr.quantile(0.75)].index.tolist()
        if not top:
            continue
        weight_schedule[rd] = {s: 1.0 / len(top) for s in top}

    gross_ret, to = portfolio_returns_from_weights(weight_schedule, price_df, start, end)
    net_ret = apply_costs(gross_ret, to, cost_bps, slip_bps)
    spy = load_price_series("SPY", start, end, ADJ_TOTALRETURN, cache)
    bm = spy["Close"].pct_change(fill_method=None).reindex(net_ret.index)
    ann_to = float(to.sum() / max(len(to) / TRADING_DAYS, 1))
    return {"returns": net_ret, "benchmark": bm, "description": DESCRIPTION, "turnover_annual": ann_to}
