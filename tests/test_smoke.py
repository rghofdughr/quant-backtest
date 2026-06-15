"""
tests/test_smoke.py — basic sanity checks on data.py and engine.py.
Run with: python -m pytest tests/ -v
"""

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import pandas as pd
import numpy as np
import pytest

import norgatedata


# ---------------------------------------------------------------------------
# Data layer
# ---------------------------------------------------------------------------

def test_norgate_connected():
    assert norgatedata.status() is True, "Norgate NDU not running"


def test_load_spy():
    from data import load_price_series
    df = load_price_series("SPY", start="2020-01-01", end="2020-12-31")
    assert not df.empty, "SPY load returned empty DataFrame"
    assert "Close" in df.columns
    assert len(df) > 200, f"Expected 200+ rows, got {len(df)}"
    assert df["Close"].isna().sum() == 0, "NaN closes in SPY"


def test_load_futures():
    from data import load_futures_series
    df = load_futures_series("&ES", start="2020-01-01", end="2020-12-31")
    assert not df.empty, "ES futures returned empty"
    assert "Close" in df.columns


def test_watchlist_symbols():
    from data import watchlist_symbols
    syms = watchlist_symbols("S&P 500 Current & Past")
    assert len(syms) > 500, f"Expected 500+ S&P 500 symbols, got {len(syms)}"


def test_constituent_mask():
    from data import index_constituent_mask
    mask = index_constituent_mask("AAPL", "S&P 500", start="2010-01-01", end="2020-12-31")
    # AAPL has been in S&P 500 since 1982 — should be True throughout
    assert mask.any(), "AAPL should be in S&P 500"
    assert mask.dtype == bool


def test_dollar_volume():
    from data import load_price_series, compute_dollar_volume
    df = load_price_series("AAPL", start="2020-01-01", end="2020-12-31")
    dv = compute_dollar_volume(df)
    assert dv.dropna().gt(0).all(), "Dollar volume should be positive"


# ---------------------------------------------------------------------------
# Engine
# ---------------------------------------------------------------------------

def test_metrics():
    from engine import compute_metrics
    rng = np.random.default_rng(42)
    ret = pd.Series(rng.normal(0.0004, 0.01, 500), index=pd.bdate_range("2020-01-01", periods=500))
    m = compute_metrics(ret)
    assert "sharpe" in m
    assert "cagr" in m
    assert "max_dd" in m
    assert m["max_dd"] <= 0, "Max drawdown must be non-positive"


def test_apply_costs():
    from engine import apply_costs
    gross = pd.Series([0.01, 0.02, -0.005, 0.003])
    turnover = pd.Series([0.1, 0.1, 0.1, 0.1])
    net = apply_costs(gross, turnover, cost_bps=5, slippage_bps=5)
    assert (net < gross).all(), "Net returns should be lower than gross after costs"


def test_monthly_returns_table():
    from engine import monthly_returns_table
    rng = np.random.default_rng(0)
    ret = pd.Series(rng.normal(0, 0.01, 756), index=pd.bdate_range("2020-01-01", periods=756))
    tbl = monthly_returns_table(ret)
    assert "Annual" in tbl.columns
    assert len(tbl) == ret.index.year.nunique()


def test_is_oos_split():
    from engine import is_oos_split
    ret = pd.Series(np.zeros(1000), index=pd.bdate_range("2010-01-01", periods=1000))
    is_, oos = is_oos_split(ret, oos_fraction=0.30)
    assert len(is_) == 700
    assert len(oos) == 300
    assert is_.index[-1] < oos.index[0]


def test_vol_weight():
    from engine import vol_weight
    rng = np.random.default_rng(1)
    ret_df = pd.DataFrame(
        rng.normal(0, 0.01, (200, 3)),
        columns=["A", "B", "C"],
        index=pd.bdate_range("2023-01-01", periods=200),
    )
    w = vol_weight(["A", "B", "C"], ret_df, lookback=63, vol_target=0.10)
    assert set(w.keys()) == {"A", "B", "C"}
    assert all(v > 0 for v in w.values())


# ---------------------------------------------------------------------------
# Cache round-trip
# ---------------------------------------------------------------------------

def test_cache_roundtrip(tmp_path):
    from data import load_price_series
    df1 = load_price_series("SPY", "2021-01-01", "2021-06-30", cache_dir=str(tmp_path))
    df2 = load_price_series("SPY", "2021-01-01", "2021-06-30", cache_dir=str(tmp_path))
    pd.testing.assert_frame_equal(df1, df2)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
