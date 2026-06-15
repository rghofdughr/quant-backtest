"""
engine.py — backtesting engine: cost model, metrics, plotting.

All returns are daily unless otherwise noted.
"""

from __future__ import annotations
import logging
import warnings
from pathlib import Path
from typing import Dict, Optional, Tuple

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker

log = logging.getLogger(__name__)

TRADING_DAYS = 252


# ---------------------------------------------------------------------------
# Cost model
# ---------------------------------------------------------------------------

def apply_costs(
    gross_returns: pd.Series,
    turnover: pd.Series,
    cost_bps: float = 5.0,
    slippage_bps: float = 5.0,
) -> pd.Series:
    """
    Deduct round-trip transaction costs proportional to turnover.
    turnover: fraction of portfolio turned over each period (0–1 per side).
    cost_bps + slippage_bps applied one-way; multiply by 2 for round-trip.
    """
    one_way = (cost_bps + slippage_bps) / 10_000
    cost_drag = turnover * one_way * 2   # round-trip
    return gross_returns - cost_drag


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------

def compute_metrics(
    returns: pd.Series,
    benchmark: Optional[pd.Series] = None,
    rf: float = 0.0,
    label: str = "Strategy",
) -> Dict:
    """
    Full tearsheet metrics from a daily returns series.
    """
    r = returns.dropna()
    if r.empty:
        return {}

    ann_factor = TRADING_DAYS
    cum = (1 + r).cumprod()

    # CAGR
    n_years = len(r) / ann_factor
    cagr = float(cum.iloc[-1] ** (1 / n_years) - 1) if n_years > 0 else np.nan

    # Annualized vol
    vol = float(r.std() * np.sqrt(ann_factor))

    # Sharpe (annualized, using rf=0 by default per config)
    excess = r - rf / ann_factor
    sharpe = float(excess.mean() / excess.std() * np.sqrt(ann_factor)) if excess.std() > 0 else np.nan

    # Sortino
    downside = r[r < 0].std()
    sortino = float(r.mean() / downside * np.sqrt(ann_factor)) if downside > 0 else np.nan

    # Max drawdown
    peak = cum.cummax()
    dd = (cum - peak) / peak
    mdd = float(dd.min())

    # Calmar
    calmar = float(cagr / abs(mdd)) if mdd != 0 else np.nan

    # Hit rate / win-loss
    wins  = r[r > 0]
    losses = r[r < 0]
    hit_rate  = float(len(wins) / len(r)) if len(r) > 0 else np.nan
    avg_win   = float(wins.mean())  if len(wins)   > 0 else 0.0
    avg_loss  = float(losses.mean()) if len(losses) > 0 else 0.0
    win_loss_ratio = float(abs(avg_win / avg_loss)) if avg_loss != 0 else np.nan

    result = dict(
        label=label,
        cagr=cagr,
        vol=vol,
        sharpe=sharpe,
        sortino=sortino,
        max_dd=mdd,
        calmar=calmar,
        hit_rate=hit_rate,
        avg_win=avg_win,
        avg_loss=avg_loss,
        win_loss=win_loss_ratio,
        n_days=len(r),
        start=str(r.index[0].date()),
        end=str(r.index[-1].date()),
    )

    # Alpha / beta vs benchmark
    if benchmark is not None:
        b = benchmark.reindex(r.index).dropna()
        common = r.reindex(b.index).dropna()
        b = b.reindex(common.index)
        if len(common) > 20:
            cov = np.cov(common, b)
            beta = float(cov[0, 1] / cov[1, 1]) if cov[1, 1] > 0 else np.nan
            alpha_daily = float(common.mean() - beta * b.mean())
            alpha_ann = float((1 + alpha_daily) ** ann_factor - 1)
            result.update(beta=beta, alpha=alpha_ann)

    return result


def is_oos_split(returns: pd.Series, oos_fraction: float = 0.30) -> Tuple[pd.Series, pd.Series]:
    """Split returns into in-sample and out-of-sample portions."""
    n = len(returns)
    split = int(n * (1 - oos_fraction))
    return returns.iloc[:split], returns.iloc[split:]


def portfolio_returns_from_weights(
    weight_schedule: dict,
    close_df: pd.DataFrame,
    start: str,
    end: str,
    execution_lag: int = 1,
) -> Tuple[pd.Series, pd.Series]:
    """
    Build daily portfolio returns from a monthly (or any frequency) weight schedule.

    Parameters
    ----------
    weight_schedule : {pd.Timestamp: {symbol: float}}
        Target weights set at each rebalance date (close of that day).
    close_df : wide DataFrame of close prices (index=dates, cols=symbols).
    start, end : date range for output.
    execution_lag : days between signal date and first day of holding (default 1 →
                    weights signaled at close of d, applied from close of d+1).

    Returns
    -------
    (port_ret, turnover) — both daily Series indexed to business-day range.
    """
    idx = pd.bdate_range(start, end)
    reb_dates = sorted(weight_schedule.keys())

    all_syms = list({s for w in weight_schedule.values() for s in w})
    avail = [s for s in all_syms if s in close_df.columns]

    ret_mat = close_df[avail].reindex(idx).pct_change(fill_method=None).fillna(0.0)

    port_ret = pd.Series(0.0, index=idx)
    to_series = pd.Series(0.0, index=idx)

    prev_w: dict = {}
    for ri, reb_d in enumerate(reb_dates):
        w = weight_schedule[reb_d]

        # First day of holding = execution_lag trading days after rebalance date
        future = idx[idx > reb_d]
        if len(future) < execution_lag:
            continue
        hold_start = future[execution_lag - 1]

        # Last day of holding = day before next rebalance's execution start
        if ri + 1 < len(reb_dates):
            next_future = idx[idx > reb_dates[ri + 1]]
            hold_end = next_future[execution_lag - 1] if len(next_future) >= execution_lag else idx[-1]
        else:
            hold_end = idx[-1]

        hold_mask = (idx >= hold_start) & (idx <= hold_end)

        # Vectorised return contribution
        w_s = {s: v for s, v in w.items() if s in avail}
        if w_s:
            syms = list(w_s.keys())
            wts  = np.array([w_s[s] for s in syms])
            port_ret[hold_mask] += (ret_mat.loc[hold_mask, syms] * wts).sum(axis=1).values

        # Turnover at execution date
        all_s = set(list(w.keys()) + list(prev_w.keys()))
        to = sum(abs(w.get(s, 0.0) - prev_w.get(s, 0.0)) for s in all_s) / 2.0
        if hold_start in to_series.index:
            to_series[hold_start] += to

        prev_w = dict(w)

    return port_ret, to_series


def compute_turnover(weights_df: pd.DataFrame) -> pd.Series:
    """
    Given a DataFrame of daily portfolio weights (rows=dates, cols=assets),
    returns daily one-way turnover as a fraction of portfolio.
    """
    delta = weights_df.diff().abs().sum(axis=1) / 2
    return delta


# ---------------------------------------------------------------------------
# Monthly returns table
# ---------------------------------------------------------------------------

def monthly_returns_table(daily_returns: pd.Series) -> pd.DataFrame:
    """Aggregate daily returns to monthly, pivot to year×month."""
    mo = (1 + daily_returns).resample("ME").prod() - 1
    mo.index = pd.to_datetime(mo.index)
    df = mo.to_frame("ret")
    df["year"]  = df.index.year
    df["month"] = df.index.month
    pivot = df.pivot(index="year", columns="month", values="ret")
    pivot.columns = ["Jan","Feb","Mar","Apr","May","Jun",
                     "Jul","Aug","Sep","Oct","Nov","Dec"][:len(pivot.columns)]
    pivot["Annual"] = (1 + mo).resample("YE").prod().values - 1
    return pivot


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------

def plot_tearsheet(
    returns: pd.Series,
    metrics: Dict,
    benchmark_returns: Optional[pd.Series] = None,
    title: str = "Strategy",
    save_path: Optional[str] = None,
) -> plt.Figure:
    fig = plt.figure(figsize=(14, 10))
    fig.suptitle(title, fontsize=14, fontweight="bold")

    gs = fig.add_gridspec(3, 2, hspace=0.45, wspace=0.3)
    ax_eq   = fig.add_subplot(gs[0, :])
    ax_dd   = fig.add_subplot(gs[1, 0])
    ax_dist = fig.add_subplot(gs[1, 1])
    ax_heat = fig.add_subplot(gs[2, :])

    cum = (1 + returns).cumprod()
    ax_eq.plot(cum.index, cum.values, color="steelblue", lw=1.5, label=title)
    if benchmark_returns is not None:
        bm = (1 + benchmark_returns.reindex(returns.index).fillna(0)).cumprod()
        ax_eq.plot(bm.index, bm.values, color="gray", lw=1, ls="--", label="SPY")
    ax_eq.set_title("Equity Curve")
    ax_eq.set_ylabel("Growth of $1")
    ax_eq.legend(fontsize=8)
    ax_eq.grid(True, alpha=0.3)

    peak = cum.cummax()
    dd   = (cum - peak) / peak
    ax_dd.fill_between(dd.index, dd.values, 0, color="salmon", alpha=0.7)
    ax_dd.set_title(f"Drawdown  (Max: {metrics.get('max_dd', 0):.1%})")
    ax_dd.set_ylabel("Drawdown")
    ax_dd.yaxis.set_major_formatter(mticker.PercentFormatter(xmax=1))
    ax_dd.grid(True, alpha=0.3)

    ax_dist.hist(returns.dropna() * 100, bins=60, color="steelblue", edgecolor="none", alpha=0.8)
    ax_dist.axvline(0, color="k", lw=0.8)
    ax_dist.set_title("Daily Return Distribution")
    ax_dist.set_xlabel("Daily Return (%)")
    ax_dist.grid(True, alpha=0.3)

    mo_tbl = monthly_returns_table(returns)
    _heat_table(ax_heat, mo_tbl)

    stats = (
        f"CAGR: {metrics.get('cagr', 0):.1%}  |  "
        f"Vol: {metrics.get('vol', 0):.1%}  |  "
        f"Sharpe: {metrics.get('sharpe', 0):.2f}  |  "
        f"Sortino: {metrics.get('sortino', 0):.2f}  |  "
        f"MaxDD: {metrics.get('max_dd', 0):.1%}  |  "
        f"Calmar: {metrics.get('calmar', 0):.2f}  |  "
        f"Hit: {metrics.get('hit_rate', 0):.1%}"
    )
    fig.text(0.5, 0.01, stats, ha="center", fontsize=8.5, color="dimgray")

    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=120, bbox_inches="tight")
        log.info("Saved tearsheet → %s", save_path)

    return fig


def _heat_table(ax: plt.Axes, tbl: pd.DataFrame) -> None:
    """Draw monthly-returns heatmap on *ax*."""
    ax.axis("off")
    vals = tbl.values
    n_rows, n_cols = vals.shape

    col_labels = list(tbl.columns)
    row_labels  = [str(y) for y in tbl.index]

    cmap = plt.get_cmap("RdYlGn")
    vmax = max(abs(np.nanmax(vals)), abs(np.nanmin(vals)), 0.01)

    table = ax.table(
        cellText=[[f"{v:.1%}" if not np.isnan(v) else "" for v in row] for row in vals],
        rowLabels=row_labels,
        colLabels=col_labels,
        loc="center",
        cellLoc="center",
    )
    table.auto_set_font_size(True)
    table.scale(1.0, 1.2)

    for (r, c), cell in table._cells.items():
        if r == 0 or c == -1:
            cell.set_facecolor("#404040")
            cell.set_text_props(color="white", fontweight="bold")
        else:
            v = vals[r - 1, c]
            if not np.isnan(v):
                norm = max(-1, min(1, v / vmax))
                cell.set_facecolor(cmap((norm + 1) / 2))
            else:
                cell.set_facecolor("#f5f5f5")

    ax.set_title("Monthly Returns", pad=4)


# ---------------------------------------------------------------------------
# Comparison table (master results)
# ---------------------------------------------------------------------------

def build_master_table(all_metrics: list[Dict]) -> pd.DataFrame:
    cols = ["label", "cagr", "vol", "sharpe", "sortino",
            "max_dd", "calmar", "hit_rate", "win_loss", "alpha", "beta",
            "start", "end"]
    rows = []
    for m in all_metrics:
        rows.append({c: m.get(c, np.nan) for c in cols})
    df = pd.DataFrame(rows).set_index("label")
    # Format percentages
    for col in ["cagr", "vol", "max_dd", "hit_rate", "alpha"]:
        if col in df.columns:
            df[col] = df[col].map(lambda x: f"{x:.2%}" if pd.notna(x) else "")
    for col in ["sharpe", "sortino", "calmar", "win_loss", "beta"]:
        if col in df.columns:
            df[col] = df[col].map(lambda x: f"{x:.2f}" if pd.notna(x) else "")
    return df


# ---------------------------------------------------------------------------
# Position sizing
# ---------------------------------------------------------------------------

def equal_weight(symbols: list, long: list, short: list = None) -> Dict[str, float]:
    """Equal-weight dollar-neutral (if short given) or long-only."""
    w = {}
    n_long  = len(long)
    n_short = len(short) if short else 0
    for s in long:
        w[s] = 1.0 / n_long if n_long else 0.0
    if short:
        for s in short:
            w[s] = -1.0 / n_short if n_short else 0.0
    return w


def vol_weight(
    symbols: list,
    returns_df: pd.DataFrame,
    lookback: int = 63,
    vol_target: float = 0.10,
) -> Dict[str, float]:
    """
    Inverse-vol weighting scaled so each position targets *vol_target*
    (annualized). Returns raw weights (not necessarily summing to 1).
    """
    w = {}
    for s in symbols:
        if s not in returns_df.columns:
            continue
        r = returns_df[s].dropna().iloc[-lookback:]
        if len(r) < 20:
            continue
        v = r.std() * np.sqrt(TRADING_DAYS)
        if v > 0:
            w[s] = vol_target / v
    return w
