"""
analysis/montecarlo.py
----------------------
Block-bootstrap Monte Carlo on the survivor sleeve.

Resampling method: fixed-block circular bootstrap, block length L=50 trading days
(~2.5 months). Whole rows of the aligned multi-strategy + SPY return matrix are
resampled together, preserving cross-strategy and strategy-vs-benchmark correlation
by construction. Strategies are NEVER resampled independently.

Portfolio specs simulated:
  Core  (3 strategies): S08, S46, S30
  Full  (7 strategies): S08, S46, S30 + S31, S35, S02, S49

Weighting schemes per portfolio:
  1. Equal-weight
  2. Inverse-vol (in-sample vol, no look-ahead: computed on first 70% of dates)
  3. Correlation-aware cluster parity (see CLUSTER_WEIGHTS below)

Cluster rule (from correlation.py Task 1 findings):
  Equity-beta cluster:  S08, S30, S31, S35, S49  — lock up in crashes (N_eff 1.9)
  Multi-asset cluster:  S46, S02                  — genuinely lower correlation
  Equal budget between clusters (50/50).
  Within cluster: equal weight.

  CORE  (S08, S30, S46): S08=25%, S30=25%, S46=50%
  FULL  (all 7):         S08=10%, S30=10%, S31=10%, S35=10%, S49=10%,
                         S46=25%, S02=25%

For each scheme: 10,000 simulated paths at 25-year and 5-year horizons.
Report: CAGR/vol/Sharpe/MDD percentile distribution, probability of ruin events,
probability of underperforming SPY on same bootstrapped blocks.

Outputs:
  results/mc_fan_*.png          — fan charts
  results/mc_metrics_*.csv      — percentile tables
  (MONTECARLO_SUMMARY.md written by a separate final step)

Usage:
  cd C:\\Users\\Owner\\quant50
  python analysis/montecarlo.py
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
import pandas as pd

REPO_ROOT   = Path(__file__).parent.parent
sys.path.insert(0, str(REPO_ROOT))
RETURNS_DIR = REPO_ROOT / "results" / "returns"
RESULTS_DIR = REPO_ROOT / "results"

# ---- config ----------------------------------------------------------------
N_SIMS      = 10_000
BLOCK_LEN   = 50      # trading days per block (~2.5 months)
BATCH_SIZE  = 500     # paths per batch (memory management)
TRADING_DAYS= 252
HORIZON_25  = 25 * TRADING_DAYS    # ~6300 trading days
HORIZON_5   =  5 * TRADING_DAYS    # ~1260 trading days

SURVIVORS = ["s08", "s46", "s30", "s31", "s35", "s02", "s49"]
LABELS = {
    "s02": "S02 TS-Mom",
    "s08": "S08 Sector",
    "s30": "S30 LowVol",
    "s31": "S31 VolTgt",
    "s35": "S35 SellMay",
    "s46": "S46 RiskPar",
    "s49": "S49 DollarR",
}

IS_CUTOFF = 0.70   # in-sample fraction for vol estimation (no look-ahead)

# Cluster-aware weights (from Task 1 correlation analysis)
# Equity-beta cluster: S08, S30, S31, S35, S49 (50% budget total)
# Multi-asset cluster: S46, S02 (50% budget total)
CLUSTER_WEIGHTS_FULL = {
    "s08": 0.10, "s30": 0.10, "s31": 0.10, "s35": 0.10, "s49": 0.10,
    "s46": 0.25, "s02": 0.25,
}
# Core: S08 + S30 (equity cluster, 50%) vs S46 (multi-asset, 50%)
CLUSTER_WEIGHTS_CORE = {
    "s08": 0.25, "s30": 0.25, "s46": 0.50,
}


# ===========================================================================
# Load data
# ===========================================================================

def load_data() -> tuple[pd.DataFrame, pd.Series]:
    frames = {}
    for sid in SURVIVORS:
        path = RETURNS_DIR / f"{sid}.parquet"
        if not path.exists():
            raise FileNotFoundError(f"{path} — run analysis/save_returns.py first")
        s = pd.read_parquet(path).squeeze()
        s.index = pd.to_datetime(s.index)
        frames[sid] = s

    spy_path = RETURNS_DIR / "spy.parquet"
    spy = pd.read_parquet(spy_path).squeeze()
    spy.index = pd.to_datetime(spy.index)

    df = pd.DataFrame(frames).dropna()
    spy = spy.reindex(df.index).fillna(0.0)
    return df, spy


# ===========================================================================
# Compute in-sample vols (no look-ahead)
# ===========================================================================

def compute_is_vols(df: pd.DataFrame) -> pd.Series:
    n_is = int(len(df) * IS_CUTOFF)
    return df.iloc[:n_is].std() * np.sqrt(TRADING_DAYS)


def inverse_vol_weights(sids: list[str], vols: pd.Series) -> dict[str, float]:
    inv = {s: 1.0 / vols[s] for s in sids if vols[s] > 0}
    total = sum(inv.values())
    return {s: v / total for s, v in inv.items()}


# ===========================================================================
# Portfolio specs
# ===========================================================================

def build_specs(df: pd.DataFrame, spy: pd.Series) -> list[dict]:
    vols = compute_is_vols(df)
    core  = ["s08", "s46", "s30"]
    full  = SURVIVORS

    specs = []
    for label, sids in [("Core (S08+S46+S30)", core), ("Full (all 7)", full)]:
        k = len(sids)
        # 1. Equal weight
        specs.append({
            "name": f"{label} | Equal-weight",
            "sids": sids,
            "weights": {s: 1/k for s in sids},
            "is_core": (sids == core),
        })
        # 2. Inverse-vol
        ivw = inverse_vol_weights(sids, vols)
        specs.append({
            "name": f"{label} | Inverse-vol",
            "sids": sids,
            "weights": ivw,
            "is_core": (sids == core),
        })
        # 3. Cluster-aware
        cw = CLUSTER_WEIGHTS_CORE if sids == core else CLUSTER_WEIGHTS_FULL
        cw_sub = {s: cw[s] for s in sids}
        total = sum(cw_sub.values())
        cw_sub = {s: v/total for s, v in cw_sub.items()}
        specs.append({
            "name": f"{label} | Cluster-parity",
            "sids": sids,
            "weights": cw_sub,
            "is_core": (sids == core),
        })
    return specs


# ===========================================================================
# Fixed-block circular bootstrap (vectorized)
# ===========================================================================

def block_bootstrap_batch(data: np.ndarray, T_out: int,
                           n_paths: int, block_len: int) -> np.ndarray:
    """
    Fixed-block circular bootstrap.
    data:    (T_in, K) — all strategies + SPY in last column
    Returns: (n_paths, T_out, K)
    """
    T_in, K = data.shape
    n_blocks = (T_out + block_len - 1) // block_len

    # Draw block starts: (n_paths, n_blocks)
    starts = np.random.randint(0, T_in, size=(n_paths, n_blocks))

    # Build index matrix: (n_paths, n_blocks, block_len)
    offsets = np.arange(block_len)
    idx = (starts[:, :, np.newaxis] + offsets[np.newaxis, np.newaxis, :]) % T_in

    # Flatten to (n_paths, n_blocks * block_len) then trim to T_out
    idx = idx.reshape(n_paths, -1)[:, :T_out]   # (n_paths, T_out)

    # Index into data: (n_paths, T_out, K)
    return data[idx]


# ===========================================================================
# Path metrics
# ===========================================================================

def path_metrics(port_rets: np.ndarray, spy_rets: np.ndarray,
                 T_days: int) -> dict[str, np.ndarray]:
    """
    port_rets: (n_paths, T_days)
    spy_rets:  (n_paths, T_days)
    Returns dict of metric arrays (n_paths,).
    """
    n_years = T_days / TRADING_DAYS

    # Portfolio equity curves
    cum  = np.cumprod(1 + port_rets, axis=1)           # (n_paths, T)
    peak = np.maximum.accumulate(cum, axis=1)
    dd   = (cum - peak) / peak
    mdd  = dd.min(axis=1)                               # (n_paths,)

    cagr   = cum[:, -1] ** (1 / n_years) - 1
    vol    = port_rets.std(axis=1) * np.sqrt(TRADING_DAYS)
    sharpe = np.where(vol > 0, (port_rets.mean(axis=1) * TRADING_DAYS) / vol, np.nan)

    # SPY equity curves
    spy_cum = np.cumprod(1 + spy_rets, axis=1)
    spy_mdd_arr = (spy_cum - np.maximum.accumulate(spy_cum, axis=1)).min(axis=1) / \
                   np.maximum.accumulate(spy_cum, axis=1).max(axis=1)

    spy_cagr  = spy_cum[:, -1] ** (1 / n_years) - 1

    underperf = (cum[:, -1] < spy_cum[:, -1]).astype(float)

    return {
        "cagr":       cagr,
        "vol":        vol,
        "sharpe":     sharpe,
        "mdd":        mdd,
        "spy_cagr":   spy_cagr,
        "underperf":  underperf,
        "dd_30":      (mdd < -0.30).astype(float),
        "dd_40":      (mdd < -0.40).astype(float),
        "dd_50":      (mdd < -0.50).astype(float),
        # equity curve sample points (11 evenly spaced incl t=0)
        "eq_sample":  np.concatenate([np.ones((len(cum), 1)),
                                      cum[:, np.linspace(0, T_days-1, 10, dtype=int)]],
                                     axis=1),
    }


def pct(arr: np.ndarray, p: float) -> float:
    return float(np.percentile(arr, p * 100))


def metrics_table(m: dict[str, np.ndarray], name: str, horizon_yr: int) -> str:
    lines = [f"\n### {name}  |  {horizon_yr}-year horizon\n"]
    lines.append(
        f"| Metric         |  5th  | 25th  | median | 75th  | 95th  |\n"
        f"|----------------|-------|-------|--------|-------|-------|\n"
    )
    for key, label in [("cagr", "CAGR"), ("vol", "Ann Vol"), ("sharpe", "Sharpe"), ("mdd", "Max DD")]:
        arr = m[key]
        sign = "" if key != "mdd" else ""
        fmt = ".1%" if key in ("cagr", "vol", "mdd") else ".2f"
        row = f"| {label:<14} | {arr[arr == arr].min():^5} "  # placeholder
        vals = [pct(arr, p) for p in [0.05, 0.25, 0.50, 0.75, 0.95]]
        if key in ("cagr", "vol", "mdd"):
            row = "| {:<14} |{:>6.1%} |{:>6.1%} |{:>7.1%} |{:>6.1%} |{:>6.1%} |".format(
                label, *vals)
        else:
            row = "| {:<14} |{:>6.2f} |{:>6.2f} |{:>7.2f} |{:>6.2f} |{:>6.2f} |".format(
                label, *vals)
        lines.append(row + "\n")

    lines.append("\n**Risk events:**\n")
    lines.append(f"- P(underperform SPY): **{m['underperf'].mean():.1%}**\n")
    lines.append(f"- P(MDD > 30%):        **{m['dd_30'].mean():.1%}**\n")
    lines.append(f"- P(MDD > 40%):        **{m['dd_40'].mean():.1%}**\n")
    lines.append(f"- P(MDD > 50%):        **{m['dd_50'].mean():.1%}**\n")
    lines.append(f"- 95th-pct MDD (size against): **{pct(m['mdd'], 0.95):.1%}**\n")
    return "".join(lines)


# ===========================================================================
# Fan chart
# ===========================================================================

def plot_fan(all_eq: np.ndarray, spy_eq: np.ndarray,
             title: str, horizon_yr: int, save_path: Path) -> None:
    """
    all_eq:  (N_SIMS, 11) equity curve samples
    spy_eq:  (N_SIMS, 11)
    """
    xs = np.linspace(0, horizon_yr, all_eq.shape[1])
    pcts = [5, 25, 50, 75, 95]
    colors = ["#c6dbef", "#6baed6", "#2171b5", "#6baed6", "#c6dbef"]

    fig, ax = plt.subplots(figsize=(11, 5.5))

    perc = np.percentile(all_eq, pcts, axis=0)  # (5, 11)
    spy_perc = np.percentile(spy_eq, [5, 50, 95], axis=0)

    ax.fill_between(xs, perc[0], perc[4], alpha=0.20, color="#2171b5", label="5-95th pct")
    ax.fill_between(xs, perc[1], perc[3], alpha=0.35, color="#2171b5", label="25-75th pct")
    ax.plot(xs, perc[2], color="#08519c", lw=2.0, label="Median")

    ax.plot(xs, spy_perc[1], color="gray", lw=1.5, ls="--", label="SPY median")
    ax.fill_between(xs, spy_perc[0], spy_perc[2], alpha=0.10, color="gray", label="SPY 5-95th")

    ax.set_yscale("log")
    ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda y, _: f"{y:.1f}x"))
    ax.set_xlabel(f"Years (block bootstrap, L=50d, N={N_SIMS:,})", fontsize=9)
    ax.set_ylabel("Growth of $1 (log scale)", fontsize=9)
    ax.set_title(title, fontsize=10, fontweight="bold")
    ax.legend(fontsize=7.5, loc="upper left", ncol=2)
    ax.grid(True, alpha=0.3, which="both")
    plt.tight_layout()
    fig.savefig(save_path, dpi=130, bbox_inches="tight")
    plt.close(fig)
    print(f"  Fan chart saved: {save_path.name}")


# ===========================================================================
# Main
# ===========================================================================

def main():
    print("Loading return series ...")
    df, spy = load_data()
    T_in = len(df)
    print(f"  {T_in} days  ({df.index[0].date()} to {df.index[-1].date()})")
    print(f"  Strategies: {list(df.columns)}")
    print()

    specs = build_specs(df, spy)
    print(f"Portfolio specs to simulate ({len(specs)} total):")
    for sp in specs:
        w_str = "  ".join(f"{s}: {w:.0%}" for s, w in sorted(sp["weights"].items()))
        print(f"  [{sp['name']}]  {w_str}")
    print()

    # ---- Combined data matrix (strategies + SPY) ----
    combined = np.column_stack([df.values, spy.values.reshape(-1, 1)])
    K_strat = df.shape[1]

    results_md = ["# Monte Carlo Summary\n\n"]
    results_md.append(
        f"Block-bootstrap Monte Carlo: N={N_SIMS:,} paths, block L={BLOCK_LEN} days "
        f"({BLOCK_LEN/TRADING_DAYS:.1f} months)  \n"
        f"Common window bootstrapped: {df.index[0].date()} to {df.index[-1].date()}  "
        f"({T_in:,} trading days)  \n"
        "Strategies and SPY resampled together (same block draws) — SPY comparison is apples-to-apples.  \n\n"
    )

    for sp in specs:
        sids    = sp["sids"]
        weights = sp["weights"]
        name    = sp["name"]
        col_idx = [list(df.columns).index(s) for s in sids]

        w_arr = np.array([weights[s] for s in sids])   # (k,)
        spy_col = K_strat   # last column index in combined

        print(f"\n{'='*64}")
        print(f"Simulating: {name}")
        print(f"  Weights: { {s: f'{w:.1%}' for s, w in weights.items()} }")
        t0 = time.time()

        # Storage
        all_metrics_25 = {k: [] for k in ["cagr","vol","sharpe","mdd","underperf","dd_30","dd_40","dd_50"]}
        all_metrics_5  = {k: [] for k in ["cagr","vol","sharpe","mdd","underperf","dd_30","dd_40","dd_50"]}
        all_eq_25  = []
        all_eq_5   = []
        spy_eq_25  = []
        spy_eq_5   = []

        n_batches = N_SIMS // BATCH_SIZE
        for b in range(n_batches):
            # Bootstrap: (BATCH_SIZE, HORIZON_25, K_strat+1)
            paths = block_bootstrap_batch(combined, HORIZON_25, BATCH_SIZE, BLOCK_LEN)

            # Portfolio returns: (BATCH_SIZE, HORIZON_25)
            strat_paths = paths[:, :, col_idx]   # (B, T, k)
            port_rets_25 = (strat_paths * w_arr[np.newaxis, np.newaxis, :]).sum(axis=2)
            spy_rets_25  = paths[:, :, spy_col]

            # 5-year slice
            port_rets_5 = port_rets_25[:, :HORIZON_5]
            spy_rets_5  = spy_rets_25[:, :HORIZON_5]

            # 25-year metrics
            m25 = path_metrics(port_rets_25, spy_rets_25, HORIZON_25)
            for k in all_metrics_25:
                all_metrics_25[k].append(m25[k])
            all_eq_25.append(m25["eq_sample"])
            spy_eq_25.append(np.concatenate([
                np.ones((BATCH_SIZE, 1)),
                np.cumprod(1+spy_rets_25, axis=1)[:, np.linspace(0, HORIZON_25-1, 10, dtype=int)]
            ], axis=1))

            # 5-year metrics
            m5 = path_metrics(port_rets_5, spy_rets_5, HORIZON_5)
            for k in all_metrics_5:
                all_metrics_5[k].append(m5[k])
            all_eq_5.append(m5["eq_sample"])
            spy_eq_5.append(np.concatenate([
                np.ones((BATCH_SIZE, 1)),
                np.cumprod(1+spy_rets_5, axis=1)[:, np.linspace(0, HORIZON_5-1, 10, dtype=int)]
            ], axis=1))

            if (b+1) % 5 == 0:
                print(f"  batch {b+1}/{n_batches}  ({time.time()-t0:.0f}s elapsed)")

        elapsed = time.time() - t0
        print(f"  Done in {elapsed:.0f}s")

        # Concatenate
        for k in all_metrics_25:
            all_metrics_25[k] = np.concatenate(all_metrics_25[k])
        for k in all_metrics_5:
            all_metrics_5[k] = np.concatenate(all_metrics_5[k])

        eq_25_arr  = np.concatenate(all_eq_25,  axis=0)
        eq_5_arr   = np.concatenate(all_eq_5,   axis=0)
        spy25_arr  = np.concatenate(spy_eq_25,  axis=0)
        spy5_arr   = np.concatenate(spy_eq_5,   axis=0)

        # Print summary
        print(f"\n  25-year:  median CAGR {np.median(all_metrics_25['cagr']):.1%}  "
              f"med Sharpe {np.median(all_metrics_25['sharpe']):.2f}  "
              f"95pct MDD {np.percentile(all_metrics_25['mdd'],95):.1%}  "
              f"P(>30%DD) {all_metrics_25['dd_30'].mean():.1%}")
        print(f"  5-year:   median CAGR {np.median(all_metrics_5['cagr']):.1%}  "
              f"med Sharpe {np.median(all_metrics_5['sharpe']):.2f}  "
              f"95pct MDD {np.percentile(all_metrics_5['mdd'],95):.1%}  "
              f"P(>30%DD) {all_metrics_5['dd_30'].mean():.1%}")

        # Save metrics CSV
        safe_name = name.replace(" ", "_").replace("|", "").replace("(", "").replace(")", "").replace("/", "").replace("+", "")[:40]
        csv_path = RESULTS_DIR / f"mc_metrics_{safe_name}.csv"
        rows = []
        for horizon, m in [(25, all_metrics_25), (5, all_metrics_5)]:
            for metric, arr in m.items():
                if metric in ("eq_sample",):
                    continue
                rows.append({
                    "portfolio": name, "horizon_yr": horizon, "metric": metric,
                    "p05": pct(arr, 0.05), "p25": pct(arr, 0.25),
                    "p50": pct(arr, 0.50), "p75": pct(arr, 0.75), "p95": pct(arr, 0.95),
                    "mean": float(arr.mean()),
                })
        pd.DataFrame(rows).to_csv(csv_path, index=False)

        # Fan charts
        tag = safe_name[:30]
        plot_fan(eq_25_arr, spy25_arr, f"{name}  |  25-Year Horizon", 25,
                 RESULTS_DIR / f"mc_fan_{tag}_25yr.png")
        plot_fan(eq_5_arr, spy5_arr, f"{name}  |  5-Year Horizon", 5,
                 RESULTS_DIR / f"mc_fan_{tag}_5yr.png")

        # Append to markdown
        results_md.append(f"\n---\n\n## {name}\n\n")
        results_md.append(metrics_table(all_metrics_25, name, 25))
        results_md.append(metrics_table(all_metrics_5,  name, 5))

    # Save combined MD
    md_path = RESULTS_DIR / "mc_tables.md"
    with open(md_path, "w", encoding="utf-8") as f:
        f.write("".join(results_md))
    print(f"\nMetrics tables saved -> {md_path}")


if __name__ == "__main__":
    main()
