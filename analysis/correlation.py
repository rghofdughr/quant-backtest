"""
analysis/correlation.py
-----------------------
Correlation structure analysis for the 7 survivor strategies.

Outputs:
  results/corr_full.png       -- full-sample annotated heatmap
  results/corr_crash.png      -- crash-regime heatmap
  results/corr_rolling.png    -- rolling 252d avg pairwise correlation
  results/corr_eigenvalues.png -- eigenvalue / effective-bets bar chart
  Printed: verdict paragraph + N_eff + crash-vs-full comparison
"""
from __future__ import annotations

import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
import pandas as pd
import seaborn as sns

REPO_ROOT   = Path(__file__).parent.parent
sys.path.insert(0, str(REPO_ROOT))
RETURNS_DIR = REPO_ROOT / "results" / "returns"
RESULTS_DIR = REPO_ROOT / "results"

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

# crash-regime windows
CRASH_WINDOWS = [
    ("2008-09-01", "2009-03-31"),  # GFC
    ("2020-02-15", "2020-04-15"),  # COVID crash
    ("2022-01-01", "2022-10-31"),  # rate-hike drawdown
]

EQUITY_BETA_CLUSTER = {"s08", "s30", "s31", "s35"}   # for fair-weather verdict


# ---------------------------------------------------------------------------
# Load + align
# ---------------------------------------------------------------------------

def load_returns() -> pd.DataFrame:
    frames = {}
    for sid in SURVIVORS:
        path = RETURNS_DIR / f"{sid}.parquet"
        if not path.exists():
            print(f"WARNING: {path} not found — run analysis/save_returns.py first")
            continue
        s = pd.read_parquet(path).squeeze()
        s.index = pd.to_datetime(s.index)
        frames[LABELS[sid]] = s

    df = pd.DataFrame(frames)
    df = df.dropna()   # inner-join alignment
    return df


# ---------------------------------------------------------------------------
# Heatmap helper
# ---------------------------------------------------------------------------

def plot_corr_heatmap(corr: pd.DataFrame, title: str, save_path: Path,
                      n_obs: int = 0) -> None:
    fig, ax = plt.subplots(figsize=(8, 6.5))
    mask = np.zeros_like(corr, dtype=bool)
    mask[np.triu_indices_from(mask, k=1)] = False   # show full matrix

    sns.heatmap(
        corr,
        ax=ax,
        annot=True,
        fmt=".2f",
        cmap="RdBu_r",
        center=0,
        vmin=-1, vmax=1,
        linewidths=0.5,
        linecolor="white",
        annot_kws={"size": 9},
        square=True,
    )
    subtitle = f"n = {n_obs:,} days" if n_obs else ""
    ax.set_title(f"{title}\n{subtitle}", fontsize=11, fontweight="bold", pad=10)
    plt.xticks(rotation=30, ha="right", fontsize=8)
    plt.yticks(rotation=0, fontsize=8)
    plt.tight_layout()
    fig.savefig(save_path, dpi=130, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {save_path.name}")


# ---------------------------------------------------------------------------
# Effective number of bets
# ---------------------------------------------------------------------------

def effective_n_bets(corr: pd.DataFrame) -> tuple[float, np.ndarray]:
    """N_eff = (sum lambda)^2 / sum(lambda^2) = n^2 / sum(lambda^2)."""
    eigvals = np.linalg.eigvalsh(corr.values)
    eigvals = np.sort(eigvals)[::-1]
    n_eff = (eigvals.sum() ** 2) / (eigvals ** 2).sum()
    return float(n_eff), eigvals


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    print("Loading return series ...")
    df = load_returns()
    if df.empty or len(df.columns) < 2:
        print("ERROR: not enough return series loaded.")
        sys.exit(1)

    n, k = len(df), len(df.columns)
    print(f"Aligned: {n} trading days  ({df.index[0].date()} to {df.index[-1].date()})")
    print(f"Strategies: {k}  ({', '.join(df.columns.tolist())})")
    print()

    # -----------------------------------------------------------------------
    # 1. Full-sample correlation
    # -----------------------------------------------------------------------
    print("=" * 64)
    print("1. FULL-SAMPLE PAIRWISE CORRELATIONS")
    print("=" * 64)
    corr_full = df.corr()
    print(corr_full.round(3).to_string())
    print()
    plot_corr_heatmap(corr_full, "Full-Sample Correlation (2000–2024)",
                      RESULTS_DIR / "corr_full.png", n_obs=n)

    # -----------------------------------------------------------------------
    # 2. Crash-conditional correlation
    # -----------------------------------------------------------------------
    print("=" * 64)
    print("2. CRASH-CONDITIONAL CORRELATIONS")
    print("   (GFC 2008-09/2009-03, COVID 2020-02/04, Rates 2022-01/10)")
    print("=" * 64)
    crash_mask = pd.Series(False, index=df.index)
    for start, end in CRASH_WINDOWS:
        crash_mask |= (df.index >= pd.Timestamp(start)) & (df.index <= pd.Timestamp(end))

    df_crash = df[crash_mask]
    n_crash = len(df_crash)
    corr_crash = df_crash.corr()
    print(f"Crash observations: {n_crash} days ({n_crash/n:.1%} of sample)")
    print()
    print(corr_crash.round(3).to_string())
    print()
    plot_corr_heatmap(corr_crash, "Crash-Conditional Correlation (3 bear regimes)",
                      RESULTS_DIR / "corr_crash.png", n_obs=n_crash)

    # Delta: crash vs full
    print("CRASH vs FULL DELTA (crash_corr - full_corr) -- positive = correlation RISES in crash:")
    delta = corr_crash - corr_full
    print(delta.round(3).to_string())
    print()

    # Flag pairs with material increase
    upper = [(corr_full.columns[i], corr_full.columns[j],
              float(corr_full.iloc[i, j]), float(corr_crash.iloc[i, j]),
              float(delta.iloc[i, j]))
             for i in range(k) for j in range(i+1, k)]
    flagged = [(a, b, fc, cc, d) for a, b, fc, cc, d in upper if d > 0.10]
    if flagged:
        print("FAIR-WEATHER DIVERSIFICATION flags (delta > 0.10):")
        for a, b, fc, cc, d in sorted(flagged, key=lambda x: -x[4]):
            print(f"  {a} x {b}: full={fc:+.2f} -> crash={cc:+.2f}  (delta={d:+.2f})")
    else:
        print("No material fair-weather diversification (no pair rises > 0.10 in crash).")
    print()

    # -----------------------------------------------------------------------
    # 3. Rolling 252-day average pairwise correlation
    # -----------------------------------------------------------------------
    print("=" * 64)
    print("3. ROLLING 252-DAY AVERAGE PAIRWISE CORRELATION")
    print("=" * 64)

    # All upper-triangle pairs
    pairs = [(df.columns[i], df.columns[j])
             for i in range(k) for j in range(i+1, k)]

    rolling_avg = pd.Series(dtype=float, index=df.index)
    pair_rolling = {}
    for (a, b) in pairs:
        pair_rolling[(a, b)] = df[a].rolling(252).corr(df[b])

    rolling_avg = pd.concat(pair_rolling.values(), axis=1).mean(axis=1)

    fig, ax = plt.subplots(figsize=(12, 4))
    ax.plot(rolling_avg.index, rolling_avg.values, color="steelblue", lw=1.2, label="Avg pairwise corr")
    ax.axhline(float(rolling_avg.mean()), color="k", lw=0.8, ls="--", label=f"Mean {float(rolling_avg.mean()):.2f}")
    ax.axhline(0, color="gray", lw=0.5)
    ax.fill_between(rolling_avg.index, rolling_avg.values,
                    float(rolling_avg.mean()), alpha=0.15, color="steelblue")
    ax.set_title("Rolling 252-Day Average Pairwise Correlation — Survivor Sleeve", fontsize=11)
    ax.set_ylabel("Avg Pairwise Correlation")
    ax.legend(fontsize=8)
    ax.yaxis.set_major_formatter(mticker.FormatStrFormatter("%.2f"))
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    fig.savefig(RESULTS_DIR / "corr_rolling.png", dpi=130, bbox_inches="tight")
    plt.close(fig)
    print(f"  Mean rolling avg corr: {rolling_avg.mean():.3f}")
    print(f"  Max rolling avg corr:  {rolling_avg.max():.3f}  (on {rolling_avg.idxmax().date()})")
    print(f"  Min rolling avg corr:  {rolling_avg.min():.3f}  (on {rolling_avg.idxmin().date()})")
    print(f"  Saved: corr_rolling.png")
    print()

    # -----------------------------------------------------------------------
    # 4. Effective number of independent bets
    # -----------------------------------------------------------------------
    print("=" * 64)
    print("4. EFFECTIVE NUMBER OF INDEPENDENT BETS (N_eff)")
    print("=" * 64)
    n_eff_full, eigvals_full = effective_n_bets(corr_full)
    n_eff_crash, eigvals_crash = effective_n_bets(corr_crash)

    print(f"Full sample:  N_eff = {n_eff_full:.2f}  out of {k} strategies")
    print(f"Crash regime: N_eff = {n_eff_crash:.2f}  out of {k} strategies")
    print()
    print(f"Eigenvalues (full): {' '.join(f'{v:.3f}' for v in eigvals_full)}")
    pct_var = eigvals_full / eigvals_full.sum() * 100
    print(f"  PC1 explains {pct_var[0]:.1f}% of variance  (first 2: {pct_var[:2].sum():.1f}%)")
    print()

    # Eigenvalue bar chart
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    for ax, eigv, label in zip(axes,
                                [eigvals_full, eigvals_crash],
                                ["Full Sample", "Crash Regime"]):
        pv = eigv / eigv.sum() * 100
        n_eff_val = (eigv.sum()**2) / (eigv**2).sum()
        ax.bar(range(1, len(eigv)+1), pv, color="steelblue", alpha=0.8, edgecolor="white")
        ax.axhline(100/k, color="red", lw=1, ls="--", label=f"Equal ({100/k:.0f}%)")
        ax.set_title(f"{label}  (N_eff = {n_eff_val:.2f}/{k})", fontsize=9)
        ax.set_xlabel("Principal Component")
        ax.set_ylabel("% Variance Explained")
        ax.legend(fontsize=7)
        ax.grid(True, alpha=0.3, axis="y")
    plt.tight_layout()
    fig.savefig(RESULTS_DIR / "corr_eigenvalues.png", dpi=130, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: corr_eigenvalues.png")
    print()

    # -----------------------------------------------------------------------
    # 5. Verdict paragraph
    # -----------------------------------------------------------------------
    print("=" * 64)
    print("5. VERDICT")
    print("=" * 64)

    avg_corr_full  = np.mean([corr_full.iloc[i,j] for i in range(k) for j in range(i+1,k)])
    avg_corr_crash = np.mean([corr_crash.iloc[i,j] for i in range(k) for j in range(i+1,k)])

    # Identify the equity-beta cluster average correlation
    eq_labels = [LABELS[s] for s in EQUITY_BETA_CLUSTER if LABELS[s] in df.columns]
    if len(eq_labels) >= 2:
        eq_corrs = [corr_full.loc[a, b]
                    for i, a in enumerate(eq_labels)
                    for b in eq_labels[i+1:]]
        avg_eq_corr = float(np.mean(eq_corrs))
    else:
        avg_eq_corr = np.nan

    # S46 vs equity cluster correlation
    s46_label = LABELS["s46"]
    s46_vs_eq = [float(corr_full.loc[s46_label, lbl])
                 for lbl in eq_labels if lbl in corr_full.columns]
    avg_s46_vs_eq = float(np.mean(s46_vs_eq)) if s46_vs_eq else np.nan

    fair_weather = len(flagged) > 0
    concentration_risk = n_eff_full < 3.0
    crash_corr_rises = avg_corr_crash > avg_corr_full + 0.08

    print()
    print("DIVERSIFICATION VERDICT:")
    print("-" * 60)

    verdict_lines = [
        f"Full-sample average pairwise correlation: {avg_corr_full:.2f}  "
        f"(crash-regime: {avg_corr_crash:.2f}{'  [RISES IN STRESS]' if crash_corr_rises else ''}).",
        "",
        f"Effective independent bets: {n_eff_full:.1f} out of {k} strategies full-sample, "
        f"{n_eff_crash:.1f} in crash regimes.",
        "",
    ]

    if not np.isnan(avg_eq_corr):
        verdict_lines.append(
            f"The EQUITY-BETA CLUSTER (S08 Sector, S30 LowVol, S31 VolTgt, S35 SellMay) "
            f"has average within-cluster correlation {avg_eq_corr:.2f}. "
        )
        if not np.isnan(avg_s46_vs_eq):
            verdict_lines.append(
                f"S46 Risk Parity correlates {avg_s46_vs_eq:.2f} on average to that cluster, "
                f"making it the primary real diversifier. "
                f"S02 TS-Momentum and S49 Dollar Regime sit between the two groups."
            )
        verdict_lines.append("")

    if concentration_risk:
        verdict_lines.append(
            f"WARNING: N_eff = {n_eff_full:.1f} < 3.0. With {k} strategies, fewer than "
            f"3 effective independent bets means this sleeve is more concentrated than it looks. "
            "The first principal component dominates."
        )
    else:
        verdict_lines.append(
            f"N_eff = {n_eff_full:.1f} is reasonable for {k} strategies. "
            "Diversification is real but not as broad as raw strategy count implies."
        )

    if fair_weather:
        verdict_lines.append(
            f"\nFAIR-WEATHER WARNING: {len(flagged)} pair(s) show correlation rising "
            f"> 0.10 during crash regimes. The diversification partially collapses "
            f"precisely when it is needed most. Factor this into position sizing."
        )
    else:
        verdict_lines.append(
            "\nCorrelations do NOT materially rise during crash regimes relative to full-sample. "
            "The diversification holds up in stress — this is the desirable property."
        )

    for line in verdict_lines:
        print(f"  {line}")

    print()
    print("BOTTOM LINE:")
    if n_eff_full < 2.5:
        print("  This is essentially ONE equity-beta bet (N_eff < 2.5). S46 is the only hedge.")
    elif n_eff_full < 3.5:
        print("  Roughly 3 independent bets: the equity-beta cluster, S46 (bonds/real-assets),")
        print("  and the trend/momentum strategies (S02/S49). Not 7 independent sources of alpha.")
    else:
        print(f"  {n_eff_full:.1f} effective bets — meaningful diversification, but less than strategy count.")


if __name__ == "__main__":
    main()
