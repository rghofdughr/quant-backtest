"""Quick Task 1 table from existing Group 2 metrics JSONs — no re-running needed."""
import json, sys, numpy as np, pandas as pd
from pathlib import Path

REPO = Path(__file__).parent.parent
RESULTS_DIR = REPO / "results"

G2_NAMES = {
    "s66":"Vol-Confirmed Mom","s67":"Amihud Illiquidity","s68":"Mom Ensemble",
    "s69":"Sharpe Rank","s70":"MaxDD Quality","s71":"52-Wk Breakout",
    "s72":"Reversal Demeaned","s73":"Residual Mom","s74":"Accel Breadth",
    "s75":"Donchian Equity","s76":"MA200 Band","s77":"Dual Momentum",
    "s78":"Vol Trend ETF","s79":"Adaptive Trend","s80":"IBS Reversion",
    "s81":"Jan Reversal PIT","s82":"Month-End Flow","s83":"DOW Conditional",
    "s84":"FOMC Week","s85":"Gap Go","s86":"Range Expansion","s87":"Vol Spike",
    "s88":"NR7","s89":"DV Momentum","s90":"Credit Regime","s91":"Inflation Tilt",
    "s92":"Country ETF Mom","s93":"Defensive Rotation","s94":"Index Deletion",
    "s95":"R2000 Promotion","s96":"Deep Value","s97":"Div Capture",
    "s98":"Ex-Date Drift","s99":"Div Initiation","s100":"Distressed",
    "s101":"Sector Pairs","s102":"ETF Basket Arb",
}

def classify(is_sh, oos_sh):
    if is_sh is None or np.isnan(is_sh): return "STUB"
    if is_sh <= 0.3: return "WEAK"
    if oos_sh is None or np.isnan(oos_sh): return "WEAK"
    if oos_sh < 0: return "FRAGILE"
    if oos_sh < 0.5 * is_sh: return "MIRAGE"
    if oos_sh >= 0.7 * is_sh: return "ROBUST"
    return "DECAY"

def safe(v):
    return float(v) if v is not None else np.nan

rows = []
all_ids = [f"s{i}" for i in range(66, 103)] + ["s102"]
for sid in sorted(set(all_ids), key=lambda x: int(x[1:])):
    path = RESULTS_DIR / f"{sid}_metrics.json"
    if not path.exists():
        rows.append({"id":sid,"name":G2_NAMES.get(sid,sid),"IS_SR":np.nan,"OOS_SR":np.nan,
                     "decay":np.nan,"IS_CAGR":np.nan,"OOS_CAGR":np.nan,"OOS_MDD":np.nan,
                     "IS_MDD":np.nan,"Full_SR":np.nan,"Full_CAGR":np.nan,
                     "TO":np.nan,"cost_drag":np.nan,"flag":"MISSING"})
        continue
    with open(path) as f:
        d = json.load(f)
    is_m = d.get("is",{}); oos_m = d.get("oos",{}); full_m = d.get("full",{})
    is_sh  = safe(is_m.get("sharpe"))
    oos_sh = safe(oos_m.get("sharpe"))
    to     = safe(d.get("turnover_annual"))
    cd     = to * 20 / 10000 if not np.isnan(to) else np.nan
    rows.append({
        "id":sid, "name":G2_NAMES.get(sid,sid),
        "IS_SR":is_sh, "OOS_SR":oos_sh,
        "decay":oos_sh - is_sh if not(np.isnan(is_sh) or np.isnan(oos_sh)) else np.nan,
        "IS_CAGR":safe(is_m.get("cagr")), "OOS_CAGR":safe(oos_m.get("cagr")),
        "Full_SR":safe(full_m.get("sharpe")), "Full_CAGR":safe(full_m.get("cagr")),
        "OOS_MDD":safe(oos_m.get("max_dd")), "IS_MDD":safe(is_m.get("max_dd")),
        "TO":to, "cost_drag":cd,
        "flag":classify(is_sh, oos_sh),
    })

df = pd.DataFrame(rows).sort_values("OOS_SR", ascending=False)

print()
print("=" * 120)
print("  TASK 1 — GROUP 2 IS/OOS TABLE  (full 2000-01-03 to 2024-12-31, 70/30 split)")
print("  IS: 2000-01-03 to 2017-06-30  |  OOS: 2017-07-03 to 2024-12-31 (~7.8yr)")
print("  Costs: 5+5 bps/side already applied in each strategy. cost_drag = TO * 20bps (reference only).")
print("=" * 120)
hdr = f"{'ID':<7}{'Name':<24}{'IS_SR':>7}{'OOS_SR':>8}{'Decay':>7}{'IS_CAGR':>9}{'OOS_CAGR':>10}{'OOS_MDD':>9}{'IS_MDD':>9}{'TO/yr':>7}{'CostDrag':>9}  Flag"
print(hdr)
print("-" * 120)

for _, r in df.iterrows():
    def fs(v): return f"{v:+6.2f}" if not np.isnan(v) else "   N/A"
    def fp(v): return f"{v:+7.1%}" if not np.isnan(v) else "    N/A"
    def ft(v): return f"{v:5.1f}x" if not np.isnan(v) else "  N/A"
    def fc(v): return f"{v:.1%}" if not np.isnan(v) else "  N/A"
    hi  = " [HIGH-TO]" if (not np.isnan(r.TO) and r.TO > 10) else ""
    dc  = f"{r.decay:+.2f}" if not np.isnan(r.decay) else "   N/A"
    print(f"{r.id:<7}{r['name'][:24]:<24}{fs(r.IS_SR):>7}{fs(r.OOS_SR):>8}{dc:>7}"
          f"{fp(r.IS_CAGR):>9}{fp(r.OOS_CAGR):>10}{fp(r.OOS_MDD):>9}{fp(r.IS_MDD):>9}"
          f"{ft(r.TO):>7}{fc(r.cost_drag):>9}  {r.flag}{hi}")

print()
print("Flag: ROBUST=OOS>=70%IS (both>0.3) | DECAY=OOS 50-70%IS | MIRAGE=OOS<50%IS | FRAGILE=OOS<0 | WEAK=IS<=0.3")
print()
for flag in ["ROBUST","DECAY","MIRAGE","FRAGILE","WEAK","MISSING"]:
    sub = df[df["flag"] == flag]
    if sub.empty: continue
    ids_list = " ".join(sorted(sub["id"].tolist(), key=lambda x: int(x[1:])))
    print(f"  {flag:<9}({len(sub):2d}): {ids_list}")

print()
print("HIGH-TURNOVER cost death (costs already applied — shown for execution-risk awareness):")
for _, r in df[df["TO"] > 10].iterrows():
    print(f"  {r.id} {r['name']}: {r.TO:.0f}x/yr turnover, {r.cost_drag:.1%}/yr theoretical drag, "
          f"OOS_CAGR={r.OOS_CAGR:.1%}, flag={r.flag}")

print()
# Survivors
survivors = df[df["OOS_SR"] >= 0.5]["id"].tolist()
print(f"Survivors (OOS SR >= 0.5): {len(survivors)}")
for sid in survivors:
    r = df[df["id"] == sid].iloc[0]
    print(f"  {sid} {r['name']}: IS {r.IS_SR:.2f} -> OOS {r.OOS_SR:.2f} ({r.flag}), "
          f"OOS CAGR {r.OOS_CAGR:.1%}, MDD {r.OOS_MDD:.1%}")

# Save CSV
df.to_csv(RESULTS_DIR / "group2_is_oos_table.csv", index=False)
print()
print(f"Saved -> results/group2_is_oos_table.csv")
