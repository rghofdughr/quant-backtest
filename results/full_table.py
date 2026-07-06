"""full_table.py — all strategies, every metric, sorted by OOS SR"""
import json, glob, os

NAMES = {
    's01':'12-1 Momentum (R1k)','s02':'TS Momentum (Futures)','s03':'3-Month Mom (R1k)',
    's04':'6-Month Mom (R1k)','s05':'1-Month Mom (R1k)','s06':'Reversal 1wk (R1k)',
    's07':'Donchian (Futures)','s08':'Sector Rotation (SPDRs)','s09':'Short-Term Rev (R1k)',
    's10':'Bollinger Mean Rev','s11':'RSI Mean Rev (R1k)','s12':'Pairs Trading',
    's13':'Beta Neutral Mom','s14':'Frog-in-Pan Mom','s15':'Industry Mom (R1k)',
    's16':'Book/Market','s17':'Earnings Yield','s18':'EV/EBITDA',
    's19':'Piotroski F-Score','s20':'Gross Profit/Assets','s21':'Asset Growth Rev',
    's22':'Accruals','s23':'Earnings Revision','s24':'Analyst Upgrade',
    's25':'Earnings Momentum','s26':'Dividend Yield','s27':'VIX Carry',
    's28':'Short Straddle','s29':'Covered Call','s30':'Low Volatility (R1k)',
    's31':'Vol Targeting (SPY)','s32':'Vol Dispersion','s33':'IV Crush',
    's34':'Skewness Anomaly','s35':'Sell in May','s36':'Halloween Effect',
    's37':'Turn-of-Month (old)','s38':'Jan Effect (R2k)','s39':'Pre-Holiday (old)',
    's40':'Post-Earnings Drift','s41':'Conf Interval Mom','s42':'Insider Buying',
    's43':'Analyst Revision','s44':'Merger Arb','s45':'Short Squeeze',
    's46':'Risk Parity (5-asset)','s47':'Equal Risk Contrib','s48':'Min Variance',
    's49':'Dollar Regime','s50':'Managed Futures Trend','s51':'Commodity Mom',
    's52':'Idiosyncratic Vol','s53':'Lottery Demand','s54':'ADX Momentum',
    's55':'Price Oscillator','s56':'MACD Cross','s57':'Keltner Breakout',
    's58':'Chandelier Exit','s59':'Vol-of-Vol Regime','s60':'Corr Regime',
    's61':'Sortino Momentum','s62':'Tail Risk Hedge','s63':'ETF Breakout',
    's64':'Pairs Cointegration','s65':'Stat Arb ETF','s66':'Vol-Confirmed Mom',
    's67':'Quality Factor','s68':'Mom Ensemble','s69':'Sharpe Rank',
    's70':'Risk-Adj Mom','s71':'52wk Breakout (R1k)','s72':'Overnight vs Day',
    's73':'Residual Momentum','s74':'Sector Breadth','s75':'Donchian Equity',
    's76':'MA200 Band (R1k)','s77':'Dual Momentum (GEM)','s78':'Vol Trend ETF',
    's79':'Adaptive Trend','s80':'Vol Regime Switch','s81':'Jan Reversal PIT',
    's82':'Earnings Season','s83':'Options Exp Week','s84':'FOMC Drift',
    's85':'Earnings Whisper','s86':'Supply Chain Mom','s87':'ESG Tilt',
    's88':'Multi-Factor','s89':'Alt Risk Parity','s90':'Credit Regime (HYG/LQD)',
    's91':'Inflation Tilt','s92':'Dollar Carry','s93':'Defensive Rotation',
    's94':'Index Deletion','s95':'S&P Rebalance','s96':'Earnings Drift v2',
    's97':'Div Capture','s98':'Ex-Date Drift','s99':'Div Initiation',
    's100':'Sector Pairs','s101':'Factor Zoo','s102':'ETF Basket Arb',
    # New batch
    's103':'Yield Curve Regime','s104':'Gold Regime','s105':'Oil/Energy Regime',
    's106':'Global Momentum','s107':'Real Estate (VNQ)','s108':'Weekly Reversal (R1k)',
    's109':'Sector RSI Reversion','s110':'Monthly Reversal (R1k)','s111':'ETF Z-score Rev',
    's112':'Turn of Month','s113':'January Barometer','s114':'Pre-Holiday',
    's115':'Year-End Reversal (R1k)','s116':'Weekday Effect','s117':'Vol Spike Recovery',
    's118':'Low-Vol Rotation','s119':'Vol Compression','s120':'EMA Cross Multi-Asset',
    's121':'Breadth Thrust','s122':'52wk High Prox (R1k)',
}

STATUS = {
    's08':'VALIDATED','s31':'VALIDATED','s46':'VALIDATED','s30':'VALIDATED',
    's02':'VALIDATED','s35':'VALIDATED','s49':'VALIDATED','s90':'VALIDATED',
    's115':'SURVIVOR*',
    's07':'ARTIFACT','s78':'ARTIFACT','s79':'ARTIFACT','s81':'ARTIFACT',
    's63':'ARTIFACT','s93':'ARTIFACT','s91':'ARTIFACT',
    's75':'REDUNDANT','s71':'REDUNDANT',
    's52':'REDUNDANT','s54':'REDUNDANT','s59':'REDUNDANT','s61':'REDUNDANT',
    's66':'REDUNDANT','s68':'REDUNDANT','s69':'REDUNDANT','s73':'REDUNDANT',
    's76':'REDUNDANT','s77':'REDUNDANT',
    's104':'REDUNDANT','s105':'REDUNDANT','s118':'REDUNDANT',
    's121':'BORDERLINE','s122':'REDUNDANT',
    's60':'REGIME-TIMER','s119':'MIRAGE',
    's26':'PROXY','s45':'PROXY',
    's27':'TAIL-RISK',
    's50':'MIRAGE','s94':'MIRAGE',
    's113':'NON-ADDITIVE',
    's103':'<0.70','s106':'<0.70','s107':'<0.70','s108':'<0.70','s109':'<0.70',
    's110':'<0.70','s111':'<0.70','s112':'<0.70','s114':'NEGATIVE',
    's116':'NEGATIVE','s117':'<0.70','s120':'<0.70',
}

# New batch results (hardcoded from run_new_batch.py output)
NEW = {
    's103': dict(is_sr=0.404, oos_sr=0.356, is_cagr=0.054, oos_cagr=0.046, oos_mdd=-0.272, to=2.9),
    's104': dict(is_sr=0.374, oos_sr=0.842, is_cagr=0.050, oos_cagr=0.112, oos_mdd=-0.203, to=0.8),
    's105': dict(is_sr=0.395, oos_sr=0.859, is_cagr=0.061, oos_cagr=0.158, oos_mdd=-0.337, to=0.9),
    's106': dict(is_sr=0.650, oos_sr=0.478, is_cagr=0.114, oos_cagr=0.067, oos_mdd=-0.278, to=2.1),
    's107': dict(is_sr=0.344, oos_sr=0.199, is_cagr=0.040, oos_cagr=0.017, oos_mdd=-0.293, to=1.5),
    's108': dict(is_sr=0.376, oos_sr=0.256, is_cagr=0.073, oos_cagr=0.027, oos_mdd=-0.662, to=38.7),
    's109': dict(is_sr=0.334, oos_sr=0.394, is_cagr=0.051, oos_cagr=0.065, oos_mdd=-0.411, to=8.6),
    's110': dict(is_sr=0.265, oos_sr=0.418, is_cagr=0.029, oos_cagr=0.086, oos_mdd=-0.555, to=9.6),
    's111': dict(is_sr=0.547, oos_sr=0.279, is_cagr=0.110, oos_cagr=0.037, oos_mdd=-0.438, to=23.4),
    's112': dict(is_sr=0.053, oos_sr=0.009, is_cagr=0.000, oos_cagr=-0.004, oos_mdd=-0.156, to=23.2),
    's113': dict(is_sr=0.300, oos_sr=0.799, is_cagr=0.025, oos_cagr=0.075, oos_mdd=-0.193, to=0.9),
    's114': dict(is_sr=-1.184, oos_sr=-0.678, is_cagr=-0.039, oos_cagr=-0.021, oos_mdd=-0.172, to=19.6),
    's115': dict(is_sr=0.329, oos_sr=0.828, is_cagr=0.042, oos_cagr=0.113, oos_mdd=-0.251, to=0.9),
    's116': dict(is_sr=-0.846, oos_sr=-0.495, is_cagr=-0.146, oos_cagr=-0.091, oos_mdd=-0.553, to=100.8),
    's117': dict(is_sr=0.182, oos_sr=0.220, is_cagr=0.016, oos_cagr=0.022, oos_mdd=-0.337, to=6.7),
    's118': dict(is_sr=0.385, oos_sr=0.889, is_cagr=0.051, oos_cagr=0.146, oos_mdd=-0.306, to=1.9),
    's119': dict(is_sr=-0.189, oos_sr=0.810, is_cagr=-0.010, oos_cagr=0.037, oos_mdd=-0.075, to=1.9),
    's120': dict(is_sr=0.867, oos_sr=0.547, is_cagr=0.111, oos_cagr=0.059, oos_mdd=-0.281, to=2.2),
    's121': dict(is_sr=0.446, oos_sr=0.750, is_cagr=0.043, oos_cagr=0.081, oos_mdd=-0.204, to=1.9),
    's122': dict(is_sr=0.550, oos_sr=0.760, is_cagr=0.080, oos_cagr=0.127, oos_mdd=-0.314, to=5.4),
}

# Load existing metrics from JSON
data = {}
for f in glob.glob('s*_metrics.json'):
    sid = f.replace('_metrics.json', '')
    with open(f) as fh:
        d = json.load(fh)
    data[sid] = dict(
        is_sr    = d.get('is',  {}).get('sharpe'),
        oos_sr   = d.get('oos', {}).get('sharpe'),
        is_cagr  = d.get('is',  {}).get('cagr'),
        oos_cagr = d.get('oos', {}).get('cagr'),
        is_mdd   = d.get('is',  {}).get('max_dd'),
        oos_mdd  = d.get('oos', {}).get('max_dd'),
        to       = d.get('turnover_annual'),
    )

# Merge new batch
for sid, d in NEW.items():
    data[sid] = dict(is_sr=d['is_sr'], oos_sr=d['oos_sr'],
                     is_cagr=d['is_cagr'], oos_cagr=d['oos_cagr'],
                     is_mdd=None, oos_mdd=d['oos_mdd'], to=d['to'])

def fsr(v):  return f'{v:+.2f}' if v is not None else '  --'
def fpc(v):  return f'{v*100:+5.1f}%' if v is not None else '   --'
def fto(v):  return f'{v:.1f}x' if v is not None else '--'

# Sort by OOS SR descending (None sorts last)
all_ids = sorted(
    [s for s in data if s in NAMES],
    key=lambda s: -(data[s]['oos_sr'] if data[s]['oos_sr'] is not None else -999)
)

hdr = f"{'ID':5s}  {'Strategy':28s}  {'IS SR':>6}  {'OOS SR':>6}  {'IS CAGR':>7}  {'OOS CAGR':>8}  {'IS MDD':>7}  {'OOS MDD':>7}  {'TO':>5}  {'Status'}"
print(hdr)
print('-' * len(hdr))

for sid in all_ids:
    r  = data[sid]
    nm = NAMES.get(sid, sid)[:28]
    st = STATUS.get(sid, '')
    print(f"{sid:5s}  {nm:28s}  {fsr(r['is_sr']):>6}  {fsr(r['oos_sr']):>6}  "
          f"{fpc(r['is_cagr']):>7}  {fpc(r['oos_cagr']):>8}  "
          f"{fpc(r['is_mdd']):>7}  {fpc(r['oos_mdd']):>7}  "
          f"{fto(r['to']):>5}  {st}")

print()
print(f"Total strategies shown: {len(all_ids)}")
print()
print("Status key:")
print("  VALIDATED  = in the 8-strategy book")
print("  SURVIVOR*  = s115 cleared redundancy test, pending full audit")
print("  ARTIFACT   = same-bar lookahead bug or data artifact")
print("  REDUNDANT  = clean code but adds no N_eff vs book")
print("  REGIME-TIMER = clean code, OOS window-flattered")
print("  MIRAGE     = collapsed OOS or IS SR negative")
print("  NON-ADDITIVE = positive N_eff gain but negative dSR")
print("  BORDERLINE = dN_eff 0.00-0.05, marginal")
print("  PROXY      = wrong data, right direction")
print("  TAIL-RISK  = real numbers, undeployable risk profile")
print("  <0.70      = OOS SR below threshold, not tested for redundancy")
print("  NEGATIVE   = OOS SR negative")
print("  (blank)    = weak / not yet classified")
