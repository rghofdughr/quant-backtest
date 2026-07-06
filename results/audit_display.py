import json, glob

NAMES = {
    's07':'Donchian Turtle (Futures)','s08':'Sector Rotation (SPDRs)',
    's30':'Low Volatility (LF)','s31':'Vol Targeting (SPY)',
    's35':'Sell in May','s46':'Risk Parity (5-asset)',
    's49':'Dollar Regime (EEM/SPY)','s52':'Idiosyncratic Volatility',
    's54':'TS Mom Long-Term','s59':'Vol-of-Vol Regime',
    's60':'Corr. Regime','s61':'Factor Combo',
    's63':'ETF Breakout (Donchian)','s66':'Vol-Confirmed Mom',
    's68':'Mom Ensemble','s69':'Sharpe Rank',
    's71':'52-Week Breakout (R1000)','s73':'Residual Momentum',
    's76':'MA200 Band','s77':'Dual Momentum (GEM)',
    's78':'Vol Trend ETF','s79':'Adaptive Trend',
    's90':'Credit Regime (HYG/LQD)','s91':'Inflation Tilt',
    's93':'Defensive Rotation',
}

STATUS = {
    's08':'VALIDATED','s31':'VALIDATED','s46':'VALIDATED','s30':'VALIDATED',
    's02':'VALIDATED','s35':'VALIDATED','s49':'VALIDATED','s90':'VALIDATED',
    's07':'ARTIFACT','s78':'ARTIFACT','s79':'ARTIFACT','s81':'ARTIFACT',
    's63':'ARTIFACT','s93':'ARTIFACT',
    's75':'REDUNDANT','s71':'REDUNDANT',
    's60':'REGIME-TIMER',
    's26':'PROXY','s45':'PROXY','s27':'TAIL-RISK',
    's50':'MIRAGE','s94':'MIRAGE',
}

CORRECTED = {
    's63': 0.280,
    's93': 0.726,
}

data = {}
for f in glob.glob('s*_metrics.json'):
    sid = f.replace('_metrics.json','')
    with open(f) as fh:
        d = json.load(fh)
    data[sid] = {
        'is_sr':   d.get('is',{}).get('sharpe'),
        'oos_sr':  d.get('oos',{}).get('sharpe'),
        'oos_cagr':d.get('oos',{}).get('cagr'),
        'is_mdd':  d.get('is',{}).get('max_dd'),
        'oos_mdd': d.get('oos',{}).get('max_dd'),
        'to':      d.get('turnover_annual'),
    }

def fsr(v):
    return ('%+.2f' % v) if v is not None else '  ---'
def fpc(v):
    return ('%+.0f%%' % (v*100)) if v is not None else '  ---'
def fto(v):
    return ('%.0fx' % v) if v is not None else '---'

ids = sorted(
    [s for s in data if s in NAMES
     and data[s]['oos_sr'] is not None
     and data[s]['oos_sr'] >= 0.70],
    key=lambda s: -(data[s]['oos_sr'] or 0)
)

print('%-6s  %-28s  %7s  %8s  %7s  %7s  %8s  %5s  %-14s' %
      ('ID','Strategy','IS SR','OOS SR','IS MDD','OOS MDD','OOS CAG','TO','Status'))
print('-' * 107)
for sid in ids:
    r  = data[sid]
    st = STATUS.get(sid, '')
    corr = CORRECTED.get(sid)
    if corr is not None:
        oos_col = '%+.2f*' % r['oos_sr']
    else:
        oos_col = fsr(r['oos_sr'])
    print('%-6s  %-28s  %7s  %8s  %7s  %7s  %8s  %5s  %-14s' % (
        sid, NAMES[sid][:28],
        fsr(r['is_sr']), oos_col,
        fpc(r['is_mdd']), fpc(r['oos_mdd']),
        fpc(r['oos_cagr']),
        fto(r['to']), st,
    ))

print()
print('* raw (buggy) OOS SR shown; corrected SR in brackets:')
for sid, c in CORRECTED.items():
    print('  %s corrected -> %.3f' % (sid, c))
print()
print('ARTIFACT/REDUNDANT/REGIME-TIMER = do not deploy despite raw OOS SR')
