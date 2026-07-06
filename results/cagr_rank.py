import json
BOOK = {'s08':'Sector Rotation','s46':'Risk Parity','s30':'Low Volatility',
        's02':'TS Momentum','s31':'Vol Targeting','s35':'Sell in May',
        's49':'Dollar Regime','s90':'Credit Regime'}
rows = []
for sid, name in BOOK.items():
    try:
        with open('%s_metrics.json' % sid) as f:
            d = json.load(f)
        cagr = d.get('oos', {}).get('cagr')
        sr   = d.get('oos', {}).get('sharpe')
        mdd  = d.get('oos', {}).get('max_dd')
        rows.append((cagr or 0, sid, name, cagr, sr, mdd))
    except Exception as e:
        print('  %s: ERROR %s' % (sid, e))
rows.sort(reverse=True)
print('  %-4s  %-22s  %9s  %7s  %8s' % ('ID', 'Name', 'OOS CAGR', 'OOS SR', 'OOS MDD'))
print('  ' + '-'*58)
for _, sid, name, cagr, sr, mdd in rows:
    print('  %-4s  %-22s  %+8.1f%%  %+7.3f  %+8.1f%%' % (sid, name, cagr*100, sr, mdd*100))
