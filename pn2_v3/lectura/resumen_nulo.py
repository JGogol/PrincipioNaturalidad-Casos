import json, sys, numpy as np
for caso in ['gerd', 'selene']:
    d = json.load(open('nulo_%s.json' % caso)); obs = d['observado']
    print('\n==', caso, '|F| =', d['F'], 'campo =', d['campo'], 'reps =', d['reps'], 'swaps medios', np.mean(d['swaps']))
    print('%-26s %10s | %-28s | %-28s | %-28s' % ('cantidad', 'observado', 'campo: med [p2.5,p97.5] pct', 'grados', 'uniforme'))
    for k in obs:
        fila = '%-26s %10s' % (k, ('%.3f' % obs[k]) if isinstance(obs[k], float) else obs[k])
        for nulo in ('campo', 'grados', 'uniforme'):
            x = np.array([r[k] for r in d['nulos'][nulo] if r[k] is not None], dtype=float)
            pct = (x < obs[k]).mean() * 100
            fila += ' | %7.2f [%7.2f,%7.2f] %3.0f' % (np.median(x), np.percentile(x, 2.5), np.percentile(x, 97.5), pct)
        print(fila)
