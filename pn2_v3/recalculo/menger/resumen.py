# -*- coding: utf-8 -*-
import sys, os, json, collections as C, numpy as np, networkx as nx
sys.path.insert(0, '../../lectura')
import menger
from objeto_min import Objeto
from nulo_F import campo
LEC = '../../lectura'
d = json.load(open('menger.json'))
for _k,_run in d.items():
    for _a,_rr in _run['raices'].items():
        _rr['res']={t:[(None if (x is None or x==float('inf')) else x) for x in v] for t,v in _rr['res'].items()}
cab = json.load(open(os.path.join(LEC, 'cabezas.json')))
def hist(v):
    c = C.Counter(['inf' if x is None else (str(int(x)) if x < 4 else '4+') for x in v]); n = len(v)
    return ' '.join('%s:%d(%.0f%%)' % (k, c[k], 100 * c[k] / n) for k in ['1', '2', '3', '4+', 'inf'] if c[k])
out = []
for key, run in d.items():
    nombre, var = key.split('|'); caso = 'gerd' if nombre == 'Nilo' else 'selene'
    o = Objeto('../../objeto/' + caso)
    out.append('== %s  (%s, |F|=%d)' % (nombre, var, run['nF']))
    for act, rr in run['raices'].items():
        a = o.actores.index(act); res = {int(k): v for k, v in rr['res'].items()}
        aj = [v[0] for t, v in res.items() if o.act_idx[t] != a]
        pr = [v[0] for t, v in res.items() if o.act_idx[t] == a]
        out.append('  %-42s propias %4d (kF=inf todas: %s) | ajenas %4d  kF: %s' % (act[:42], len(pr), all(x is None for x in pr), len(aj), hist(aj)))
# coincidencias (sin D, estricto) y nodo de la mediacion
for caso, nombre in (('gerd', 'Nilo'), ('selene', 'Selene')):
    o = Objeto('../../objeto/' + caso)
    run = d.get('%s|sinD' % nombre)
    if not run: continue
    t2i = {tx: i for i, tx in enumerate(o.nodos['texto'].values)}
    # F sin D reconstruido igual que en menger.correr
    here = os.getcwd()
    F = o.F_pares()
    os.chdir(LEC); from clase_D import clases; cl = clases(nombre); os.chdir(here)
    tt = {tx.replace('|', '/'): i for i, tx in enumerate(o.nodos['texto'].values)}
    D = set()
    for i in range(len(cl)):
        c, ta, tb = cl[i]
        if c == 'D': x, y = tt[ta], tt[tb]; D.add((min(x, y), max(x, y)))
    F = [p for p in F if p not in D]
    out.append('== %s, coincidencias del terreno común (sin D, estricto): costo de desconocer por actor' % nombre)
    for gi, g in enumerate(cab[caso]['sin D|estricto']['grupos']):
        miembros = [t2i[tx] for _, tx in g]; acts = sorted({ac for ac, _ in g})
        fila = []
        for a, r in enumerate(o.raices):
            G = menger.grafo(o, F, a, 'kF'); G.add_node('T')
            for m in miembros: G.add_edge(m, 'T')
            kf = menger.corte(G, r, 'T')
            alcanza = nx.has_path(G, r, 'T')
            fila.append('%s=%s' % (o.actores[a].split()[-1][:8], 'no alcanza' if not alcanza else ('propia' if kf == float('inf') else int(kf))))
        out.append('  %d. [%s] %s :: %s' % (gi + 1, ','.join(x.split()[-1][:8] for x in acts), g[0][1][:60], '  '.join(fila)))
    if nombre == 'Nilo':
        t = 1069 if o.nodos['texto'].values[1069].startswith('Sudan requests African') else t2i.get('Sudan requests African Union mediation intervention')
        fila = []
        for a, r in enumerate(o.raices):
            for var in ('F', 'sinD'):
                res = d['Nilo|' + var]['raices'][o.actores[a]]['res']; v = res.get(str(t))
                fila.append('%s/%s=%s' % (o.actores[a].split()[-1][:8], var, 'propia' if v and v[0] is None else (int(v[0]) if v else 'no alcanza')))
        out.append('  Mediación UA (nodo %d): %s' % (t, '  '.join(fila)))
if os.environ.get('SIN_NULO'):
    open('RESULTADOS_menger.txt', 'w', encoding='utf-8').write('\n'.join(out) + '\n'); print('\n'.join(out)); sys.exit()
# nulo: fraccion de compromisos ajenos con kF>=2, F real vs |F| pares al azar del campo (10 rep x 150 objetivos por raiz)
rng = np.random.default_rng(2026)
for caso, nombre in (('gerd', 'Nilo'), ('selene', 'Selene')):
    o = Objeto('../../objeto/' + caso); F = o.F_pares(); P = campo(o)
    def frac(FF):
        tot = ok = 0
        for a, r in enumerate(o.raices):
            G = menger.grafo(o, FF, a, 'kF')
            alc = [n for n in nx.descendants(G, r) if not isinstance(n, tuple) and o.act_idx[n] != a]
            if not alc: continue
            for t in rng.choice(alc, size=min(150, len(alc)), replace=False):
                tot += 1; ok += menger.corte(G, r, int(t)) >= 2
        return ok / tot if tot else float('nan'), tot
    real = frac(F); nul = []
    for k in range(10):
        idx = rng.choice(len(P), size=len(F), replace=False); nul.append(frac([P[i] for i in idx])[0])
    out.append('== %s nulo: fracción de compromisos ajenos con kF>=2 (no se caen impugnando un solo par): real %.3f (n=%d) vs nulo mediana %.3f [%.3f, %.3f]' % (nombre, real[0], real[1], np.median(nul), min(nul), max(nul)))
open('RESULTADOS_menger.txt', 'w', encoding='utf-8').write('\n'.join(out) + '\n'); print('\n'.join(out))
