import sys, os, json
sys.path.insert(0, '../../lectura')
from objeto_min import Objeto
cab = json.load(open('../../lectura/cabezas.json'))
out = []
for caso, nombre in (('gerd', 'Nilo'), ('selene', 'Selene')):
    o = Objeto('../../objeto/' + caso)
    t2i = {tx: i for i, tx in enumerate(o.nodos['texto'].values)}
    prof = o.nodos['profundidad'].values
    for gi, g in enumerate(cab[caso]['sin D|estricto']['grupos']):
        acts = {ac for ac, _ in g}
        if len(acts) < 2: continue
        fila = []
        for ac, tx in g:
            t = t2i[tx]; a = o.act_idx[t]; Va = int((o.act_idx == a).sum())
            fila.append('%s: prof %d, rama hasta la premisa %d pasos, lo que cuelga %d (%.1f%% de su árbol)' % (ac.split()[-1][:8], prof[t], prof[t], len(o.Sub[t]), 100 * len(o.Sub[t]) / Va))
        out.append('%s coincidencia %d: %s\n    ' % (nombre, gi + 1, g[0][1][:60]) + '\n    '.join(fila))
open('RESULTADOS_rama.txt', 'w', encoding='utf-8').write('\n'.join(out) + '\n'); print('\n'.join(out))
