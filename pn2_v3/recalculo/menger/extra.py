import sys, os, json, collections as C, networkx as nx
sys.path.insert(0, '../../lectura')
import menger
from objeto_min import Objeto
LEC = '../../lectura'
d = json.load(open('menger.json')); cab = json.load(open(os.path.join(LEC, 'cabezas.json')))
out = []
def isinf(x): return x is None or x == float('inf')
# kP: pasos propios a retirar, sobre proposiciones propias
for key, run in d.items():
    nombre, var = key.split('|'); caso = 'gerd' if nombre == 'Nilo' else 'selene'
    o = Objeto('../../objeto/' + caso)
    for act, rr in run['raices'].items():
        a = o.actores.index(act)
        kp = [v[1] for t, v in rr['res'].items() if o.act_idx[int(t)] == a and int(t) != rr['r']]
        c = C.Counter('inf' if isinf(x) else ('1' if x == 1 else '2+') for x in kp)
        out.append('%s %-6s %-40s propias %d: retirar 1 paso alcanza %d; hacen falta 2+ pasos %d (otro razonamiento lo devuelve)' % (nombre, var, act[:40], len(kp), c['1'], c['2+']))
def Fsin(o, nombre):
    F = o.F_pares(); here = os.getcwd(); os.chdir(LEC); from clase_D import clases; cl = clases(nombre); os.chdir(here)
    tt = {tx.replace('|', '/'): i for i, tx in enumerate(o.nodos['texto'].values)}; D = set()
    for i in range(len(cl)):
        c, ta, tb = cl[i]
        if c == 'D': x, y = tt[ta], tt[tb]; D.add((min(x, y), max(x, y)))
    return [p for p in F if p not in D]
# coincidencias de varios actores: cuántos dictámenes del juez hay que impugnar para separar mi proposición de las de los otros
for caso, nombre in (('gerd', 'Nilo'), ('selene', 'Selene')):
    o = Objeto('../../objeto/' + caso); F = Fsin(o, nombre)
    t2i = {tx: i for i, tx in enumerate(o.nodos['texto'].values)}
    for gi, g in enumerate(cab[caso]['sin D|estricto']['grupos']):
        acts = {ac for ac, _ in g}
        if len(acts) < 2: continue
        fila = []
        for a in sorted(acts):
            ai = o.actores.index(a)
            G = menger.grafo(o, F, ai, 'kF')
            G.add_node('S'); G.add_node('T')
            for ac, tx in g:
                if ac == a: G.add_edge('S', t2i[tx])
                else: G.add_edge(t2i[tx], 'T')
            fila.append('%s=%s' % (a.split()[-1][:8], int(menger.corte(G, 'S', 'T'))))
        out.append('%s coincidencia %d (%d actores) %s :: impugnar %s' % (nombre, gi + 1, len(acts), g[0][1][:50], ' '.join(fila)))
open('RESULTADOS_menger_extra.txt', 'w', encoding='utf-8').write('\n'.join(out) + '\n'); print('\n'.join(out))
