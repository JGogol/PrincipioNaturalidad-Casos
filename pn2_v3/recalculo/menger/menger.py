# -*- coding: utf-8 -*-
"""Costo de desconocer una coincidencia (Menger). Para cada premisa r y cada proposición t que r
alcanza (Reg), corte mínimo entre r y t en el grafo bosque + fusiones:
  kF  : cuántos pares del juez hay que impugnar si se aceptan todos los pasos (propios y ajenos)
        = máximo número de caminos que no comparten ninguna fusión (Menger). inf si t cuelga de r.
  kP  : cuántos pasos PROPIOS hay que retirar si se aceptan el juez y los pasos ajenos.
Una fusión es un par no dirigido: se modela como un nodo de capacidad 1 que se corta entero."""
import sys, json, time, numpy as np, networkx as nx
sys.path.insert(0, '../../lectura')
from objeto_min import Objeto
from clase_D import clases
INF = None

def grafo(o, F, r_actor, modo):
    G = nx.DiGraph()
    A = o.A.tocoo()
    for p, c in zip(A.row, A.col):
        propio = o.act_idx[p] == r_actor
        if modo == 'kP' and propio: G.add_edge(int(p), int(c), capacity=1)
        else: G.add_edge(int(p), int(c))            # infinita
    for i, (u, w) in enumerate(F):
        fi, fo = ('fi', i), ('fo', i)
        G.add_edge(u, fi); G.add_edge(w, fi)
        if modo == 'kF': G.add_edge(fi, fo, capacity=1)
        else: G.add_edge(fi, fo)
        G.add_edge(fo, u); G.add_edge(fo, w)
    return G

def corte(G, s, t):
    try: return nx.minimum_cut_value(G, s, t)
    except nx.NetworkXUnbounded: return float('inf')

def correr(caso, nombre, quitarD):
    o = Objeto('../../objeto/' + caso); F = o.F_pares()
    if quitarD:
        import os; here=os.getcwd(); os.chdir('../../lectura'); cl = clases(nombre); os.chdir(here); txt2i = {tx.replace('|', '/'): i for i, tx in enumerate(o.nodos['texto'].values)}
        D = set()
        for i in range(len(cl)):
            c, ta, tb = cl[i]
            if c == 'D': a, b = txt2i[ta], txt2i[tb]; D.add((min(a, b), max(a, b)))
        F = [p for p in F if p not in D]
    out = {'nF': len(F), 'raices': {}}
    for k, r in enumerate(o.raices):
        a = o.act_idx[r]
        GF = grafo(o, F, a, 'kF'); GP = grafo(o, F, a, 'kP')
        alc = nx.descendants(GF, r) - {n for n in GF if isinstance(n, tuple)}
        res = {}
        t0 = time.time()
        for t in sorted(alc):
            res[int(t)] = (corte(GF, r, t), corte(GP, r, t) if o.act_idx[t] == a or True else None)
        out['raices'][o.actores[a]] = {'r': int(r), 'res': res, 'seg': round(time.time() - t0, 1)}
        print(nombre, 'sinD' if quitarD else 'F', o.actores[a], len(res), 'objetivos', round(time.time() - t0, 1), 's', flush=True)
    return o, out

def correr_raiz(caso, nombre, quitarD, k):
    import os
    o = Objeto('../../objeto/' + caso); F = o.F_pares()
    if quitarD:
        here=os.getcwd(); os.chdir('../../lectura'); cl = clases(nombre); os.chdir(here)
        txt2i = {tx.replace('|', '/'): i for i, tx in enumerate(o.nodos['texto'].values)}
        D = set()
        for i in range(len(cl)):
            c, ta, tb = cl[i]
            if c == 'D': a, b = txt2i[ta], txt2i[tb]; D.add((min(a, b), max(a, b)))
        F = [p for p in F if p not in D]
    r = o.raices[k]; a = o.act_idx[r]
    GF = grafo(o, F, a, 'kF'); GP = grafo(o, F, a, 'kP')
    alc = nx.descendants(GF, r) - {n for n in GF if isinstance(n, tuple)}
    t0 = time.time(); res = {}
    for t in sorted(alc): res[int(t)] = (corte(GF, r, t), corte(GP, r, t))
    return len(F), o.actores[a], {'r': int(r), 'res': res, 'seg': round(time.time() - t0, 1)}

if __name__ == '__main__':
    caso, nombre, var, k = sys.argv[1], sys.argv[2], sys.argv[3], int(sys.argv[4])
    nF, act, rr = correr_raiz(caso, nombre, var == 'sinD', k)
    try: todo = json.load(open('menger.json'))
    except Exception: todo = {}
    key = '%s|%s' % (nombre, var)
    if not isinstance(todo.get(key), dict) or 'raices' not in todo.get(key, {}): todo[key] = {'nF': nF, 'raices': {}}
    todo[key]['nF'] = nF; todo[key]['raices'][act] = rr
    json.dump(todo, open('menger.json', 'w'), default=lambda x: None if x == float('inf') else x)
    print(key, act, len(rr['res']), rr['seg'], 's', flush=True)
