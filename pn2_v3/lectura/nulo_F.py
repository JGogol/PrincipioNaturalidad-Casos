# -*- coding: utf-8 -*-
"""Modelo nulo de F. Pregunta: ¿las cantidades estructurales de §3–§4 (terreno común, puentes, base,
regiones distintas, modularidad, reparto) son propiedad de QUÉ pares certifica F, o salen de tener
|F| pares inter-actor cualesquiera sobre estos bosques?

Tres nulos, cada uno con |F| pares y actores distintos en cada par:
  campo   : pares tomados al azar del campo que el juez vio (pasaron el prefiltro, dos direcciones).
  grados  : F reconectada por intercambios dobles de aristas preservando el grado de cada nodo en F
            (los mismos nodos, con la misma cantidad de fusiones cada uno; cambia con quién).
  uniforme: pares inter-actor cualesquiera (referencia).
Salida: nulo_<caso>.json con el observado y la distribución de cada cantidad bajo cada nulo."""
import sys, json, time
import numpy as np
from objeto_min import Objeto

def campo(o):
    M = o.impl.tocsr(); B = (M != 0).multiply((M != 0).T).tocoo()
    P = [(int(u), int(v)) for u, v in zip(B.row, B.col) if u < v and o.act_idx[u] != o.act_idx[v]]
    return P

def swaps(F, act, rng, k):
    F = [tuple(p) for p in F]; S = set(F); m = len(F)
    hechos = 0; intentos = 0
    while hechos < k and intentos < 50 * k:
        intentos += 1
        i, j = rng.integers(m, size=2)
        if i == j: continue
        (a, b), (c, d) = F[i], F[j]
        if rng.random() < 0.5: c, d = d, c
        # nuevo: (a,d), (c,b)
        e1, e2 = (min(a, d), max(a, d)), (min(c, b), max(c, b))
        if a == d or c == b or act[a] == act[d] or act[c] == act[b]: continue
        if e1 in S or e2 in S or e1 == e2: continue
        S.discard(F[i]); S.discard(F[j]); S.add(e1); S.add(e2)
        F[i], F[j] = e1, e2; hechos += 1
    return sorted(F), hechos

def medir(o, F):
    Reg = o.regiones(F)
    L = o.lecturas(Reg, F, muestra_modularidad=200)
    rep = np.array(list(L['reparto'].values()), dtype=float)
    L['reparto_desigualdad'] = float((rep.max() - rep.min()) / rep.sum()) if rep.sum() else None
    L['mediana_reg_puentes'] = float(np.median(Reg.sum(1)[np.any(Reg & ~o._subm(), axis=1)])) if L['puentes'] else None
    return {k: L[k] for k in ('terreno_comun', 'puentes', 'base', 'regiones_distintas', 'frac_vacias',
                              'intersecciones_distintas', 'max_reg', 'reparto_desigualdad', 'mediana_reg_puentes')}

if __name__ == '__main__':
    caso = sys.argv[1]; R = int(sys.argv[2]) if len(sys.argv) > 2 else 100
    rng = np.random.default_rng(2026)
    o = Objeto('../objeto/' + caso)
    F = o.F_pares(); m = len(F)
    obs = medir(o, F)
    P = campo(o)
    n = o.n
    todos = [(u, v) for u in range(n) for v in range(u + 1, n) if o.act_idx[u] != o.act_idx[v]]
    out = {'caso': caso, 'F': m, 'campo': len(P), 'reps': R, 'observado': obs, 'nulos': {'campo': [], 'grados': [], 'uniforme': []}, 'swaps': []}
    t0 = time.time()
    for r in range(R):
        idx = rng.choice(len(P), size=m, replace=False)
        out['nulos']['campo'].append(medir(o, sorted(P[i] for i in idx)))
        G, h = swaps(F, o.act_idx, rng, 10 * m); out['swaps'].append(h)
        out['nulos']['grados'].append(medir(o, G))
        idx = rng.choice(len(todos), size=m, replace=False)
        out['nulos']['uniforme'].append(medir(o, sorted(todos[i] for i in idx)))
        json.dump(out, open('nulo_%s.json' % caso, 'w'), indent=0)
        print(r, '%.0f s' % (time.time() - t0), flush=True)
    print('listo')
