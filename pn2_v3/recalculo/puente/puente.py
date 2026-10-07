# -*- coding: utf-8 -*-
"""§3.8: sostén de una propuesta por actor, sobre el Nilo.
sosten_a(X) = |{u en V_a : Reg(u) toca X}| / |V_a|  (qué fracción del razonamiento de a desemboca en X).
X = cada proposición del caso (puentes internos) y cada propuesta inyectada (puente externo).
Certificado propuesta-caso: promedio de implicación juez 1 >= tau y contradicción máxima juez 1 < gamma
(sin prefiltro de coseno ni verificación Λ, que no se corrieron sobre las propuestas). Uso: python puente.py"""
import json, sys, numpy as np
sys.path.insert(0, '../../lectura')
from objeto_min import Objeto
R = '../../'
o = Objeto(R + 'objeto/gerd'); F = o.F_pares(); n = o.n
idx = {k: i for i, k in enumerate(o.nodos['id'].values)}
tau, gamma = o.tau, o.gamma

def grafo(extra_hijos=(), extra_fus=(), N=None):
    N = N or n
    hijos = [[] for _ in range(N)]; fus = [[] for _ in range(N)]
    A = o.A.tocoo()
    for p, c in zip(A.row, A.col): hijos[p].append(c)
    for p, c in extra_hijos: hijos[p].append(c)
    for u, w in list(F) + list(extra_fus): fus[u].append(w); fus[w].append(u)
    return hijos, fus

def regiones(hijos, fus, N):
    M = np.zeros((N, N), bool)
    for v in range(N):
        vis = M[v]; vis[v] = True; st = [v]
        while st:
            x = st.pop()
            for y in hijos[x] + fus[x]:
                if not vis[y]: vis[y] = True; st.append(y)
    return M

def propuesta(cert, arbol):
    E = json.load(open(R + arbol, encoding='utf-8'))['nodos']
    eid = {x['id']: n + k for k, x in enumerate(E)}
    fe = set()
    for l in open(R + cert, encoding='utf-8'):
        d = json.loads(l); e = eid[d['E']]
        for r in d['r']:
            if (r[1] + r[2]) / 2 >= tau and max(r[3], r[4]) < gamma: fe.add((idx[r[0]], e))
    eh = [(eid[x['padre']], eid[x['id']]) for x in E if x['padre']]
    N = n + len(E)
    h, f = grafo(eh, fe, N)
    M = regiones(h, f, N)
    toca = M[:n, n:].any(1)
    s = [float(toca[o.act_idx == i].sum() / (o.act_idx == i).sum()) for i in range(3)]
    return dict(nodos=len(E), pares_certificados=len(fe), sosten=s)

h, f = grafo(); M = regiones(h, f, n)
col = np.array([[M[o.act_idx == i, v].sum() / (o.act_idx == i).sum() for i in range(3)] for v in range(n)])
out = dict(actores=o.actores, criterio='tau=%.2f gamma=%.2f, juez 1, sin prefiltro ni Lambda' % (tau, gamma),
           legitima=propuesta('complementos/juez_propuesta/cert_gerd_v4.jsonl', 'complementos/propuestas/E_arbol_gerd.json'),
           ajena=propuesta('complementos/juez_propuesta/cert_cruzado_selene_en_gerd.jsonl', 'complementos/propuestas/E_arbol_selene.json'))
mn = col.min(1); orden = np.argsort(-mn)
out['internos_top'] = [dict(i=int(v), texto=o.nodos['texto'].values[v], actor=o.actor[v], sosten=[round(x, 4) for x in col[v]]) for v in orden[:15]]
out['n_tocados_por_3'] = int((mn > 0).sum())
json.dump(out, open('puente.json', 'w', encoding='utf-8'), ensure_ascii=False, indent=1)
np.save('sosten_nodos.npy', col)
print(json.dumps({k: out[k] for k in ['legitima', 'ajena', 'n_tocados_por_3']}, ensure_ascii=False))
for d in out['internos_top'][:10]: print(d['sosten'], d['actor'].split()[-1], d['texto'][:80])
