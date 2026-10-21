# -*- coding: utf-8 -*-
"""§3.8 v3.8: el sostén con la versión estricta (una sola fusión), que §3.1 fija como la cifra en firme.
R1(u) = Sub(u) ∪ Sub·F·Sub  (desde lo propio, se cruza UNA fusión y se baja por el árbol ajeno).
Se reporta: (a) región completa Reg = Sub·(F·Sub)* (reproduce puente.json), (b) estricta, (c) estricta sin los pares
que la lectura rotuló D (contenido distinto). Mismo certificado propuesta-caso que puente.py. Uso: python puente_estricto.py
Escribe puente_estricto.json y sosten_nodos_estricto.npy (sostén estricto por nodo, con D; lo lee fig_puente.py)."""
import json, sys, os, numpy as np
sys.path.insert(0, '../../lectura')
from objeto_min import Objeto
from clase_D import clases
R = '../../'
o = Objeto(R + 'objeto/gerd'); F = o.F_pares(); n = o.n; act = o.act_idx
idx = {k: i for i, k in enumerate(o.nodos['id'].values)}
tau, gamma = o.tau, o.gamma
# pares D (misma correspondencia que clase_D.py)
_cwd = os.getcwd(); os.chdir(R + 'lectura'); cl = clases('Nilo'); os.chdir(_cwd); txt2i = {tx.replace('|', '/'): i for i, tx in enumerate(o.nodos['texto'].values)}
Fset = set(F); D = set()
for i, (c, ta, tb) in cl.items():
    a, b = txt2i[ta], txt2i[tb]; p = (min(a, b), max(a, b)); assert p in Fset
    if c == 'D': D.add(p)
FsinD = [p for p in F if p not in D]

def sub_matrix(N, hijos):
    S = np.zeros((N, N), bool)
    for v in range(N):
        st = [v]; S[v, v] = True
        while st:
            x = st.pop()
            for y in hijos[x]:
                if not S[v, y]: S[v, y] = True; st.append(y)
    return S

def matrices(Fp, extra_hijos=(), extra_fus=(), N=None):
    N = N or n
    hijos = [[] for _ in range(N)]
    A = o.A.tocoo()
    for p, c in zip(A.row, A.col): hijos[p].append(c)
    for p, c in extra_hijos: hijos[p].append(c)
    S = sub_matrix(N, hijos)
    Fm = np.zeros((N, N), bool)
    for u, w in list(Fp) + list(extra_fus): Fm[u, w] = Fm[w, u] = True
    R1 = S | ((S.astype(np.int32) @ Fm.astype(np.int32) @ S.astype(np.int32)) > 0)
    # región completa: alcanzabilidad por hijos y fusiones
    M = np.zeros((N, N), bool); fus = [np.flatnonzero(Fm[v]) for v in range(N)]
    for v in range(N):
        vis = M[v]; vis[v] = True; st = [v]
        while st:
            x = st.pop()
            for y in list(hijos[x]) + list(fus[x]):
                if not vis[y]: vis[y] = True; st.append(y)
    return R1, M

def sosten(Mx, cols):
    toca = Mx[:n][:, cols].any(1)
    return [round(float(toca[act == i].sum() / (act == i).sum()), 4) for i in range(3)]

def propuesta(cert, arbol, Fp):
    E = json.load(open(R + arbol, encoding='utf-8'))['nodos']; eid = {x['id']: n + k for k, x in enumerate(E)}
    fe = set()
    for l in open(R + cert, encoding='utf-8'):
        d = json.loads(l); e = eid[d['E']]
        for r in d['r']:
            if (r[1] + r[2]) / 2 >= tau and max(r[3], r[4]) < gamma: fe.add((idx[r[0]], e))
    eh = [(eid[x['padre']], eid[x['id']]) for x in E if x['padre']]
    N = n + len(E); R1, M = matrices(Fp, eh, fe, N); cols = list(range(n, N))
    return dict(completa=sosten(M, cols), estricta=sosten(R1, cols), pares=len(fe))

out = {'actores': o.actores, 'F': len(F), 'D': len(D)}
for nombre, Fp in (('con_D', F), ('sin_D', FsinD)):
    R1, M = matrices(Fp)
    colR1 = np.array([[R1[act == i, v].sum() / (act == i).sum() for i in range(3)] for v in range(n)])
    colM = np.array([[M[act == i, v].sum() / (act == i).sum() for i in range(3)] for v in range(n)])
    if nombre == 'con_D': np.save('sosten_nodos_estricto.npy', colR1)  # para fig_puente.py
    res = {}
    for k, col in (('completa', colM), ('estricta', colR1)):
        mn = col.min(1); orden = np.argsort(-mn, kind='stable')
        res[k] = dict(n_tocados_por_3=int((mn > 0).sum()),
                      top=[dict(i=int(v), actor=o.actor[v].split()[-1], texto=o.nodos['texto'].values[v][:90], sosten=[round(float(x), 4) for x in col[v]]) for v in orden[:8]])
    res['legitima'] = propuesta('complementos/juez_propuesta/cert_gerd_v4.jsonl', 'complementos/propuestas/E_arbol_gerd.json', Fp)
    res['ajena'] = propuesta('complementos/juez_propuesta/cert_cruzado_selene_en_gerd.jsonl', 'complementos/propuestas/E_arbol_selene.json', Fp)
    out[nombre] = res
ref = json.load(open('puente.json', encoding='utf-8'))
assert [round(x, 4) for x in ref['legitima']['sosten']] == out['con_D']['legitima']['completa'], (ref['legitima']['sosten'], out['con_D']['legitima'])
assert out['con_D']['completa']['n_tocados_por_3'] == ref['n_tocados_por_3']
json.dump(out, open('puente_estricto.json', 'w', encoding='utf-8'), ensure_ascii=False, indent=1)
for nombre in ('con_D', 'sin_D'):
    r = out[nombre]; print('==', nombre, '| legítima', r['legitima'], '| ajena', r['ajena'])
    for k in ('completa', 'estricta'):
        print('  ', k, 'tocados por los 3:', r[k]['n_tocados_por_3'])
        for d in r[k]['top'][:5]: print('     ', d['sosten'], d['actor'], d['texto'])
