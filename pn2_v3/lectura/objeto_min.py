# -*- coding: utf-8 -*-
"""Lectura mínima del objeto (A, T) de PN II y cálculo de Reg = Sub·(F·Sub)*, reimplementado
desde las definiciones del paper (§2.3, §3.2) sin usar dinamica/objeto.py: sirve como segunda
implementación independiente. Se valida contra las cifras de §3 antes de usarse (validar.py)."""
import json, os
import numpy as np
import scipy.sparse as sp
import pandas as pd

Q = 65534

def _csr(z, pref, shape):
    return sp.csr_matrix((z[pref + '_data'], z[pref + '_indices'], z[pref + '_indptr']), shape=shape)

def decod(m):
    """uint16 -> float en [0,1]; 0 = no evaluado -> NaN."""
    m = m.astype(np.float64)
    d = m.copy(); d.data = (d.data - 1) / Q
    return d

class Objeto:
    def __init__(self, ruta):
        self.ruta = ruta
        self.meta = json.load(open(os.path.join(ruta, 'meta.json'), encoding='utf-8'))
        self.nodos = pd.read_csv(os.path.join(ruta, 'nodos.csv'))
        self.n = len(self.nodos)
        a = np.load(os.path.join(ruta, 'A.npz'))
        self.A = sp.csr_matrix((a['data'].astype(np.int8), a['indices'], a['indptr']), shape=tuple(a['shape']))
        t = np.load(os.path.join(ruta, 'T.npz'))
        sh = (self.n, self.n)
        self.impl = decod(_csr(t, 'j1_impl', sh))
        self.contr = decod(_csr(t, 'j1_contr', sh))
        self.sim = decod(_csr(t, 'sim', sh))
        self.rechazo = _csr(t, 'rechazo', sh)
        self.actor = self.nodos['actor'].values
        self.actores = list(self.meta['actores'])
        self.act_idx = np.array([self.actores.index(x) for x in self.actor])
        self.raices = [int(self.nodos.index[(self.nodos['actor'] == a) & (self.nodos['profundidad'] == 0)][0]) for a in self.actores]
        u = self.meta['umbrales']
        self.tau, self.gamma, self.delta, self.cos = u['tau'], u['gamma'], u['delta'], u['coseno']
        self.Sub = self._sub()

    # ---- certificado F, según §2.3: promedio de implicación >= tau, max contradicción < gamma,
    #      coseno >= corte (todo par almacenado lo cumple), sin rechazo, actores distintos ----
    def F_pares(self):
        I = self.impl.tocoo()
        pares = {}
        for u, v, x in zip(I.row, I.col, I.data):
            if u < v:
                pares[(u, v)] = [x, None]
            else:
                pares.setdefault((v, u), [None, None])[1] = x
        C = self.contr.tocsr(); R = self.rechazo.tocsr(); S = self.sim.tocsr()
        out = []
        for (u, v), (x, y) in pares.items():
            if x is None or y is None: continue
            if self.act_idx[u] == self.act_idx[v]: continue
            if (x + y) / 2 < self.tau: continue
            if max(C[u, v], C[v, u]) >= self.gamma: continue
            if max(S[u, v], S[v, u]) < self.cos: continue
            if R[u, v] != 0 or R[v, u] != 0: continue
            out.append((int(u), int(v)))
        return sorted(out)

    # ---- gránulo Sub(v): v y todo lo que cuelga de v ----
    def _sub(self):
        n = self.n
        hijos = [[] for _ in range(n)]
        A = self.A.tocoo()
        for p, c in zip(A.row, A.col):
            hijos[p].append(c)
        # orden por profundidad descendente para acumular
        prof = self.nodos['profundidad'].values
        orden = np.argsort(-prof)
        Sub = [None] * n
        for v in orden:
            s = {int(v)}
            for c in hijos[v]:
                s |= Sub[c]
            Sub[int(v)] = s
        return Sub

    # ---- Reg(v) = alcanzabilidad: aristas del bosque hacia abajo + pares de F en ambos sentidos ----
    def regiones(self, F):
        n = self.n
        fus = [[] for _ in range(n)]
        for u, v in F:
            fus[u].append(v); fus[v].append(u)
        hijos = [[] for _ in range(n)]
        A = self.A.tocoo()
        for p, c in zip(A.row, A.col):
            hijos[p].append(c)
        Reg = []
        for v in range(n):
            vis = np.zeros(n, dtype=bool); vis[v] = True
            pila = [v]
            while pila:
                x = pila.pop()
                for y in hijos[x]:
                    if not vis[y]: vis[y] = True; pila.append(y)
                for y in fus[x]:
                    if not vis[y]: vis[y] = True; pila.append(y)
            Reg.append(vis)
        return np.array(Reg)          # matriz booleana n x n: fila v = Reg(v)

    def _subm(self):
        if not hasattr(self, '_subm_cache'):
            m = np.zeros((self.n, self.n), dtype=bool)
            for v in range(self.n):
                m[v, list(self.Sub[v])] = True
            self._subm_cache = m
        return self._subm_cache

    # ---- lecturas de §3 y §4 ----
    def lecturas(self, Reg, F, muestra_modularidad=400, semilla=11):
        n = self.n
        subm = self._subm()
        tam = Reg.sum(1)
        M = np.logical_and.reduce([Reg[r] for r in self.raices])
        puentes = np.any(Reg & ~subm, axis=1)
        base = tam > 1
        # regiones distintas
        filas = {Reg[v].tobytes() for v in range(n)}
        # modularidad: intersecciones de una muestra de regiones distintas
        rng = np.random.default_rng(semilla)
        uniq = sorted(filas)
        idx = rng.choice(len(uniq), size=min(muestra_modularidad, len(uniq)), replace=False)
        Ru = np.array([np.frombuffer(uniq[i], dtype=bool) for i in idx])
        inter = set(); vacias = 0; tot = 0
        for i in range(len(Ru)):
            X = Ru[i] & Ru[i + 1:]
            tot += len(X)
            e = X.sum(1) == 0
            vacias += int(e.sum())
            for row in X[~e]:
                inter.add(row.tobytes())
        # reparto del terreno común y composición
        reparto = {a: int((M & (self.act_idx == i)).sum()) for i, a in enumerate(self.actores)}
        # saltos: nivel BFS contando 1 por fusión (para mediana de lo ajeno)
        return dict(
            F=len(F), terreno_comun=int(M.sum()), reparto=reparto,
            puentes=int(puentes.sum()), base=int(base.sum()),
            reglas_cruce=int((base & puentes).sum()),
            regiones_distintas=len(filas), mediana_reg=float(np.median(tam)), max_reg=int(tam.max()),
            frac_sin_salir=float((~puentes).mean()),
            intersecciones_distintas=len(inter), frac_vacias=vacias / tot if tot else None,
            raices={a: int(Reg[r].sum()) for a, r in zip(self.actores, self.raices)},
        )

    def reducto(self, F):
        Reg0 = self.regiones(F)
        F = list(F); i = 0
        while i < len(F):
            G = F[:i] + F[i + 1:]
            if np.array_equal(self.regiones(G), Reg0):
                F = G
            else:
                i += 1
        return F
