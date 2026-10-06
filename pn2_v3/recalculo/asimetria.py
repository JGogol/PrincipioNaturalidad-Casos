# -*- coding: utf-8 -*-
"""§2.1: el juez no es simétrico. Para cada par ordenado evaluado (u, v) compara T[u,v] con T[v,u].
La cifra depende de la regla con que se cuente "difiere"; este script reporta varias reglas para
que el texto declare la que usa. Uso: python asimetria.py"""
import numpy as np
from comun import meta, capa, valor

UMBRALES = [0.0, 0.001, 0.01, 0.05, 0.1, 0.5]   # 0.0 = cualquier diferencia a la resolución de uint16

for caso in ['gerd', 'selene']:
    n = meta(caso)['n']
    print(f'== {caso} (n = {n})')
    for nombre in ['j1_impl', 'j1_contr']:
        M = capa(caso, nombre, n)
        C = M.tocoo()
        a = valor(C.data)
        b = valor(np.asarray(M.T.tocsr()[C.row, C.col]).ravel())
        ok = ~np.isnan(a) & ~np.isnan(b)
        d = np.abs(a - b)[ok]
        print(f'  {nombre}: entradas ordenadas evaluadas en las dos direcciones = {ok.sum()} '
              f'({ok.sum() // 2} pares no ordenados); |Δ| máx = {d.max():.4f}; mediana = {np.median(d):.4f}')
        for t in UMBRALES:
            k = int((d > t).sum())
            print(f'    |Δ| > {t:<5}: {k:>7} entradas ordenadas ({k / len(d):6.1%})')
