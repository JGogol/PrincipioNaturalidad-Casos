# -*- coding: utf-8 -*-
"""§2.1: la arista del bosque no es implicación. Mediana del juez sobre las aristas padre -> hijo
(capas j*_ph_*; existen sólo en el Nilo). Uso: python padre_hijo.py"""
import numpy as np
from comun import meta, bosque, capa, valor

for caso in ['gerd', 'selene']:
    n = meta(caso)['n']
    A = bosque(caso).tocoo()
    print(f'== {caso}: {len(A.row)} aristas')
    for nombre in ['j1_ph_impl', 'j1_ph_contr', 'j2_ph_impl', 'j2_ph_contr']:
        M = capa(caso, nombre, n)
        if M is None:
            print(f'  {nombre}: la capa no existe en este caso')
            continue
        ph = valor(np.asarray(M[A.row, A.col]).ravel())
        hp = valor(np.asarray(M[A.col, A.row]).ravel())
        print(f'  {nombre}: evaluadas {np.sum(~np.isnan(ph))}; mediana padre->hijo = {np.nanmedian(ph):.4f}; '
              f'hijo->padre = {np.nanmedian(hp):.4f}; padre->hijo >= 0,55: {int(np.nansum(ph >= 0.55))}')
