# -*- coding: utf-8 -*-
"""Lectura de capas del objeto publicado (objeto/<caso>/{A,T}.npz). Sin dependencias del pipeline."""
import os, json
import numpy as np, scipy.sparse as sp

RAIZ = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..')
Q = 65534

def ruta(*p):
    return os.path.join(RAIZ, *p)

def meta(caso):
    return json.load(open(ruta('objeto', caso, 'meta.json'), encoding='utf-8'))

def bosque(caso):
    a = np.load(ruta('objeto', caso, 'A.npz'))
    n = int(a['shape'][0])
    return sp.csr_matrix((a['data'], a['indices'], a['indptr']), shape=(n, n))

def capa(caso, nombre, n):
    """Devuelve la capa como CSR de enteros uint16 (0 = no evaluado) o None si no existe."""
    z = np.load(ruta('objeto', caso, 'T.npz'))
    if nombre + '_data' not in z.files:
        return None
    return sp.csr_matrix((z[nombre + '_data'], z[nombre + '_indices'], z[nombre + '_indptr']), shape=(n, n))

def valor(q):
    """uint16 -> [0,1]; q = 0 (no evaluado) -> NaN."""
    q = np.asarray(q, dtype=float)
    return np.where(q > 0, (q - 1) / Q, np.nan)
