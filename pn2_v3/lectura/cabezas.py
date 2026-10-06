# -*- coding: utf-8 -*-
"""Cabezas del terreno común (Def. 2(i) del Apéndice A): proposiciones de M∩ cuyo padre no está en M∩,
agrupadas por indistinguibilidad (Reg mutua) = coincidencias. Con nulo de campo."""
import json, sys, numpy as np
from objeto_min import Objeto
from clase_D import clases
from nulo_F import campo

def terreno(o, F, estricto=False):
    Reg = o.regiones(F)
    if not estricto:
        M = np.flatnonzero(np.logical_and.reduce([Reg[r] for r in o.raices]))
    else:
        def alc1(r):
            vis = set(o.Sub[r]); out = set(vis)
            for u, v in F:
                for a, b in ((u, v), (v, u)):
                    if a in vis: out |= o.Sub[b]
            return out
        M = np.array(sorted(set.intersection(*[alc1(r) for r in o.raices])))
    return Reg, M

def cabezas(o, Reg, M, padre):
    Ms = set(M.tolist())
    cab = [v for v in M if padre[v] is None or padre[v] not in Ms]
    grupos = []
    for v in cab:
        for g in grupos:
            if Reg[v, g[0]] and Reg[g[0], v]: g.append(v); break
        else: grupos.append([v])
    return cab, grupos

if __name__ == '__main__':
    rng = np.random.default_rng(5); out = {}
    for caso, nombre in (('gerd', 'Nilo'), ('selene', 'Selene')):
        o = Objeto('../objeto/' + caso); F = o.F_pares(); cl = clases(nombre)
        txt2i = {tx.replace('|', '/'): i for i, tx in enumerate(o.nodos['texto'].values)}
        Fj = []
        for i in range(len(cl)):
            c, ta, tb = cl[i]; a, b = txt2i[ta], txt2i[tb]; Fj.append((min(a, b), max(a, b)))
        cl = {i: v[0] for i, v in cl.items()}
        id2i = {x: i for i, x in enumerate(o.nodos['id'].values)}
        padre = {i: (id2i[p] if isinstance(p, str) else None) for i, p in enumerate(o.nodos['padre'].values)}
        res = {}
        print('==', nombre)
        for et, quitar in (('F', set()), ('sin D', {'D'}), ('solo S', {'C', 'P', 'D'})):
            G = sorted(p for i, p in enumerate(Fj) if cl[i] not in quitar)
            for estr in (False, True):
                Reg, M = terreno(o, G, estr); cab, gr = cabezas(o, Reg, M, padre)
                k = '%s|%s' % (et, 'estricto' if estr else 'completo')
                res[k] = dict(M=int(len(M)), cabezas=len(cab), coincidencias=len(gr),
                              de_tres=sum(1 for g in gr if len({o.actor[v] for v in g}) == 3),
                              de_dos=sum(1 for g in gr if len({o.actor[v] for v in g}) == 2),
                              de_uno=sum(1 for g in gr if len({o.actor[v] for v in g}) == 1),
                              grupos=[[(o.actor[v], o.nodos['texto'].values[v]) for v in g] for g in gr])
                print('  %-7s %-9s |M|=%4d cabezas=%3d coincidencias=%3d (de tres %d, de dos %d, de uno %d)' % (et, 'estricto' if estr else 'completo', len(M), len(cab), len(gr), res[k]['de_tres'], res[k]['de_dos'], res[k]['de_uno']))
        # nulo de campo: coincidencias del terreno común completo y estricto con |F| pares al azar
        P = campo(o); nc = {'completo': [], 'estricto': []}; nt = {'completo': [], 'estricto': []}
        for r in range(100):
            idx = rng.choice(len(P), size=len(F), replace=False); G = sorted(P[i] for i in idx)
            for estr in (False, True):
                Reg, M = terreno(o, G, estr); cab, gr = cabezas(o, Reg, M, padre)
                nc['estricto' if estr else 'completo'].append(len(gr))
                nt['estricto' if estr else 'completo'].append(sum(1 for g in gr if len({o.actor[v] for v in g}) == 3))
        for k in nc:
            print('  nulo de campo, %-8s: coincidencias mediana %.0f [%.0f, %.0f]; de los tres actores %.0f [%.0f, %.0f]' % (k, np.median(nc[k]), np.percentile(nc[k], 2.5), np.percentile(nc[k], 97.5), np.median(nt[k]), np.percentile(nt[k], 2.5), np.percentile(nt[k], 97.5)))
        res['nulo'] = {k: dict(coincidencias=nc[k], de_tres=nt[k]) for k in nc}
        out[caso] = res
    json.dump(out, open('cabezas.json', 'w'), ensure_ascii=False, indent=0)
