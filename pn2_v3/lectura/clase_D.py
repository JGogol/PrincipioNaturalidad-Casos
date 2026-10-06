# -*- coding: utf-8 -*-
"""Propagación del error medida: quitar de F los pares que la lectura de
paper/notas/clasificacion_F_2026-09-24.md rotuló D (contenido distinto) y recalcular §3;
comparar con quitar la misma cantidad de pares al azar (200 repeticiones). También cuenta los
triángulos inter-actor de F (tres actores, tres pares) contra el nulo de campo."""
import re, sys, json, numpy as np
from objeto_min import Objeto
from nulo_F import campo, medir

def clases(caso_nombre):
    """Devuelve {i: (clase, texto_a, texto_b)} leyendo la tabla del archivo de clasificación."""
    t = open('../anotacion/clasificacion_F_2026-09-24.md', encoding='utf-8').read()
    sec = t.split('## ' + caso_nombre)[1].split('\n## ')[0]
    out = {}
    for m in re.finditer(r'^\| (\d+) \| ([SCPD]) \| ([^|]*) \| ([^|]*) \| ([^|]*) \| ([^|]*) \|$', sec, re.M):
        out[int(m.group(1))] = (m.group(2), m.group(4).strip(), m.group(6).strip())
    return out

def triangulos(F, act):
    S = set(F); ady = {}
    for u, v in F: ady.setdefault(u, set()).add(v); ady.setdefault(v, set()).add(u)
    t = 0
    for u, v in F:
        for w in ady[u] & ady[v]:
            if w > v and len({act[u], act[v], act[w]}) == 3: t += 1
    return t

if __name__ == '__main__':
    rng = np.random.default_rng(7)
    res = {}
    for caso, nombre in (('gerd', 'Nilo'), ('selene', 'Selene')):
        o = Objeto('../objeto/' + caso); F = o.F_pares()
        # el orden de fusiones.json coincide con el de F_pares? se verifica por texto
        cl = clases(nombre)
        assert len(cl) == len(F), (len(cl), len(F))
        txt2i = {tx.replace('|', '/'): i for i, tx in enumerate(o.nodos['texto'].values)}
        Fset = set(F); Fj = []
        for i in range(len(cl)):
            c, ta, tb = cl[i]
            a, b = txt2i[ta], txt2i[tb]
            p = (min(a, b), max(a, b)); assert p in Fset, p; Fj.append(p)
        assert len(set(Fj)) == len(F)
        cl = {i: v[0] for i, v in cl.items()}
        D = [Fj[i] for i, c in cl.items() if c == 'D']
        sinD = sorted(set(F) - set(D))
        obs = medir(o, F); obs['triangulos'] = triangulos(F, o.act_idx)
        sd = medir(o, sinD); sd['triangulos'] = triangulos(sinD, o.act_idx)
        az = []
        for r in range(200):
            idx = rng.choice(len(F), size=len(D), replace=False)
            G = sorted(set(F) - {F[i] for i in idx}); m = medir(o, G); m['triangulos'] = triangulos(G, o.act_idx); az.append(m)
        # triángulos bajo nulo de campo
        P = campo(o); tn = []
        for r in range(200):
            idx = rng.choice(len(P), size=len(F), replace=False); tn.append(triangulos(sorted(P[i] for i in idx), o.act_idx))
        res[caso] = dict(F=len(F), D=len(D), observado=obs, sin_D=sd, azar=az, triangulos_nulo_campo=tn,
                         D_por_clase={c: sum(1 for x in cl.values() if x == c) for c in 'SCPD'})
        print('==', nombre, '|F| =', len(F), 'D =', len(D))
        print('%-26s %10s %10s | %-30s' % ('cantidad', 'observado', 'sin D', 'quitando %d al azar: med [p2.5,p97.5] pct(sin D)' % len(D)))
        for k in obs:
            x = np.array([m[k] for m in az if m[k] is not None], dtype=float)
            f = lambda v: ('%.3f' % v) if isinstance(v, float) else str(v)
            print('%-26s %10s %10s | %8.2f [%8.2f,%8.2f] %3.0f' % (k, f(obs[k]), f(sd[k]), np.median(x), np.percentile(x, 2.5), np.percentile(x, 97.5), (x < sd[k]).mean() * 100))
        print('triángulos inter-actor: observado %d · nulo de campo med %.1f [%.1f, %.1f]' % (obs['triangulos'], np.median(tn), np.percentile(tn, 2.5), np.percentile(tn, 97.5)))
    json.dump(res, open('clase_D.json', 'w'), indent=0)
