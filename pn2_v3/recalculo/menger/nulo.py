import sys, json, numpy as np, networkx as nx
sys.path.insert(0, '../../lectura')
import menger
from objeto_min import Objeto
from nulo_F import campo
caso, reps, seed = sys.argv[1], int(sys.argv[2]), int(sys.argv[3])
o = Objeto('../../objeto/' + caso); F = o.F_pares(); P = campo(o)
rng = np.random.default_rng(seed)
def frac(FF, n=100):
    tot = ok = 0
    for a, r in enumerate(o.raices):
        G = menger.grafo(o, FF, a, 'kF')
        alc = [x for x in nx.descendants(G, r) if not isinstance(x, tuple) and o.act_idx[x] != a]
        if not alc: continue
        for t in rng.choice(alc, size=min(n, len(alc)), replace=False):
            tot += 1; ok += menger.corte(G, r, int(t)) >= 2
    return ok / tot if tot else float('nan')
res = {'real': frac(F) if seed == 0 else None, 'nulo': []}
for k in range(reps):
    idx = rng.choice(len(P), size=len(F), replace=False); res['nulo'].append(frac([P[i] for i in idx]))
json.dump(res, open('nulo_%s_%d.json' % (caso, seed), 'w')); print(caso, seed, res)
