# -*- coding: utf-8 -*-
"""§4.2 y Suplemento S4.2: control cruzado del certificado (Nilo x Selene).
Lee controles/control_cruzado/cruz_F.jsonl (juez 1 sobre 45.000 pares cruzados) y el objeto de cada
caso. Separa lo observado de lo estimado y compara contra el certificado interno ANTES y DESPUÉS de
la verificación Λ, porque el campo cruzado no pasó por Λ. Uso: python control_cruzado.py"""
import json
import numpy as np, pandas as pd
from comun import ruta, meta, capa, valor

# ---- campo cruzado ----
filas = []
for linea in open(ruta('controles', 'control_cruzado', 'cruz_F.jsonl'), encoding='utf-8'):
    filas += json.loads(linea)['r']
m = json.load(open(ruta('controles', 'control_cruzado', 'cruz_muestra.json'), encoding='utf-8'))
u = meta('gerd')['umbrales']
tau, gamma = u['tau'], u['gamma']
df = pd.DataFrame(filas, columns=['estrato', 'i', 'j', 'cos', 'imp_ij', 'imp_ji', 'con_ij', 'con_ji'])
df['cert'] = ((df.imp_ij + df.imp_ji) / 2 >= tau) & (df[['con_ij', 'con_ji']].max(axis=1) < gamma)
pasan = m['pasan']
nA = int((df.estrato == 'A').sum()); nB = int((df.estrato == 'B').sum())
cA = int(df.cert[df.estrato == 'A'].sum()); cB = int(df.cert[df.estrato == 'B'].sum())
restoB = pasan - nA
est = cA + cB * restoB / nB
print('== campo cruzado (juez 1, sin verificación Λ)')
print(f'  pares que pasan el prefiltro: {pasan}')
print(f'  estrato A (coseno más alto, exhaustivo): {nA} evaluados, {cA} certificados; coseno mínimo de A = {df.cos[df.estrato=="A"].min():.4f}')
print(f'  estrato B (muestra al azar de {restoB}): {nB} evaluados, {cB} certificados')
print(f'  OBSERVADO: {cA + cB} certificados sobre {nA + nB} pares evaluados')
print(f'  ESTIMADO sobre todo el campo: {cA} + {cB}·{restoB}/{nB} = {est:.1f}  -> tasa {est / pasan:.3e}')
print(f'  nodos distintos del Nilo entre los certificados: {df.i[df.cert].nunique()}')

# ---- certificado interno, antes y después de Λ ----
def interno(caso):
    M = meta(caso); n = M['n']
    act = pd.read_csv(ruta('objeto', caso, 'nodos.csv'))['actor'].values
    I = capa(caso, 'j1_impl', n); C = capa(caso, 'j1_contr', n)
    S = capa(caso, 'sim', n); R = capa(caso, 'rechazo', n)
    Ic = I.tocoo(); sel = Ic.row < Ic.col
    r, c = Ic.row[sel], Ic.col[sel]
    imp = (valor(Ic.data[sel]) + valor(np.asarray(I[c, r]).ravel())) / 2
    con = np.fmax(valor(np.asarray(C[r, c]).ravel()), valor(np.asarray(C[c, r]).ravel()))
    sim = np.fmax(valor(np.asarray(S[r, c]).ravel()), valor(np.asarray(S[c, r]).ravel()))
    rech = (np.asarray(R[r, c]).ravel() != 0) | (np.asarray(R[c, r]).ravel() != 0)
    ok = (act[r] != act[c]) & (imp >= tau) & (con < gamma) & (sim >= M['umbrales']['coseno'])
    return dict(pares=len(r), pre=int(ok.sum()), post=int((ok & ~rech).sum()), sim=sim, ok=ok, rech=rech)

print('\n== certificado interno (mismo juez 1, mismos umbrales)')
res = {}
for caso in ['gerd', 'selene']:
    x = interno(caso); res[caso] = x
    print(f'  {caso}: {x["pares"]} pares; certificados antes de Λ = {x["pre"]}, después de Λ = {x["post"]}')
g = res['gerd']
print('\n== comparación cruzado / Nilo')
print(f'  tasa cruzada estimada / tasa Nilo después de Λ (lo que compara el texto actual): '
      f'{(est / pasan) / (g["post"] / g["pares"]):.2f}')
print(f'  tasa cruzada estimada / tasa Nilo ANTES de Λ (comparación homogénea): '
      f'{(est / pasan) / (g["pre"] / g["pares"]):.2f}')

# ---- por bandas de coseno ----
print('\n== por bandas de coseno (tasa de certificados; cruzado bajo 0,503 desde la muestra B)')
cortes = [0.35, 0.40, 0.45, float(df.cos[df.estrato == 'A'].min()), 0.60, 0.70, 1.01]
print(f'  {"banda":<16}{"Nilo post-Λ":>12}{"Nilo pre-Λ":>12}{"cruzado":>12}{"/ post":>8}{"/ pre":>8}')
for lo, hi in zip(cortes[:-1], cortes[1:]):
    b = (g['sim'] >= lo) & (g['sim'] < hi)
    if b.sum() == 0: continue
    tpost = (g['ok'] & ~g['rech'] & b).sum() / b.sum(); tpre = (g['ok'] & b).sum() / b.sum()
    est_ = 'B' if hi <= cortes[3] + 1e-9 else 'A'
    xb = df[(df.estrato == est_) & (df.cos >= lo) & (df.cos < hi)]
    tc = xb.cert.mean() if len(xb) else float('nan')
    f = lambda a, b_: f'{a / b_:.1f}' if b_ > 0 else '—'
    print(f'  [{lo:.3f}, {hi:.3f}){"":<2}{tpost:>12.2e}{tpre:>12.2e}{tc:>12.2e}{f(tc, tpost):>8}{f(tc, tpre):>8}   (n cruzado {len(xb)})')
