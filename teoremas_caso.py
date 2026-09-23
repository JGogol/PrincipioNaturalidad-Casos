#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import json, os, sys, glob, random, statistics
from collections import defaultdict
random.seed(20260902)
RUN=sys.argv[1]
nodos={}
for f in glob.glob(os.path.join(RUN,'arboles','*.json')):
    a=json.load(open(f,encoding='utf-8'))
    actor=a['actor']
    def rec(n):
        nodos[n['id']]={'actor':actor,'nivel':n['nivel'],'hijos':[h['id'] for h in n.get('hijos',[])]}
        for h in n.get('hijos',[]): rec(h)
    rec(a['raiz'])
fj=json.load(open(os.path.join(RUN,'fusiones.json'),encoding='utf-8'))
fl=fj['fusiones'] if isinstance(fj,dict) else fj
EDGES=[(e['nodo_a_id'],e['nodo_b_id']) for e in fl if e['nodo_a_id'] in nodos and e['nodo_b_id'] in nodos]
IDS=list(nodos); actores=sorted({nodos[i]['actor'] for i in IDS})
print("CASO:", os.path.basename(RUN.rstrip('/')))
print("actores:%d  nodos:%d  fusiones:%d"%(len(actores),len(IDS),len(EDGES)))
_sc={}
def sub(n):
    if n in _sc: return _sc[n]
    o=set([n]); p=[n]
    while p:
        x=p.pop()
        for h in nodos[x]['hijos']:
            if h not in o: o.add(h); p.append(h)
    _sc[n]=o; return o
def build(edges):
    a=defaultdict(set)
    for x,y in edges: a[x].add(y); a[y].add(x)
    return a
def cl(seed,adj):
    Y=set(seed); ch=True
    while ch:
        ch=False
        for u in list(Y):
            for w in adj[u]:
                if w not in Y:
                    g=sub(w)
                    if not g<=Y: Y|=g; ch=True
    return Y
full=build(EDGES)
R={v:cl(sub(v),full) for v in IDS}
tam=[len(R[v]) for v in IDS]
print("\n[SATURACION] region: media=%.1f mediana=%d max=%d (de %d) -> %s"
      %(statistics.mean(tam),statistics.median(tam),max(tam),len(IDS),
        "RALO (no satura)" if statistics.median(tam)<len(IDS)*0.3 else "denso/satura"))
ve=vi=vm=0
def semilla(): return set().union(*[sub(random.choice(IDS)) for _ in range(random.randint(1,5))])
for _ in range(150):
    X=semilla(); cX=cl(X,full)
    if not X<=cX: ve+=1
    if cl(cX,full)!=cX: vi+=1
    Y=X|semilla()
    if not cl(X,full)<=cl(Y,full): vm+=1
print("[CLAUSURA] ext=%d idem=%d mono=%d violaciones (150 pruebas)"%(ve,vi,vm))
Rl=list(R.values()); vint=0
for _ in range(800):
    A=random.choice(Rl); B=random.choice(Rl); I=A&B
    if cl(I,full)!=I: vint+=1
print("[SIST. CLAUSURA] R(u)inter R(v) definible: %d violaciones (800 pares)"%vint)
adj=build(EDGES); base=[]
for (a,b) in EDGES:
    adj[a].discard(b); adj[b].discard(a)
    if not((sub(b)<=cl({a},adj)) and (sub(a)<=cl({b},adj))): base.append((a,b)); adj[a].add(b); adj[b].add(a)
print("[BASE MINIMA] %d/%d fusiones (%.1f%% base, %.1f%% redundante)"
      %(len(base),len(EDGES),100*len(base)/len(EDGES),100*(1-len(base)/len(EDGES))))
raices={a:next(i for i in IDS if nodos[i]['actor']==a and nodos[i]['nivel']==0) for a in actores}
meet=set(R[raices[actores[0]]])
for a in actores[1:]: meet&=R[raices[a]]
print("[NUCLEO COMUN] meet de los %d actores: %d nodos (definible=%s)"%(len(actores),len(meet),cl(meet,full)==meet))
badj=build(base); col=[]
for (a,b) in base:
    Rc=cl(sub(a),badj); badj[a].discard(b); badj[b].discard(a)
    Rs=cl(sub(a),badj); badj[a].add(b); badj[b].add(a)
    d=len(Rc-Rs)
    if d>0: col.append(d)
print("[RECOVERY] renunciar a 1 convergencia arrastra media=%.1f max=%d colaterales"%(statistics.mean(col) if col else 0, max(col) if col else 0))
