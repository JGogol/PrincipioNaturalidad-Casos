# -*- coding: utf-8 -*-
"""Valida objeto_min contra las cifras publicadas en PN II v3.2 (§2.2, §3.6, §3.7, §4.6, §4.7)."""
import sys, json, time
from objeto_min import Objeto
REF = {'gerd':   dict(F=102, terreno_comun=370, puentes=304, base=610, reglas_cruce=304, regiones_distintas=1403, reducto=94,
                      raices={'Arab Republic of Egypt': 822, 'Federal Democratic Republic of Ethiopia': 823, 'Republic of Sudan': 701}),
       'selene': dict(F=218, terreno_comun=551, puentes=442, base=673, reglas_cruce=442, regiones_distintas=1271, reducto=183)}
for caso in sys.argv[1:] or ['gerd', 'selene']:
    t0 = time.time()
    o = Objeto('../objeto/' + caso)
    F = o.F_pares()
    Reg = o.regiones(F)
    L = o.lecturas(Reg, F)
    red = o.reducto(F)
    L['reducto'] = len(red)
    print(caso, json.dumps(L, ensure_ascii=False))
    for k, v in REF[caso].items():
        print('  %-20s calculado %-40s publicado %-40s %s' % (k, str(L.get(k)), str(v), 'OK' if L.get(k) == v else 'DIFIERE'))
    print('  %.0f s' % (time.time() - t0))
