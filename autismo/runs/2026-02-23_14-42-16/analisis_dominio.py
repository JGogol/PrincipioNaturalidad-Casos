"""
Análisis completo BLOCK y AUTO_FILTER por dominio y nivel
Principle of Naturality - Autistic and Neurotypical Cognition
Ejecutar: python3 analisis_completo_dominios_niveles.py
Requiere: grafo.json en el mismo directorio

Método: el dominio se propaga desde el nodo L1 (que tiene ID de situación)
hacia todos sus descendientes via BFS sobre las aristas del grafo.
Esto permite análisis por dominio en todos los niveles, no solo en L1.
"""

import json, re
from collections import defaultdict

# ─── CARGA ────────────────────────────────────────────────────────────────────
with open('grafo.json') as f:
    data = json.load(f)

nodos   = data['nodos']
aristas = data['aristas']

# ─── DOMINIO DESDE ID ─────────────────────────────────────────────────────────
def get_domain_from_id(node_id):
    m = re.search(r'_s(\d+)_', node_id)
    if not m:
        return None
    s = int(m.group(1))
    if  1 <= s <= 10:  return 'D1'
    if 11 <= s <= 20:  return 'D2'
    if 21 <= s <= 30:  return 'D3'
    if 31 <= s <= 50:  return 'D4'
    if 51 <= s <= 60:  return 'D5'
    if 61 <= s <= 75:  return 'D6'
    if 76 <= s <= 90:  return 'D7'
    if 91 <= s <= 100: return 'D8'
    return None

DOMAIN_NAMES = {
    'D1': 'School — Sensory/Routine   (S1–S10)',
    'D2': 'School — Social/Language   (S11–S20)',
    'D3': 'School — Cognitive/Learn   (S21–S30)',
    'D4': 'Home / Family              (S31–S50)',
    'D5': 'Medical / Public           (S51–S60)',
    'D6': 'Social Peer — Learning     (S61–S75)',
    'D7': 'Social Peer — Interaction  (S76–S90)',
    'D8': 'Self-Regulation            (S91–S100)',
}
DOMS   = ['D1','D2','D3','D4','D5','D6','D7','D8']
LEVELS = [1, 2, 3, 4, 5, 6, 7]

# ─── PROPAGACIÓN DE DOMINIO POR BFS ──────────────────────────────────────────
children = defaultdict(list)
for a in aristas:
    children[a['origen']].append(a['destino'])

node_domain = {}
roots = ['AutisticCognition_0_0', 'NeurotypicalCognition_0_0']

for root in roots:
    queue = [(root, None)]
    while queue:
        current, parent_domain = queue.pop(0)
        d = get_domain_from_id(current)
        node_domain[current] = d if d else parent_domain
        for child in children[current]:
            queue.append((child, node_domain[current]))

# ─── CLASIFICADORES ───────────────────────────────────────────────────────────
def is_block(t):
    t = t.lower()
    return any(x in t for x in [
        'cannot ', 'lacks explicit', 'lacks predictable', 'lacks rules',
        'lacks concrete', 'loses predictable', 'loses access',
        'experiences disruption', 'experiences disrupted',
        'breaks down', 'becomes disrupted', 'is interrupted',
        'becomes stuck', 'no explicit rule', 'has no explicit'
    ])

def is_autofilter(t):
    t = t.lower()
    return any(x in t for x in ['automatically ', 'automatic relevance'])

def is_social_infer(t):
    t = t.lower()
    return any(x in t for x in [
        'infers', 'social norm', 'social cue', 'social inference',
        'social expectation', 'social signal', 'social context',
        'social meaning', 'social relevance', 'begins inferring',
        'social script', 'social framework', 'social rules'
    ])

def is_rule_seek(t):
    t = t.lower()
    return any(x in t for x in [
        'searches for explicit', 'requires explicit',
        'seeks explicit', 'requires concrete-visual representation',
        'requires concrete instructions', 'requires concrete examples',
        'requires predictable feedback to proceed',
        'requires predictable structure before',
        'requires concrete-visual information'
    ])

def is_fixation(t):
    t = t.lower()
    return any(x in t for x in [
        'fixates on', 'locks onto', 'attention locks',
        'focuses on specific', 'focuses on individual',
        'focuses on the specific', 'focuses on concrete details'
    ])

def is_generalize(t):
    t = t.lower()
    return any(x in t for x in [
        'generaliz', 'applies previous', 'applies familiar',
        'applies learned', 'draws on previous', 'recalls previous',
        'references similar past', 'recognizes the structural similarity',
        'searches for structural similarities'
    ])

# ─── ACUMULACIÓN ──────────────────────────────────────────────────────────────
# stats[domain][nivel][metric]
stats = defaultdict(lambda: defaultdict(lambda: defaultdict(int)))

for n in nodos:
    nid   = n['id']
    nivel = n.get('nivel', 0)
    texto = n.get('texto', '')
    if nivel == 0:
        continue

    actor  = 'AC' if 'Autistic' in nid else 'NT'
    domain = node_domain.get(nid)
    if domain is None:
        domain = 'UNKNOWN'

    s = stats[domain][nivel]
    if actor == 'AC':
        s['ac_total']    += 1
        if is_block(texto):     s['ac_block']    += 1
        if is_rule_seek(texto): s['ac_rule_seek'] += 1
        if is_fixation(texto):  s['ac_fixation']  += 1
    else:
        s['nt_total'] += 1
        if is_autofilter(texto):   s['nt_af']  += 1
        if is_social_infer(texto): s['nt_si']  += 1
        if is_generalize(texto):   s['nt_gen'] += 1

# ─── HELPER ───────────────────────────────────────────────────────────────────
def pct(num, den):
    return f"{100*num/den:.1f}" if den > 0 else "—"

def print_row(label, s, width=8):
    at = s['ac_total'] or 1
    nt = s['nt_total'] or 1
    print(
        f"  {label:<{width}} "
        f"{s['ac_total']:<7} "
        f"{pct(s['ac_block'],at):<8} "
        f"{pct(s['ac_rule_seek'],at):<8} "
        f"{pct(s['ac_fixation'],at):<8} | "
        f"{s['nt_total']:<7} "
        f"{pct(s['nt_af'],nt):<8} "
        f"{pct(s['nt_si'],nt):<8} "
        f"{pct(s['nt_gen'],nt)}"
    )

HDR = (
    f"  {'Nivel':<8} {'AC_n':<7} {'BLK%':<8} {'RSK%':<8} {'FIX%':<8} | "
    f"{'NT_n':<7} {'AF%':<8} {'SI%':<8} {'GEN%'}"
)
SEP = "  " + "-" * 78

# ─── REPORTE 1: por dominio y nivel ──────────────────────────────────────────
print()
print("=" * 82)
print("REPORTE 1 — BLOCK / RULE_SEEK / FIXATION (AC)  y  AF / SI / GEN (NT)")
print("           por DOMINIO y NIVEL  (dominio propagado desde L1 por árbol)")
print("=" * 82)
print("Claves: BLK=BLOCK  RSK=RULE_SEEK  FIX=FIXATION  AF=AUTO_FILTER  SI=SOCIAL_INFER  GEN=GENERALIZE")

for d in DOMS:
    print()
    print(f"  {'='*78}")
    print(f"  {DOMAIN_NAMES[d]}")
    print(f"  {'='*78}")
    print(HDR)
    print(SEP)

    acc = defaultdict(int)
    for nivel in LEVELS:
        s = stats[d][nivel]
        if s['ac_total'] == 0 and s['nt_total'] == 0:
            continue
        print_row(str(nivel), s)
        for k in s: acc[k] += s[k]

    print(SEP)
    print_row('TOT', acc, width=8)

# ─── REPORTE 2: global por nivel (todos los dominios) ─────────────────────────
print()
print()
print("=" * 82)
print("REPORTE 2 — Global por NIVEL (suma de todos los dominios)")
print("=" * 82)
print(HDR)
print(SEP)

grand = defaultdict(int)
for nivel in LEVELS:
    acc = defaultdict(int)
    for d in DOMS:
        s = stats[d][nivel]
        for k in s: acc[k] += s[k]
    if acc['ac_total'] == 0 and acc['nt_total'] == 0:
        continue
    print_row(str(nivel), acc)
    for k in acc: grand[k] += acc[k]

print(SEP)
print_row('TOTAL', grand)

# ─── REPORTE 3: resumen por dominio (todos los niveles sumados) ────────────────
print()
print()
print("=" * 82)
print("REPORTE 3 — Resumen por DOMINIO (suma todos los niveles)")
print("=" * 82)
print(HDR)
print(SEP)

for d in DOMS:
    acc = defaultdict(int)
    for nivel in LEVELS:
        for k, v in stats[d][nivel].items():
            acc[k] += v
    print_row(DOMAIN_NAMES[d][:40], acc, width=42)

print(SEP)
grand2 = defaultdict(int)
for d in DOMS:
    for nivel in LEVELS:
        for k, v in stats[d][nivel].items():
            grand2[k] += v
print_row('TOTAL', grand2, width=42)

print()
print("Análisis completado.")
print("Todos los valores son auditables desde grafo.json (campo 'texto', 'nivel', 'id').")