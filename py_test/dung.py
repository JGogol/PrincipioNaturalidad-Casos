#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Principle of Naturality - Dung Argumentation Framework v8.8
============================================================

v8.8: Agrega conteo de nodos nunca contradichos en el árbol completo
      ¿Cuántos nodos del total nunca aparecieron en ninguna contradicción?
      Estos son la "zona ciega" — ni fusionaron ni fueron atacados.

Uso:
    python dung.py --path /ruta/al/run --caso jamison
    python dung.py --path /ruta/al/run --caso gerd
    python dung.py --path /ruta/al/run --caso lialee

Necesita en --path:
    estadisticas_fusiones.json
    fusiones.json
    hubs.json          (opcional — activa cruce grounded↔HUBs)
    arboles/           (opcional — activa conteo zona ciega exacto)
      Actor1.json
      Actor2.json
      ...
"""

import json, collections, argparse, os, glob


def actor_from_id(node_id):
    parts = node_id.split('_')
    return '_'.join(parts[:-2])


def load_framework(run_path):
    path = os.path.join(run_path, "estadisticas_fusiones.json")
    print(f"Cargando {path} ...")
    with open(path, encoding='utf-8') as f:
        data = json.load(f)
    return data


def load_eq_space(run_path):
    eq_nodes, fusion_pairs = set(), []
    path = os.path.join(run_path, "fusiones.json")
    if not os.path.exists(path):
        return eq_nodes, fusion_pairs
    print(f"Cargando {path} ...")
    with open(path, encoding='utf-8') as f:
        fdata = json.load(f)
    items = fdata if isinstance(fdata, list) else fdata.get('items', fdata.get('fusiones', []))
    for item in items:
        a = item.get('nodo_a_id') or item.get('node_a_id') or item.get('id_a')
        b = item.get('nodo_b_id') or item.get('node_b_id') or item.get('id_b')
        if a and b:
            fusion_pairs.append((a, b))
            eq_nodes.add(a)
            eq_nodes.add(b)
    return eq_nodes, fusion_pairs


def load_hubs(run_path):
    path = os.path.join(run_path, "hubs.json")
    if not os.path.exists(path):
        return {}, {}
    print(f"Cargando {path} ...")
    with open(path, encoding='utf-8') as f:
        hdata = json.load(f)
    items = hdata if isinstance(hdata, list) else hdata.get('items', hdata.get('hubs', []))
    items_sorted = sorted(items, key=lambda h: (h.get('ftt_sum', 9999), h.get('ftt_max', 9999)))
    hub_convergencia, nodo_a_hub = {}, {}
    for rank, hub in enumerate(items_sorted, start=1):
        clase_id    = hub.get('clase_id', f'hub_{rank}')
        clase_nodos = set(hub.get('clase_nodos', []))
        hub_convergencia[clase_id] = {
            'rank': rank,
            'ftt_sum': hub.get('ftt_sum', 0),
            'ftt_max': hub.get('ftt_max', 0),
            'clase_nodos': clase_nodos,
            'nodo_por_actor': hub.get('nodo_por_actor', {}),
        }
        for nodo_id in clase_nodos:
            nodo_a_hub[nodo_id] = clase_id
    print(f"  HUBs cargados: {len(hub_convergencia)}")
    return hub_convergencia, nodo_a_hub


def load_all_nodes(run_path):
    """
    Carga todos los nodos de los árboles causales.
    Busca en arboles/*.json y los recorre recursivamente.
    Devuelve dict { node_id -> texto } con TODOS los nodos del caso.
    """
    all_nodes = {}
    arboles_dir = os.path.join(run_path, "arboles")

    if not os.path.exists(arboles_dir):
        print("  arboles/ no encontrado — zona ciega calculada solo desde contradicciones")
        return all_nodes

    archivos = glob.glob(os.path.join(arboles_dir, "*.json"))
    print(f"Cargando árboles: {len(archivos)} archivos ...")

    def recorrer(nodo_data, actor):
        nid  = nodo_data.get('id')
        txt  = nodo_data.get('texto', '')
        if nid:
            all_nodes[nid] = txt
        for hijo in nodo_data.get('hijos', []):
            recorrer(hijo, actor)

    for path in archivos:
        with open(path, encoding='utf-8') as f:
            arbol = json.load(f)
        actor = arbol.get('actor', '')
        raiz  = arbol.get('raiz', {})
        recorrer(raiz, actor)

    print(f"  Total nodos en árboles: {len(all_nodes)}")
    return all_nodes


def build_attack_graph(contradictions):
    """Solo contradicciones NLI son ataques Dung."""
    attacks_to = collections.defaultdict(set)
    for item in contradictions:
        na, nb = item['node_a_id'], item['node_b_id']
        attacks_to[na].add(nb)
        attacks_to[nb].add(na)
    return attacks_to


def grounded_extension(nodes, attacks_to):
    grounded, defeated = set(), set()
    changed = True
    while changed:
        changed = False
        for node in nodes:
            if node in grounded or node in defeated:
                continue
            if (attacks_to.get(node, set()) & nodes).issubset(defeated):
                grounded.add(node)
                changed = True
        new_def = set()
        for g in grounded:
            new_def.update(attacks_to.get(g, set()) & nodes)
        newly = new_def - defeated
        if newly:
            defeated |= newly
            changed = True
    return grounded, defeated


def conflict_components(nodes, attacks_to):
    hub_set   = set(nodes)
    conflicts = {n: attacks_to.get(n, set()) & hub_set for n in nodes}
    visited, components = set(), []
    for start in nodes:
        if start not in visited:
            comp, queue = set(), [start]
            while queue:
                n = queue.pop()
                if n in visited:
                    continue
                visited.add(n)
                comp.add(n)
                queue.extend(conflicts[n] - visited)
            components.append(comp)
    isolated = {n for n in nodes if not conflicts[n]}
    return components, isolated


def get_node_texts(run_path):
    texts = {}
    path = os.path.join(run_path, "fusiones.json")
    if not os.path.exists(path):
        return texts
    with open(path, encoding='utf-8') as f:
        fdata = json.load(f)
    items = fdata if isinstance(fdata, list) else fdata.get('items', fdata.get('fusiones', []))
    for item in items:
        for id_k, txt_k in [('nodo_a_id','nodo_a_texto'),('nodo_b_id','nodo_b_texto'),
                             ('node_a_id','node_a_texto'),('node_b_id','node_b_texto')]:
            nid, txt = item.get(id_k), item.get(txt_k)
            if nid and txt:
                texts[nid] = txt
    return texts


def cruzar_grounded_hubs(grounded, hub_convergencia, nodo_a_hub):
    if not hub_convergencia:
        return [], {}
    resultados = []
    resumen = {'en_algun_hub': 0, 'fuera_de_hubs': 0}
    for node_id in sorted(grounded):
        clase_id = nodo_a_hub.get(node_id)
        if clase_id is None:
            resultados.append({'node_id': node_id, 'actor': actor_from_id(node_id),
                                'en_hub': False, 'clase_id': None, 'hub_rank': None,
                                'ftt_sum': None, 'es_optimo': False, 'situacion': 'FUERA_DE_HUBS'})
            resumen['fuera_de_hubs'] += 1
        else:
            hub = hub_convergencia[clase_id]
            es_optimo = node_id in hub['nodo_por_actor'].values()
            resultados.append({'node_id': node_id, 'actor': actor_from_id(node_id),
                                'en_hub': True, 'clase_id': clase_id,
                                'hub_rank': hub['rank'], 'ftt_sum': hub['ftt_sum'],
                                'ftt_max': hub['ftt_max'], 'es_optimo': es_optimo,
                                'situacion': 'EN_HUB'})
            resumen['en_algun_hub'] += 1
    return resultados, resumen


def analizar_zona_ciega(all_nodes, contradictions, eq_nodes):
    """
    Calcula nodos nunca contradichos en el árbol completo.

    Tres categorías:
      contradicted     : aparece en al menos 1 par de contradicción
      en_E_no_atacado  : está en E (fusionó) pero nadie lo contradijo
      zona_ciega       : nunca fusionó NI fue contradicho — invisible al sistema
    """
    if not all_nodes:
        return {}

    # Nodos que aparecen en alguna contradicción
    contradicted = set()
    for item in contradictions:
        contradicted.add(item['node_a_id'])
        contradicted.add(item['node_b_id'])

    total          = len(all_nodes)
    n_contradicted = len(contradicted & all_nodes.keys())

    # Nodos en E que no fueron atacados
    en_E_no_atacado = eq_nodes - contradicted

    # Zona ciega: fuera de E Y nunca contradichos
    zona_ciega = set(all_nodes.keys()) - contradicted - eq_nodes

    # Desglose por actor
    por_actor = collections.defaultdict(lambda: {
        'total': 0, 'contradicted': 0, 'en_E_no_atacado': 0, 'zona_ciega': 0
    })
    for nid in all_nodes:
        a = actor_from_id(nid)
        por_actor[a]['total'] += 1
        if nid in contradicted:
            por_actor[a]['contradicted'] += 1
        elif nid in eq_nodes:
            por_actor[a]['en_E_no_atacado'] += 1
        else:
            por_actor[a]['zona_ciega'] += 1

    return {
        'total_nodos_arbol':    total,
        'nodos_contradicted':   n_contradicted,
        'nodos_en_E_no_atacado': len(en_E_no_atacado),
        'nodos_zona_ciega':     len(zona_ciega),
        'ratio_zona_ciega':     round(len(zona_ciega) / total, 4) if total else 0,
        'por_actor':            dict(por_actor),
    }


def analyze(run_path, caso):
    print()
    print("=" * 65)
    print(f"DUNG AF ANALYSIS v8.8 — {caso.upper()}")
    print("=" * 65)

    data      = load_framework(run_path)
    stats     = data['stats']
    contrad   = data['contradictions']['items']
    rejected  = data['rejected_fusions']['items']
    evaluated = stats['pairs_evaluated']

    eq_nodes, fusion_pairs       = load_eq_space(run_path)
    hub_convergencia, nodo_a_hub = load_hubs(run_path)
    all_nodes                    = load_all_nodes(run_path)

    if not eq_nodes:
        print("ERROR: fusiones.json no encontrado o vacío")
        return

    attacks_to           = build_attack_graph(contrad)
    eq_conf              = {h: attacks_to.get(h, set()) & eq_nodes for h in eq_nodes}
    internal             = sum(len(v) for v in eq_conf.values()) // 2
    grounded, defeated   = grounded_extension(eq_nodes, eq_conf)
    components, isolated = conflict_components(eq_nodes, eq_conf)
    comp_sizes           = sorted([len(c) for c in components], reverse=True)
    texts                = get_node_texts(run_path)
    cruce, resumen_cruce = cruzar_grounded_hubs(grounded, hub_convergencia, nodo_a_hub)
    zona_ciega           = analizar_zona_ciega(all_nodes, contrad, eq_nodes)

    # ── Consola ──────────────────────────────────────────────────────────
    print(f"\nEspacio E:           {len(eq_nodes)} nodos, {len(fusion_pairs)} pares fusión")
    print(f"Contradicciones NLI: {len(contrad)}  →  ratio = {len(contrad)/evaluated:.1%}")
    print(f"Conflictos internos: {internal}")
    print(f"Grounded:            {len(grounded)} ({len(grounded)/len(eq_nodes):.1%} de E)")
    print(f"Defeated:            {len(defeated)}")
    print(f"Conflict components: {len(components)}  {comp_sizes[:8]}")

    print("\nExtensión fundamentada:")
    for n in sorted(grounded):
        txt = texts.get(n, '')
        print(f"  [{actor_from_id(n)[:25]}] {n.split('_')[-2]}_{n.split('_')[-1]}")
        if txt: print(f"    \"{txt[:85]}\"")

    if cruce:
        print(f"\nCruce grounded↔HUBs: en_algun_hub={resumen_cruce['en_algun_hub']}  fuera={resumen_cruce['fuera_de_hubs']}")

    if zona_ciega:
        print(f"\nZONA CIEGA (nunca fusionó NI fue contradicho):")
        print(f"  Total nodos árbol:     {zona_ciega['total_nodos_arbol']}")
        print(f"  Contradichos:          {zona_ciega['nodos_contradicted']}")
        print(f"  En E, no atacados:     {zona_ciega['nodos_en_E_no_atacado']}")
        print(f"  Zona ciega:            {zona_ciega['nodos_zona_ciega']}  ({zona_ciega['ratio_zona_ciega']:.1%} del árbol)")
        print(f"\n  Por actor:")
        for actor, d in zona_ciega['por_actor'].items():
            print(f"    {actor[:45]:45}  total={d['total']}  contradichos={d['contradicted']}  "
                  f"E_no_atacado={d['en_E_no_atacado']}  zona_ciega={d['zona_ciega']}")

    # ── JSON ─────────────────────────────────────────────────────────────
    resultado = {
        'caso': caso, 'version': '8.8',
        'nli_stats': {
            'pairs_total':             stats['pairs_total'],
            'pairs_evaluated':         evaluated,
            'fusions_approved':        stats['fusions_approved'],
            'contradictions_detected': stats['contradictions_detected'],
            'nli_neutral':             stats['nli_neutral'],
            'nli_low_entailment':      stats['nli_low_entailment'],
        },
        'metricas': {
            'eq_space_nodes':        len(eq_nodes),
            'fusion_pairs':          len(fusion_pairs),
            'contradictions_nli':    len(contrad),
            'role_inversion_pairs':  len(rejected),
            'ratio_conflict_nli':    round(len(contrad)/evaluated, 4),
            'ratio_conflict_total':  round((len(contrad)+len(rejected))/evaluated, 4),
            'internal_conflicts':    internal,
            'grounded_size':         len(grounded),
            'grounded_ratio':        round(len(grounded)/len(eq_nodes), 4),
            'defeated_size':         len(defeated),
            'indeterminate_size':    len(eq_nodes)-len(grounded)-len(defeated),
            'conflict_components':   len(components),
            'component_sizes':       comp_sizes,
            'isolated_nodes':        len(isolated),
            'fusion_hub_ratio':      round(len(fusion_pairs)/len(eq_nodes), 4),
        },
        'grounded_extension': [
            {'node_id': n, 'actor': actor_from_id(n), 'texto': texts.get(n, '')}
            for n in sorted(grounded)
        ],
        'cruce_grounded_hubs': {'resumen': resumen_cruce, 'detalle': cruce},
        'zona_ciega': zona_ciega,
        'actores': {}
    }

    actor_data = collections.defaultdict(lambda: {'nodos': 0, 'conflictos': 0})
    for h in eq_nodes:
        a = actor_from_id(h)
        actor_data[a]['nodos'] += 1
        actor_data[a]['conflictos'] += len(eq_conf[h])
    for a, d in actor_data.items():
        resultado['actores'][a] = {
            'nodos_en_E':          d['nodos'],
            'conflictos_internos': d['conflictos'] // 2,
            'en_grounded':         sum(1 for n in grounded if actor_from_id(n) == a),
            'en_defeated':         sum(1 for n in defeated if actor_from_id(n) == a),
        }

    out_path = os.path.join(run_path, f"dung_resultado_{caso}.json")
    with open(out_path, 'w', encoding='utf-8') as f:
        json.dump(resultado, f, ensure_ascii=False, indent=2)

    size_kb = os.path.getsize(out_path) / 1024
    print(f"\nGuardado: {out_path}  ({size_kb:.1f} KB)")


if __name__ == '__main__':
    p = argparse.ArgumentParser(description='Dung AF Analysis v8.8')
    p.add_argument('--path', required=True, help='Run directory del caso')
    p.add_argument('--caso', required=True, help='jamison / gerd / lialee')
    a = p.parse_args()
    analyze(a.path, a.caso)