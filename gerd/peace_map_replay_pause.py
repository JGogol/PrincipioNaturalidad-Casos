"""
Peace Field Replay — Presentation Mode
=======================================
Versión con pausas manuales para charlas y presentaciones.

Cada segmento corre animado y luego PAUSA esperando click del usuario.

Segmentos:
1. Π aparece
2-4. Cada árbol crece nodo por nodo
5-7. Fusiones por par de actores
8-10. HUBs uno por uno
11. Vista final

Uso:
    python peace_map_presentation.py --run-path casos/gerd/runs/2026-02-05_15-29-23
"""

import argparse
import json
from pathlib import Path
from datetime import datetime
from collections import defaultdict
import numpy as np
import networkx as nx
from sklearn.decomposition import PCA
import plotly.graph_objects as go

# ============================================================================
# CONFIGURACIÓN
# ============================================================================

COLOR_ROTATION = [
    '#E63946',  # Rojo (Egypt)
    '#2A9D8F',  # Cyan (Ethiopia)
    '#E9C46A',  # Amarillo (Sudan)
    '#9B5DE5',  # Púrpura
    '#00BBF9',  # Cyan claro
    '#F15BB5',  # Rosa
]

COLOR_PI = '#FFD700'      # Dorado
COLOR_HUB = '#FFFFFF'     # Blanco
COLOR_FUSION = '#9B59B6'  # Púrpura

ACTOR_COLORS = {}
ACTOR_SHORT = {}


def initialize_actors(actors):
    global ACTOR_COLORS, ACTOR_SHORT
    common_words = ['Republic', 'Democratic', 'Federal', 'of', 'the', 'The',
                    'Kingdom', 'State', 'United', 'People', 'Arab', 'Islamic']
    
    for i, actor in enumerate(sorted(actors)):
        ACTOR_COLORS[actor] = COLOR_ROTATION[i % len(COLOR_ROTATION)]
        words = [w for w in actor.split() if w not in common_words]
        ACTOR_SHORT[actor] = words[-1] if words else actor[:10]
        ACTOR_COLORS[ACTOR_SHORT[actor]] = ACTOR_COLORS[actor]


def get_short(actor):
    return ACTOR_SHORT.get(actor, actor.split()[-1] if actor else 'Unknown')


def get_color(actor):
    return ACTOR_COLORS.get(actor, ACTOR_COLORS.get(get_short(actor), '#888888'))


def cosine_sim(a, b):
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-9)


# ============================================================================
# CARGA DE DATOS (igual que antes)
# ============================================================================

def load_data(run_path: Path):
    """Carga todos los datos necesarios."""
    print(f"\nLoading data from: {run_path}")
    
    # Grafo
    grafo_path = run_path / 'grafo_con_fusiones.graphml'
    if not grafo_path.exists():
        grafo_path = run_path / 'grafo.graphml'
    G = nx.read_graphml(str(grafo_path))
    print(f"  Graph: {G.number_of_nodes()} nodes, {G.number_of_edges()} edges")
    
    # Embeddings - intentar múltiples estrategias de matching
    emb_path = run_path / 'embeddings_cache.npz'
    node_emb = {}
    pi_emb = None
    if emb_path.exists():
        import hashlib
        
        data = np.load(str(emb_path), allow_pickle=True)
        keys = list(data['keys'])
        embeddings = data['embeddings']
        
        print(f"  Embeddings file: {len(keys)} keys")
        
        # Crear mapeos del grafo
        node_ids = list(G.nodes())
        texto_to_node = {}
        hash_to_node = {}
        for node_id in node_ids:
            texto = G.nodes[node_id].get('texto', '')
            if texto:
                texto_to_node[texto] = node_id
                texto_to_node[texto.lower().strip()] = node_id
                texto_to_node[texto[:100]] = node_id
                # MD5 hash (como usa replay)
                h = hashlib.md5(texto.lower().encode('utf-8')).hexdigest()
                hash_to_node[h] = node_id
        
        # Intentar matching
        for i, key in enumerate(keys):
            key_str = str(key)
            
            if key_str.startswith('pi:'):
                pi_emb = embeddings[i]
                continue
            
            # Estrategia 1: key es node_id directo
            if key_str in G.nodes():
                node_emb[key_str] = embeddings[i]
                continue
            
            # Estrategia 2: key es hash MD5 (como en replay)
            if key_str in hash_to_node:
                node_emb[hash_to_node[key_str]] = embeddings[i]
                continue
            
            # Estrategia 3: key es texto
            if key_str in texto_to_node:
                node_emb[texto_to_node[key_str]] = embeddings[i]
                continue
            
            # Estrategia 4: key normalizado
            key_norm = key_str.lower().strip()
            if key_norm in texto_to_node:
                node_emb[texto_to_node[key_norm]] = embeddings[i]
                continue
            
            # Estrategia 5: key truncado
            if key_str[:100] in texto_to_node:
                node_emb[texto_to_node[key_str[:100]]] = embeddings[i]
        
        print(f"  Embeddings matched: {len(node_emb)} nodes")
    
    # HUBs
    hubs_data = None
    hubs_path = run_path / 'hubs.json'
    if hubs_path.exists():
        with open(hubs_path, 'r', encoding='utf-8') as f:
            hubs_data = json.load(f)
        print(f"  HUBs: {len(hubs_data.get('hubs', []))}")
    
    # Eventos
    eventos = []
    eventos_path = run_path / 'eventos.json'
    if eventos_path.exists():
        with open(eventos_path, 'r', encoding='utf-8') as f:
            eventos_data = json.load(f)
        eventos = eventos_data.get('eventos', [])
        print(f"  Eventos: {len(eventos)}")
    
    # Pi text
    pi_text = None
    for p in [run_path / 'input.json', run_path.parent / 'input.json', 
              run_path.parent.parent / 'input.json']:
        if p.exists():
            with open(p, 'r', encoding='utf-8') as f:
                input_data = json.load(f)
            pi_text = input_data.get('problema', {}).get('decision_central', '')
            break
    
    return G, node_emb, pi_emb, hubs_data, eventos, pi_text


def project_to_3d(node_emb, G, pi_embedding=None):
    """Proyecta embeddings a 3D, centrado en origen. 
    Si no hay embeddings, usa posiciones basadas en actor/nivel."""
    
    all_node_ids = list(G.nodes())
    max_nivel = max(int(G.nodes[n].get('nivel', 0)) for n in G.nodes()) or 1
    
    # Si no hay embeddings suficientes, usar fallback basado en actor/nivel
    if len(node_emb) < 10:
        print(f"  ⚠ Few embeddings ({len(node_emb)}), using actor-based positions")
        
        actors = sorted(set(G.nodes[n].get('actor', 'Unknown') for n in G.nodes()))
        actor_angle = {a: (i / len(actors)) * 2 * np.pi for i, a in enumerate(actors)}
        
        node_pos = {}
        for node_id in all_node_ids:
            actor = G.nodes[node_id].get('actor', 'Unknown')
            nivel = int(G.nodes[node_id].get('nivel', 0))
            
            angle = actor_angle.get(actor, 0)
            # Radio decrece con nivel (convergencia)
            radius = 2.0 - (nivel / max_nivel) * 1.2
            
            x = np.cos(angle) * radius + (np.random.random() - 0.5) * 0.3
            y = nivel / max_nivel * 10
            z = np.sin(angle) * radius + (np.random.random() - 0.5) * 0.3
            
            node_pos[node_id] = (x, y, z)
        
        pi_pos = (0.0, -0.5, 0.0)
        return node_pos, pi_pos, max_nivel
    
    # Con embeddings: usar PCA
    roots = [n for n, d in G.nodes(data=True) if d.get('nivel', 0) == 0 or d.get('nivel', '0') == '0']
    
    if pi_embedding is not None:
        origin = pi_embedding
    else:
        root_embs = np.array([node_emb[r] for r in roots if r in node_emb])
        origin = root_embs.mean(axis=0) if len(root_embs) > 0 else np.zeros(len(list(node_emb.values())[0]))
    
    all_nodes = list(node_emb.keys())
    all_embs = np.array([node_emb[n] for n in all_nodes])
    all_embs_centered = all_embs - origin
    
    pca = PCA(n_components=2)
    coords_2d = pca.fit_transform(all_embs_centered)
    
    node_pos = {}
    for i, node_id in enumerate(all_nodes):
        nivel = int(G.nodes[node_id].get('nivel', 0))
        y = nivel / max_nivel * 10
        node_pos[node_id] = (coords_2d[i, 0], y, coords_2d[i, 1])
    
    # Agregar nodos sin embedding usando interpolación por actor
    actors = sorted(set(G.nodes[n].get('actor', 'Unknown') for n in G.nodes()))
    actor_centers = {}
    for actor in actors:
        actor_nodes = [n for n in node_pos if G.nodes[n].get('actor') == actor]
        if actor_nodes:
            cx = np.mean([node_pos[n][0] for n in actor_nodes])
            cz = np.mean([node_pos[n][2] for n in actor_nodes])
            actor_centers[actor] = (cx, cz)
    
    for node_id in all_node_ids:
        if node_id not in node_pos:
            actor = G.nodes[node_id].get('actor', 'Unknown')
            nivel = int(G.nodes[node_id].get('nivel', 0))
            y = nivel / max_nivel * 10
            
            if actor in actor_centers:
                cx, cz = actor_centers[actor]
                x = cx + (np.random.random() - 0.5) * 0.5
                z = cz + (np.random.random() - 0.5) * 0.5
            else:
                x = (np.random.random() - 0.5) * 2
                z = (np.random.random() - 0.5) * 2
            
            node_pos[node_id] = (x, y, z)
    
    # Centrar en origen
    all_x = [p[0] for p in node_pos.values()]
    all_z = [p[2] for p in node_pos.values()]
    cx, cz = np.mean(all_x), np.mean(all_z)
    
    for node_id in node_pos:
        x, y, z = node_pos[node_id]
        node_pos[node_id] = (x - cx, y, z - cz)
    
    pi_pos = (0.0, -0.5, 0.0)
    
    return node_pos, pi_pos, max_nivel


# ============================================================================
# GENERACIÓN DE SEGMENTOS
# ============================================================================

def create_presentation_html(G, node_emb, node_pos, pi_pos, pi_emb, hubs_data, 
                             eventos, output_path, frames_per_tree=50):
    """
    Genera HTML con animación por segmentos y pausas manuales.
    """
    actors = sorted(set(G.nodes[n].get('actor', 'Unknown') for n in G.nodes()))
    initialize_actors(actors)
    
    print(f"\n  Actors: {actors}")
    
    # Agrupar nodos por actor y ordenar por nivel
    actor_nodes = {actor: [] for actor in actors}
    for node_id in node_pos:
        actor = G.nodes[node_id].get('actor', 'Unknown')
        nivel = int(G.nodes[node_id].get('nivel', 0))
        if actor in actor_nodes:
            actor_nodes[actor].append((node_id, nivel))
    
    # Ordenar por nivel
    for actor in actors:
        actor_nodes[actor].sort(key=lambda x: x[1])
    
    # Agrupar aristas por actor
    actor_edges = {actor: [] for actor in actors}
    for src, tgt in G.edges():
        actor = G.nodes[src].get('actor', 'Unknown')
        if actor in actor_edges:
            actor_edges[actor].append((src, tgt))
    
    # Extraer fusiones del grafo
    fusion_pairs = defaultdict(list)  # (actor_a, actor_b) -> [(node_a_pos, node_b_pos)]
    for node_id, attrs in G.nodes(data=True):
        fusion_con = attrs.get('fusion_con', '')
        if fusion_con:
            actor_a = attrs.get('actor', '')
            # Encontrar el nodo con el que fusiona
            for other_id, other_attrs in G.nodes(data=True):
                if other_attrs.get('texto', '') == fusion_con:
                    actor_b = other_attrs.get('actor', '')
                    if actor_a and actor_b and actor_a != actor_b:
                        key = tuple(sorted([actor_a, actor_b]))
                        if node_id in node_pos and other_id in node_pos:
                            fusion_pairs[key].append((node_pos[node_id], node_pos[other_id]))
                    break
    
    # Si no hay fusiones en el grafo, buscar en eventos
    if not fusion_pairs and eventos:
        texto_to_pos = {}
        for node_id in node_pos:
            texto = G.nodes[node_id].get('texto', '')
            if texto:
                texto_to_pos[texto.lower().strip()[:100]] = node_pos[node_id]
        
        for ev in eventos:
            if ev.get('evento') == 'fusion':
                datos = ev.get('datos', {})
                actor_a = datos.get('actor_a', '')
                actor_b = datos.get('actor_b', '')
                node_a_text = datos.get('node_a', '').lower().strip()[:100]
                node_b_text = datos.get('node_b', '').lower().strip()[:100]
                
                pos_a = texto_to_pos.get(node_a_text)
                pos_b = texto_to_pos.get(node_b_text)
                
                if pos_a and pos_b and actor_a and actor_b:
                    key = tuple(sorted([actor_a, actor_b]))
                    fusion_pairs[key].append((pos_a, pos_b))
    
    print(f"  Fusion pairs: {[(get_short(k[0]) + '↔' + get_short(k[1]), len(v)) for k, v in fusion_pairs.items()]}")
    
    # HUB positions
    hub_positions = []
    if hubs_data:
        for hub in hubs_data.get('hubs', []):
            positions = []
            for node_id in hub.get('clase_nodos', []):
                if node_id in node_pos:
                    positions.append(node_pos[node_id])
            if positions:
                cx = np.mean([p[0] for p in positions])
                cy = np.mean([p[1] for p in positions])
                cz = np.mean([p[2] for p in positions])
                hub_positions.append({
                    'pos': (cx, cy, cz),
                    'ttf': hub.get('ftt_sum', 0),
                    'node_positions': positions[:15]  # Max 15 líneas por HUB
                })
    
    print(f"  HUBs: {len(hub_positions)}")
    
    # ========================================================================
    # GENERAR SEGMENTOS DE FRAMES
    # ========================================================================
    
    all_frames = []
    segment_info = []  # (start_frame, end_frame, title)
    
    frame_idx = 0
    
    # --- Segmento 0: Solo Π ---
    segment_info.append((frame_idx, frame_idx, "Π — Central Problem"))
    all_frames.append(create_frame(
        frame_idx, pi_pos, {}, {}, [], [], [],
        "Π — The Central Problem", actors
    ))
    frame_idx += 1
    
    # --- Segmentos 1-3: Cada árbol crece ---
    visible_nodes = {}
    visible_edges = {}
    
    for actor in actors:
        nodes = actor_nodes[actor]
        edges = actor_edges[actor]
        
        start_frame = frame_idx
        n_nodes = len(nodes)
        
        # Crear frames para el crecimiento de este árbol
        for i in range(frames_per_tree):
            progress = (i + 1) / frames_per_tree
            nodes_to_show = int(n_nodes * progress)
            
            # Agregar nodos de este actor
            for node_id, nivel in nodes[:nodes_to_show]:
                visible_nodes[node_id] = actor
            
            # Agregar aristas donde ambos nodos son visibles
            for src, tgt in edges:
                if src in visible_nodes and tgt in visible_nodes:
                    visible_edges[(src, tgt)] = actor
            
            all_frames.append(create_frame(
                frame_idx, pi_pos, visible_nodes, visible_edges, 
                [], [], [],
                f"Building {get_short(actor)} tree... ({nodes_to_show}/{n_nodes})",
                actors, node_pos, G, node_emb, pi_emb
            ))
            frame_idx += 1
        
        segment_info.append((start_frame, frame_idx - 1, f"{get_short(actor)} Tree Complete ({n_nodes} nodes)"))
    
    # --- Segmentos 4-6: Fusiones por par (ANIMADAS) ---
    visible_fusions = []
    
    for (actor_a, actor_b), fusions in sorted(fusion_pairs.items()):
        start_frame = frame_idx
        n_fusions = len(fusions)
        
        if n_fusions == 0:
            continue
        
        # Frames para animar las fusiones de este par
        frames_for_fusions = min(30, max(10, n_fusions // 2))  # 10-30 frames
        
        for i in range(frames_for_fusions):
            progress = (i + 1) / frames_for_fusions
            fusions_to_show = int(n_fusions * progress)
            
            # Agregar fusiones gradualmente
            for pos_a, pos_b in fusions[:fusions_to_show]:
                fusion_key = (pos_a, pos_b)
                if fusion_key not in [f['key'] for f in visible_fusions]:
                    visible_fusions.append({
                        'pos_a': pos_a,
                        'pos_b': pos_b,
                        'color': get_color(actor_a),
                        'actor_a': actor_a,
                        'actor_b': actor_b,
                        'key': fusion_key
                    })
            
            all_frames.append(create_frame(
                frame_idx, pi_pos, visible_nodes, visible_edges,
                visible_fusions, [], [],
                f"Fusions {get_short(actor_a)}↔{get_short(actor_b)}: {len([f for f in visible_fusions if f['color'] == get_color(actor_a)])}/{n_fusions}",
                actors, node_pos, G, node_emb, pi_emb
            ))
            frame_idx += 1
        
        segment_info.append((start_frame, frame_idx - 1, 
                            f"Fusions {get_short(actor_a)}↔{get_short(actor_b)}: {n_fusions}"))
    
    # --- Segmentos 7-9: HUBs ---
    visible_hubs = []
    
    for i, hub_data in enumerate(hub_positions):
        start_frame = frame_idx
        
        visible_hubs.append(hub_data)
        
        label = "★ OPTIMAL HUB" if i == 0 else f"HUB #{i+1}"
        
        all_frames.append(create_frame(
            frame_idx, pi_pos, visible_nodes, visible_edges,
            visible_fusions, visible_hubs, [],
            f"{label} — TTF: {hub_data['ttf']:.2f}",
            actors, node_pos, G, node_emb, pi_emb
        ))
        frame_idx += 1
        
        segment_info.append((start_frame, frame_idx - 1, f"{label}"))
    
    # --- Segmento final ---
    segment_info.append((frame_idx, frame_idx, "Analysis Complete"))
    all_frames.append(create_frame(
        frame_idx, pi_pos, visible_nodes, visible_edges,
        visible_fusions, visible_hubs, [],
        f"✓ Analysis Complete — {len(visible_nodes)} nodes, {len(visible_fusions)} fusions, {len(visible_hubs)} HUBs",
        actors, node_pos, G, node_emb, pi_emb
    ))
    
    print(f"\n  Total frames: {len(all_frames)}")
    print(f"  Segments: {len(segment_info)}")
    for i, (start, end, title) in enumerate(segment_info):
        print(f"    [{i}] {start}-{end}: {title}")
    
    # Generar HTML
    html = generate_presentation_html(all_frames, segment_info, actors)
    
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write(html)
    
    print(f"\n✓ Saved: {output_path}")


def create_frame(frame_idx, pi_pos, visible_nodes, visible_edges, 
                 visible_fusions, visible_hubs, hub_lines,
                 title, actors, node_pos=None, G=None, node_emb=None, pi_emb=None):
    """Crea un frame con los elementos visibles."""
    
    data = []
    
    # Π siempre visible
    data.append({
        'type': 'scatter3d',
        'x': [pi_pos[0]], 'y': [pi_pos[1]], 'z': [pi_pos[2]],
        'mode': 'markers+text',
        'marker': {'size': 14, 'color': COLOR_PI, 'symbol': 'diamond',
                   'line': {'color': 'white', 'width': 2}},
        'text': ['Π'],
        'textposition': 'top center',
        'textfont': {'size': 14, 'color': COLOR_PI},
        'name': 'Π (Problem)',
        'showlegend': True
    })
    
    if not node_pos:
        return {'data': data, 'title': title}
    
    # Aristas del árbol por actor
    for actor in actors:
        edge_x, edge_y, edge_z = [], [], []
        for (src, tgt), edge_actor in visible_edges.items():
            if edge_actor == actor:
                if src in node_pos and tgt in node_pos:
                    p1, p2 = node_pos[src], node_pos[tgt]
                    edge_x.extend([p1[0], p2[0], None])
                    edge_y.extend([p1[1], p2[1], None])
                    edge_z.extend([p1[2], p2[2], None])
        
        if edge_x:
            data.append({
                'type': 'scatter3d',
                'x': edge_x, 'y': edge_y, 'z': edge_z,
                'mode': 'lines',
                'line': {'color': get_color(actor), 'width': 1.5},
                'opacity': 0.4,
                'name': f'Tree {get_short(actor)}',
                'showlegend': False,
                'hoverinfo': 'skip'
            })
    
    # Nodos por actor
    for actor in actors:
        xs, ys, zs, sizes = [], [], [], []
        for node_id, node_actor in visible_nodes.items():
            if node_actor == actor and node_id in node_pos:
                pos = node_pos[node_id]
                xs.append(pos[0])
                ys.append(pos[1])
                zs.append(pos[2])
                
                if node_emb and pi_emb is not None and node_id in node_emb:
                    sim = cosine_sim(node_emb[node_id], pi_emb)
                else:
                    sim = 0.5
                sizes.append(4 + sim * 6)
        
        count = len(xs)
        data.append({
            'type': 'scatter3d',
            'x': xs if xs else [None], 
            'y': ys if ys else [None], 
            'z': zs if zs else [None],
            'mode': 'markers',
            'marker': {'size': sizes if sizes else [1], 'color': get_color(actor), 
                      'opacity': 0.85, 'line': {'color': 'white', 'width': 0.5}},
            'name': f'{get_short(actor)} ({count})',
            'showlegend': True
        })
    
    # Fusiones - agrupar por par de actores
    fusion_by_pair = {}
    for f in visible_fusions:
        actor_a = f.get('actor_a', '')
        actor_b = f.get('actor_b', '')
        pair_key = tuple(sorted([actor_a, actor_b]))
        if pair_key not in fusion_by_pair:
            fusion_by_pair[pair_key] = {'fusions': [], 'color': f.get('color', COLOR_FUSION)}
        fusion_by_pair[pair_key]['fusions'].append(f)
    
    for (actor_a, actor_b), pair_data in fusion_by_pair.items():
        fusions_group = pair_data['fusions']
        color = pair_data['color']
        
        fusion_x, fusion_y, fusion_z = [], [], []
        for f in fusions_group:
            p1, p2 = f['pos_a'], f['pos_b']
            mid = ((p1[0]+p2[0])/2, max(p1[1], p2[1]) + 0.8, (p1[2]+p2[2])/2)
            
            for i in range(9):
                s = i / 8
                x = (1-s)**2 * p1[0] + 2*(1-s)*s * mid[0] + s**2 * p2[0]
                y = (1-s)**2 * p1[1] + 2*(1-s)*s * mid[1] + s**2 * p2[1]
                z = (1-s)**2 * p1[2] + 2*(1-s)*s * mid[2] + s**2 * p2[2]
                fusion_x.append(x)
                fusion_y.append(y)
                fusion_z.append(z)
            fusion_x.append(None)
            fusion_y.append(None)
            fusion_z.append(None)
        
        if fusion_x:
            name_a = get_short(actor_a) if actor_a else '?'
            name_b = get_short(actor_b) if actor_b else '?'
            
            data.append({
                'type': 'scatter3d',
                'x': fusion_x, 'y': fusion_y, 'z': fusion_z,
                'mode': 'lines',
                'line': {'color': color, 'width': 2},
                'opacity': 0.35,
                'name': f'↔ {name_a}↔{name_b} ({len(fusions_group)})',
                'showlegend': True,
                'hoverinfo': 'skip'
            })
    
    # HUBs con líneas
    hub_line_x, hub_line_y, hub_line_z = [], [], []
    hub_x, hub_y, hub_z, hub_sizes, hub_texts = [], [], [], [], []
    
    for i, hub in enumerate(visible_hubs):
        pos = hub['pos']
        hub_x.append(pos[0])
        hub_y.append(pos[1])
        hub_z.append(pos[2])
        hub_sizes.append(24 if i == 0 else 18)
        hub_texts.append(f'{"★ " if i==0 else ""}HUB #{i+1}')
        
        # Líneas a nodos
        for node_p in hub.get('node_positions', []):
            hub_line_x.extend([pos[0], node_p[0], None])
            hub_line_y.extend([pos[1], node_p[1], None])
            hub_line_z.extend([pos[2], node_p[2], None])
    
    if hub_line_x:
        data.append({
            'type': 'scatter3d',
            'x': hub_line_x, 'y': hub_line_y, 'z': hub_line_z,
            'mode': 'lines',
            'line': {'color': COLOR_PI, 'width': 1.5},
            'opacity': 0.5,
            'name': 'HUB connections',
            'showlegend': False,
            'hoverinfo': 'skip'
        })
    
    if hub_x:
        data.append({
            'type': 'scatter3d',
            'x': hub_x, 'y': hub_y, 'z': hub_z,
            'mode': 'markers+text',
            'marker': {'size': hub_sizes, 'color': COLOR_HUB, 'symbol': 'diamond',
                      'line': {'color': COLOR_PI, 'width': 3}},
            'text': hub_texts,
            'textposition': 'top center',
            'textfont': {'size': 10, 'color': COLOR_PI},
            'name': f'HUBs ({len(hub_x)})',
            'showlegend': True
        })
    
    return {'data': data, 'title': title}


def generate_presentation_html(frames, segment_info, actors):
    """Genera HTML con controles de presentación."""
    
    # Convertir frames a JSON
    frames_json = json.dumps(frames)
    segments_json = json.dumps(segment_info)
    
    html = f'''<!DOCTYPE html>
<html>
<head>
    <meta charset="utf-8">
    <title>Peace Field — Presentation Mode</title>
    <script src="https://cdn.plot.ly/plotly-2.27.0.min.js"></script>
    <style>
        * {{ margin: 0; padding: 0; box-sizing: border-box; }}
        body {{ 
            background: #0a0a1a; 
            color: white; 
            font-family: 'Segoe UI', Arial, sans-serif;
            overflow: hidden;
        }}
        #container {{ width: 100vw; height: 100vh; position: relative; }}
        #plot {{ width: 100%; height: calc(100% - 80px); }}
        
        #controls {{
            position: fixed;
            bottom: 0;
            left: 0;
            right: 0;
            height: 80px;
            background: rgba(20, 20, 40, 0.95);
            border-top: 1px solid #333;
            display: flex;
            align-items: center;
            justify-content: center;
            gap: 20px;
            padding: 0 30px;
        }}
        
        .btn {{
            padding: 12px 30px;
            font-size: 16px;
            border: none;
            border-radius: 8px;
            cursor: pointer;
            transition: all 0.2s;
            font-weight: 600;
        }}
        
        .btn-primary {{
            background: linear-gradient(135deg, #FFD700, #FFA500);
            color: #000;
        }}
        .btn-primary:hover {{ transform: scale(1.05); }}
        .btn-primary:disabled {{ 
            background: #444; 
            color: #888;
            cursor: not-allowed;
            transform: none;
        }}
        
        .btn-secondary {{
            background: #333;
            color: white;
            border: 1px solid #555;
        }}
        .btn-secondary:hover {{ background: #444; }}
        
        #segment-info {{
            flex: 1;
            text-align: center;
        }}
        #segment-title {{
            font-size: 18px;
            font-weight: 600;
            color: #FFD700;
        }}
        #segment-progress {{
            font-size: 12px;
            color: #888;
            margin-top: 4px;
        }}
        
        #status {{
            position: fixed;
            top: 20px;
            left: 50%;
            transform: translateX(-50%);
            background: rgba(0,0,0,0.8);
            padding: 10px 25px;
            border-radius: 25px;
            font-size: 14px;
            border: 1px solid #FFD700;
            z-index: 100;
        }}
        
        .playing {{ color: #4CAF50; }}
        .paused {{ color: #FFD700; }}
        
        /* Instrucciones iniciales */
        #instructions {{
            position: fixed;
            top: 50%;
            left: 50%;
            transform: translate(-50%, -50%);
            background: rgba(0,0,0,0.9);
            padding: 40px;
            border-radius: 15px;
            border: 2px solid #FFD700;
            text-align: center;
            z-index: 200;
        }}
        #instructions h2 {{ color: #FFD700; margin-bottom: 20px; }}
        #instructions p {{ margin: 10px 0; color: #ccc; }}
        #instructions .key {{ 
            display: inline-block;
            background: #333;
            padding: 5px 15px;
            border-radius: 5px;
            margin: 5px;
            border: 1px solid #555;
        }}
    </style>
</head>
<body>
    <div id="container">
        <div id="status"><span class="paused">PAUSED</span> — Press SPACE or click NEXT</div>
        <div id="plot"></div>
        <div id="controls">
            <button class="btn btn-secondary" id="btn-prev">← Previous</button>
            <div id="segment-info">
                <div id="segment-title">Ready to Start</div>
                <div id="segment-progress">Segment 0 / 0</div>
            </div>
            <button class="btn btn-primary" id="btn-next">Next →</button>
        </div>
    </div>
    
    <div id="instructions">
        <h2>🎬 Peace Field — Presentation Mode</h2>
        <p>Control the animation with:</p>
        <p>
            <span class="key">SPACE</span> or <span class="key">→</span> Next segment
        </p>
        <p>
            <span class="key">←</span> Previous segment
        </p>
        <p>
            <span class="key">Mouse drag</span> Rotate view
        </p>
        <p style="margin-top: 20px;">
            <button class="btn btn-primary" onclick="startPresentation()">Start Presentation</button>
        </p>
    </div>
    
    <script>
        const frames = {frames_json};
        const segments = {segments_json};
        
        let currentSegment = -1;
        let currentFrame = 0;
        let isAnimating = false;
        let animationTimer = null;
        
        // Plotly layout
        const layout = {{
            scene: {{
                xaxis: {{ title: 'PCA 1', gridcolor: '#333', showbackground: true, backgroundcolor: '#0a0a1a' }},
                yaxis: {{ title: 'Level', gridcolor: '#333', showbackground: true, backgroundcolor: '#0a0a1a' }},
                zaxis: {{ title: 'PCA 2', gridcolor: '#333', showbackground: true, backgroundcolor: '#0a0a1a' }},
                bgcolor: '#0a0a1a',
                camera: {{ eye: {{ x: 1.5, y: 1.5, z: 1.5 }} }},
                aspectmode: 'data'
            }},
            paper_bgcolor: '#0a0a1a',
            plot_bgcolor: '#0a0a1a',
            font: {{ color: 'white' }},
            showlegend: true,
            legend: {{
                bgcolor: 'rgba(20,20,40,0.95)',
                bordercolor: 'rgba(255,215,0,0.3)',
                x: 1.02, y: 0.98
            }},
            margin: {{ l: 0, r: 150, t: 50, b: 0 }},
            title: {{
                text: '<b>Peace Field Replay</b>',
                x: 0.5,
                font: {{ size: 16 }}
            }}
        }};
        
        function startPresentation() {{
            document.getElementById('instructions').style.display = 'none';
            Plotly.newPlot('plot', frames[0].data, layout);
            updateUI();
            goToSegment(0);
        }}
        
        function updateUI() {{
            const seg = segments[currentSegment] || [0, 0, 'Ready'];
            document.getElementById('segment-title').textContent = seg[2];
            document.getElementById('segment-progress').textContent = 
                `Segment ${{currentSegment + 1}} / ${{segments.length}}`;
            
            document.getElementById('btn-prev').disabled = currentSegment <= 0;
            document.getElementById('btn-next').disabled = currentSegment >= segments.length - 1;
            
            const status = document.getElementById('status');
            if (isAnimating) {{
                status.innerHTML = '<span class="playing">▶ PLAYING</span>';
            }} else {{
                status.innerHTML = '<span class="paused">⏸ PAUSED</span> — Press SPACE or click NEXT';
            }}
        }}
        
        function goToSegment(segIdx) {{
            if (segIdx < 0 || segIdx >= segments.length) return;
            
            // Detener animación actual
            if (animationTimer) {{
                clearInterval(animationTimer);
                animationTimer = null;
            }}
            
            currentSegment = segIdx;
            const [startFrame, endFrame, title] = segments[segIdx];
            
            if (startFrame === endFrame) {{
                // Segmento de un solo frame
                currentFrame = startFrame;
                Plotly.react('plot', frames[currentFrame].data, {{
                    ...layout,
                    title: {{ text: '<b>' + frames[currentFrame].title + '</b>', x: 0.5, font: {{ size: 16 }} }}
                }});
                isAnimating = false;
                updateUI();
            }} else {{
                // Segmento animado
                currentFrame = startFrame;
                isAnimating = true;
                updateUI();
                
                animationTimer = setInterval(() => {{
                    if (currentFrame < endFrame) {{
                        currentFrame++;
                        Plotly.react('plot', frames[currentFrame].data, {{
                            ...layout,
                            title: {{ text: '<b>' + frames[currentFrame].title + '</b>', x: 0.5, font: {{ size: 16 }} }}
                        }});
                    }} else {{
                        // Fin del segmento
                        clearInterval(animationTimer);
                        animationTimer = null;
                        isAnimating = false;
                        updateUI();
                    }}
                }}, 80);  // 80ms por frame = ~12 fps
            }}
        }}
        
        function nextSegment() {{
            if (currentSegment < segments.length - 1) {{
                goToSegment(currentSegment + 1);
            }}
        }}
        
        function prevSegment() {{
            if (currentSegment > 0) {{
                goToSegment(currentSegment - 1);
            }}
        }}
        
        // Event listeners
        document.getElementById('btn-next').addEventListener('click', nextSegment);
        document.getElementById('btn-prev').addEventListener('click', prevSegment);
        
        document.addEventListener('keydown', (e) => {{
            if (e.code === 'Space' || e.code === 'ArrowRight') {{
                e.preventDefault();
                nextSegment();
            }} else if (e.code === 'ArrowLeft') {{
                e.preventDefault();
                prevSegment();
            }}
        }});
        
        // Iniciar si no hay instrucciones visibles
        if (window.location.hash === '#autostart') {{
            startPresentation();
        }}
    </script>
</body>
</html>'''
    
    return html


# ============================================================================
# MAIN
# ============================================================================

def main():
    parser = argparse.ArgumentParser(description='Peace Field — Presentation Mode')
    parser.add_argument('--run-path', type=str, required=True)
    parser.add_argument('--output', type=str, default=None)
    parser.add_argument('--frames-per-tree', type=int, default=50,
                       help='Frames para animar cada árbol (default: 50)')
    
    args = parser.parse_args()
    
    run_path = Path(args.run_path)
    if not run_path.exists():
        print(f"Error: {run_path} not found")
        return 1
    
    output_path = Path(args.output) if args.output else run_path / 'peace_map_presentation.html'
    
    print("\n" + "=" * 60)
    print("PEACE FIELD — PRESENTATION MODE")
    print("=" * 60)
    
    # Cargar datos
    G, node_emb, pi_emb, hubs_data, eventos, pi_text = load_data(run_path)
    
    # Proyectar a 3D
    print("\nProjecting to 3D...")
    node_pos, pi_pos, max_nivel = project_to_3d(node_emb, G, pi_emb)
    print(f"  Nodes with positions: {len(node_pos)}")
    
    # Crear presentación
    print("\nCreating presentation...")
    create_presentation_html(
        G, node_emb, node_pos, pi_pos, pi_emb, hubs_data, eventos,
        output_path, args.frames_per_tree
    )
    
    print("\n" + "=" * 60)
    print("READY FOR PRESENTATION")
    print("=" * 60)
    print(f"\nOpen: {output_path}")
    print("\nControls:")
    print("  SPACE / → : Next segment")
    print("  ←        : Previous segment")
    print("  Mouse    : Rotate view")
    
    return 0


if __name__ == '__main__':
    exit(main())