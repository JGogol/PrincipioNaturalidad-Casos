"""
Peace Map Replay — Principio de Naturalidad v9
==============================================
Animación temporal del proceso de construcción de árboles causales
y descubrimiento de HUBs.

Basado en peace_field_map.py, agrega:
  - Reproducción temporal frame-by-frame
  - Nodos apareciendo según eventos.json
  - Aristas del árbol creciendo
  - Líneas de fusión conectando actores
  - Zoom cinematográfico al HUB final

Uso:
    python peace_map_replay.py --run-path casos/gerd/runs/2026-02-05_15-29-23
    python peace_map_replay.py --run-path casos/gerd/runs/2026-02-05_15-29-23 --duration 180 --fps 30

Requiere en run-path:
    - eventos.json (OBLIGATORIO para animación)
    - grafo.graphml (o grafo_con_fusiones.graphml)
    - embeddings_cache.npz
    - hubs.json
    - input.json (para Π)

Autor: Javier Gogol Merletti / Claude
Febrero 2026
"""

import argparse
import hashlib
import json
from pathlib import Path
from datetime import datetime
from collections import defaultdict

import networkx as nx
import numpy as np
from sklearn.decomposition import PCA
import plotly.graph_objects as go


# ============================================================================
# CONFIGURACIÓN
# ============================================================================

COLOR_ROTATION = [
    '#E63946',  # Rojo (Egypt)
    '#2A9D8F',  # Verde-azul (Ethiopia)
    '#E9C46A',  # Amarillo (Sudan)
    '#9B5DE5',  # Púrpura
    '#00BBF9',  # Cyan
    '#F15BB5',  # Rosa
    '#00F5D4',  # Turquesa
]

COLOR_CONVERGE = '#00FF88'
COLOR_DIVERGE = '#FF4444'
COLOR_PI = '#FFD700'
COLOR_HUB = '#FFFFFF'
COLOR_FUSION = '#9B5DE5'  # Púrpura para fusiones

ACTOR_COLORS = {}
ACTOR_SHORT = {}


def initialize_actors(actors):
    """Inicializa colores y nombres cortos para actores detectados."""
    global ACTOR_COLORS, ACTOR_SHORT
    
    common_words = ['Republic', 'Democratic', 'Federal', 'of', 'the', 'The',
                    'Kingdom', 'State', 'United', 'People', 'Arab', 'Islamic']
    
    for i, actor in enumerate(sorted(actors)):
        ACTOR_COLORS[actor] = COLOR_ROTATION[i % len(COLOR_ROTATION)]
        words = [w for w in actor.split() if w not in common_words]
        ACTOR_SHORT[actor] = words[-1] if words else actor[:10]
        ACTOR_COLORS[ACTOR_SHORT[actor]] = ACTOR_COLORS[actor]
    
    print(f"  Actors: {list(ACTOR_SHORT.values())}")


def get_short(actor):
    return ACTOR_SHORT.get(actor, actor.split()[-1] if actor else 'Unknown')


def get_color(actor):
    return ACTOR_COLORS.get(actor, ACTOR_COLORS.get(get_short(actor), '#888888'))


# ============================================================================
# CARGA DE DATOS
# ============================================================================

def load_data(run_path: Path, input_path: Path = None):
    """Carga todos los datos necesarios."""
    print(f"\nLoading from: {run_path}")
    
    # Grafo
    grafo_path = run_path / 'grafo_con_fusiones.graphml'
    if not grafo_path.exists():
        grafo_path = run_path / 'grafo.graphml'
    
    G = nx.read_graphml(str(grafo_path))
    print(f"  Graph: {G.number_of_nodes()} nodes, {G.number_of_edges()} edges")
    
    # Embeddings
    emb_data = np.load(str(run_path / 'embeddings_cache.npz'), allow_pickle=True)
    keys = list(emb_data['keys'])
    embeddings = emb_data['embeddings']
    emb_dim = embeddings.shape[1]
    print(f"  Embeddings: {len(keys)} vectors × {emb_dim}d")
    
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
    else:
        print(f"  ⚠ eventos.json no encontrado - animación no disponible")
    
    # Π (problema central)
    pi_text = None
    pi_embedding = None
    
    if input_path and input_path.exists():
        possible_paths = [input_path]
    else:
        possible_paths = [
            run_path / 'input.json',
            run_path.parent / 'input.json',
            run_path.parent.parent / 'input.json',
            run_path.parent.parent.parent / 'input.json',
        ]
    
    for p in possible_paths:
        if p.exists():
            with open(p, 'r', encoding='utf-8') as f:
                input_data = json.load(f)
            pi_text = input_data.get('problema', {}).get('decision_central_embedding') or \
                      input_data.get('problema', {}).get('decision_central')
            if pi_text:
                print(f"  Π: {pi_text[:50]}...")
                break
    
    # Buscar embedding de Π
    if pi_text:
        pi_hash_lower = hashlib.md5(pi_text.lower().encode('utf-8')).hexdigest()
        pi_hash_original = hashlib.md5(pi_text.encode('utf-8')).hexdigest()
        keys_str = [str(k) for k in keys]
        
        if pi_hash_lower in keys_str:
            idx = keys_str.index(pi_hash_lower)
            pi_embedding = embeddings[idx]
        elif pi_hash_original in keys_str:
            idx = keys_str.index(pi_hash_original)
            pi_embedding = embeddings[idx]
    
    return G, keys, embeddings, emb_dim, hubs_data, pi_embedding, eventos


def map_embeddings(G, keys, embeddings):
    """Mapea embeddings a nodos del grafo con múltiples estrategias."""
    keys_str = [str(k) for k in keys]
    
    # Crear mapeos del grafo
    node_emb = {}
    
    # Mapeo texto -> node_id
    texto_to_node = {}
    for node_id, attrs in G.nodes(data=True):
        texto = attrs.get('texto', '')
        if texto:
            texto_to_node[texto] = node_id
            texto_to_node[texto.lower()] = node_id
            texto_to_node[texto.lower().strip()] = node_id
            # También hash MD5
            h = hashlib.md5(texto.lower().encode('utf-8')).hexdigest()
            texto_to_node[h] = node_id
    
    # Intentar matchear cada key
    for i, key in enumerate(keys_str):
        if key.startswith('pi:'):
            continue
        
        # Estrategia 1: key es node_id directo
        if key in G.nodes():
            node_emb[key] = embeddings[i]
            continue
        
        # Estrategia 2: key es texto o hash
        if key in texto_to_node:
            node_emb[texto_to_node[key]] = embeddings[i]
            continue
        
        # Estrategia 3: key lowercase
        key_lower = key.lower().strip()
        if key_lower in texto_to_node:
            node_emb[texto_to_node[key_lower]] = embeddings[i]
            continue
        
        # Estrategia 4: key es hash MD5
        h = hashlib.md5(key.lower().encode('utf-8')).hexdigest()
        if h in texto_to_node:
            node_emb[texto_to_node[h]] = embeddings[i]
    
    print(f"  Mapped: {len(node_emb)}/{G.number_of_nodes()} nodes")
    return node_emb


# ============================================================================
# GEOMETRÍA
# ============================================================================

def cosine_sim(a, b):
    """Similitud coseno entre dos vectores."""
    dot = np.dot(a, b)
    na, nb = np.linalg.norm(a), np.linalg.norm(b)
    if na == 0 or nb == 0:
        return 0.0
    return float(np.clip(dot / (na * nb), -1, 1))


def project_to_3d(node_emb, G, pi_embedding=None):
    """Proyecta embeddings a 3D: X,Z = PCA semántico, Y = nivel topológico.
    CENTRA los datos en el origen para que la cámara los vea correctamente.
    Si no hay embeddings, usa fallback basado en actor/nivel."""
    
    max_nivel = max(int(G.nodes[n].get('nivel', 0)) for n in G.nodes()) or 1
    
    # Si no hay embeddings suficientes, usar fallback
    if len(node_emb) < 10:
        print(f"  ⚠ Few embeddings ({len(node_emb)}), using actor-based positions")
        
        actors = sorted(set(G.nodes[n].get('actor', 'Unknown') for n in G.nodes()))
        actor_angle = {a: (i / len(actors)) * 2 * np.pi for i, a in enumerate(actors)}
        
        node_pos = {}
        for node_id in G.nodes():
            actor = G.nodes[node_id].get('actor', 'Unknown')
            nivel = int(G.nodes[node_id].get('nivel', 0))
            
            angle = actor_angle.get(actor, 0)
            radius = 2.0 - (nivel / max_nivel) * 1.2
            
            x = np.cos(angle) * radius + (np.random.random() - 0.5) * 0.3
            y = nivel / max_nivel * 10
            z = np.sin(angle) * radius + (np.random.random() - 0.5) * 0.3
            
            node_pos[node_id] = (x, y, z)
        
        pi_pos = (0.0, -0.5, 0.0)
        return node_pos, pi_pos, None, None, max_nivel, 0.0
    
    # Con embeddings: usar PCA
    roots = [n for n, d in G.nodes(data=True) if d.get('nivel', 0) == 0 or d.get('nivel', '0') == '0']
    
    if pi_embedding is not None:
        origin = pi_embedding
    else:
        root_embs = np.array([node_emb[r] for r in roots if r in node_emb])
        if len(root_embs) > 0:
            origin = root_embs.mean(axis=0)
        else:
            # Usar el primer embedding como origen
            origin = list(node_emb.values())[0]
    
    all_nodes = list(node_emb.keys())
    all_embs = np.array([node_emb[n] for n in all_nodes])
    all_embs_centered = all_embs - origin
    
    pca = PCA(n_components=2)
    coords_2d = pca.fit_transform(all_embs_centered)
    variance = sum(pca.explained_variance_ratio_)
    print(f"  PCA variance: {variance:.1%}")
    
    node_pos = {}
    for i, node_id in enumerate(all_nodes):
        nivel = int(G.nodes[node_id].get('nivel', 0))
        y = nivel / max(max_nivel, 1) * 10
        node_pos[node_id] = (coords_2d[i, 0], y, coords_2d[i, 1])
    
    # Agregar nodos sin embedding usando posición del actor
    actors = sorted(set(G.nodes[n].get('actor', 'Unknown') for n in G.nodes()))
    actor_centers = {}
    for actor in actors:
        actor_nodes = [n for n in node_pos if G.nodes[n].get('actor') == actor]
        if actor_nodes:
            cx = np.mean([node_pos[n][0] for n in actor_nodes])
            cz = np.mean([node_pos[n][2] for n in actor_nodes])
            actor_centers[actor] = (cx, cz)
    
    for node_id in G.nodes():
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
    
    # CENTRAR en el origen: restar el centroide XZ
    all_x = [p[0] for p in node_pos.values()]
    all_z = [p[2] for p in node_pos.values()]
    centroid_x = np.mean(all_x)
    centroid_z = np.mean(all_z)
    
    for node_id in node_pos:
        x, y, z = node_pos[node_id]
        node_pos[node_id] = (x - centroid_x, y, z - centroid_z)
    
    print(f"  Data centered at origin (shifted by {centroid_x:.2f}, {centroid_z:.2f})")
    
    # Pi también centrado
    pi_pos = (0.0, -0.5, 0.0)
    if pi_embedding is not None:
        pi_centered = pi_embedding - origin
        pi_2d = pca.transform([pi_centered])[0]
        pi_pos = (pi_2d[0] - centroid_x, -0.5, pi_2d[1] - centroid_z)
    
    return node_pos, pi_pos, pca, origin, max_nivel, variance


# ============================================================================
# PROCESAMIENTO DE EVENTOS
# ============================================================================

def parse_timestamp(ts_str):
    """Parsea timestamp ISO."""
    try:
        return datetime.fromisoformat(ts_str)
    except:
        return None


def process_eventos(eventos, G, node_pos, hubs_data=None):
    """
    Procesa eventos y genera estructura para animación.
    
    TIMING: 
    - Fase 1 (árbol): 0% - 60% del tiempo de animación
    - Fase 2 (fusiones): 60% - 90%
    - Fase 3+ (HUBs y final): 90% - 100%
    """
    if not eventos:
        return {}, [], [], []
    
    # Calcular rango temporal
    timestamps = [parse_timestamp(ev.get('timestamp', '')) for ev in eventos]
    timestamps = [t for t in timestamps if t]
    
    if not timestamps:
        return {}, [], [], []
    
    t_min, t_max = min(timestamps), max(timestamps)
    t_range = (t_max - t_min).total_seconds()
    
    print(f"  Time range: {t_range/3600:.2f} hours")
    
    # Crear mapeo texto -> node_id del grafo (normalizado)
    texto_to_node = {}
    for node_id, attrs in G.nodes(data=True):
        texto = attrs.get('texto', '')
        if texto:
            texto_norm = texto.lower().strip()
            texto_to_node[texto_norm] = node_id
            texto_to_node[texto_norm[:100]] = node_id
            texto_to_node[texto_norm[:80]] = node_id
            texto_to_node[texto] = node_id
            texto_to_node[texto[:100]] = node_id
    
    print(f"  Text mappings: {len(texto_to_node)}")
    
    # Primera pasada: identificar rangos de tiempo por fase
    fase_1_end = None  # Fin de construcción de árboles
    fase_2_start = None  # Inicio de fusiones
    fase_2_end = None  # Fin de fusiones
    
    for ev in eventos:
        ts = parse_timestamp(ev.get('timestamp', ''))
        if not ts:
            continue
        tipo = ev.get('evento', '')
        datos = ev.get('datos', {})
        
        if tipo == 'fase_inicio':
            fase_num = str(datos.get('numero', ''))
            if fase_num == '2' or 'fusion' in datos.get('fase', '').lower():
                fase_2_start = ts
            elif fase_num in ['3', '4'] or 'hub' in datos.get('fase', '').lower():
                fase_2_end = ts
        
        if tipo == 'nodo_agregado' and fase_1_end is None:
            fase_1_end = ts  # Se actualiza hasta el último nodo
        if tipo == 'nodo_agregado':
            fase_1_end = ts
    
    # Calcular tiempos de transición
    if fase_2_start:
        t_tree_end = (fase_2_start - t_min).total_seconds() / t_range
    else:
        t_tree_end = 0.5  # Default: 50%
    
    if fase_2_end:
        t_fusion_end = (fase_2_end - t_min).total_seconds() / t_range
    else:
        t_fusion_end = 0.9  # Default: 90%
    
    print(f"  Tree phase ends at: {t_tree_end*100:.0f}% real time")
    print(f"  Fusion phase ends at: {t_fusion_end*100:.0f}% real time")
    
    # Función para remapear tiempo real a tiempo de animación
    # Árbol: 0-t_tree_end real → 0-0.60 animación (más lento)
    # Fusiones: t_tree_end-t_fusion_end real → 0.60-0.90 animación
    # Final: t_fusion_end-1.0 real → 0.90-1.0 animación
    def remap_time(t_real):
        if t_real <= t_tree_end:
            # Fase árbol: expandir a 60% del tiempo de animación
            return (t_real / t_tree_end) * 0.60 if t_tree_end > 0 else 0
        elif t_real <= t_fusion_end:
            # Fase fusiones: 30% del tiempo
            progress = (t_real - t_tree_end) / (t_fusion_end - t_tree_end) if (t_fusion_end - t_tree_end) > 0 else 0
            return 0.60 + progress * 0.30
        else:
            # Fase final: 10% del tiempo
            progress = (t_real - t_fusion_end) / (1.0 - t_fusion_end) if (1.0 - t_fusion_end) > 0 else 0
            return 0.90 + progress * 0.10
    
    # Procesar eventos
    node_to_time = {}
    fusion_lines = []
    hub_times = []
    phase_times = []
    
    actor_nodes = defaultdict(list)
    matched = 0
    unmatched = 0
    
    for ev in eventos:
        ts = parse_timestamp(ev.get('timestamp', ''))
        if not ts:
            continue
        
        t_real = (ts - t_min).total_seconds() / t_range if t_range > 0 else 0
        t_anim = remap_time(t_real)  # Tiempo remapeado para animación
        
        tipo = ev.get('evento', '')
        datos = ev.get('datos', {})
        
        if tipo == 'fase_inicio':
            phase_times.append({
                't': t_anim,
                'numero': datos.get('numero', '?'),
                'fase': datos.get('fase', '')
            })
        
        elif tipo in ['arbol_inicio', 'nodo_agregado']:
            texto = datos.get('raiz', '') if tipo == 'arbol_inicio' else datos.get('nodo', '')
            actor = datos.get('actor', '')
            nivel = datos.get('nivel', 0)
            
            texto_norm = texto.lower().strip()
            node_id = (texto_to_node.get(texto_norm) or 
                      texto_to_node.get(texto_norm[:100]) or
                      texto_to_node.get(texto_norm[:80]) or
                      texto_to_node.get(texto) or
                      texto_to_node.get(texto[:100]))
            
            if node_id and node_id in node_pos:
                node_to_time[node_id] = t_anim
                actor_nodes[actor].append((node_id, nivel, t_anim))
                matched += 1
            else:
                unmatched += 1
        
        elif tipo == 'fusion':
            actor_a = datos.get('actor_a', '')
            actor_b = datos.get('actor_b', '')
            node_a_text = datos.get('node_a', '')
            node_b_text = datos.get('node_b', '')
            
            node_a_norm = node_a_text.lower().strip()
            node_b_norm = node_b_text.lower().strip()
            
            node_a_id = (texto_to_node.get(node_a_norm) or 
                        texto_to_node.get(node_a_norm[:100]) or
                        texto_to_node.get(node_a_text))
            node_b_id = (texto_to_node.get(node_b_norm) or 
                        texto_to_node.get(node_b_norm[:100]) or
                        texto_to_node.get(node_b_text))
            
            if node_a_id and node_b_id and node_a_id in node_pos and node_b_id in node_pos:
                fusion_lines.append({
                    't': t_anim,
                    'actor_a': actor_a,
                    'actor_b': actor_b,
                    'color_a': get_color(actor_a),
                    'color_b': get_color(actor_b),
                    'node_a': node_a_id,
                    'node_b': node_b_id,
                    'pos_a': node_pos[node_a_id],
                    'pos_b': node_pos[node_b_id],
                    'sbert': datos.get('sbert_similarity', 0),
                    'nli': datos.get('nli_score', 0)
                })
            else:
                nodes_a = actor_nodes.get(actor_a, [])
                nodes_b = actor_nodes.get(actor_b, [])
                
                if nodes_a and nodes_b:
                    best_a = max(nodes_a, key=lambda x: x[1])
                    best_b = max(nodes_b, key=lambda x: x[1])
                    
                    if best_a[0] in node_pos and best_b[0] in node_pos:
                        fusion_lines.append({
                            't': t_anim,
                            'actor_a': actor_a,
                            'actor_b': actor_b,
                            'color_a': get_color(actor_a),
                            'color_b': get_color(actor_b),
                            'node_a': best_a[0],
                            'node_b': best_b[0],
                            'pos_a': node_pos[best_a[0]],
                            'pos_b': node_pos[best_b[0]],
                            'sbert': datos.get('sbert_similarity', 0),
                            'nli': datos.get('nli_score', 0)
                        })
        
        elif tipo == 'hub_encontrado':
            hub_times.append({
                't': t_anim,
                'clase_id': datos.get('clase_id', ''),
                'ftt_sum': datos.get('ftt_sum', 0),
                'num_nodos': datos.get('num_nodos', 0)
            })
    
    print(f"  Nodes matched: {matched}, unmatched: {unmatched}")
    print(f"  Nodes timed: {len(node_to_time)}")
    print(f"  Fusions: {len(fusion_lines)}")
    print(f"  HUBs: {len(hub_times)}")
    print(f"  Phases: {len(phase_times)}")
    
    if len(node_to_time) < 10:
        print(f"  ⚠ Few matches, using level-based fallback...")
        max_nivel = max(int(G.nodes[n].get('nivel', 0)) for n in G.nodes())
        for node_id in node_pos:
            nivel = int(G.nodes[node_id].get('nivel', 0))
            node_to_time[node_id] = (nivel / max(max_nivel, 1)) * 0.60
        print(f"  Nodes timed (fallback): {len(node_to_time)}")
    
    return node_to_time, fusion_lines, hub_times, phase_times


# ============================================================================
# GENERACIÓN DE FRAMES
# ============================================================================

def generate_frames(G, node_emb, node_pos, pi_pos, pi_emb, hubs_data,
                    node_to_time, fusion_lines, hub_times, phase_times,
                    max_nivel, num_frames=200):
    """
    Genera frames para la animación de Plotly.
    - Cámara automática que sigue el centro de actividad
    - Aristas del árbol coloreadas por actor
    - Líneas conectando nodos a HUBs
    - Fusiones sutiles
    """
    actors = sorted(set(G.nodes[n].get('actor', 'Unknown') for n in G.nodes()))
    
    # Calcular centroides de HUBs y sus nodos
    hub_positions = []
    hub_node_ids = []  # Lista de nodos por cada HUB
    if hubs_data:
        for hub in hubs_data.get('hubs', []):
            positions = []
            node_ids = []
            for node_id in hub.get('clase_nodos', []):
                if node_id in node_pos:
                    positions.append(node_pos[node_id])
                    node_ids.append(node_id)
            if positions:
                cx = np.mean([p[0] for p in positions])
                cy = np.mean([p[1] for p in positions])
                cz = np.mean([p[2] for p in positions])
                hub_positions.append({
                    'pos': (cx, cy, cz),
                    'ttf': hub.get('ftt_sum', 0),
                    'actores': hub.get('actores', []),
                    'node_positions': positions
                })
                hub_node_ids.append(node_ids)
    
    frames = []
    slider_steps = []
    
    # Agrupar nodos por actor
    actor_node_list = {actor: [] for actor in actors}
    for node_id in node_pos:
        actor = G.nodes[node_id].get('actor', 'Unknown')
        if actor in actor_node_list:
            actor_node_list[actor].append(node_id)
    
    # Agrupar aristas por actor
    actor_edges = {actor: [] for actor in actors}
    for src, tgt in G.edges():
        actor_src = G.nodes[src].get('actor', 'Unknown')
        actor_tgt = G.nodes[tgt].get('actor', 'Unknown')
        if actor_src == actor_tgt and actor_src in actor_edges:
            actor_edges[actor_src].append((src, tgt))
    
    print(f"  Nodes per actor: {[(get_short(a), len(ns)) for a, ns in actor_node_list.items()]}")
    print(f"  Edges per actor: {[(get_short(a), len(es)) for a, es in actor_edges.items()]}")
    
    # Función para calcular centro de actividad (solo para referencia, no para cámara)
    def get_activity_center(visible_nodes, t, window=0.1):
        """Calcula el centro de los nodos visibles."""
        nodes_list = [nid for nid in visible_nodes if nid in node_pos]
        if not nodes_list:
            return (0, 3, 0)
        positions = [node_pos[nid] for nid in nodes_list]
        cx = np.mean([p[0] for p in positions])
        cy = np.mean([p[1] for p in positions])
        cz = np.mean([p[2] for p in positions])
        return (cx, cy, cz)
    
    for frame_idx in range(num_frames + 1):
        t = frame_idx / num_frames
        
        # Nodos visibles hasta este momento
        visible_nodes = {nid for nid, nt in node_to_time.items() if nt <= t}
        
        # Fusiones visibles
        visible_fusions = [f for f in fusion_lines if f['t'] <= t]
        
        # HUBs visibles
        visible_hubs = [h for h in hub_times if h['t'] <= t]
        
        # Fase actual
        current_phase = {'numero': 0, 'fase': 'Initializing...'}
        for ph in phase_times:
            if ph['t'] <= t:
                current_phase = ph
        
        phase_num_str = str(current_phase.get('numero', '0'))
        # Extraer solo dígitos del número de fase (ej: '2b' -> 2)
        phase_num = int(''.join(c for c in phase_num_str if c.isdigit()) or '0')
        
        frame_data = []
        
        # --- Π (siempre visible) ---
        frame_data.append(go.Scatter3d(
            x=[pi_pos[0]], y=[pi_pos[1]], z=[pi_pos[2]],
            mode='markers+text',
            marker=dict(size=14, color=COLOR_PI, symbol='diamond',
                       line=dict(color='white', width=2)),
            text=['Π'],
            textposition='top center',
            textfont=dict(size=14, color=COLOR_PI),
            name='Π (Problem)',
            showlegend=True,
            legendgroup='reference',
            hovertemplate='<b>Π — Problem Core</b><extra></extra>'
        ))
        
        # --- Aristas del árbol POR ACTOR (coloreadas) ---
        for actor in actors:
            color = get_color(actor)
            short = get_short(actor)
            
            tree_x, tree_y, tree_z = [], [], []
            for src, tgt in actor_edges[actor]:
                if src in visible_nodes and tgt in visible_nodes:
                    if src in node_pos and tgt in node_pos:
                        p1, p2 = node_pos[src], node_pos[tgt]
                        tree_x.extend([p1[0], p2[0], None])
                        tree_y.extend([p1[1], p2[1], None])
                        tree_z.extend([p1[2], p2[2], None])
            
            frame_data.append(go.Scatter3d(
                x=tree_x if tree_x else [None],
                y=tree_y if tree_y else [None],
                z=tree_z if tree_z else [None],
                mode='lines',
                line=dict(color=color, width=1.5),
                opacity=0.4,
                name=f'Tree {short}',
                showlegend=False,
                legendgroup=f'actor_{short}',
                hoverinfo='skip'
            ))
        
        # --- Nodos por actor ---
        for actor in actors:
            actor_visible = [n for n in actor_node_list[actor] if n in visible_nodes]
            color = get_color(actor)
            short = get_short(actor)
            
            if actor_visible:
                xs, ys, zs, sizes, hovers = [], [], [], [], []
                
                for node_id in actor_visible:
                    pos = node_pos[node_id]
                    xs.append(pos[0])
                    ys.append(pos[1])
                    zs.append(pos[2])
                    
                    if node_id in node_emb and pi_emb is not None:
                        sim_pi = cosine_sim(node_emb[node_id], pi_emb)
                    else:
                        sim_pi = 0.5
                    sizes.append(4 + sim_pi * 6)
                    
                    nivel = G.nodes[node_id].get('nivel', '?')
                    texto = G.nodes[node_id].get('texto', '')[:50]
                    hovers.append(f'<b>{short}</b> L{nivel}<br>{texto}...')
                
                frame_data.append(go.Scatter3d(
                    x=xs, y=ys, z=zs,
                    mode='markers',
                    marker=dict(size=sizes, color=color, opacity=0.85,
                               line=dict(color='white', width=0.5)),
                    name=f'{short} ({len(actor_visible)})',
                    showlegend=True,
                    legendgroup=f'actor_{short}',
                    hovertemplate='%{hovertext}<extra></extra>',
                    hovertext=hovers
                ))
            else:
                frame_data.append(go.Scatter3d(
                    x=[None], y=[None], z=[None],
                    mode='markers',
                    marker=dict(size=1, color=color, opacity=0),
                    name=f'{short} (0)',
                    showlegend=True,
                    legendgroup=f'actor_{short}',
                    hoverinfo='skip'
                ))
        
        # --- Líneas de fusión (SUTILES) ---
        fusion_pairs = defaultdict(list)
        for f in visible_fusions:
            key = tuple(sorted([get_short(f['actor_a']), get_short(f['actor_b'])]))
            fusion_pairs[key].append(f)
        
        actor_pairs = []
        for i, a1 in enumerate(actors):
            for a2 in actors[i+1:]:
                actor_pairs.append((get_short(a1), get_short(a2)))
        
        for pair in actor_pairs:
            pair_fusions = fusion_pairs.get(pair, [])
            
            if pair_fusions:
                fusion_x, fusion_y, fusion_z = [], [], []
                color_a = pair_fusions[0].get('color_a', '#888')
                
                for f in pair_fusions:
                    p1, p2 = f['pos_a'], f['pos_b']
                    mid = ((p1[0]+p2[0])/2, max(p1[1], p2[1]) + 0.8, (p1[2]+p2[2])/2)
                    
                    # Curva bezier más suave
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
                
                frame_data.append(go.Scatter3d(
                    x=fusion_x, y=fusion_y, z=fusion_z,
                    mode='lines',
                    line=dict(color=color_a, width=2),  # Más fino
                    opacity=0.35,  # Más sutil
                    name=f"⟷ {pair[0]}↔{pair[1]} ({len(pair_fusions)})",
                    showlegend=True,
                    legendgroup=f'fusion_{pair[0]}_{pair[1]}',
                    hoverinfo='skip'
                ))
            else:
                frame_data.append(go.Scatter3d(
                    x=[None], y=[None], z=[None],
                    mode='lines',
                    line=dict(color='#666', width=1),
                    opacity=0,
                    name=f"⟷ {pair[0]}↔{pair[1]} (0)",
                    showlegend=True,
                    legendgroup=f'fusion_{pair[0]}_{pair[1]}',
                    hoverinfo='skip'
                ))
        
        # --- Líneas hacia HUBs (conectando nodos al centro del HUB) ---
        hub_lines_x, hub_lines_y, hub_lines_z = [], [], []
        for i, hub_pos_data in enumerate(hub_positions):
            is_visible = len(visible_hubs) > i or t > 0.95
            if is_visible:
                hub_center = hub_pos_data['pos']
                # Conectar algunos nodos representativos al centro
                for node_p in hub_pos_data['node_positions'][:20]:  # Max 20 líneas por HUB
                    hub_lines_x.extend([hub_center[0], node_p[0], None])
                    hub_lines_y.extend([hub_center[1], node_p[1], None])
                    hub_lines_z.extend([hub_center[2], node_p[2], None])
        
        frame_data.append(go.Scatter3d(
            x=hub_lines_x if hub_lines_x else [None],
            y=hub_lines_y if hub_lines_y else [None],
            z=hub_lines_z if hub_lines_z else [None],
            mode='lines',
            line=dict(color=COLOR_PI, width=1.5),
            opacity=0.5,
            name='HUB connections',
            showlegend=False,
            hoverinfo='skip'
        ))
        
        # --- HUBs (diamantes) ---
        hub_x, hub_y, hub_z, hub_sizes, hub_texts = [], [], [], [], []
        for i, hub_pos_data in enumerate(hub_positions):
            is_visible = len(visible_hubs) > i or t > 0.95
            if is_visible:
                pos = hub_pos_data['pos']
                hub_x.append(pos[0])
                hub_y.append(pos[1])
                hub_z.append(pos[2])
                hub_sizes.append(24 if i == 0 else 18)
                hub_texts.append(f'{"★ OPTIMAL " if i==0 else ""}HUB #{i+1}<br>TTF: {hub_pos_data["ttf"]:.2f}')
        
        frame_data.append(go.Scatter3d(
            x=hub_x if hub_x else [None],
            y=hub_y if hub_y else [None],
            z=hub_z if hub_z else [None],
            mode='markers',
            marker=dict(
                size=hub_sizes if hub_sizes else [1],
                color=COLOR_HUB,
                symbol='diamond',
                line=dict(color=COLOR_PI, width=3)
            ),
            name=f'HUBs ({len(hub_x)})',
            showlegend=True,
            legendgroup='hubs',
            hovertemplate='<b>%{text}</b><extra></extra>',
            text=hub_texts if hub_texts else ['']
        ))
        
        # Crear frame SIN cámara - control 100% manual
        phase_text = f"Phase {current_phase['numero']}: {current_phase['fase'][:40]}..."
        
        frames.append(go.Frame(
            data=frame_data,
            name=str(frame_idx),
            layout=go.Layout(
                title=dict(
                    text=f'<b>Peace Field Replay</b> — {phase_text}<br>'
                         f'<sup>Nodes: {len(visible_nodes)} | Fusions: {len(visible_fusions)} | '
                         f'HUBs: {len(visible_hubs)} | Progress: {t*100:.0f}%</sup>',
                )
            )
        ))
        
        if frame_idx % max(1, num_frames // 25) == 0:
            slider_steps.append({
                'args': [[str(frame_idx)], {'frame': {'duration': 50, 'redraw': True}, 'mode': 'immediate'}],
                'label': f'{int(t*100)}%',
                'method': 'animate'
            })
    
    return frames, slider_steps


# ============================================================================
# CREACIÓN DE FIGURA
# ============================================================================

def create_animated_figure(G, node_emb, node_pos, pi_pos, pi_emb, hubs_data,
                           node_to_time, fusion_lines, hub_times, phase_times,
                           max_nivel, variance, num_frames=200):
    """Crea la figura animada de Plotly con cámara automática."""
    
    actors = sorted(set(G.nodes[n].get('actor', 'Unknown') for n in G.nodes()))
    initialize_actors(actors)
    
    print("\nGenerating animation frames...")
    frames, slider_steps = generate_frames(
        G, node_emb, node_pos, pi_pos, pi_emb, hubs_data,
        node_to_time, fusion_lines, hub_times, phase_times,
        max_nivel, num_frames
    )
    
    if not frames:
        print("  ⚠ No frames generated!")
        return go.Figure()
    
    print(f"  Generated {len(frames)} frames")
    print(f"  Traces per frame: {len(frames[0].data)}")
    
    # Figura inicial usa el primer frame
    fig = go.Figure(
        data=frames[0].data,
        frames=frames
    )
    
    # Layout - cámara fija simple mirando al origen (datos están centrados ahí)
    fig.update_layout(
        title=dict(
            text='<b>Peace Field Replay</b> — Click ▶ Play | Drag to rotate | Scroll to zoom<br>'
                 f'<sup>{len(node_to_time)} nodes | {len(fusion_lines)} fusions</sup>',
            x=0.5,
            font=dict(size=16, color='white')
        ),
        scene=dict(
            xaxis_title='PCA 1',
            yaxis_title='Level',
            zaxis_title='PCA 2',
            bgcolor='#0a0a1a',
            xaxis=dict(gridcolor='#333', showbackground=True, backgroundcolor='#0a0a1a'),
            yaxis=dict(gridcolor='#333', showbackground=True, backgroundcolor='#0a0a1a'),
            zaxis=dict(gridcolor='#333', showbackground=True, backgroundcolor='#0a0a1a'),
            camera=dict(
                eye=dict(x=1.5, y=1.5, z=1.5)
            ),
            aspectmode='data',
        ),
        paper_bgcolor='#0a0a1a',
        plot_bgcolor='#0a0a1a',
        font=dict(color='white'),
        
        # LEYENDA
        showlegend=True,
        legend=dict(
            bgcolor='rgba(20,20,40,0.95)',
            bordercolor='rgba(255,215,0,0.3)',
            borderwidth=1,
            x=1.02,
            y=0.98,
            xanchor='left',
            yanchor='top',
            font=dict(size=11, color='white'),
            itemsizing='constant',
            tracegroupgap=5,
        ),
        
        margin=dict(l=0, r=200, t=80, b=100),
        height=850,
        
        # Controles de animación
        updatemenus=[
            dict(
                type='buttons',
                showactive=True,
                y=-0.05,
                x=0.08,
                xanchor='left',
                yanchor='top',
                pad=dict(t=0, r=10),
                bgcolor='rgba(40,40,60,0.9)',
                font=dict(color='white'),
                buttons=[
                    dict(
                        label='▶ Play',
                        method='animate',
                        args=[None, {
                            'frame': {'duration': 180, 'redraw': True},
                            'fromcurrent': True,
                            'transition': {'duration': 100}
                        }]
                    ),
                    dict(
                        label='⏸ Pause',
                        method='animate',
                        args=[[None], {
                            'frame': {'duration': 0, 'redraw': False},
                            'mode': 'immediate',
                            'transition': {'duration': 0}
                        }]
                    ),
                    dict(
                        label='⏮ Reset',
                        method='animate',
                        args=[['0'], {
                            'frame': {'duration': 0, 'redraw': True},
                            'mode': 'immediate',
                            'transition': {'duration': 0}
                        }]
                    )
                ]
            ),
            # Velocidad
            dict(
                type='buttons',
                direction='right',
                showactive=True,
                y=-0.05,
                x=0.35,
                xanchor='left',
                yanchor='top',
                pad=dict(t=0),
                bgcolor='rgba(40,40,60,0.9)',
                font=dict(color='#FFD700', size=10),
                buttons=[
                    dict(label='0.5x', method='animate',
                         args=[None, {'frame': {'duration': 360, 'redraw': True}, 'fromcurrent': True, 'transition': {'duration': 200}}]),
                    dict(label='1x', method='animate',
                         args=[None, {'frame': {'duration': 180, 'redraw': True}, 'fromcurrent': True, 'transition': {'duration': 100}}]),
                    dict(label='2x', method='animate',
                         args=[None, {'frame': {'duration': 90, 'redraw': True}, 'fromcurrent': True, 'transition': {'duration': 50}}]),
                    dict(label='4x', method='animate',
                         args=[None, {'frame': {'duration': 45, 'redraw': True}, 'fromcurrent': True, 'transition': {'duration': 25}}]),
                ],
            ),
        ],
        
        # Slider de tiempo
        sliders=[{
            'active': 0,
            'yanchor': 'top',
            'xanchor': 'left',
            'currentvalue': {
                'font': {'size': 14, 'color': '#FFD700'},
                'prefix': 'Progress: ',
                'visible': True,
                'xanchor': 'center'
            },
            'transition': {'duration': 100},
            'pad': {'b': 10, 't': 40},
            'len': 0.85,
            'x': 0.08,
            'y': 0,
            'steps': slider_steps,
            'bgcolor': '#333',
            'activebgcolor': '#FFD700',
            'bordercolor': '#555',
            'font': {'color': 'white', 'size': 10}
        }]
    )
    
    return fig


# ============================================================================
# MAIN
# ============================================================================

def main():
    parser = argparse.ArgumentParser(
        description='Peace Map Replay — Animación temporal del Principio de Naturalidad'
    )
    parser.add_argument('--run-path', type=str, required=True,
                        help='Path al directorio del run')
    parser.add_argument('--input-path', type=str, default=None,
                        help='Path a input.json (opcional)')
    parser.add_argument('--output', type=str, default='peace_map_replay.html',
                        help='Nombre del archivo de salida')
    parser.add_argument('--frames', type=int, default=200,
                        help='Número de frames de animación (default: 200)')
    
    args = parser.parse_args()
    run_path = Path(args.run_path)
    input_path = Path(args.input_path) if args.input_path else None
    
    print("\n" + "=" * 60)
    print("PEACE MAP REPLAY — Principio de Naturalidad")
    print("=" * 60)
    
    # Cargar datos
    G, keys, embeddings, emb_dim, hubs_data, pi_emb, eventos = load_data(run_path, input_path)
    
    if not eventos:
        print("\n⚠ ERROR: Se requiere eventos.json para la animación")
        return 1
    
    # Mapear embeddings
    node_emb = map_embeddings(G, keys, embeddings)
    
    # Proyectar a 3D
    print("\nProjecting to 3D...")
    node_pos, pi_pos, pca, origin, max_nivel, variance = project_to_3d(node_emb, G, pi_emb)
    
    # Procesar eventos
    print("\nProcessing eventos...")
    node_to_time, fusion_lines, hub_times, phase_times = process_eventos(
        eventos, G, node_pos, hubs_data
    )
    
    # Crear figura animada
    print("\nCreating animated figure...")
    fig = create_animated_figure(
        G, node_emb, node_pos, pi_pos, pi_emb, hubs_data,
        node_to_time, fusion_lines, hub_times, phase_times,
        max_nivel, variance, args.frames
    )
    
    # Guardar
    output_path = run_path / args.output
    fig.write_html(str(output_path), include_plotlyjs='cdn')
    print(f"\n✓ Saved: {output_path}")
    
    # Stats
    print(f"\nStats:")
    print(f"  Nodes: {G.number_of_nodes()}")
    print(f"  Fusions: {len(fusion_lines)}")
    print(f"  Frames: {args.frames}")
    print(f"  HUBs: {len(hub_times)}")
    
    print("\n" + "=" * 60)
    print("Press Play to watch the causal trees grow!")
    print("=" * 60 + "\n")
    
    return 0


if __name__ == '__main__':
    exit(main())