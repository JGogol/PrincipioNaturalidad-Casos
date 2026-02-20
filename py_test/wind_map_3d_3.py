"""
Mapa de Vientos 3D v3b - Principio de Naturalidad v8.5
Layout Híbrido: Topología + Semántica

CAUSA del problema en v2/v3:
  PCA 3D de embeddings ignora la estructura del grafo.
  Nodos semánticamente similares colapsan juntos aunque estén
  en ramas distintas del árbol. Se pierde la topología padre→hijo.

SOLUCIÓN (v3b):
  Eje Y = nivel en el árbol (0=raíz, 1, 2, 3...) → estructura jerárquica REAL
  Ejes X, Z = PCA 2D de embeddings centrado en Π → posición semántica REAL
  
  Resultado: los árboles se ven como árboles (ramifican por nivel),
  la convergencia al HUB se ve como embudo natural (los actores se
  acercan en X,Z conforme bajan de nivel).

Vectores: cada flecha nace de su nodo padre real.
Ángulos: θ_Π y θ_root calculados en este espacio híbrido.

Requisitos:
    pip install networkx numpy scikit-learn plotly

Uso:
    python wind_map_3d_v3b.py --run-path casos/gerd/runs/2026-02-04_11-13-40

Autor: Javier Gogol Merletti
"""

import argparse
import hashlib
import json
from pathlib import Path

import networkx as nx
import numpy as np
from sklearn.decomposition import PCA
import plotly.graph_objects as go


# ============================================================================
# CONFIGURACIÓN
# ============================================================================

COLOR_PALETTE = [
    '#E63946',  # Rojo
    '#2A9D8F',  # Verde-azul
    '#E9C46A',  # Amarillo
    '#9B5DE5',  # Púrpura
    '#00BBF9',  # Cyan
    '#F15BB5',  # Rosa
    '#00F5D4',  # Turquesa
    '#FEE440',  # Amarillo brillante
    '#FF6B6B',  # Coral
    '#4ECDC4',  # Teal
]


def get_actor_colors(G):
    actors = sorted(set(G.nodes[n]['actor'] for n in G.nodes()))
    return {actor: COLOR_PALETTE[i % len(COLOR_PALETTE)] for i, actor in enumerate(actors)}


def get_actor_short_names(actors):
    short = {}
    for actor in actors:
        words = actor.replace('Republic of', '').replace('Federal Democratic', '').strip().split()
        short[actor] = words[-1] if words else actor[:10]
    return short


# ============================================================================
# CARGA DE DATOS
# ============================================================================

def load_data(run_path: Path, input_path: Path = None):
    print(f"Loading data from: {run_path}")
    
    G = nx.read_graphml(str(run_path / 'grafo.graphml'))
    print(f"  Graph: {G.number_of_nodes()} nodes, {G.number_of_edges()} edges")
    
    emb_data = np.load(str(run_path / 'embeddings_cache.npz'), allow_pickle=True)
    keys = list(emb_data['keys'])
    embeddings = emb_data['embeddings']
    print(f"  Embeddings: {len(keys)} vectors")
    
    with open(run_path / 'clases_equivalencia.json', 'r', encoding='utf-8') as f:
        clases = json.load(f)
    print(f"  Equivalence classes: {clases['total_classes']}")
    
    hubs_data = None
    hubs_path = run_path / 'hubs.json'
    if hubs_path.exists():
        with open(hubs_path, 'r', encoding='utf-8') as f:
            hubs_data = json.load(f)
        print(f"  HUBs: {len(hubs_data.get('hubs', []))}")
    
    input_data = None
    if input_path and input_path.exists():
        with open(input_path, 'r', encoding='utf-8') as f:
            input_data = json.load(f)
        print(f"  Input: loaded (Π available)")
    
    return G, keys, embeddings, clases, hubs_data, input_data


def get_pi_embedding(input_data, keys, embeddings):
    if not input_data:
        return None, None
    
    pi_text = input_data['problema'].get('decision_central_embedding') or \
              input_data['problema'].get('decision_central')
    
    if not pi_text:
        return None, None
    
    pi_hash = hashlib.md5(pi_text.lower().encode('utf-8')).hexdigest()
    
    if pi_hash in keys:
        idx = keys.index(pi_hash)
        print(f"  Π found in cache (hash: {pi_hash[:12]}...)")
        return embeddings[idx], pi_text
    else:
        print(f"  Warning: Π not in cache")
        return None, pi_text


def map_embeddings_to_nodes(G, keys, embeddings):
    hash_to_emb = {k: embeddings[i] for i, k in enumerate(keys)}
    
    node_embeddings = {}
    for node_id, attrs in G.nodes(data=True):
        text = attrs['texto'].lower()
        h = hashlib.md5(text.encode('utf-8')).hexdigest()
        if h in hash_to_emb:
            node_embeddings[node_id] = hash_to_emb[h]
    
    print(f"  Mapped: {len(node_embeddings)} / {G.number_of_nodes()} nodes")
    return node_embeddings


# ============================================================================
# GEOMETRÍA
# ============================================================================

def cosine_sim(a, b):
    dot = np.dot(a, b)
    na, nb = np.linalg.norm(a), np.linalg.norm(b)
    if na == 0 or nb == 0:
        return 0.0
    return float(np.clip(dot / (na * nb), -1, 1))


def angle_between_vectors_3d(v1, v2):
    n1, n2 = np.linalg.norm(v1), np.linalg.norm(v2)
    if n1 == 0 or n2 == 0:
        return 0.0
    cos_a = np.clip(np.dot(v1, v2) / (n1 * n2), -1, 1)
    return float(np.degrees(np.arccos(cos_a)))


def compute_edge_geometry(parent_pos, child_pos, root_pos, pi_pos):
    parent_pos = np.array(parent_pos)
    child_pos = np.array(child_pos)
    root_pos = np.array(root_pos)
    pi_pos = np.array(pi_pos)
    
    vec_edge = child_pos - parent_pos
    vec_to_pi = pi_pos - parent_pos
    vec_to_root = root_pos - parent_pos
    
    theta_pi = angle_between_vectors_3d(vec_edge, vec_to_pi)
    theta_root = angle_between_vectors_3d(vec_edge, vec_to_root)
    
    dist_parent_pi = float(np.linalg.norm(parent_pos - pi_pos))
    dist_child_pi = float(np.linalg.norm(child_pos - pi_pos))
    
    acercamiento = dist_child_pi < dist_parent_pi
    
    return {
        'vec_edge': vec_edge,
        'theta_pi': theta_pi,
        'theta_root': theta_root,
        'dist_parent_pi': dist_parent_pi,
        'dist_child_pi': dist_child_pi,
        'acercamiento': acercamiento,
        'delta_dist': dist_parent_pi - dist_child_pi,
    }


# ============================================================================
# PROYECCIÓN HÍBRIDA: Topología + Semántica
# ============================================================================

def project_hybrid_3d(node_embeddings, G, pi_embedding=None):
    """
    Layout híbrido:
      - Eje Y = nivel en el árbol (invertido: raíz arriba, hojas abajo)
                Esto respeta la TOPOLOGÍA real del grafo.
      - Ejes X, Z = PCA 2D de embeddings centrado en Π
                Esto respeta la SEMÁNTICA real.
    
    El resultado muestra los árboles como árboles (ramificaciones visibles),
    y la convergencia semántica como acercamiento en el plano X,Z.
    """
    
    roots = [n for n, d in G.nodes(data=True) if d['nivel'] == 0]
    
    # --- Eje Y: nivel del árbol ---
    max_nivel = max(int(G.nodes[n].get('nivel', 0)) for n in G.nodes())
    print(f"  Max tree depth: {max_nivel}")
    
    # --- Ejes X, Z: PCA 2D de embeddings ---
    if pi_embedding is not None:
        origin = pi_embedding
        print(f"  Semantic origin: Π (real embedding)")
    else:
        root_embs = np.array([node_embeddings[r] for r in roots if r in node_embeddings])
        origin = root_embs.mean(axis=0)
        print(f"  Semantic origin: centroid of roots")
    
    all_nodes = list(node_embeddings.keys())
    all_embs = np.array([node_embeddings[n] for n in all_nodes])
    all_embs_centered = all_embs - origin
    
    # PCA 2D para el plano semántico
    matrix_with_origin = np.vstack([np.zeros_like(origin), all_embs_centered])
    
    pca = PCA(n_components=2)
    coords_2d = pca.fit_transform(matrix_with_origin)
    
    origin_2d = coords_2d[0]  # ~[0, 0]
    node_coords_2d = coords_2d[1:]
    
    print(f"  PCA 2D variance: {pca.explained_variance_ratio_.sum():.2%}")
    print(f"  Semantic origin in 2D: [{origin_2d[0]:.4f}, {origin_2d[1]:.4f}]")
    
    # --- Componer 3D: X=PC1, Y=nivel, Z=PC2 ---
    # Y invertido: raíz en la parte superior (Y alto), hojas abajo
    # Escalar Y para que sea comparable a X,Z
    
    # Rango de X,Z
    x_range = node_coords_2d[:, 0].max() - node_coords_2d[:, 0].min()
    z_range = node_coords_2d[:, 1].max() - node_coords_2d[:, 1].min()
    semantic_range = max(x_range, z_range, 0.001)
    
    # Escalar nivel para que ocupe un rango similar
    y_scale = semantic_range / max(max_nivel, 1)
    
    node_to_pos = {}
    for i, n in enumerate(all_nodes):
        nivel = int(G.nodes[n].get('nivel', 0))
        x = node_coords_2d[i, 0]
        y = (max_nivel - nivel) * y_scale  # Raíz arriba, hojas abajo
        z = node_coords_2d[i, 1]
        node_to_pos[n] = np.array([x, y, z])
    
    # Π en el plano semántico, a nivel "debajo de todo" (base del embudo)
    pi_y = -1.0 * y_scale  # Debajo de la máxima profundidad
    origin_3d = np.array([origin_2d[0], pi_y, origin_2d[1]])
    
    print(f"  Π position: [{origin_3d[0]:.4f}, {origin_3d[1]:.4f}, {origin_3d[2]:.4f}]")
    print(f"  Y scale factor: {y_scale:.4f}")
    
    # Reportar posición de raíces
    for r in roots:
        if r in node_to_pos:
            actor = G.nodes[r]['actor']
            pos = node_to_pos[r]
            print(f"  Root {actor}: [{pos[0]:.3f}, {pos[1]:.3f}, {pos[2]:.3f}]")
    
    return node_to_pos, roots, origin_3d, max_nivel, y_scale


# ============================================================================
# CONSTRUCCIÓN DEL ÁRBOL
# ============================================================================

def build_parent_map(G):
    parent_map = {}
    for src, tgt in G.edges():
        parent_map[tgt] = src
    return parent_map


def get_root_of_node(node_id, parent_map, G):
    current = node_id
    visited = set()
    while current in parent_map and current not in visited:
        visited.add(current)
        current = parent_map[current]
    return current


# ============================================================================
# VISUALIZACIÓN v3b
# ============================================================================

def create_wind_map(G, node_to_pos, roots, clases, hubs_data, origin_3d,
                    pi_text=None, node_embeddings=None, max_nivel=0, y_scale=1.0):
    
    ACTOR_COLORS = get_actor_colors(G)
    ACTOR_SHORT = get_actor_short_names(ACTOR_COLORS.keys())
    print(f"  Actors: {list(ACTOR_SHORT.values())}")
    
    parent_map = build_parent_map(G)
    
    root_of_actor = {}
    for r in roots:
        if r in node_to_pos:
            actor = G.nodes[r]['actor']
            root_of_actor[actor] = r
    
    # Identificar HUBs
    hub_classes = {}
    if hubs_data and 'hubs' in hubs_data:
        for i, hub in enumerate(hubs_data['hubs']):
            class_id = hub.get('class_id') or hub.get('clase_id')
            members = hub.get('clase_nodos', [])
            hub_classes[class_id] = {
                'rank': i,
                'ftt_sum': hub['ftt_sum'],
                'ftt_max': hub['ftt_max'],
                'members': set(members),
                'is_optimal': (i == 0),
                'texto_por_actor': hub.get('texto_por_actor', {}),
                'distancias': hub.get('distancias', {}),
                'caminos': hub.get('caminos', {}),
            }
    
    for class_id, hc in hub_classes.items():
        if not hc['members'] and class_id in clases.get('classes', {}):
            hc['members'] = set(clases['classes'][class_id]['members'])
    
    all_hub_nodes = set()
    for hc in hub_classes.values():
        all_hub_nodes.update(hc['members'])
    
    pi_pos = np.array(origin_3d)
    
    fig = go.Figure()
    
    # =========================================================================
    # 1. ARISTAS DE FONDO: topología real del grafo
    #    Cada línea conecta padre→hijo REAL
    # =========================================================================
    for actor, color in ACTOR_COLORS.items():
        edge_x, edge_y, edge_z = [], [], []
        
        for src, tgt in G.edges():
            if G.nodes[src]['actor'] != actor:
                continue
            if src not in node_to_pos or tgt not in node_to_pos:
                continue
            
            p1, p2 = node_to_pos[src], node_to_pos[tgt]
            edge_x.extend([p1[0], p2[0], None])
            edge_y.extend([p1[1], p2[1], None])
            edge_z.extend([p1[2], p2[2], None])
        
        if edge_x:
            fig.add_trace(go.Scatter3d(
                x=edge_x, y=edge_y, z=edge_z,
                mode='lines',
                line=dict(color=color, width=0.8),
                opacity=0.15,
                showlegend=False,
                hoverinfo='skip'
            ))
    
    # =========================================================================
    # 2. FLECHAS DE VIENTO en cada arista
    #    Dirección = padre→hijo (dato real)
    #    Opacidad = acercamiento a Π
    # =========================================================================
    stats = {'acerca': 0, 'aleja': 0, 'total': 0}
    
    for actor, color in ACTOR_COLORS.items():
        root_id = root_of_actor.get(actor)
        if not root_id or root_id not in node_to_pos:
            continue
        
        root_pos = node_to_pos[root_id]
        
        # Separar por acercamiento
        cones = {True: {'x': [], 'y': [], 'z': [], 'u': [], 'v': [], 'w': [], 'hover': []},
                 False: {'x': [], 'y': [], 'z': [], 'u': [], 'v': [], 'w': [], 'hover': []}}
        
        for src, tgt in G.edges():
            if G.nodes[src]['actor'] != actor:
                continue
            if src not in node_to_pos or tgt not in node_to_pos:
                continue
            
            p1 = node_to_pos[src]
            p2 = node_to_pos[tgt]
            
            geo = compute_edge_geometry(p1, p2, root_pos, pi_pos)
            nivel_src = int(G.nodes[src].get('nivel', 0))
            nivel_tgt = int(G.nodes[tgt].get('nivel', 0))
            
            sim_cos = None
            if node_embeddings and src in node_embeddings and tgt in node_embeddings:
                sim_cos = cosine_sim(node_embeddings[src], node_embeddings[tgt])
            
            stats['total'] += 1
            if geo['acercamiento']:
                stats['acerca'] += 1
            else:
                stats['aleja'] += 1
            
            # Flecha al 60% de la arista, dirección real
            vec = geo['vec_edge']
            norm_v = np.linalg.norm(vec)
            if norm_v == 0:
                continue
            
            vec_unit = vec / norm_v
            arrow_pos = np.array(p1) + vec * 0.6
            arrow_len = norm_v * 0.25
            
            sim_str = f"simCos={sim_cos:.3f}" if sim_cos is not None else "simCos=N/A"
            hover = (
                f"<b>{ACTOR_SHORT.get(actor, actor)}</b> L{nivel_src}→L{nivel_tgt}<br>"
                f"{sim_str}<br>"
                f"θ_Π={geo['theta_pi']:.1f}° | θ_root={geo['theta_root']:.1f}°<br>"
                f"Δdist_Π={geo['delta_dist']:+.3f} ({'↓acerca' if geo['acercamiento'] else '↑aleja'})<br>"
                f"d→Π: {geo['dist_parent_pi']:.3f}→{geo['dist_child_pi']:.3f}"
            )
            
            bucket = cones[geo['acercamiento']]
            bucket['x'].append(arrow_pos[0])
            bucket['y'].append(arrow_pos[1])
            bucket['z'].append(arrow_pos[2])
            bucket['u'].append(vec_unit[0] * arrow_len)
            bucket['v'].append(vec_unit[1] * arrow_len)
            bucket['w'].append(vec_unit[2] * arrow_len)
            bucket['hover'].append(hover)
        
        for acerca in [True, False]:
            bucket = cones[acerca]
            if not bucket['x']:
                continue
            
            fig.add_trace(go.Cone(
                x=bucket['x'], y=bucket['y'], z=bucket['z'],
                u=bucket['u'], v=bucket['v'], w=bucket['w'],
                colorscale=[[0, color], [1, color]],
                sizemode='absolute',
                sizeref=0.025,
                showscale=False,
                opacity=0.55 if acerca else 0.18,
                showlegend=False,
                hovertext=bucket['hover'],
                hoverinfo='text',
                anchor='tail',
            ))
    
    print(f"  Vectors: {stats['total']} total, {stats['acerca']} approach Π, {stats['aleja']} recede")
    
    # =========================================================================
    # 3. CAMINOS DE CONVERGENCIA → HUB
    #    Grosor decreciente (embudo): ancho en raíz, fino en HUB
    # =========================================================================
    if hubs_data and 'hubs' in hubs_data:
        for hub_idx, hub in enumerate(hubs_data['hubs']):
            caminos = hub.get('caminos', {})
            is_optimal = (hub_idx == 0)
            
            for actor, path in caminos.items():
                if len(path) < 2:
                    continue
                
                color = ACTOR_COLORS.get(actor, 'gray')
                root_id = root_of_actor.get(actor)
                root_pos = node_to_pos.get(root_id, origin_3d) if root_id else origin_3d
                
                total_steps = len(path) - 1
                
                for j in range(total_steps):
                    src_id = path[j]['id']
                    tgt_id = path[j + 1]['id']
                    
                    if src_id not in node_to_pos or tgt_id not in node_to_pos:
                        continue
                    
                    p1 = node_to_pos[src_id]
                    p2 = node_to_pos[tgt_id]
                    
                    geo = compute_edge_geometry(p1, p2, root_pos, pi_pos)
                    
                    sim, sim_str = None, "N/A"
                    if node_embeddings and src_id in node_embeddings and tgt_id in node_embeddings:
                        sim = cosine_sim(node_embeddings[src_id], node_embeddings[tgt_id])
                        sim_str = f"{sim:.3f}"
                    
                    progress = j / total_steps
                    if is_optimal:
                        width = 9 - progress * 6  # 9 → 3
                        opacity = 0.85
                        dash = None
                    else:
                        width = 5 - progress * 3  # 5 → 2
                        opacity = 0.45
                        dash = 'dash'
                    
                    src_text = path[j].get('texto', '')[:50]
                    tgt_text = path[j + 1].get('texto', '')[:50]
                    hover = (
                        f"<b>{'★ ' if is_optimal else ''}Path {ACTOR_SHORT.get(actor, actor)} → HUB</b><br>"
                        f"L{j}→L{j+1} | simCos={sim_str}<br>"
                        f"θ_Π={geo['theta_pi']:.1f}° | θ_root={geo['theta_root']:.1f}°<br>"
                        f"Δdist_Π={geo['delta_dist']:+.3f} ({'↓acerca' if geo['acercamiento'] else '↑aleja'})<br>"
                        f"<i>{src_text}...</i><br>→ <i>{tgt_text}...</i>"
                    )
                    
                    fig.add_trace(go.Scatter3d(
                        x=[p1[0], p2[0]], y=[p1[1], p2[1]], z=[p1[2], p2[2]],
                        mode='lines',
                        line=dict(color=color, width=width, dash=dash),
                        opacity=opacity,
                        showlegend=(j == 0 and hub_idx < 2),
                        name=f"{'★ ' if is_optimal else ''}Path: {ACTOR_SHORT.get(actor, actor)} → HUB" if j == 0 else None,
                        text=[hover, hover],
                        hoverinfo='text'
                    ))
                    
                    # Flechas de dirección en cada arista del camino principal
                    direction = np.array(p2) - np.array(p1)
                    norm_d = np.linalg.norm(direction)
                    if norm_d > 0 and is_optimal:
                        direction_unit = direction / norm_d
                        arrow_pos = np.array(p1) + direction * 0.7
                        arrow_scale = norm_d * 0.3
                        
                        fig.add_trace(go.Cone(
                            x=[arrow_pos[0]], y=[arrow_pos[1]], z=[arrow_pos[2]],
                            u=[direction_unit[0] * arrow_scale],
                            v=[direction_unit[1] * arrow_scale],
                            w=[direction_unit[2] * arrow_scale],
                            colorscale=[[0, color], [1, color]],
                            sizemode='absolute',
                            sizeref=0.04,
                            showscale=False,
                            opacity=0.85,
                            showlegend=False,
                            hoverinfo='skip',
                            anchor='tail',
                        ))
    
    # =========================================================================
    # 4. NODOS NORMALES
    # =========================================================================
    for actor, color in ACTOR_COLORS.items():
        normal = [n for n in G.nodes()
                  if G.nodes[n]['actor'] == actor
                  and n in node_to_pos
                  and n not in all_hub_nodes]
        
        if not normal:
            continue
        
        x = [node_to_pos[n][0] for n in normal]
        y = [node_to_pos[n][1] for n in normal]
        z = [node_to_pos[n][2] for n in normal]
        
        texts = []
        for n in normal:
            pos = node_to_pos[n]
            d_pi = float(np.linalg.norm(np.array(pos) - pi_pos))
            nivel = G.nodes[n].get('nivel', '?')
            texto = G.nodes[n]['texto'][:60]
            texts.append(f"L{nivel} | d_Π={d_pi:.3f}<br>{texto}...")
        
        fig.add_trace(go.Scatter3d(
            x=x, y=y, z=z,
            mode='markers',
            marker=dict(size=2.5, color=color, opacity=0.35),
            text=texts,
            hoverinfo='text',
            name=ACTOR_SHORT.get(actor, actor)
        ))
    
    # =========================================================================
    # 5. HUBs
    # =========================================================================
    for class_id, hc in hub_classes.items():
        members = [n for n in hc['members'] if n in node_to_pos]
        if not members:
            continue
        
        is_optimal = hc['is_optimal']
        
        if is_optimal:
            size, hub_color, line_width = 14, 'gold', 3
            name_prefix = "★ HUB Óptimo"
        else:
            size, hub_color, line_width = 9, '#FF6B35', 2
            name_prefix = f"HUB #{hc['rank']+1}"
        
        for idx, node in enumerate(members):
            pos = node_to_pos[node]
            actor = G.nodes[node]['actor']
            actor_color = ACTOR_COLORS.get(actor, 'gray')
            dist = hc['distancias'].get(actor, 0)
            texto = G.nodes[node]['texto']
            d_pi = float(np.linalg.norm(np.array(pos) - pi_pos))
            nivel = G.nodes[node].get('nivel', '?')
            
            fig.add_trace(go.Scatter3d(
                x=[pos[0]], y=[pos[1]], z=[pos[2]],
                mode='markers',
                marker=dict(size=size, color=hub_color, symbol='circle',
                            line=dict(color=actor_color, width=line_width)),
                text=[f"<b>{name_prefix}</b> - {ACTOR_SHORT.get(actor, actor)}<br>"
                      f"L{nivel} | FTT_dist={dist:.2f} | d_Π={d_pi:.3f}<br>{texto}"],
                hoverinfo='text',
                name=f"{name_prefix} (FTT={hc['ftt_sum']:.2f})" if idx == 0 else None,
                showlegend=(idx == 0)
            ))
        
        if len(members) > 1:
            for i_m in range(len(members)):
                for j_m in range(i_m + 1, len(members)):
                    p1, p2 = node_to_pos[members[i_m]], node_to_pos[members[j_m]]
                    sim_str = ""
                    if node_embeddings:
                        n1, n2 = members[i_m], members[j_m]
                        if n1 in node_embeddings and n2 in node_embeddings:
                            sim = cosine_sim(node_embeddings[n1], node_embeddings[n2])
                            sim_str = f"simCos={sim:.3f}"
                    
                    fig.add_trace(go.Scatter3d(
                        x=[p1[0], p2[0]], y=[p1[1], p2[1]], z=[p1[2], p2[2]],
                        mode='lines',
                        line=dict(color=hub_color, width=2, dash='dot'),
                        opacity=0.5,
                        showlegend=False,
                        text=[sim_str, sim_str] if sim_str else None,
                        hoverinfo='text' if sim_str else 'skip'
                    ))
    
    # =========================================================================
    # 6. RAÍCES (parte superior del embudo)
    # =========================================================================
    for root in roots:
        if root not in node_to_pos:
            continue
        pos = node_to_pos[root]
        actor = G.nodes[root]['actor']
        color = ACTOR_COLORS.get(actor, 'gray')
        d_pi = float(np.linalg.norm(np.array(pos) - pi_pos))
        
        fig.add_trace(go.Scatter3d(
            x=[pos[0]], y=[pos[1]], z=[pos[2]],
            mode='markers',
            marker=dict(size=13, color=color, symbol='diamond',
                        line=dict(color='white', width=2)),
            text=[f"<b>ROOT {ACTOR_SHORT.get(actor, '?')}</b><br>"
                  f"d_Π={d_pi:.3f}<br>{G.nodes[root]['texto'][:80]}..."],
            hoverinfo='text',
            name=f"Root: {ACTOR_SHORT.get(actor, actor)}",
            showlegend=True
        ))
    
    # =========================================================================
    # 7. Π (base del embudo)
    # =========================================================================
    pi_label = f"Π: {pi_text[:60]}..." if pi_text else "Π (problem)"
    
    fig.add_trace(go.Scatter3d(
        x=[pi_pos[0]], y=[pi_pos[1]], z=[pi_pos[2]],
        mode='markers+text',
        marker=dict(size=14, color='white', symbol='x',
                    line=dict(color='black', width=2)),
        text=['Π'],
        textposition='top center',
        textfont=dict(size=16, color='white'),
        hovertext=[pi_label],
        hoverinfo='text',
        name='Π (problem)'
    ))
    
    # =========================================================================
    # 8. LÍNEAS DE REFERENCIA Root → Π (eje del embudo)
    # =========================================================================
    for root in roots:
        if root not in node_to_pos:
            continue
        pos = node_to_pos[root]
        actor = G.nodes[root]['actor']
        color = ACTOR_COLORS.get(actor, 'gray')
        
        fig.add_trace(go.Scatter3d(
            x=[pos[0], pi_pos[0]], y=[pos[1], pi_pos[1]], z=[pos[2], pi_pos[2]],
            mode='lines',
            line=dict(color=color, width=1, dash='dot'),
            opacity=0.2,
            showlegend=False,
            hoverinfo='skip'
        ))
    
    # =========================================================================
    # 9. PLANOS DE NIVEL (guías horizontales para ver la jerarquía)
    # =========================================================================
    # Solo dibujar planos para niveles clave: 0 (raíz) y nivel del HUB
    hub_niveles = set()
    for n in all_hub_nodes:
        if n in node_to_pos:
            hub_niveles.add(int(G.nodes[n].get('nivel', 0)))
    
    # Rango X, Z para los planos
    all_x = [node_to_pos[n][0] for n in node_to_pos]
    all_z = [node_to_pos[n][2] for n in node_to_pos]
    x_min, x_max = min(all_x) - 0.1, max(all_x) + 0.1
    z_min, z_max = min(all_z) - 0.1, max(all_z) + 0.1
    
    for nivel in sorted(hub_niveles):
        y_nivel = (max_nivel - nivel) * y_scale
        # Plano sutil
        fig.add_trace(go.Mesh3d(
            x=[x_min, x_max, x_max, x_min],
            y=[y_nivel, y_nivel, y_nivel, y_nivel],
            z=[z_min, z_min, z_max, z_max],
            i=[0, 0], j=[1, 2], k=[2, 3],
            color='gold',
            opacity=0.03,
            showlegend=False,
            hoverinfo='skip'
        ))
    
    # =========================================================================
    # LAYOUT
    # =========================================================================
    fig.update_layout(
        title=dict(
            text='Wind Map v3b: Hybrid Layout (Topology + Semantics) - Principio de Naturalidad v8.5',
            font=dict(size=16, color='white')
        ),
        scene=dict(
            xaxis_title='PC1 (semantic)',
            yaxis_title='Level (tree depth ↓)',
            zaxis_title='PC2 (semantic)',
            bgcolor='rgb(15, 15, 25)',
            xaxis=dict(gridcolor='rgba(100,100,100,0.3)',
                       showbackground=True, backgroundcolor='rgb(20, 20, 35)'),
            yaxis=dict(gridcolor='rgba(100,100,100,0.3)',
                       showbackground=True, backgroundcolor='rgb(20, 20, 35)'),
            zaxis=dict(gridcolor='rgba(100,100,100,0.3)',
                       showbackground=True, backgroundcolor='rgb(20, 20, 35)'),
            aspectmode='data',
            camera=dict(
                eye=dict(x=1.5, y=1.0, z=1.5),  # Vista desde arriba-lateral
                up=dict(x=0, y=1, z=0),  # Y es "arriba"
            ),
        ),
        paper_bgcolor='rgb(15, 15, 25)',
        font=dict(color='white'),
        legend=dict(
            yanchor="top", y=0.99,
            xanchor="left", x=0.01,
            bgcolor='rgba(0,0,0,0.7)',
            font=dict(size=11)
        ),
        margin=dict(l=0, r=0, t=50, b=0)
    )
    
    return fig


# ============================================================================
# MAIN
# ============================================================================

def main():
    parser = argparse.ArgumentParser(description='Mapa de Vientos 3D v3b - Hybrid Layout')
    parser.add_argument('--run-path', '-r', type=str, required=True)
    parser.add_argument('--input-path', '-i', type=str, default=None)
    parser.add_argument('--output', '-o', type=str, default=None)
    
    args = parser.parse_args()
    
    run_path = Path(args.run_path)
    input_path = Path(args.input_path) if args.input_path else None
    
    if not input_path:
        possible = run_path.parent.parent / 'input.json'
        if possible.exists():
            input_path = possible
    
    print("\n" + "=" * 70)
    print("  WIND MAP v3b - HYBRID LAYOUT (Topology + Semantics)")
    print("=" * 70)
    
    G, keys, embeddings, clases, hubs_data, input_data = load_data(run_path, input_path)
    
    print("\n  PROCESSING Π")
    pi_embedding, pi_text = get_pi_embedding(input_data, keys, embeddings)
    
    print("\n  MAPPING EMBEDDINGS")
    node_embeddings = map_embeddings_to_nodes(G, keys, embeddings)
    
    print("\n  HYBRID PROJECTION")
    node_to_pos, roots, origin_3d, max_nivel, y_scale = project_hybrid_3d(
        node_embeddings, G, pi_embedding)
    
    print("\n  BUILDING WIND MAP")
    fig = create_wind_map(G, node_to_pos, roots, clases, hubs_data, origin_3d,
                          pi_text, node_embeddings, max_nivel, y_scale)
    
    output_path = Path(args.output) if args.output else (run_path / 'wind_map_3d_v3b.html')
    fig.write_html(str(output_path))
    print(f"\n✓ Saved: {output_path}")
    
    import webbrowser
    webbrowser.open(f'file://{output_path.absolute()}')
    
    return 0


if __name__ == '__main__':
    exit(main())