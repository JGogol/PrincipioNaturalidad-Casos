"""
Peace Field Map — Principio de Naturalidad v9
=============================================
Visualización de campo vectorial de convergencia/divergencia.

Muestra en un solo gráfico 3D interactivo:
  - Todos los nodos coloreados por distancia a Π
  - Flechas hacia Π desde cada nodo
  - Flechas hacia Raíz (identidad) desde cada nodo
  - Color de aristas por ΔsimΠ (verde=converge, rojo=diverge)
  - Controles interactivos para filtrar por tipo y actor
  - HUBs como puntos de encuentro
  - Π como atractor central visible

"Ver la paz en vivo — donde los actores difieren y donde convergen"

Uso:
    python peace_field_map.py --run-path casos/gerd/runs/2026-02-05_15-29-23

Requiere en run-path:
    - grafo.graphml (o grafo_con_fusiones.graphml)
    - embeddings_cache.npz
    - hubs.json
    - clases_equivalencia.json
    - input.json (para Π) o resultado.json

Autor: Javier Gogol Merletti / Claude
Febrero 2026
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

COLOR_ROTATION = [
    '#E63946',  # Rojo
    '#2A9D8F',  # Verde-azul
    '#E9C46A',  # Amarillo
    '#9B5DE5',  # Púrpura
    '#00BBF9',  # Cyan
    '#F15BB5',  # Rosa
    '#00F5D4',  # Turquesa
]

# Colores para convergencia/divergencia
COLOR_CONVERGE = '#00FF88'  # Verde brillante
COLOR_DIVERGE = '#FF4444'   # Rojo brillante
COLOR_PI = '#FFD700'        # Dorado
COLOR_HUB = '#FFFFFF'       # Blanco

# Mapeos dinámicos
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
    return ACTOR_SHORT.get(actor, actor.split()[-1])


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
    
    # Π (problema central)
    pi_text = None
    pi_embedding = None
    
    # Buscar input.json en múltiples ubicaciones
    if input_path and input_path.exists():
        possible_paths = [input_path]
    else:
        possible_paths = [
            run_path / 'input.json',
            run_path.parent / 'input.json',
            run_path.parent.parent / 'input.json',
            run_path.parent.parent.parent / 'input.json',
        ]
    
    found_input_path = None
    for p in possible_paths:
        if p.exists():
            found_input_path = p
            print(f"  Π source: {p}")
            break
    
    if found_input_path:
        with open(found_input_path, 'r', encoding='utf-8') as f:
            input_data = json.load(f)
        # Priorizar decision_central_embedding
        pi_text = input_data.get('problema', {}).get('decision_central_embedding') or \
                  input_data.get('problema', {}).get('decision_central')
        if pi_text:
            print(f"  Π text: {pi_text[:60]}...")
    
    # Si no, intentar desde resultado.json
    if not pi_text:
        resultado_path = run_path / 'resultado.json'
        if resultado_path.exists():
            with open(resultado_path, 'r', encoding='utf-8') as f:
                resultado = json.load(f)
            if 'problema' in resultado:
                pi_text = resultado['problema'].get('decision_central_embedding') or \
                          resultado['problema'].get('decision_central')
    
    # Buscar embedding de Π - probar múltiples formas de hash
    if pi_text:
        # Probar hash con lowercase (como lo hace el código actual)
        pi_hash_lower = hashlib.md5(pi_text.lower().encode('utf-8')).hexdigest()
        # Probar hash sin lowercase
        pi_hash_original = hashlib.md5(pi_text.encode('utf-8')).hexdigest()
        
        # Convertir keys a strings nativos para comparación
        keys_str = [str(k) for k in keys]
        
        if pi_hash_lower in keys_str:
            idx = keys_str.index(pi_hash_lower)
            pi_embedding = embeddings[idx]
            print(f"  Π: Found (lowercase hash)")
        elif pi_hash_original in keys_str:
            idx = keys_str.index(pi_hash_original)
            pi_embedding = embeddings[idx]
            print(f"  Π: Found (original hash)")
        else:
            print(f"  Π: Not in cache, attempting to compute...")
            try:
                from sentence_transformers import SentenceTransformer
                model = SentenceTransformer('all-mpnet-base-v2')
                pi_embedding = model.encode(pi_text)
                print(f"  Π: Computed using SentenceTransformer")
            except ImportError:
                print(f"  Π: SentenceTransformer not available")
                print(f"     Will use centroid of roots as fallback")
            except Exception as e:
                print(f"  Π: Error computing embedding: {e}")
                print(f"     Will use centroid of roots as fallback")
    else:
        print(f"  Π: Text not found in input/resultado")
        print(f"     Searched: {[str(p) for p in possible_paths]}")
    
    return G, keys, embeddings, emb_dim, hubs_data, pi_embedding


def map_embeddings(G, keys, embeddings):
    """Mapea embeddings a nodos del grafo."""
    hash_to_emb = {k: embeddings[i] for i, k in enumerate(keys)}
    
    node_emb = {}
    for node_id, attrs in G.nodes(data=True):
        text = attrs.get('texto', '').lower()
        h = hashlib.md5(text.encode('utf-8')).hexdigest()
        if h in hash_to_emb:
            node_emb[node_id] = hash_to_emb[h]
    
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
    """
    Proyecta embeddings a 3D:
      X, Z = PCA 2D (semántica)
      Y = nivel topológico (estructura)
    """
    # Obtener raíces y sus embeddings
    roots = [n for n, d in G.nodes(data=True) if d.get('nivel', 0) == 0 or d.get('nivel', '0') == '0']
    
    # Origen: Π si existe, sino centroide de raíces
    if pi_embedding is not None:
        origin = pi_embedding
        print(f"  Origin: Π")
    else:
        root_embs = np.array([node_emb[r] for r in roots if r in node_emb])
        origin = root_embs.mean(axis=0) if len(root_embs) > 0 else np.zeros(768)
        print(f"  Origin: Centroid of roots")
    
    # PCA centrado en origen
    all_nodes = list(node_emb.keys())
    all_embs = np.array([node_emb[n] for n in all_nodes])
    all_embs_centered = all_embs - origin
    
    pca = PCA(n_components=2)
    coords_2d = pca.fit_transform(all_embs_centered)
    variance = sum(pca.explained_variance_ratio_)
    print(f"  PCA variance: {variance:.1%}")
    
    # Máximo nivel
    max_nivel = max(int(G.nodes[n].get('nivel', 0)) for n in G.nodes())
    
    # Coordenadas 3D
    node_pos = {}
    for i, node_id in enumerate(all_nodes):
        nivel = int(G.nodes[node_id].get('nivel', 0))
        y = nivel / max(max_nivel, 1) * 10  # Escalar Y
        node_pos[node_id] = (coords_2d[i, 0], y, coords_2d[i, 1])
    
    # Posición de Π (origen en PCA = 0,0)
    pi_pos = (0.0, -0.5, 0.0)  # Ligeramente debajo del nivel 0
    
    # Posición de Π proyectada para vectores
    if pi_embedding is not None:
        pi_centered = pi_embedding - origin
        pi_2d = pca.transform([pi_centered])[0]
        pi_pos = (pi_2d[0], -0.5, pi_2d[1])
    
    return node_pos, pi_pos, pca, origin, max_nivel, variance


def compute_metrics(parent_emb, child_emb, pi_emb, root_emb):
    """Calcula métricas de convergencia/divergencia."""
    sim_parent_pi = cosine_sim(parent_emb, pi_emb)
    sim_child_pi = cosine_sim(child_emb, pi_emb)
    delta_sim_pi = sim_child_pi - sim_parent_pi
    
    sim_parent_root = cosine_sim(parent_emb, root_emb)
    sim_child_root = cosine_sim(child_emb, root_emb)
    delta_sim_root = sim_child_root - sim_parent_root
    
    return {
        'sim_child_pi': sim_child_pi,
        'sim_parent_pi': sim_parent_pi,
        'delta_sim_pi': delta_sim_pi,
        'sim_child_root': sim_child_root,
        'delta_sim_root': delta_sim_root,
        'converges': delta_sim_pi > 0,
    }


# ============================================================================
# VISUALIZACIÓN
# ============================================================================

def create_peace_field_map(G, node_emb, node_pos, pi_pos, pi_emb, hubs_data, 
                           pca, origin, max_nivel, emb_dim, variance):
    """Crea la visualización del campo de paz."""
    
    fig = go.Figure()
    
    # Obtener actores y raíces
    actors = sorted(set(G.nodes[n].get('actor', 'Unknown') for n in G.nodes()))
    initialize_actors(actors)
    
    roots = {G.nodes[n].get('actor'): n for n in G.nodes() 
             if G.nodes[n].get('nivel', '0') in [0, '0']}
    root_emb = {actor: node_emb[node_id] for actor, node_id in roots.items() 
                if node_id in node_emb}
    
    # HUBs
    hub_nodes = set()
    if hubs_data:
        for hub in hubs_data.get('hubs', []):
            for node_id in hub.get('clase_nodos', []):
                hub_nodes.add(node_id)
    
    # =========================================================================
    # 1. Π — EL ATRACTOR CENTRAL
    # =========================================================================
    fig.add_trace(go.Scatter3d(
        x=[pi_pos[0]], y=[pi_pos[1]], z=[pi_pos[2]],
        mode='markers+text',
        marker=dict(size=15, color=COLOR_PI, symbol='diamond',
                    line=dict(color='white', width=2)),
        text=['Π'],
        textposition='top center',
        textfont=dict(size=14, color=COLOR_PI),
        name='Π (Problem Core)',
        legendgroup='reference',
        legendgrouptitle_text='Reference Points',
        hovertemplate='<b>Π — Problem Core</b><br>All actors should converge here<extra></extra>'
    ))
    
    # =========================================================================
    # 2. RAÍCES — POSICIONES INICIALES
    # =========================================================================
    for actor, node_id in roots.items():
        if node_id not in node_pos:
            continue
        pos = node_pos[node_id]
        color = get_color(actor)
        texto = G.nodes[node_id].get('texto', '')[:80]
        
        fig.add_trace(go.Scatter3d(
            x=[pos[0]], y=[pos[1]], z=[pos[2]],
            mode='markers',
            marker=dict(size=12, color=color, symbol='square',
                        line=dict(color='white', width=2)),
            name=f'Root: {get_short(actor)}',
            legendgroup='reference',
            hovertemplate=f'<b>ROOT — {get_short(actor)}</b><br>{texto}...<extra></extra>'
        ))
    
    # =========================================================================
    # 3. CAMPO VECTORIAL — PRIMERO PARA QUE APAREZCA ARRIBA EN LEYENDA
    # =========================================================================
    
    # Estructuras por actor
    actor_conv = {actor: {'x': [], 'y': [], 'z': [], 'u': [], 'v': [], 'w': [], 'hover': []} for actor in actors}
    actor_div = {actor: {'x': [], 'y': [], 'z': [], 'u': [], 'v': [], 'w': [], 'hover': []} for actor in actors}
    tree_lines = {'x': [], 'y': [], 'z': []}
    
    total_conv, total_div = 0, 0
    
    for actor in actors:
        actor_root_emb = root_emb.get(actor, pi_emb if pi_emb is not None else np.zeros(emb_dim))
        
        for src, tgt in G.edges():
            if G.nodes[src].get('actor') != actor:
                continue
            if src not in node_pos or tgt not in node_pos:
                continue
            if src not in node_emb or tgt not in node_emb:
                continue
            
            p_src = np.array(node_pos[src])
            p_tgt = np.array(node_pos[tgt])
            p_pi = np.array(pi_pos)
            
            tree_lines['x'].extend([p_src[0], p_tgt[0], None])
            tree_lines['y'].extend([p_src[1], p_tgt[1], None])
            tree_lines['z'].extend([p_src[2], p_tgt[2], None])
            
            m = compute_metrics(node_emb[src], node_emb[tgt], 
                               pi_emb if pi_emb is not None else actor_root_emb,
                               actor_root_emb)
            
            pos = p_tgt
            vec_to_pi = p_pi - pos
            dist_to_pi = np.linalg.norm(vec_to_pi)
            
            if dist_to_pi < 0.001:
                continue
            
            vec_unit = vec_to_pi / dist_to_pi
            magnitude = min(abs(m['delta_sim_pi']) * 3, 0.8)
            
            nivel_tgt = G.nodes[tgt].get('nivel', '?')
            texto = G.nodes[tgt].get('texto', '')[:50]
            hover = (f'<b>{get_short(actor)}</b> L{nivel_tgt}<br>'
                    f'simΠ: {m["sim_child_pi"]:.3f}<br>'
                    f'ΔsimΠ: {m["delta_sim_pi"]:+.4f}<br>'
                    f'{texto}...')
            
            if m['converges']:
                # CONVERGENCIA: flecha HACIA Π
                bucket = actor_conv[actor]
                bucket['x'].append(pos[0])
                bucket['y'].append(pos[1])
                bucket['z'].append(pos[2])
                bucket['u'].append(vec_unit[0] * magnitude)
                bucket['v'].append(vec_unit[1] * magnitude)
                bucket['w'].append(vec_unit[2] * magnitude)
                bucket['hover'].append(hover)
                total_conv += 1
            else:
                # DIVERGENCIA: flecha DESDE Π (invertida)
                bucket = actor_div[actor]
                bucket['x'].append(pos[0])
                bucket['y'].append(pos[1])
                bucket['z'].append(pos[2])
                bucket['u'].append(-vec_unit[0] * magnitude)
                bucket['v'].append(-vec_unit[1] * magnitude)
                bucket['w'].append(-vec_unit[2] * magnitude)
                bucket['hover'].append(hover)
                total_div += 1
    
    print(f"  Vector field: {total_conv} convergence, {total_div} divergence")
    
    # CONVERGENCIA por actor (flechas hacia Π)
    for actor in actors:
        bucket = actor_conv[actor]
        if not bucket['x']:
            continue
        
        actor_color = get_color(actor)
        
        # Líneas
        conv_lines_x, conv_lines_y, conv_lines_z = [], [], []
        for i in range(len(bucket['x'])):
            x0, y0, z0 = bucket['x'][i], bucket['y'][i], bucket['z'][i]
            dx, dy, dz = bucket['u'][i], bucket['v'][i], bucket['w'][i]
            conv_lines_x.extend([x0, x0+dx, None])
            conv_lines_y.extend([y0, y0+dy, None])
            conv_lines_z.extend([z0, z0+dz, None])
        
        fig.add_trace(go.Scatter3d(
            x=conv_lines_x, y=conv_lines_y, z=conv_lines_z,
            mode='lines',
            line=dict(color=actor_color, width=3),
            opacity=0.8,
            name=f'🟢 {get_short(actor)} → Π',
            legendgroup='convergence',
            legendgrouptitle_text='═══ CONVERGENCE → Π ═══',
            hoverinfo='skip',
        ))
        
        # Puntas de flecha
        fig.add_trace(go.Cone(
            x=[bucket['x'][i] + bucket['u'][i] for i in range(len(bucket['x']))],
            y=[bucket['y'][i] + bucket['v'][i] for i in range(len(bucket['y']))],
            z=[bucket['z'][i] + bucket['w'][i] for i in range(len(bucket['z']))],
            u=bucket['u'], v=bucket['v'], w=bucket['w'],
            colorscale=[[0, actor_color], [1, actor_color]],
            sizemode='absolute',
            sizeref=0.04,
            showscale=False,
            opacity=0.8,
            name=f'{get_short(actor)} conv arrows',
            legendgroup='convergence',
            showlegend=False,
            hovertext=bucket['hover'],
            hoverinfo='text',
            anchor='tip',
        ))
    
    # DIVERGENCIA por actor (flechas desde Π)
    for actor in actors:
        bucket = actor_div[actor]
        if not bucket['x']:
            continue
        
        actor_color = get_color(actor)
        
        # Líneas
        div_lines_x, div_lines_y, div_lines_z = [], [], []
        for i in range(len(bucket['x'])):
            x0, y0, z0 = bucket['x'][i], bucket['y'][i], bucket['z'][i]
            dx, dy, dz = bucket['u'][i], bucket['v'][i], bucket['w'][i]
            div_lines_x.extend([x0, x0+dx, None])
            div_lines_y.extend([y0, y0+dy, None])
            div_lines_z.extend([z0, z0+dz, None])
        
        fig.add_trace(go.Scatter3d(
            x=div_lines_x, y=div_lines_y, z=div_lines_z,
            mode='lines',
            line=dict(color=actor_color, width=3, dash='dash'),  # Dash para divergencia
            opacity=0.6,
            name=f'🔴 {get_short(actor)} ← Π',
            legendgroup='divergence',
            legendgrouptitle_text='═══ DIVERGENCE ← Π ═══',
            hoverinfo='skip',
        ))
        
        # Puntas de flecha
        fig.add_trace(go.Cone(
            x=[bucket['x'][i] + bucket['u'][i] for i in range(len(bucket['x']))],
            y=[bucket['y'][i] + bucket['v'][i] for i in range(len(bucket['y']))],
            z=[bucket['z'][i] + bucket['w'][i] for i in range(len(bucket['z']))],
            u=bucket['u'], v=bucket['v'], w=bucket['w'],
            colorscale=[[0, actor_color], [1, actor_color]],
            sizemode='absolute',
            sizeref=0.04,
            showscale=False,
            opacity=0.6,
            name=f'{get_short(actor)} div arrows',
            legendgroup='divergence',
            showlegend=False,
            hovertext=bucket['hover'],
            hoverinfo='text',
            anchor='tip',
        ))
    
    # Estructura del árbol
    fig.add_trace(go.Scatter3d(
        x=tree_lines['x'], y=tree_lines['y'], z=tree_lines['z'],
        mode='lines',
        line=dict(color='#444444', width=1),
        opacity=0.3,
        name='Tree Structure',
        legendgroup='structure',
        legendgrouptitle_text='Structure',
        hoverinfo='skip',
    ))
    
    # =========================================================================
    # 4. HUBs — PUNTOS DE CONVERGENCIA
    # =========================================================================
    if hubs_data:
        for i, hub in enumerate(hubs_data.get('hubs', [])):
            hub_positions = []
            for node_id in hub.get('clase_nodos', []):
                if node_id in node_pos:
                    hub_positions.append(node_pos[node_id])
            
            if hub_positions:
                # Centroide del HUB
                cx = np.mean([p[0] for p in hub_positions])
                cy = np.mean([p[1] for p in hub_positions])
                cz = np.mean([p[2] for p in hub_positions])
                
                is_optimal = (i == 0)
                size = 18 if is_optimal else 12
                symbol = 'star' if is_optimal else 'circle'
                
                fig.add_trace(go.Scatter3d(
                    x=[cx], y=[cy], z=[cz],
                    mode='markers',
                    marker=dict(size=size, color=COLOR_HUB, symbol='diamond',
                                line=dict(color=COLOR_PI, width=3)),
                    name=f'{"★ Optimal " if is_optimal else ""}HUB #{i+1} (TTF={hub.get("ftt_sum", 0):.2f})',
                    legendgroup='reference',
                    hovertemplate=f'<b>{"★ OPTIMAL " if is_optimal else ""}HUB #{i+1}</b><br>'
                                  f'TTF: {hub.get("ftt_sum", 0):.2f}<br>'
                                  f'Actors: {", ".join([get_short(a) for a in hub.get("actores", [])])}<extra></extra>'
                ))
    
    # =========================================================================
    # 3.5 PATHS HACIA HUBs — CAMINOS DE CONVERGENCIA
    # =========================================================================
    if hubs_data and 'hubs' in hubs_data:
        for hub_idx, hub in enumerate(hubs_data['hubs']):
            caminos = hub.get('caminos', {})
            is_optimal = (hub_idx == 0)
            
            for actor, path in caminos.items():
                if len(path) < 2:
                    continue
                
                base_color = get_color(actor)
                total_steps = len(path) - 1
                
                path_x, path_y, path_z = [], [], []
                
                for j in range(len(path)):
                    node_id = path[j].get('id')
                    if node_id and node_id in node_pos:
                        pos = node_pos[node_id]
                        path_x.append(pos[0])
                        path_y.append(pos[1])
                        path_z.append(pos[2])
                
                if len(path_x) >= 2:
                    # Línea del path
                    fig.add_trace(go.Scatter3d(
                        x=path_x, y=path_y, z=path_z,
                        mode='lines',
                        line=dict(
                            color=base_color, 
                            width=8 if is_optimal else 4,
                            dash=None if is_optimal else 'dash'
                        ),
                        opacity=0.9 if is_optimal else 0.5,
                        name=f'{"★ " if is_optimal else ""}Path {get_short(actor)} → HUB #{hub_idx+1}',
                        legendgroup='paths',
                        legendgrouptitle_text='Paths to HUBs',
                        hovertemplate=f'<b>{"★ " if is_optimal else ""}Path {get_short(actor)} → HUB #{hub_idx+1}</b><extra></extra>'
                    ))
    
    # =========================================================================
    # 4. NODOS — COLOREADOS POR ACTOR
    # =========================================================================
    for actor in actors:
        actor_nodes = [n for n in G.nodes() if G.nodes[n].get('actor') == actor 
                       and n in node_pos and n not in hub_nodes and n not in roots.values()]
        
        if not actor_nodes:
            continue
        
        xs, ys, zs, sizes, hovers = [], [], [], [], []
        actor_color = get_color(actor)
        
        for node_id in actor_nodes:
            pos = node_pos[node_id]
            xs.append(pos[0])
            ys.append(pos[1])
            zs.append(pos[2])
            
            # Similitud a Π para tamaño
            if node_id in node_emb and pi_emb is not None:
                sim_pi = cosine_sim(node_emb[node_id], pi_emb)
            else:
                sim_pi = 0.5
            
            # Tamaño proporcional a cercanía a Π
            sizes.append(3 + sim_pi * 5)
            
            nivel = G.nodes[node_id].get('nivel', '?')
            texto = G.nodes[node_id].get('texto', '')[:60]
            hovers.append(f'<b>{get_short(actor)}</b> L{nivel}<br>'
                         f'simΠ: {sim_pi:.3f}<br>{texto}...')
        
        fig.add_trace(go.Scatter3d(
            x=xs, y=ys, z=zs,
            mode='markers',
            marker=dict(
                size=sizes,
                color=actor_color,
                opacity=0.7,
                line=dict(color='white', width=0.5),
            ),
            name=f'{get_short(actor)} nodes',
            legendgroup=f'nodes_{get_short(actor)}',
            legendgrouptitle_text='Nodes by Actor',
            hovertemplate='%{hovertext}<extra></extra>',
            hovertext=hovers,
        ))
    
    # =========================================================================
    # 6. (Vectores hacia Π integrados en el campo vectorial anterior)
    # =========================================================================
    
    # =========================================================================
    # LAYOUT
    # =========================================================================
    
    fig.update_layout(
        title=dict(
            text='<b>Peace Field Map</b> — Semantic Vector Field<br>'
                 f'<sup>🟢 Convergence = toward Π | 🔴 Divergence = away from Π | PCA {variance:.1%} | '
                 f'<b>Click legend items to toggle</b></sup>',
            x=0.5,
            font=dict(size=18)
        ),
        scene=dict(
            xaxis_title='PCA 1 (Semantic)',
            yaxis_title='Tree Level (Topological)',
            zaxis_title='PCA 2 (Semantic)',
            bgcolor='#0a0a1a',
            xaxis=dict(gridcolor='#333', zerolinecolor='#555'),
            yaxis=dict(gridcolor='#333', zerolinecolor='#555'),
            zaxis=dict(gridcolor='#333', zerolinecolor='#555'),
            camera=dict(
                eye=dict(x=1.5, y=1.5, z=1.5),
                up=dict(x=0, y=1, z=0)
            ),
            aspectmode='manual',
            aspectratio=dict(x=1, y=1.2, z=1),
        ),
        paper_bgcolor='#0a0a1a',
        plot_bgcolor='#0a0a1a',
        font=dict(color='white'),
        legend=dict(
            bgcolor='rgba(20,20,40,0.9)',
            bordercolor='#444',
            borderwidth=1,
            font=dict(size=11),
            itemsizing='constant',
            groupclick='togglegroup',  # Click en grupo oculta todo el grupo
            x=1.02,
            y=0.98,
            xanchor='left',
            yanchor='top',
        ),
        margin=dict(l=0, r=280, t=80, b=0),
        height=900,
    )
    
    # Solo botones de cámara
    fig.update_layout(
        updatemenus=[
            dict(
                type='buttons',
                direction='left',
                x=0.02, y=0.98,
                xanchor='left', yanchor='top',
                bgcolor='rgba(40,40,60,0.9)',
                font=dict(color='white', size=10),
                pad=dict(t=5, b=5),
                buttons=[
                    dict(label='Top', method='relayout',
                         args=[{'scene.camera.eye': {'x': 0, 'y': 2.5, 'z': 0}}]),
                    dict(label='Side', method='relayout',
                         args=[{'scene.camera.eye': {'x': 2, 'y': 0.5, 'z': 0}}]),
                    dict(label='3D', method='relayout',
                         args=[{'scene.camera.eye': {'x': 1.5, 'y': 1.5, 'z': 1.5}}]),
                    dict(label='From Π', method='relayout',
                         args=[{'scene.camera.eye': {'x': 0, 'y': -0.5, 'z': 2}}]),
                ],
            ),
        ]
    )
    
    return fig


# ============================================================================
# MAIN
# ============================================================================

def main():
    parser = argparse.ArgumentParser(description='Peace Field Map - Principio de Naturalidad')
    parser.add_argument('--run-path', type=str, required=True,
                        help='Path to run directory with grafo.graphml and embeddings_cache.npz')
    parser.add_argument('--input-path', type=str, default=None,
                        help='Path to input.json (optional, will search automatically)')
    parser.add_argument('--output', type=str, default='peace_field_map.html',
                        help='Output HTML filename')
    
    args = parser.parse_args()
    run_path = Path(args.run_path)
    input_path = Path(args.input_path) if args.input_path else None
    
    print("\n" + "=" * 60)
    print("PEACE FIELD MAP — PRINCIPIO DE NATURALIDAD")
    print("=" * 60)
    
    # Cargar datos
    G, keys, embeddings, emb_dim, hubs_data, pi_emb = load_data(run_path, input_path)
    
    # Mapear embeddings
    node_emb = map_embeddings(G, keys, embeddings)
    
    # Proyectar a 3D
    print("\nProjecting to 3D...")
    node_pos, pi_pos, pca, origin, max_nivel, variance = project_to_3d(node_emb, G, pi_emb)
    
    # Crear visualización
    print("\nCreating visualization...")
    fig = create_peace_field_map(G, node_emb, node_pos, pi_pos, pi_emb, hubs_data,
                                  pca, origin, max_nivel, emb_dim, variance)
    
    # Guardar
    output_path = run_path / args.output
    fig.write_html(str(output_path))
    print(f"\n✓ Saved: {output_path}")
    
    # Stats
    actors = set(G.nodes[n].get('actor', 'Unknown') for n in G.nodes())
    print(f"\nStats:")
    print(f"  Actors: {len(actors)}")
    print(f"  Nodes: {G.number_of_nodes()}")
    print(f"  Edges: {G.number_of_edges()}")
    print(f"  HUBs: {len(hubs_data.get('hubs', [])) if hubs_data else 0}")
    
    print("\n" + "=" * 60)
    print("Click legend items to show/hide convergence/divergence")
    print("=" * 60 + "\n")


if __name__ == '__main__':
    main()