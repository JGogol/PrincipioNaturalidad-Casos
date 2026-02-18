"""
Mapa de Vientos 3D v6 - Principio de Naturalidad v8.5
Layout Híbrido: Topología + Semántica
MÉTRICAS: ΔsimΠ (similitud coseno con Π) — SIN θ_Π (artefacto dimensional)

Corrección epistemológica v5→v6:
  θ_Π < 90° resultó ser artefacto geométrico de alta dimensionalidad.
  Diagnóstico demostró que:
    • Shuffle aleatorio: 99.9% →Π (θ̄=62°)
    • Π falso (random): 100% →Π (θ̄=72°)
    • Centroide:         100% →Π (θ̄=57°)
  En 768d, CUALQUIER punto actúa como "atractor" por θ<90°.
  El criterio correcto es ΔsimΠ: cambio en similitud coseno con Π.
    • ΔsimΠ > 0 → hijo MÁS similar a Π que padre (acerca)
    • ΔsimΠ < 0 → hijo MENOS similar a Π que padre (aleja)
  Dato real: solo 43.2% de aristas acercan. Honesto.

Arquitectura:
  ┌─────────────────────────────────────────────────────────────┐
  │  MÉTRICAS (768d, fidelidad 100%):                           │
  │    • ΔsimΠ = cosSim(hijo,Π) - cosSim(padre,Π)              │
  │    • simCos padre↔hijo (coherencia del paso)                │
  │    • simΠ por nodo (posición absoluta respecto a Π)         │
  │  ELIMINADO (artefacto demostrado):                          │
  │    • θ_Π: ángulo en Nd entre vec_arista y vec_hacia_Π       │
  ├─────────────────────────────────────────────────────────────┤
  │  VISUALIZACIÓN (con pérdida declarada):                     │
  │    • PCA 2D → ejes X, Z (~22% varianza explicada)           │
  │    • Y = nivel topológico (dato real)                       │
  │    • Flechas: dirección visual (PCA), opacidad por ΔsimΠ   │
  └─────────────────────────────────────────────────────────────┘

Uso:
    python wind_map_3d_v6.py --run-path casos/gerd/runs/2026-02-04_11-13-40

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


def hex_to_rgb(hex_color):
    h = hex_color.lstrip('#')
    return tuple(int(h[i:i+2], 16) for i in (0, 2, 4))


def rgb_to_hex(r, g, b):
    return f'#{int(r):02x}{int(g):02x}{int(b):02x}'


def color_for_level(base_hex, nivel, max_nivel):
    r, g, b = hex_to_rgb(base_hex)
    if max_nivel == 0:
        factor = 0
    else:
        factor = (nivel / max_nivel) * 0.7
    r2 = r + (255 - r) * factor
    g2 = g + (255 - g) * factor
    b2 = b + (255 - b) * factor
    return rgb_to_hex(r2, g2, b2)


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
    emb_dim = embeddings.shape[1] if len(embeddings.shape) > 1 else 0
    print(f"  Embeddings: {len(keys)} vectors × {emb_dim}d")
    
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
# GEOMETRÍA — ESPACIO ORIGINAL (768d, sin pérdida)
# ============================================================================

def cosine_sim(a, b):
    dot = np.dot(a, b)
    na, nb = np.linalg.norm(a), np.linalg.norm(b)
    if na == 0 or nb == 0:
        return 0.0
    return float(np.clip(dot / (na * nb), -1, 1))


def compute_edge_metrics(parent_emb, child_emb, pi_emb):
    """
    Métricas REALES de una arista padre→hijo en espacio original (768d).
    
    Criterio de convergencia: ΔsimΠ = cosSim(hijo,Π) - cosSim(padre,Π)
      > 0 → hijo MÁS similar a Π que padre (el paso acerca)
      < 0 → hijo MENOS similar a Π que padre (el paso aleja)
      = 0 → neutro
    
    θ_Π fue ELIMINADO: demostrado artefacto dimensional
    (shuffle random: 99.9% →Π, Π falso: 100% →Π).
    """
    parent_emb = np.asarray(parent_emb, dtype=np.float64)
    child_emb = np.asarray(child_emb, dtype=np.float64)
    pi_emb = np.asarray(pi_emb, dtype=np.float64)
    
    # Similitud coseno con Π (métrica angular, sin pérdida, sin sesgo dimensional)
    sim_parent_pi = cosine_sim(parent_emb, pi_emb)
    sim_child_pi = cosine_sim(child_emb, pi_emb)
    delta_sim_pi = sim_child_pi - sim_parent_pi
    
    # Similitud coseno padre↔hijo (coherencia del paso)
    sim_edge = cosine_sim(parent_emb, child_emb)
    
    return {
        'sim_parent_pi': sim_parent_pi,      # cosSim(padre, Π)
        'sim_child_pi': sim_child_pi,        # cosSim(hijo, Π)
        'delta_sim_pi': delta_sim_pi,        # >0 = hijo más similar a Π
        'sim_edge': sim_edge,                # cosSim(padre, hijo)
        'acerca_pi': delta_sim_pi > 0,       # criterio limpio
    }


# ============================================================================
# PROYECCIÓN HÍBRIDA (PCA para visualización, Π en 0,0)
# ============================================================================

def project_hybrid_3d(node_embeddings, G, pi_embedding=None):
    """
    Layout híbrido VISUAL:
      Y = nivel del árbol (topología real)
      X, Z = PCA 2D centrado en Π (SOLO VISUALIZACIÓN)
    """
    
    roots = [n for n, d in G.nodes(data=True) if d['nivel'] == 0]
    
    max_nivel = max(int(G.nodes[n].get('nivel', 0)) for n in G.nodes())
    print(f"  Max tree depth: {max_nivel}")
    
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
    
    pca = PCA(n_components=2)
    pca.fit(all_embs_centered)
    
    node_coords_2d = pca.transform(all_embs_centered)
    
    pi_2d = pca.transform(np.zeros((1, origin.shape[0])))[0]
    
    var_explained = pca.explained_variance_ratio_.sum()
    print(f"  PCA 2D variance explained: {var_explained:.2%}  ⚠️  SOLO VISUAL")
    print(f"  Π in PCA space: [{pi_2d[0]:.6f}, {pi_2d[1]:.6f}]  (should be ~0,0)")
    
    node_coords_2d = node_coords_2d - pi_2d
    
    print(f"  Π corrected: [0.000000, 0.000000]")
    
    for i, n in enumerate(all_nodes):
        if n in roots:
            actor = G.nodes[n]['actor']
            print(f"  Root {actor} in PCA: [{node_coords_2d[i, 0]:.4f}, {node_coords_2d[i, 1]:.4f}]")
    
    x_range = node_coords_2d[:, 0].max() - node_coords_2d[:, 0].min()
    z_range = node_coords_2d[:, 1].max() - node_coords_2d[:, 1].min()
    semantic_range = max(x_range, z_range, 0.001)
    
    y_scale = semantic_range / max(max_nivel, 1)
    
    node_to_pos = {}
    for i, n in enumerate(all_nodes):
        nivel = int(G.nodes[n].get('nivel', 0))
        x = node_coords_2d[i, 0]
        y = (max_nivel - nivel) * y_scale
        z = node_coords_2d[i, 1]
        node_to_pos[n] = np.array([x, y, z])
    
    pi_y = -1.0 * y_scale
    origin_3d = np.array([0.0, pi_y, 0.0])
    
    print(f"  Π 3D position: [{origin_3d[0]:.6f}, {origin_3d[1]:.6f}, {origin_3d[2]:.6f}]")
    print(f"  Y scale: {y_scale:.4f}")
    
    all_x = [node_to_pos[n][0] for n in node_to_pos]
    all_z = [node_to_pos[n][2] for n in node_to_pos]
    print(f"  X range: [{min(all_x):.4f}, {max(all_x):.4f}] (Π at x=0)")
    print(f"  Z range: [{min(all_z):.4f}, {max(all_z):.4f}] (Π at z=0)")
    
    return node_to_pos, roots, origin_3d, max_nivel, y_scale, var_explained


# ============================================================================
# ÁRBOL
# ============================================================================

def build_parent_map(G):
    parent_map = {}
    for src, tgt in G.edges():
        parent_map[tgt] = src
    return parent_map


# ============================================================================
# VISUALIZACIÓN v6 — ΔsimΠ como criterio, sin θ_Π
# ============================================================================

def create_wind_map(G, node_to_pos, roots, clases, hubs_data, origin_3d,
                    pi_text=None, node_embeddings=None, pi_embedding=None,
                    max_nivel=0, y_scale=1.0, pca_variance=0.0):
    
    ACTOR_COLORS = get_actor_colors(G)
    ACTOR_SHORT = get_actor_short_names(ACTOR_COLORS.keys())
    print(f"  Actors: {list(ACTOR_SHORT.values())}")
    
    emb_dim = 0
    if node_embeddings:
        sample = next(iter(node_embeddings.values()))
        emb_dim = len(sample)
    
    parent_map = build_parent_map(G)
    
    root_of_actor = {}
    for r in roots:
        if r in node_to_pos:
            actor = G.nodes[r]['actor']
            root_of_actor[actor] = r
    
    # HUBs
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
    
    can_compute = (node_embeddings is not None and pi_embedding is not None)
    
    # =========================================================================
    # 1. ARISTAS DE FONDO con degradé por nivel
    # =========================================================================
    for actor, base_color in ACTOR_COLORS.items():
        for src, tgt in G.edges():
            if G.nodes[src]['actor'] != actor:
                continue
            if src not in node_to_pos or tgt not in node_to_pos:
                continue
            
            p1, p2 = node_to_pos[src], node_to_pos[tgt]
            nivel_src = int(G.nodes[src].get('nivel', 0))
            edge_color = color_for_level(base_color, nivel_src, max_nivel)
            
            fig.add_trace(go.Scatter3d(
                x=[p1[0], p2[0], None], y=[p1[1], p2[1], None], z=[p1[2], p2[2], None],
                mode='lines',
                line=dict(color=edge_color, width=0.8),
                opacity=0.15,
                showlegend=False,
                hoverinfo='skip'
            ))
    
    # =========================================================================
    # 2. FLECHAS DE VIENTO — criterio: ΔsimΠ (768d, real)
    # =========================================================================
    stats_global = {'acerca': 0, 'aleja': 0, 'total': 0}
    stats_by_actor = {}
    stats_by_level = {}
    stats_by_actor_level = {}
    
    for actor, base_color in ACTOR_COLORS.items():
        root_id = root_of_actor.get(actor)
        if not root_id or root_id not in node_to_pos:
            continue
        
        actor_short = ACTOR_SHORT.get(actor, actor)
        
        if actor not in stats_by_actor:
            stats_by_actor[actor] = {
                'acerca': 0, 'aleja': 0, 'total': 0,
                'delta_sim_sum': 0, 'sim_edge_sum': 0,
            }
        
        # Separar por criterio ΔsimΠ > 0
        cones = {True: {'x': [], 'y': [], 'z': [], 'u': [], 'v': [], 'w': [], 'hover': []},
                 False: {'x': [], 'y': [], 'z': [], 'u': [], 'v': [], 'w': [], 'hover': []}}
        
        for src, tgt in G.edges():
            if G.nodes[src]['actor'] != actor:
                continue
            if src not in node_to_pos or tgt not in node_to_pos:
                continue
            
            p1 = node_to_pos[src]
            p2 = node_to_pos[tgt]
            
            nivel_src = int(G.nodes[src].get('nivel', 0))
            nivel_tgt = int(G.nodes[tgt].get('nivel', 0))
            
            # --- MÉTRICAS REALES EN 768d ---
            if can_compute and src in node_embeddings and tgt in node_embeddings:
                m = compute_edge_metrics(
                    node_embeddings[src], node_embeddings[tgt], pi_embedding
                )
            else:
                m = {
                    'sim_parent_pi': 0, 'sim_child_pi': 0,
                    'delta_sim_pi': 0, 'sim_edge': 0, 'acerca_pi': False,
                }
            
            # --- Acumular stats ---
            stats_global['total'] += 1
            stats_by_actor[actor]['total'] += 1
            stats_by_actor[actor]['delta_sim_sum'] += m['delta_sim_pi']
            stats_by_actor[actor]['sim_edge_sum'] += m['sim_edge']
            
            level_key = f"L{nivel_src}→L{nivel_tgt}"
            if level_key not in stats_by_level:
                stats_by_level[level_key] = {
                    'acerca': 0, 'aleja': 0, 'total': 0,
                    'delta_sim_sum': 0,
                }
            stats_by_level[level_key]['total'] += 1
            stats_by_level[level_key]['delta_sim_sum'] += m['delta_sim_pi']
            
            al_key = (actor, nivel_src)
            if al_key not in stats_by_actor_level:
                stats_by_actor_level[al_key] = {
                    'acerca': 0, 'aleja': 0, 'total': 0,
                    'delta_sim_sum': 0,
                }
            stats_by_actor_level[al_key]['total'] += 1
            stats_by_actor_level[al_key]['delta_sim_sum'] += m['delta_sim_pi']
            
            if m['acerca_pi']:
                stats_global['acerca'] += 1
                stats_by_actor[actor]['acerca'] += 1
                stats_by_level[level_key]['acerca'] += 1
                stats_by_actor_level[al_key]['acerca'] += 1
            else:
                stats_global['aleja'] += 1
                stats_by_actor[actor]['aleja'] += 1
                stats_by_level[level_key]['aleja'] += 1
                stats_by_actor_level[al_key]['aleja'] += 1
            
            # --- Vector visual para flechas (PCA) ---
            vec_vis = np.array(p2) - np.array(p1)
            norm_v = np.linalg.norm(vec_vis)
            if norm_v == 0:
                continue
            
            vec_unit = vec_vis / norm_v
            arrow_pos = np.array(p1) + vec_vis * 0.6
            arrow_len = norm_v * 0.25
            
            # Hover con métricas reales
            dir_icon = "↑acerca" if m['acerca_pi'] else "↓aleja"
            
            hover = (
                f"<b>{actor_short}</b> L{nivel_src}→L{nivel_tgt}<br>"
                f"simCos(padre,hijo)={m['sim_edge']:.3f}<br>"
                f"━━━ {emb_dim}d ━━━<br>"
                f"simΠ: {m['sim_parent_pi']:.3f} → {m['sim_child_pi']:.3f}<br>"
                f"<b>ΔsimΠ={m['delta_sim_pi']:+.4f} {dir_icon}</b>"
            )
            
            bucket = cones[m['acerca_pi']]
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
                colorscale=[[0, base_color], [1, base_color]],
                sizemode='absolute',
                sizeref=0.025,
                showscale=False,
                opacity=0.55 if acerca else 0.15,
                showlegend=False,
                hovertext=bucket['hover'],
                hoverinfo='text',
                anchor='tail',
            ))
    
    # =========================================================================
    # REPORTE — ΔsimΠ en 768d (sin θ_Π)
    # =========================================================================
    t = stats_global['total']
    pct_acerca = 100 * stats_global['acerca'] / max(t, 1)
    pct_aleja = 100 * stats_global['aleja'] / max(t, 1)
    
    print(f"\n  {'='*65}")
    print(f"  ANÁLISIS CONVERGENCIA SEMÁNTICA — ΔsimΠ en {emb_dim}d")
    print(f"  Criterio: ΔsimΠ = cosSim(hijo,Π) - cosSim(padre,Π)")
    print(f"  ΔsimΠ > 0 → acerca | ΔsimΠ < 0 → aleja")
    print(f"  (θ_Π eliminado: artefacto dimensional demostrado)")
    print(f"  PCA ({pca_variance:.1%} varianza) usado SOLO para visualización")
    print(f"  {'='*65}")
    print(f"  GLOBAL: {t} aristas")
    print(f"    ↑acerca (ΔsimΠ>0): {stats_global['acerca']:4d} ({pct_acerca:.1f}%)")
    print(f"    ↓aleja  (ΔsimΠ<0): {stats_global['aleja']:4d} ({pct_aleja:.1f}%)")
    
    print(f"\n  POR ACTOR:")
    for actor in sorted(stats_by_actor.keys(), key=lambda a: ACTOR_SHORT.get(a, a)):
        s = stats_by_actor[actor]
        if s['total'] == 0:
            continue
        pct = 100 * s['acerca'] / s['total']
        delta_sim_avg = s['delta_sim_sum'] / s['total']
        sim_edge_avg = s['sim_edge_sum'] / s['total']
        bar_len = int(pct / 5)
        bar = '█' * bar_len + '░' * (20 - bar_len)
        print(f"    {ACTOR_SHORT.get(actor, actor):12s} ↑Π {pct:5.1f}% [{bar}] "
              f"Δ̄simΠ={delta_sim_avg:+.4f}  s̄imEdge={sim_edge_avg:.3f}  (n={s['total']})")
    
    print(f"\n  POR NIVEL (transición):")
    for lk in sorted(stats_by_level.keys()):
        s = stats_by_level[lk]
        if s['total'] == 0:
            continue
        pct = 100 * s['acerca'] / s['total']
        delta_sim_avg = s['delta_sim_sum'] / s['total']
        bar_len = int(pct / 5)
        bar = '█' * bar_len + '░' * (20 - bar_len)
        print(f"    {lk:8s} ↑Π {pct:5.1f}% [{bar}] Δ̄simΠ={delta_sim_avg:+.4f}  (n={s['total']})")
    
    print(f"\n  POR ACTOR × NIVEL:")
    for actor in sorted(stats_by_actor.keys(), key=lambda a: ACTOR_SHORT.get(a, a)):
        actor_short = ACTOR_SHORT.get(actor, actor)
        levels = sorted([k for k in stats_by_actor_level if k[0] == actor], key=lambda k: k[1])
        if not levels:
            continue
        print(f"    {actor_short}:")
        for al_key in levels:
            s = stats_by_actor_level[al_key]
            if s['total'] == 0:
                continue
            nivel = al_key[1]
            pct = 100 * s['acerca'] / s['total']
            delta_sim_avg = s['delta_sim_sum'] / s['total']
            bar_len = int(pct / 5)
            bar = '█' * bar_len + '░' * (20 - bar_len)
            print(f"      L{nivel}→L{nivel+1}  ↑Π {pct:5.1f}% [{bar}] Δ̄simΠ={delta_sim_avg:+.4f}  (n={s['total']})")
    
    print(f"  {'='*65}")
    
    # =========================================================================
    # 3. CAMINOS DE CONVERGENCIA → HUB
    # =========================================================================
    if hubs_data and 'hubs' in hubs_data:
        for hub_idx, hub in enumerate(hubs_data['hubs']):
            caminos = hub.get('caminos', {})
            is_optimal = (hub_idx == 0)
            
            for actor, path in caminos.items():
                if len(path) < 2:
                    continue
                
                base_color = ACTOR_COLORS.get(actor, 'gray')
                total_steps = len(path) - 1
                
                for j in range(total_steps):
                    src_id = path[j]['id']
                    tgt_id = path[j + 1]['id']
                    
                    if src_id not in node_to_pos or tgt_id not in node_to_pos:
                        continue
                    
                    p1, p2 = node_to_pos[src_id], node_to_pos[tgt_id]
                    
                    # Métricas 768d reales
                    if (can_compute and src_id in node_embeddings
                            and tgt_id in node_embeddings):
                        m = compute_edge_metrics(
                            node_embeddings[src_id], node_embeddings[tgt_id],
                            pi_embedding
                        )
                    else:
                        m = {
                            'sim_parent_pi': 0, 'sim_child_pi': 0,
                            'delta_sim_pi': 0, 'sim_edge': 0, 'acerca_pi': False,
                        }
                    
                    # Grosor decreciente: embudo
                    progress = j / total_steps
                    if is_optimal:
                        width = 5 - progress * 3
                        opacity = 0.75
                        dash = None
                    else:
                        width = 3 - progress * 1.5
                        opacity = 0.35
                        dash = 'dash'
                    
                    nivel_src = int(G.nodes[src_id].get('nivel', 0)) if src_id in G.nodes() else j
                    line_color = color_for_level(base_color, nivel_src, max_nivel)
                    
                    dir_icon = "↑acerca" if m['acerca_pi'] else "↓aleja"
                    src_text = path[j].get('texto', '')[:50]
                    tgt_text = path[j + 1].get('texto', '')[:50]
                    
                    hover = (
                        f"<b>{'★ ' if is_optimal else ''}Path {ACTOR_SHORT.get(actor, actor)} → HUB</b><br>"
                        f"L{j}→L{j+1} | simEdge={m['sim_edge']:.3f}<br>"
                        f"━━━ {emb_dim}d ━━━<br>"
                        f"simΠ: {m['sim_parent_pi']:.3f} → {m['sim_child_pi']:.3f}<br>"
                        f"<b>ΔsimΠ={m['delta_sim_pi']:+.4f} {dir_icon}</b><br>"
                        f"<i>{src_text}...</i><br>→ <i>{tgt_text}...</i>"
                    )
                    
                    fig.add_trace(go.Scatter3d(
                        x=[p1[0], p2[0]], y=[p1[1], p2[1]], z=[p1[2], p2[2]],
                        mode='lines',
                        line=dict(color=line_color, width=width, dash=dash),
                        opacity=opacity,
                        showlegend=(j == 0 and hub_idx < 2),
                        name=f"{'★ ' if is_optimal else ''}Path: {ACTOR_SHORT.get(actor, actor)} → HUB" if j == 0 else None,
                        text=[hover, hover],
                        hoverinfo='text'
                    ))
                    
                    # Flechas de dirección visual
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
                            colorscale=[[0, line_color], [1, line_color]],
                            sizemode='absolute',
                            sizeref=0.025,
                            showscale=False,
                            opacity=0.7,
                            showlegend=False,
                            hoverinfo='skip',
                            anchor='tail',
                        ))
    
    # =========================================================================
    # 4. NODOS NORMALES — con degradé por nivel
    # =========================================================================
    for actor, base_color in ACTOR_COLORS.items():
        nodes_by_level = {}
        for n in G.nodes():
            if G.nodes[n]['actor'] != actor or n not in node_to_pos or n in all_hub_nodes:
                continue
            nivel = int(G.nodes[n].get('nivel', 0))
            if nivel not in nodes_by_level:
                nodes_by_level[nivel] = []
            nodes_by_level[nivel].append(n)
        
        for nivel, nodes in sorted(nodes_by_level.items()):
            level_color = color_for_level(base_color, nivel, max_nivel)
            
            x = [node_to_pos[n][0] for n in nodes]
            y = [node_to_pos[n][1] for n in nodes]
            z = [node_to_pos[n][2] for n in nodes]
            
            texts = []
            for n in nodes:
                sim_pi_str = ""
                if can_compute and n in node_embeddings:
                    sim_pi = cosine_sim(node_embeddings[n], pi_embedding)
                    sim_pi_str = f"simΠ={sim_pi:.3f}"
                texto = G.nodes[n]['texto'][:60]
                texts.append(f"L{nivel} | {sim_pi_str}<br>{texto}...")
            
            fig.add_trace(go.Scatter3d(
                x=x, y=y, z=z,
                mode='markers',
                marker=dict(size=2.5, color=level_color, opacity=0.4),
                text=texts,
                hoverinfo='text',
                name=f"{ACTOR_SHORT.get(actor, actor)} L{nivel}" if nivel == 0 else None,
                showlegend=(nivel == 0)
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
            size, hub_color, line_width = 7, 'gold', 1.5
            name_prefix = "★ HUB Óptimo"
        else:
            size, hub_color, line_width = 5, '#FF6B35', 1
            name_prefix = f"HUB #{hc['rank']+1}"
        
        for idx, node in enumerate(members):
            pos = node_to_pos[node]
            actor = G.nodes[node]['actor']
            actor_color = ACTOR_COLORS.get(actor, 'gray')
            dist = hc['distancias'].get(actor, 0)
            texto = G.nodes[node]['texto']
            nivel = G.nodes[node].get('nivel', '?')
            
            sim_pi_str = ""
            if can_compute and node in node_embeddings:
                sim_pi = cosine_sim(node_embeddings[node], pi_embedding)
                sim_pi_str = f"simΠ={sim_pi:.3f}"
            
            fig.add_trace(go.Scatter3d(
                x=[pos[0]], y=[pos[1]], z=[pos[2]],
                mode='markers',
                marker=dict(size=size, color=hub_color, symbol='circle',
                            line=dict(color=actor_color, width=line_width)),
                text=[f"<b>{name_prefix}</b> - {ACTOR_SHORT.get(actor, actor)}<br>"
                      f"L{nivel} | FTT={dist:.2f} | {sim_pi_str}<br>{texto}"],
                hoverinfo='text',
                name=f"{name_prefix} (FTT={hc['ftt_sum']:.2f})" if idx == 0 else None,
                showlegend=(idx == 0)
            ))
        
        # Conexiones entre miembros del HUB
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
                        line=dict(color=hub_color, width=1, dash='dot'),
                        opacity=0.35, showlegend=False,
                        text=[sim_str, sim_str] if sim_str else None,
                        hoverinfo='text' if sim_str else 'skip'
                    ))
    
    # =========================================================================
    # 6. RAÍCES
    # =========================================================================
    for root in roots:
        if root not in node_to_pos:
            continue
        pos = node_to_pos[root]
        actor = G.nodes[root]['actor']
        color = ACTOR_COLORS.get(actor, 'gray')
        
        sim_pi_str = ""
        if can_compute and root in node_embeddings:
            sim_pi = cosine_sim(node_embeddings[root], pi_embedding)
            sim_pi_str = f"simΠ={sim_pi:.3f}"
        
        fig.add_trace(go.Scatter3d(
            x=[pos[0]], y=[pos[1]], z=[pos[2]],
            mode='markers',
            marker=dict(size=7, color=color, symbol='diamond',
                        line=dict(color='white', width=1)),
            text=[f"<b>ROOT {ACTOR_SHORT.get(actor, '?')}</b><br>"
                  f"{sim_pi_str}<br>{G.nodes[root]['texto'][:80]}..."],
            hoverinfo='text',
            name=f"Root: {ACTOR_SHORT.get(actor, actor)}",
            showlegend=True
        ))
    
    # =========================================================================
    # 7. Π
    # =========================================================================
    pi_label = f"Π: {pi_text[:60]}..." if pi_text else "Π (problem)"
    
    fig.add_trace(go.Scatter3d(
        x=[pi_pos[0]], y=[pi_pos[1]], z=[pi_pos[2]],
        mode='markers+text',
        marker=dict(size=8, color='white', symbol='x',
                    line=dict(color='black', width=1.5)),
        text=['Π'],
        textposition='top center',
        textfont=dict(size=12, color='white'),
        hovertext=[pi_label],
        hoverinfo='text',
        name='Π (problem)'
    ))
    
    # =========================================================================
    # 8. EJES DE REFERENCIA
    # =========================================================================
    y_top = max_nivel * y_scale
    y_bottom = pi_pos[1]
    
    all_x = [node_to_pos[n][0] for n in node_to_pos]
    all_z = [node_to_pos[n][2] for n in node_to_pos]
    x_min, x_max = min(all_x) - 0.1, max(all_x) + 0.1
    z_min, z_max = min(all_z) - 0.1, max(all_z) + 0.1
    
    fig.add_trace(go.Scatter3d(
        x=[0, 0], y=[y_bottom, y_top], z=[0, 0],
        mode='lines',
        line=dict(color='rgba(255,255,255,0.3)', width=1.5, dash='dot'),
        showlegend=False, hoverinfo='skip'
    ))
    fig.add_trace(go.Scatter3d(
        x=[x_min, x_max], y=[y_bottom, y_bottom], z=[0, 0],
        mode='lines',
        line=dict(color='rgba(255,255,255,0.15)', width=1, dash='dash'),
        showlegend=False, hoverinfo='skip'
    ))
    fig.add_trace(go.Scatter3d(
        x=[0, 0], y=[y_bottom, y_bottom], z=[z_min, z_max],
        mode='lines',
        line=dict(color='rgba(255,255,255,0.15)', width=1, dash='dash'),
        showlegend=False, hoverinfo='skip'
    ))
    
    # Líneas Root → Π
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
            opacity=0.15,
            showlegend=False,
            hoverinfo='skip'
        ))
    
    # =========================================================================
    # 9. Etiquetas de nivel
    # =========================================================================
    for nivel in range(max_nivel + 1):
        y_val = (max_nivel - nivel) * y_scale
        fig.add_trace(go.Scatter3d(
            x=[0], y=[y_val], z=[0],
            mode='text',
            text=[f'L{nivel}'],
            textposition='middle left',
            textfont=dict(size=9, color='rgba(200,200,200,0.5)'),
            showlegend=False,
            hoverinfo='skip'
        ))
    
    # =========================================================================
    # LAYOUT
    # =========================================================================
    title_text = (
        f'Wind Map v6: ΔsimΠ ({emb_dim}d) — '
        f'Visual PCA ({pca_variance:.1%}) — '
        f'Principio de Naturalidad v8.5'
    )
    
    fig.update_layout(
        title=dict(
            text=title_text,
            font=dict(size=14, color='white')
        ),
        scene=dict(
            xaxis_title=f'PC1 (visual, {pca_variance:.0%})',
            yaxis_title='Level (tree depth ↓)',
            zaxis_title=f'PC2 (visual, {pca_variance:.0%})',
            bgcolor='rgb(15, 15, 25)',
            xaxis=dict(gridcolor='rgba(100,100,100,0.3)',
                       showbackground=True, backgroundcolor='rgb(20, 20, 35)'),
            yaxis=dict(gridcolor='rgba(100,100,100,0.3)',
                       showbackground=True, backgroundcolor='rgb(20, 20, 35)'),
            zaxis=dict(gridcolor='rgba(100,100,100,0.3)',
                       showbackground=True, backgroundcolor='rgb(20, 20, 35)'),
            aspectmode='data',
            camera=dict(
                eye=dict(x=1.8, y=0.8, z=1.5),
                center=dict(x=0, y=0, z=0),
                up=dict(x=0, y=1, z=0),
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
    parser = argparse.ArgumentParser(description='Mapa de Vientos 3D v6 — ΔsimΠ')
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
    print("  WIND MAP v6 — ΔsimΠ (768d) | θ_Π ELIMINADO (artefacto)")
    print("=" * 70)
    
    G, keys, embeddings, clases, hubs_data, input_data = load_data(run_path, input_path)
    
    emb_dim = embeddings.shape[1] if len(embeddings.shape) > 1 else 0
    print(f"\n  EMBEDDING SPACE: {emb_dim} dimensions")
    
    print("\n  PROCESSING Π")
    pi_embedding, pi_text = get_pi_embedding(input_data, keys, embeddings)
    
    print("\n  MAPPING EMBEDDINGS")
    node_embeddings = map_embeddings_to_nodes(G, keys, embeddings)
    
    print("\n  HYBRID PROJECTION (Π@0,0) — SOLO VISUALIZACIÓN")
    node_to_pos, roots, origin_3d, max_nivel, y_scale, pca_var = project_hybrid_3d(
        node_embeddings, G, pi_embedding)
    
    print(f"\n  ┌──────────────────────────────────────────────────────┐")
    print(f"  │  MÉTRICA PRINCIPAL: ΔsimΠ (cosSim en {emb_dim}d, 100%)    │")
    print(f"  │  ELIMINADO: θ_Π (artefacto dimensional demostrado)  │")
    print(f"  │  VISUAL: PCA ({pca_var:.1%} varianza) solo posiciones   │")
    print(f"  │  TOPOLOGÍA: Y = nivel real (100%)                    │")
    print(f"  └──────────────────────────────────────────────────────┘")
    
    print("\n  BUILDING WIND MAP")
    fig = create_wind_map(G, node_to_pos, roots, clases, hubs_data, origin_3d,
                          pi_text, node_embeddings, pi_embedding,
                          max_nivel, y_scale, pca_var)
    
    output_path = Path(args.output) if args.output else (run_path / 'wind_map_3d_v6.html')
    fig.write_html(str(output_path))
    print(f"\n✓ Saved: {output_path}")
    
    import webbrowser
    webbrowser.open(f'file://{output_path.absolute()}')
    
    return 0


if __name__ == '__main__':
    exit(main())