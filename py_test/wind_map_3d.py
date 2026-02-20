"""
Mapa de Vientos 3D v2 - Principio de Naturalidad v8.5
Con Π real y HUBs diferenciados

Requisitos:
    pip install networkx numpy scikit-learn plotly

Uso:
    python wind_map_3d_v2.py --run-path casos/gerd/runs/2026-02-04_11-13-40 --input-path casos/gerd/input.json

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

# Paleta de colores para actores (se asignan dinámicamente)
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
    """Genera colores dinámicamente para cada actor en el grafo."""
    actors = sorted(set(G.nodes[n]['actor'] for n in G.nodes()))
    colors = {}
    for i, actor in enumerate(actors):
        colors[actor] = COLOR_PALETTE[i % len(COLOR_PALETTE)]
    return colors


def get_actor_short_names(actors):
    """Genera nombres cortos para actores."""
    short = {}
    for actor in actors:
        # Intentar extraer última palabra significativa
        words = actor.replace('Republic of', '').replace('Federal Democratic', '').strip().split()
        if words:
            short[actor] = words[-1]
        else:
            short[actor] = actor[:10]
    return short


# ============================================================================
# CARGA DE DATOS
# ============================================================================

def load_data(run_path: Path, input_path: Path = None):
    """Carga todos los archivos necesarios."""
    print(f"Loading data from: {run_path}")
    
    # Grafo
    G = nx.read_graphml(str(run_path / 'grafo.graphml'))
    print(f"  Graph: {G.number_of_nodes()} nodes, {G.number_of_edges()} edges")
    
    # Embeddings
    emb_data = np.load(str(run_path / 'embeddings_cache.npz'), allow_pickle=True)
    keys = list(emb_data['keys'])
    embeddings = emb_data['embeddings']
    print(f"  Embeddings: {len(keys)} vectors")
    
    # Clases de equivalencia
    with open(run_path / 'clases_equivalencia.json', 'r', encoding='utf-8') as f:
        clases = json.load(f)
    print(f"  Equivalence classes: {clases['total_classes']}")
    
    # HUBs
    hubs_path = run_path / 'hubs.json'
    hubs_data = None
    if hubs_path.exists():
        with open(hubs_path, 'r', encoding='utf-8') as f:
            hubs_data = json.load(f)
        print(f"  HUBs: {len(hubs_data.get('hubs', []))}")
    
    # Input (para Π)
    input_data = None
    if input_path and input_path.exists():
        with open(input_path, 'r', encoding='utf-8') as f:
            input_data = json.load(f)
        print(f"  Input: loaded (Π available)")
    
    return G, keys, embeddings, clases, hubs_data, input_data


def get_pi_embedding(input_data, keys, embeddings):
    """Obtiene el embedding real de Π."""
    if not input_data:
        return None, None
    
    # Texto de Π
    pi_text = input_data['problema'].get('decision_central_embedding') or \
              input_data['problema'].get('decision_central')
    
    if not pi_text:
        return None, None
    
    # Buscar en cache
    pi_hash = hashlib.md5(pi_text.lower().encode('utf-8')).hexdigest()
    
    if pi_hash in keys:
        idx = keys.index(pi_hash)
        print(f"  Π found in cache (hash: {pi_hash[:12]}...)")
        return embeddings[idx], pi_text
    else:
        print(f"  Warning: Π not in cache")
        return None, pi_text


def map_embeddings_to_nodes(G, keys, embeddings):
    """Mapea embeddings a nodos usando hash MD5."""
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
# PROYECCIÓN
# ============================================================================

def project_to_3d(node_embeddings, G, pi_embedding=None):
    """Proyecta a 3D centrado en Π (o centroide de raíces si no hay Π)."""
    
    roots = [n for n, d in G.nodes(data=True) if d['nivel'] == 0]
    
    # Determinar origen
    if pi_embedding is not None:
        origin = pi_embedding
        print(f"  Origin: Π (real embedding)")
    else:
        root_embs = np.array([node_embeddings[r] for r in roots])
        origin = root_embs.mean(axis=0)
        print(f"  Origin: centroid of roots (Π not available)")
    
    # Preparar matriz
    all_nodes = list(node_embeddings.keys())
    all_embs = np.array([node_embeddings[n] for n in all_nodes])
    all_embs_centered = all_embs - origin
    
    # Incluir origen en la proyección para consistencia
    matrix_with_origin = np.vstack([np.zeros(768), all_embs_centered])
    
    # PCA
    pca = PCA(n_components=3)
    coords = pca.fit_transform(matrix_with_origin)
    
    # Separar
    origin_3d = coords[0]  # Debería ser ~[0,0,0]
    node_coords = coords[1:]
    
    print(f"  PCA variance: {pca.explained_variance_ratio_.sum():.2%}")
    print(f"  Origin in 3D: [{origin_3d[0]:.4f}, {origin_3d[1]:.4f}, {origin_3d[2]:.4f}]")
    
    node_to_pos = {n: node_coords[i] for i, n in enumerate(all_nodes)}
    
    return node_to_pos, roots, origin_3d


# ============================================================================
# VISUALIZACIÓN
# ============================================================================

def create_wind_map(G, node_to_pos, roots, clases, hubs_data, origin_3d, pi_text=None):
    """Crea el mapa de vientos 3D con HUBs diferenciados."""
    
    # Generar colores y nombres dinámicamente
    ACTOR_COLORS = get_actor_colors(G)
    ACTOR_SHORT = get_actor_short_names(ACTOR_COLORS.keys())
    
    print(f"  Actors detected: {list(ACTOR_SHORT.values())}")
    
    # Identificar HUBs y sus miembros
    hub_classes = {}  # class_id -> {rank, members, is_optimal, texto_por_actor}
    
    if hubs_data and 'hubs' in hubs_data:
        for i, hub in enumerate(hubs_data['hubs']):
            # Soportar tanto 'class_id' como 'clase_id' (español)
            class_id = hub.get('class_id') or hub.get('clase_id')
            members = hub.get('clase_nodos', [])
            
            hub_classes[class_id] = {
                'rank': i,
                'ftt_sum': hub['ftt_sum'],
                'ftt_max': hub['ftt_max'],
                'members': set(members),
                'is_optimal': (i == 0),
                'texto_por_actor': hub.get('texto_por_actor', {}),
                'distancias': hub.get('distancias', {})
            }
    
    # Si no hay miembros en hubs_data, obtener desde clases de equivalencia
    for class_id, hc in hub_classes.items():
        if not hc['members'] and class_id in clases['classes']:
            hc['members'] = set(clases['classes'][class_id]['members'])
    
    # Todos los nodos que son parte de algún HUB
    all_hub_nodes = set()
    for hc in hub_classes.values():
        all_hub_nodes.update(hc['members'])
    
    print(f"  HUB classes: {len(hub_classes)}")
    print(f"  Total HUB nodes: {len(all_hub_nodes)}")
    
    fig = go.Figure()
    
    # -------------------------------------------------------------------------
    # CAMINOS DE CONVERGENCIA (datos reales: terminan en nodo miembro)
    # -------------------------------------------------------------------------
    if hubs_data and 'hubs' in hubs_data:
        for i, hub in enumerate(hubs_data['hubs']):
            caminos = hub.get('caminos', {})
            distancias = hub.get('distancias', {})
            is_optimal = (i == 0)
            
            if not is_optimal:
                continue  # Solo destacar el HUB óptimo con gradiente
            
            for actor, path in caminos.items():
                if len(path) < 2:
                    continue
                
                color = ACTOR_COLORS.get(actor, 'gray')
                total_steps = len(path) - 1
                
                # Dibujar cada arista del camino con grosor creciente
                for j in range(total_steps):
                    src_id = path[j]['id']
                    tgt_id = path[j + 1]['id']
                    
                    if src_id not in node_to_pos or tgt_id not in node_to_pos:
                        continue
                    
                    p1, p2 = node_to_pos[src_id], node_to_pos[tgt_id]
                    
                    # Gradiente: grosor aumenta hacia el HUB
                    progress = (j + 1) / total_steps
                    width = 2 + progress * 8  # De 2 a 10
                    opacity = 0.4 + progress * 0.5  # De 0.4 a 0.9
                    
                    is_last = (j == total_steps - 1)
                    
                    # Dibujar arista
                    fig.add_trace(go.Scatter3d(
                        x=[p1[0], p2[0]], y=[p1[1], p2[1]], z=[p1[2], p2[2]],
                        mode='lines',
                        line=dict(color=color, width=width),
                        opacity=opacity,
                        showlegend=(j == 0),
                        name=f"Path: {ACTOR_SHORT.get(actor, actor)} → HUB" if j == 0 else None,
                        hoverinfo='skip'
                    ))
                    
                    # Flecha en la última arista (apunta al nodo miembro REAL)
                    if is_last:
                        direction = np.array(p2) - np.array(p1)
                        norm = np.linalg.norm(direction)
                        if norm > 0:
                            direction = direction / norm * 0.06
                        
                        fig.add_trace(go.Cone(
                            x=[p2[0]], y=[p2[1]], z=[p2[2]],
                            u=[direction[0]], v=[direction[1]], w=[direction[2]],
                            colorscale=[[0, color], [1, color]],
                            sizemode='absolute',
                            sizeref=0.05,
                            showscale=False,
                            opacity=0.9,
                            showlegend=False,
                            hoverinfo='skip'
                        ))
                        
                        # Glow en última arista
                        fig.add_trace(go.Scatter3d(
                            x=[p1[0], p2[0]], y=[p1[1], p2[1]], z=[p1[2], p2[2]],
                            mode='lines',
                            line=dict(color=color, width=width + 6),
                            opacity=0.2,
                            showlegend=False,
                            hoverinfo='skip'
                        ))
        
        # Caminos del HUB secundario (más sutiles)
        for i, hub in enumerate(hubs_data['hubs']):
            if i == 0:
                continue
            
            caminos = hub.get('caminos', {})
            
            for actor, path in caminos.items():
                if len(path) < 2:
                    continue
                
                color = ACTOR_COLORS.get(actor, 'gray')
                
                path_x, path_y, path_z = [], [], []
                for j in range(len(path) - 1):
                    src_id = path[j]['id']
                    tgt_id = path[j + 1]['id']
                    
                    if src_id in node_to_pos and tgt_id in node_to_pos:
                        p1, p2 = node_to_pos[src_id], node_to_pos[tgt_id]
                        path_x.extend([p1[0], p2[0], None])
                        path_y.extend([p1[1], p2[1], None])
                        path_z.extend([p1[2], p2[2], None])
                
                if path_x:
                    fig.add_trace(go.Scatter3d(
                        x=path_x, y=path_y, z=path_z,
                        mode='lines',
                        line=dict(color=color, width=3, dash='dash'),
                        opacity=0.4,
                        showlegend=False,
                        hoverinfo='skip'
                    ))
    
    # -------------------------------------------------------------------------
    # ARISTAS (fondo, más tenues)
    # -------------------------------------------------------------------------
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
        
        fig.add_trace(go.Scatter3d(
            x=edge_x, y=edge_y, z=edge_z,
            mode='lines',
            line=dict(color=color, width=1),
            opacity=0.2,
            showlegend=False,
            hoverinfo='skip'
        ))
    
    # -------------------------------------------------------------------------
    # NODOS NORMALES (no HUB)
    # -------------------------------------------------------------------------
    for actor, color in ACTOR_COLORS.items():
        normal = [n for n in G.nodes() 
                  if G.nodes[n]['actor'] == actor 
                  and n in node_to_pos 
                  and n not in all_hub_nodes]
        
        if normal:
            x = [node_to_pos[n][0] for n in normal]
            y = [node_to_pos[n][1] for n in normal]
            z = [node_to_pos[n][2] for n in normal]
            texts = [f"L{G.nodes[n]['nivel']}: {G.nodes[n]['texto'][:60]}..." for n in normal]
            
            fig.add_trace(go.Scatter3d(
                x=x, y=y, z=z,
                mode='markers',
                marker=dict(size=2, color=color, opacity=0.4),
                text=texts,
                hoverinfo='text',
                name=ACTOR_SHORT.get(actor, actor)
            ))
    
    # -------------------------------------------------------------------------
    # HUBs: NODOS MIEMBROS REALES (sin centroide inventado)
    # -------------------------------------------------------------------------
    for class_id, hc in hub_classes.items():
        members = [n for n in hc['members'] if n in node_to_pos]
        if not members:
            continue
        
        is_optimal = hc['is_optimal']
        
        # Estilo diferenciado: óptimo vs secundario
        if is_optimal:
            size = 12
            hub_color = 'gold'
            line_width = 3
            name_prefix = "★ HUB Óptimo"
        else:
            size = 8
            hub_color = '#FF6B35'  # Naranja
            line_width = 2
            name_prefix = f"HUB #{hc['rank']+1}"
        
        # Dibujar cada nodo miembro (dato real)
        for idx, node in enumerate(members):
            pos = node_to_pos[node]
            actor = G.nodes[node]['actor']
            actor_color = ACTOR_COLORS.get(actor, 'gray')
            dist = hc['distancias'].get(actor, 0)
            texto = G.nodes[node]['texto']
            
            fig.add_trace(go.Scatter3d(
                x=[pos[0]], y=[pos[1]], z=[pos[2]],
                mode='markers',
                marker=dict(
                    size=size, 
                    color=hub_color,
                    symbol='circle',
                    line=dict(color=actor_color, width=line_width)
                ),
                text=[f"{name_prefix} - {ACTOR_SHORT.get(actor, actor)} (dist={dist:.2f}):<br>{texto}"],
                hoverinfo='text',
                name=f"{name_prefix} (FTT={hc['ftt_sum']:.2f})" if idx == 0 else None,
                showlegend=(idx == 0)
            ))
        
        # Conexiones entre miembros (equivalencia semántica - dato real de fusiones)
        if len(members) > 1:
            for i_m in range(len(members)):
                for j_m in range(i_m + 1, len(members)):
                    p1 = node_to_pos[members[i_m]]
                    p2 = node_to_pos[members[j_m]]
                    
                    fig.add_trace(go.Scatter3d(
                        x=[p1[0], p2[0]], y=[p1[1], p2[1]], z=[p1[2], p2[2]],
                        mode='lines',
                        line=dict(color=hub_color, width=1.5, dash='dot'),
                        opacity=0.4,
                        showlegend=False,
                        hoverinfo='skip'
                    ))
    
    # -------------------------------------------------------------------------
    # RAÍCES (todas en leyenda)
    # -------------------------------------------------------------------------
    for root in roots:
        if root not in node_to_pos:
            continue
        pos = node_to_pos[root]
        actor = G.nodes[root]['actor']
        color = ACTOR_COLORS.get(actor, 'gray')
        
        fig.add_trace(go.Scatter3d(
            x=[pos[0]], y=[pos[1]], z=[pos[2]],
            mode='markers',
            marker=dict(size=10, color=color, symbol='diamond',
                        line=dict(color='white', width=1.5)),
            text=[f"ROOT ({ACTOR_SHORT.get(actor, '?')}): {G.nodes[root]['texto'][:80]}..."],
            hoverinfo='text',
            name=f"Root: {ACTOR_SHORT.get(actor, actor)}",
            showlegend=True
        ))
    
    # -------------------------------------------------------------------------
    # ORIGEN Π
    # -------------------------------------------------------------------------
    pi_label = f"Π: {pi_text[:50]}..." if pi_text else "Π (problem)"
    
    fig.add_trace(go.Scatter3d(
        x=[origin_3d[0]], y=[origin_3d[1]], z=[origin_3d[2]],
        mode='markers+text',
        marker=dict(size=12, color='white', symbol='x',
                    line=dict(color='black', width=2)),
        text=['Π'],
        textposition='top center',
        textfont=dict(size=14, color='white'),
        hovertext=[pi_label],
        hoverinfo='text',
        name='Π (problem)'
    ))
    
    # -------------------------------------------------------------------------
    # LAYOUT
    # -------------------------------------------------------------------------
    fig.update_layout(
        title=dict(
            text='Wind Map: Semantic Convergence Field - Principio de Naturalidad v8.5',
            font=dict(size=18, color='white')
        ),
        scene=dict(
            xaxis_title='PC1',
            yaxis_title='PC2', 
            zaxis_title='PC3',
            bgcolor='rgb(15, 15, 25)',
            xaxis=dict(gridcolor='rgba(100,100,100,0.3)', 
                       showbackground=True, backgroundcolor='rgb(20, 20, 35)'),
            yaxis=dict(gridcolor='rgba(100,100,100,0.3)',
                       showbackground=True, backgroundcolor='rgb(20, 20, 35)'),
            zaxis=dict(gridcolor='rgba(100,100,100,0.3)',
                       showbackground=True, backgroundcolor='rgb(20, 20, 35)'),
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
    parser = argparse.ArgumentParser(description='Mapa de Vientos 3D v2')
    parser.add_argument('--run-path', '-r', type=str, required=True,
                        help='Path al directorio del run')
    parser.add_argument('--input-path', '-i', type=str, default=None,
                        help='Path al input.json (para Π real)')
    parser.add_argument('--output', '-o', type=str, default=None,
                        help='Path del HTML de salida')
    
    args = parser.parse_args()
    
    run_path = Path(args.run_path)
    input_path = Path(args.input_path) if args.input_path else None
    
    # Auto-detectar input.json si no se especifica
    if not input_path:
        # Buscar en el directorio padre del run
        possible = run_path.parent.parent / 'input.json'
        if possible.exists():
            input_path = possible
    
    # Cargar
    print("\n" + "="*70)
    print("  LOADING")
    print("="*70)
    G, keys, embeddings, clases, hubs_data, input_data = load_data(run_path, input_path)
    
    # Embedding de Π
    print("\n" + "="*70)
    print("  PROCESSING Π")
    print("="*70)
    pi_embedding, pi_text = get_pi_embedding(input_data, keys, embeddings)
    
    # Mapear
    print("\n" + "="*70)
    print("  MAPPING")
    print("="*70)
    node_embeddings = map_embeddings_to_nodes(G, keys, embeddings)
    
    # Proyectar
    print("\n" + "="*70)
    print("  PROJECTING")
    print("="*70)
    node_to_pos, roots, origin_3d = project_to_3d(node_embeddings, G, pi_embedding)
    
    # Visualizar
    print("\n" + "="*70)
    print("  VISUALIZING")
    print("="*70)
    fig = create_wind_map(G, node_to_pos, roots, clases, hubs_data, origin_3d, pi_text)
    
    # Guardar
    output_path = Path(args.output) if args.output else (run_path / 'wind_map_3d_v2.html')
    fig.write_html(str(output_path))
    print(f"\n✓ Saved: {output_path}")
    
    # Abrir
    import webbrowser
    webbrowser.open(f'file://{output_path.absolute()}')
    
    return 0


if __name__ == '__main__':
    exit(main())