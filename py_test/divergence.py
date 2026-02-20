"""
Análisis de Divergencia — Principio de Naturalidad v9
======================================================
Complemento al Wind Map: analiza zonas de NO convergencia.

Genera:
  1. Heatmap de distancia semántica inter-actor
  2. Mapa de rechazos (admisión + redundancia)  
  3. Análisis de inversiones de agencia
  4. Distribución de ΔsimΠ por actor/nivel
  5. Clusters de divergencia temática
  6. Red de tensiones (pares de máxima distancia)

Uso:
    python divergence_analysis.py --data-path ./caso_gerd/

Requiere en data-path:
    - resultado.json
    - eventos.json  
    - fusiones.json
    - rechazos_inversion_roles.json (opcional)

Autor: Javier Gogol Merletti / Claude
Febrero 2026
"""

import argparse
import json
from pathlib import Path
from collections import defaultdict
from datetime import datetime

import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import plotly.express as px


# ============================================================================
# CONFIGURACIÓN
# ============================================================================

# Paleta de colores rotativa para cualquier número de actores
COLOR_ROTATION = [
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

# Diccionarios dinámicos - se llenan al cargar datos
ACTOR_COLORS = {}
ACTOR_SHORT = {}


def initialize_actor_mappings(actors):
    """Inicializa colores y nombres cortos para los actores detectados."""
    global ACTOR_COLORS, ACTOR_SHORT
    
    for i, actor in enumerate(sorted(actors)):
        # Asignar color rotativo
        ACTOR_COLORS[actor] = COLOR_ROTATION[i % len(COLOR_ROTATION)]
        
        # Generar nombre corto inteligente
        short = generate_short_name(actor)
        ACTOR_SHORT[actor] = short
        ACTOR_COLORS[short] = ACTOR_COLORS[actor]
    
    print(f"  Actores detectados: {list(ACTOR_SHORT.values())}")


def generate_short_name(actor):
    """Genera nombre corto para un actor."""
    # Eliminar palabras comunes
    common_words = [
        'Republic', 'Democratic', 'Federal', 'of', 'the', 'The',
        'Kingdom', 'State', 'United', 'People', 'Team', 'Group',
        'Arab', 'Islamic', 'Socialist', 'Union'
    ]
    
    words = actor.split()
    filtered = [w for w in words if w not in common_words]
    
    if filtered:
        # Tomar la última palabra significativa
        return filtered[-1]
    elif words:
        return words[-1]
    else:
        return actor[:10]


def get_short_name(actor):
    """Obtiene nombre corto de un actor."""
    if actor in ACTOR_SHORT:
        return ACTOR_SHORT[actor]
    # Fallback para actores no registrados
    return generate_short_name(actor)


def get_color(actor):
    """Obtiene color de un actor."""
    if actor in ACTOR_COLORS:
        return ACTOR_COLORS[actor]
    short = get_short_name(actor)
    if short in ACTOR_COLORS:
        return ACTOR_COLORS[short]
    return '#888888'


# ============================================================================
# CARGA DE DATOS
# ============================================================================

def load_all_data(data_path: Path):
    """Carga todos los archivos de datos del caso."""
    data = {}
    
    # resultado.json - estructura principal
    resultado_path = data_path / 'resultado.json'
    if resultado_path.exists():
        with open(resultado_path, 'r', encoding='utf-8') as f:
            data['resultado'] = json.load(f)
        print(f"✓ resultado.json: {data['resultado'].get('metrics', {}).get('total_nodes', '?')} nodos")
    
    # eventos.json - trazabilidad
    eventos_path = data_path / 'eventos.json'
    if eventos_path.exists():
        with open(eventos_path, 'r', encoding='utf-8') as f:
            data['eventos'] = json.load(f)
        print(f"✓ eventos.json: {len(data['eventos'].get('eventos', []))} eventos")
    
    # fusiones.json
    fusiones_path = data_path / 'fusiones.json'
    if fusiones_path.exists():
        with open(fusiones_path, 'r', encoding='utf-8') as f:
            data['fusiones'] = json.load(f)
        print(f"✓ fusiones.json: {data['fusiones'].get('total', 0)} fusiones")
    
    # rechazos_inversion_roles.json
    rechazos_path = data_path / 'rechazos_inversion_roles.json'
    if rechazos_path.exists():
        with open(rechazos_path, 'r', encoding='utf-8') as f:
            data['rechazos_rol'] = json.load(f)
        print(f"✓ rechazos_inversion_roles.json: {data['rechazos_rol'].get('total', 0)} rechazos")
    
    # hubs.json
    hubs_path = data_path / 'hubs.json'
    if hubs_path.exists():
        with open(hubs_path, 'r', encoding='utf-8') as f:
            data['hubs'] = json.load(f)
        print(f"✓ hubs.json: {len(data['hubs'].get('hubs', []))} HUBs")
    
    return data


# ============================================================================
# EXTRACCIÓN DE NODOS
# ============================================================================

def flatten_tree(node, actor, nodes_list):
    """Recursivamente extrae todos los nodos de un árbol."""
    node_data = {
        'id': node['id'],
        'texto': node['texto'],
        'actor': actor,
        'nivel': node['nivel'],
        'sim_con_raiz': node.get('sim_con_raiz', 0),
        'sim_con_problema': node.get('sim_con_problema', 0),
        'peso_arista': node.get('peso_arista', 0),
        'peso_acumulado': node.get('peso_acumulado', 0),
        'es_hoja': node.get('es_hoja', False),
    }
    nodes_list.append(node_data)
    
    for hijo in node.get('hijos', []):
        flatten_tree(hijo, actor, nodes_list)


def extract_all_nodes(data):
    """Extrae todos los nodos de todos los árboles."""
    nodes = []
    trees = data.get('resultado', {}).get('trees', {})
    
    for actor, tree_data in trees.items():
        raiz = tree_data.get('raiz')
        if raiz:
            flatten_tree(raiz, actor, nodes)
    
    # Inicializar mapeos de actores dinámicamente
    actors = set(n['actor'] for n in nodes)
    initialize_actor_mappings(actors)
    
    print(f"  Extraídos {len(nodes)} nodos de {len(trees)} actores")
    return nodes


def extract_events_by_type(data):
    """Agrupa eventos por tipo."""
    events_by_type = defaultdict(list)
    
    for evento in data.get('eventos', {}).get('eventos', []):
        events_by_type[evento['evento']].append(evento['datos'])
    
    return events_by_type


# ============================================================================
# ANÁLISIS 1: HEATMAP DE DISTANCIA INTER-ACTOR
# ============================================================================

def analyze_inter_actor_distance(nodes):
    """Calcula distancia semántica promedio entre actores."""
    
    # Agrupar nodos por actor
    nodes_by_actor = defaultdict(list)
    for n in nodes:
        nodes_by_actor[n['actor']].append(n)
    
    actors = sorted(nodes_by_actor.keys(), key=get_short_name)
    n_actors = len(actors)
    
    # Matriz de distancia basada en sim_con_problema
    # (aproximación: nodos con simΠ similar están cerca)
    distance_matrix = np.zeros((n_actors, n_actors))
    overlap_matrix = np.zeros((n_actors, n_actors))
    
    for i, actor_a in enumerate(actors):
        sims_a = [n['sim_con_problema'] for n in nodes_by_actor[actor_a]]
        
        for j, actor_b in enumerate(actors):
            sims_b = [n['sim_con_problema'] for n in nodes_by_actor[actor_b]]
            
            if i == j:
                # Varianza interna
                distance_matrix[i, j] = np.std(sims_a)
            else:
                # Distancia = diferencia promedio en posición respecto a Π
                mean_a, mean_b = np.mean(sims_a), np.mean(sims_b)
                std_a, std_b = np.std(sims_a), np.std(sims_b)
                
                # Overlap aproximado (cuánto se solapan las distribuciones)
                overlap = 1 - abs(mean_a - mean_b) / (std_a + std_b + 0.01)
                overlap_matrix[i, j] = max(0, overlap)
                
                # Distancia = |diferencia de medias|
                distance_matrix[i, j] = abs(mean_a - mean_b)
    
    return actors, distance_matrix, overlap_matrix, nodes_by_actor


def plot_inter_actor_heatmap(actors, distance_matrix, overlap_matrix):
    """Genera heatmap de distancia/overlap inter-actor."""
    
    short_names = [get_short_name(a) for a in actors]
    
    fig = make_subplots(
        rows=1, cols=2,
        subplot_titles=('Distancia Semántica (|ΔsimΠ|)', 'Overlap de Distribuciones'),
        horizontal_spacing=0.15
    )
    
    # Heatmap de distancia
    fig.add_trace(
        go.Heatmap(
            z=distance_matrix,
            x=short_names,
            y=short_names,
            colorscale='Reds',
            text=np.round(distance_matrix, 3),
            texttemplate='%{text}',
            textfont={"size": 12},
            hovertemplate='%{y} vs %{x}<br>Distancia: %{z:.4f}<extra></extra>',
            colorbar=dict(title='Distancia', x=0.45),
        ),
        row=1, col=1
    )
    
    # Heatmap de overlap
    fig.add_trace(
        go.Heatmap(
            z=overlap_matrix,
            x=short_names,
            y=short_names,
            colorscale='Greens',
            text=np.round(overlap_matrix, 3),
            texttemplate='%{text}',
            textfont={"size": 12},
            hovertemplate='%{y} vs %{x}<br>Overlap: %{z:.4f}<extra></extra>',
            colorbar=dict(title='Overlap', x=1.0),
        ),
        row=1, col=2
    )
    
    fig.update_layout(
        title='Análisis de Distancia Semántica Inter-Actor',
        height=500,
        width=1000,
    )
    
    return fig


# ============================================================================
# ANÁLISIS 2: MAPA DE RECHAZOS
# ============================================================================

def analyze_rejections(events_by_type):
    """Analiza nodos rechazados por admisión y redundancia."""
    
    rejections = {
        'admision': [],
        'redundancia': [],
    }
    
    # Rechazos por admisión (sim_con_problema < umbral)
    for ev in events_by_type.get('nodo_rechazado_admision', []):
        rejections['admision'].append({
            'actor': ev['actor'],
            'texto': ev['nodo'],
            'sim_con_problema': ev['sim_con_problema'],
            'umbral': ev['umbral'],
            'nivel': ev['nivel'],
            'razon': 'Baja relevancia al conflicto',
        })
    
    # Rechazos por redundancia
    for ev in events_by_type.get('nodo_podado_redundancia', []):
        rejections['redundancia'].append({
            'actor': ev['actor'],
            'texto': ev['nodo'],
            'similitud': ev['similitud'],
            'nivel': ev['nivel'],
            'razon': 'Redundante con nodo existente',
        })
    
    print(f"  Rechazos admisión: {len(rejections['admision'])}")
    print(f"  Rechazos redundancia: {len(rejections['redundancia'])}")
    
    return rejections


def plot_rejections_analysis(rejections):
    """Visualiza distribución de rechazos."""
    
    fig = make_subplots(
        rows=2, cols=2,
        subplot_titles=(
            'Rechazos por Admisión (por Actor)',
            'Rechazos por Redundancia (por Actor)',
            'Rechazos por Admisión (por Nivel)',
            'Distribución simΠ en Rechazos'
        ),
        specs=[[{"type": "bar"}, {"type": "bar"}],
               [{"type": "bar"}, {"type": "histogram"}]]
    )
    
    # Por actor - admisión
    admision_by_actor = defaultdict(int)
    for r in rejections['admision']:
        admision_by_actor[get_short_name(r['actor'])] += 1
    
    actors = list(admision_by_actor.keys())
    counts = list(admision_by_actor.values())
    colors = [get_color(a) for a in actors]
    
    fig.add_trace(
        go.Bar(x=actors, y=counts, marker_color=colors, name='Admisión'),
        row=1, col=1
    )
    
    # Por actor - redundancia
    redundancia_by_actor = defaultdict(int)
    for r in rejections['redundancia']:
        redundancia_by_actor[get_short_name(r['actor'])] += 1
    
    actors_r = list(redundancia_by_actor.keys())
    counts_r = list(redundancia_by_actor.values())
    colors_r = [get_color(a) for a in actors_r]
    
    fig.add_trace(
        go.Bar(x=actors_r, y=counts_r, marker_color=colors_r, name='Redundancia'),
        row=1, col=2
    )
    
    # Por nivel - admisión
    admision_by_nivel = defaultdict(int)
    for r in rejections['admision']:
        admision_by_nivel[f"L{r['nivel']}"] += 1
    
    niveles = sorted(admision_by_nivel.keys())
    counts_n = [admision_by_nivel[n] for n in niveles]
    
    fig.add_trace(
        go.Bar(x=niveles, y=counts_n, marker_color='#E63946', name='Por Nivel'),
        row=2, col=1
    )
    
    # Histograma de simΠ en rechazos
    sims = [r['sim_con_problema'] for r in rejections['admision']]
    
    fig.add_trace(
        go.Histogram(x=sims, nbinsx=20, marker_color='#E63946', name='simΠ'),
        row=2, col=2
    )
    
    fig.update_layout(
        title='Análisis de Nodos Rechazados — Zonas de Divergencia',
        height=700,
        width=1100,
        showlegend=False,
    )
    
    fig.add_vline(x=0.15, line_dash="dash", line_color="black", row=2, col=2,
                  annotation_text="τ_adm=0.15")
    
    return fig


# ============================================================================
# ANÁLISIS 3: INVERSIONES DE AGENCIA
# ============================================================================

def analyze_role_inversions(data):
    """Analiza patrones en rechazos por inversión de roles."""
    
    rechazos = data.get('rechazos_rol', {}).get('items', [])
    
    if not rechazos:
        return None
    
    # Extraer patrones
    patterns = {
        'por_par_actores': defaultdict(list),
        'temas': [],
    }
    
    for r in rechazos:
        par = tuple(sorted([get_short_name(r['actor_a']), get_short_name(r['actor_b'])]))
        patterns['por_par_actores'][par].append({
            'node_a': r['node_a_texto'],
            'node_b': r['node_b_texto'],
            'nli_score': r['nli_score'],
            'razon': r['rejection_reason'],
        })
    
    return patterns


def plot_role_inversions(data):
    """Visualiza inversiones de agencia."""
    
    rechazos = data.get('rechazos_rol', {}).get('items', [])
    
    if not rechazos:
        return None
    
    # Detectar actores dinámicamente
    actors_in_rejections = set()
    for r in rechazos:
        actors_in_rejections.add(get_short_name(r['actor_a']))
        actors_in_rejections.add(get_short_name(r['actor_b']))
    
    actors = sorted(actors_in_rejections)
    n_actors = len(actors)
    
    # Matriz de inversiones por par de actores
    matrix = np.zeros((n_actors, n_actors))
    
    for r in rechazos:
        a = get_short_name(r['actor_a'])
        b = get_short_name(r['actor_b'])
        if a in actors and b in actors:
            i, j = actors.index(a), actors.index(b)
            matrix[i, j] += 1
            matrix[j, i] += 1  # Simétrico
    
    # Crear figura
    fig = make_subplots(
        rows=1, cols=2,
        subplot_titles=('Inversiones de Agencia por Par', 'Detalle de Inversiones'),
        specs=[[{"type": "heatmap"}, {"type": "table"}]],
        column_widths=[0.4, 0.6]
    )
    
    # Heatmap
    fig.add_trace(
        go.Heatmap(
            z=matrix,
            x=actors,
            y=actors,
            colorscale='Oranges',
            text=matrix.astype(int),
            texttemplate='%{text}',
            textfont={"size": 14},
            hovertemplate='%{y} vs %{x}<br>Inversiones: %{z}<extra></extra>',
        ),
        row=1, col=1
    )
    
    # Tabla de detalle
    headers = ['Actor A', 'Actor B', 'NLI', 'Razón']
    cells = [
        [get_short_name(r['actor_a']) for r in rechazos],
        [get_short_name(r['actor_b']) for r in rechazos],
        [f"{r['nli_score']:.2f}" for r in rechazos],
        [r['rejection_reason'][:60] + '...' for r in rechazos],
    ]
    
    fig.add_trace(
        go.Table(
            header=dict(values=headers, fill_color='#2A9D8F', font=dict(color='white')),
            cells=dict(values=cells, fill_color='lavender', align='left'),
        ),
        row=1, col=2
    )
    
    fig.update_layout(
        title='Análisis de Inversiones de Agencia — Fusiones Rechazadas por V_ctx',
        height=400,
        width=1200,
    )
    
    return fig


# ============================================================================
# ANÁLISIS 4: DISTRIBUCIÓN DE ΔsimΠ
# ============================================================================

def compute_delta_sim_pi(nodes):
    """Calcula ΔsimΠ para cada nodo respecto a su padre implícito."""
    
    # Ordenar por actor y nivel para inferir padres
    nodes_by_actor = defaultdict(list)
    for n in nodes:
        nodes_by_actor[n['actor']].append(n)
    
    deltas = []
    
    for actor, actor_nodes in nodes_by_actor.items():
        # Ordenar por peso_acumulado para aproximar jerarquía
        sorted_nodes = sorted(actor_nodes, key=lambda x: x['peso_acumulado'])
        
        for i, node in enumerate(sorted_nodes):
            if node['nivel'] == 0:
                continue
            
            # Buscar padre (nodo de nivel anterior con menor peso_acumulado)
            parent_candidates = [n for n in sorted_nodes[:i] 
                                if n['nivel'] == node['nivel'] - 1]
            
            if parent_candidates:
                # Tomar el más cercano en peso_acumulado
                parent = min(parent_candidates, 
                           key=lambda p: abs(p['peso_acumulado'] - node['peso_acumulado'] + node['peso_arista']))
                
                delta = node['sim_con_problema'] - parent['sim_con_problema']
                deltas.append({
                    'actor': actor,
                    'nivel': node['nivel'],
                    'delta_sim_pi': delta,
                    'texto': node['texto'],
                    'sim_pi': node['sim_con_problema'],
                })
    
    return deltas


def plot_delta_distribution(deltas):
    """Visualiza distribución de ΔsimΠ."""
    
    fig = make_subplots(
        rows=2, cols=2,
        subplot_titles=(
            'Distribución Global de ΔsimΠ',
            'ΔsimΠ por Actor',
            'ΔsimΠ por Nivel',
            'Convergencia vs Divergencia por Actor'
        )
    )
    
    # Histograma global
    all_deltas = [d['delta_sim_pi'] for d in deltas]
    
    fig.add_trace(
        go.Histogram(x=all_deltas, nbinsx=50, marker_color='steelblue', name='ΔsimΠ'),
        row=1, col=1
    )
    fig.add_vline(x=0, line_dash="dash", line_color="red", row=1, col=1)
    
    # Box plot por actor
    actors = sorted(set(d['actor'] for d in deltas), key=get_short_name)
    
    for actor in actors:
        actor_deltas = [d['delta_sim_pi'] for d in deltas if d['actor'] == actor]
        fig.add_trace(
            go.Box(y=actor_deltas, name=get_short_name(actor), 
                   marker_color=get_color(actor), boxmean=True),
            row=1, col=2
        )
    
    # Por nivel
    niveles = sorted(set(d['nivel'] for d in deltas))
    nivel_means = []
    nivel_stds = []
    
    for nivel in niveles:
        nivel_deltas = [d['delta_sim_pi'] for d in deltas if d['nivel'] == nivel]
        nivel_means.append(np.mean(nivel_deltas))
        nivel_stds.append(np.std(nivel_deltas))
    
    fig.add_trace(
        go.Bar(
            x=[f'L{n}' for n in niveles],
            y=nivel_means,
            error_y=dict(type='data', array=nivel_stds),
            marker_color=['#2A9D8F' if m > 0 else '#E63946' for m in nivel_means],
            name='Media ΔsimΠ'
        ),
        row=2, col=1
    )
    fig.add_hline(y=0, line_dash="dash", line_color="black", row=2, col=1)
    
    # Proporción convergencia/divergencia por actor
    conv_div = defaultdict(lambda: {'converge': 0, 'diverge': 0})
    for d in deltas:
        short = get_short_name(d['actor'])
        if d['delta_sim_pi'] > 0:
            conv_div[short]['converge'] += 1
        else:
            conv_div[short]['diverge'] += 1
    
    actors_short = list(conv_div.keys())
    converge = [conv_div[a]['converge'] for a in actors_short]
    diverge = [conv_div[a]['diverge'] for a in actors_short]
    
    fig.add_trace(
        go.Bar(name='Converge (ΔsimΠ>0)', x=actors_short, y=converge, marker_color='#2A9D8F'),
        row=2, col=2
    )
    fig.add_trace(
        go.Bar(name='Diverge (ΔsimΠ<0)', x=actors_short, y=diverge, marker_color='#E63946'),
        row=2, col=2
    )
    
    fig.update_layout(
        title='Análisis de Convergencia/Divergencia Semántica (ΔsimΠ)',
        height=800,
        width=1200,
        barmode='group',
    )
    
    return fig


# ============================================================================
# ANÁLISIS 5: CLUSTERS DE DIVERGENCIA
# ============================================================================

def identify_divergence_clusters(nodes, deltas, threshold=-0.05):
    """Identifica clusters de nodos que divergen fuertemente."""
    
    # Nodos con fuerte divergencia
    divergent_nodes = []
    
    delta_by_id = {d['texto']: d for d in deltas}
    
    for n in nodes:
        if n['texto'] in delta_by_id:
            d = delta_by_id[n['texto']]
            if d['delta_sim_pi'] < threshold:
                divergent_nodes.append({
                    **n,
                    'delta_sim_pi': d['delta_sim_pi'],
                })
    
    # Agrupar por actor y nivel
    clusters = defaultdict(list)
    for dn in divergent_nodes:
        key = (get_short_name(dn['actor']), dn['nivel'])
        clusters[key].append(dn)
    
    return divergent_nodes, clusters


def plot_divergence_clusters(divergent_nodes, clusters):
    """Visualiza clusters de divergencia."""
    
    if not divergent_nodes:
        return None
    
    # Scatter de nodos divergentes
    fig = go.Figure()
    
    for actor in set(get_short_name(n['actor']) for n in divergent_nodes):
        actor_nodes = [n for n in divergent_nodes if get_short_name(n['actor']) == actor]
        
        fig.add_trace(go.Scatter(
            x=[n['nivel'] for n in actor_nodes],
            y=[n['delta_sim_pi'] for n in actor_nodes],
            mode='markers',
            marker=dict(
                size=10,
                color=get_color(actor),
                opacity=0.7,
            ),
            text=[f"{n['texto'][:50]}...<br>ΔsimΠ={n['delta_sim_pi']:.4f}" for n in actor_nodes],
            hoverinfo='text',
            name=actor,
        ))
    
    fig.add_hline(y=0, line_dash="dash", line_color="gray")
    fig.add_hline(y=-0.05, line_dash="dot", line_color="red", 
                  annotation_text="Umbral divergencia fuerte")
    
    fig.update_layout(
        title=f'Nodos con Divergencia Fuerte (ΔsimΠ < -0.05): {len(divergent_nodes)} nodos',
        xaxis_title='Nivel del Árbol',
        yaxis_title='ΔsimΠ',
        height=500,
        width=1000,
    )
    
    return fig


# ============================================================================
# ANÁLISIS 6: RED DE FUSIONES
# ============================================================================

def analyze_fusion_network(data):
    """Analiza la red de fusiones entre actores."""
    
    fusiones = data.get('fusiones', {}).get('fusiones', [])
    
    if not fusiones:
        return None
    
    # Contar fusiones por par de actores
    fusion_counts = defaultdict(int)
    fusion_sims = defaultdict(list)
    
    for f in fusiones:
        a = get_short_name(f['nodo_a_actor'])
        b = get_short_name(f['nodo_b_actor'])
        par = tuple(sorted([a, b]))
        fusion_counts[par] += 1
        fusion_sims[par].append(f['similitud'])
    
    return fusion_counts, fusion_sims


def plot_fusion_network(fusion_counts, fusion_sims):
    """Visualiza red de fusiones entre actores."""
    
    if not fusion_counts:
        return None
    
    # Detectar actores dinámicamente
    actors_set = set()
    for (a, b) in fusion_counts.keys():
        actors_set.add(a)
        actors_set.add(b)
    
    actors = sorted(actors_set)
    n_actors = len(actors)
    
    # Matriz de fusiones
    matrix = np.zeros((n_actors, n_actors))
    sim_matrix = np.zeros((n_actors, n_actors))
    
    for (a, b), count in fusion_counts.items():
        if a in actors and b in actors:
            i, j = actors.index(a), actors.index(b)
            matrix[i, j] = count
            matrix[j, i] = count
            avg_sim = np.mean(fusion_sims[(a, b)])
            sim_matrix[i, j] = avg_sim
            sim_matrix[j, i] = avg_sim
    
    fig = make_subplots(
        rows=1, cols=2,
        subplot_titles=('Cantidad de Fusiones', 'Similitud Promedio en Fusiones'),
    )
    
    fig.add_trace(
        go.Heatmap(
            z=matrix,
            x=actors,
            y=actors,
            colorscale='Blues',
            text=matrix.astype(int),
            texttemplate='%{text}',
            textfont={"size": 14},
        ),
        row=1, col=1
    )
    
    fig.add_trace(
        go.Heatmap(
            z=sim_matrix,
            x=actors,
            y=actors,
            colorscale='Greens',
            text=np.round(sim_matrix, 3),
            texttemplate='%{text}',
            textfont={"size": 12},
        ),
        row=1, col=2
    )
    
    fig.update_layout(
        title='Red de Fusiones Entre Actores',
        height=400,
        width=900,
    )
    
    return fig


# ============================================================================
# REPORTE TEXTUAL
# ============================================================================

def generate_text_report(data, nodes, rejections, deltas, divergent_nodes):
    """Genera reporte de texto con estadísticas clave."""
    
    report = []
    report.append("=" * 70)
    report.append("ANÁLISIS DE DIVERGENCIA — PRINCIPIO DE NATURALIDAD")
    report.append(f"Generado: {datetime.now().strftime('%Y-%m-%d %H:%M')}")
    report.append("=" * 70)
    
    # Estadísticas generales
    metrics = data.get('resultado', {}).get('metrics', {})
    report.append("\n1. ESTADÍSTICAS GENERALES")
    report.append("-" * 40)
    report.append(f"   Total nodos: {len(nodes)}")
    report.append(f"   Fusiones detectadas: {metrics.get('fusions_detected', '?')}")
    report.append(f"   Clases equivalencia: {metrics.get('equivalence_classes', '?')}")
    report.append(f"   HUBs globales: {metrics.get('hubs_found', '?')}")
    
    # Rechazos
    report.append("\n2. ANÁLISIS DE RECHAZOS (Zonas de No-Convergencia)")
    report.append("-" * 40)
    report.append(f"   Rechazos por admisión: {len(rejections['admision'])}")
    report.append(f"   Rechazos por redundancia: {len(rejections['redundancia'])}")
    
    fusion_stats = metrics.get('fusion_stats', {})
    report.append(f"   Contradicciones NLI: {fusion_stats.get('contradictions_detected', '?')}")
    report.append(f"   Rechazos inversión rol: {fusion_stats.get('fusions_rejected_role_inversion', '?')}")
    
    # ΔsimΠ
    report.append("\n3. CONVERGENCIA/DIVERGENCIA (ΔsimΠ)")
    report.append("-" * 40)
    
    if deltas:
        all_d = [d['delta_sim_pi'] for d in deltas]
        converge = sum(1 for d in all_d if d > 0)
        diverge = sum(1 for d in all_d if d < 0)
        
        report.append(f"   Total transiciones: {len(all_d)}")
        report.append(f"   Convergen (ΔsimΠ>0): {converge} ({100*converge/len(all_d):.1f}%)")
        report.append(f"   Divergen (ΔsimΠ<0): {diverge} ({100*diverge/len(all_d):.1f}%)")
        report.append(f"   ΔsimΠ promedio: {np.mean(all_d):.4f}")
        report.append(f"   ΔsimΠ std: {np.std(all_d):.4f}")
    
    # Divergencia fuerte
    report.append("\n4. DIVERGENCIA FUERTE (ΔsimΠ < -0.05)")
    report.append("-" * 40)
    report.append(f"   Nodos con divergencia fuerte: {len(divergent_nodes)}")
    
    if divergent_nodes:
        by_actor = defaultdict(int)
        for n in divergent_nodes:
            by_actor[get_short_name(n['actor'])] += 1
        
        for actor, count in sorted(by_actor.items()):
            report.append(f"     {actor}: {count}")
    
    # Top divergentes
    report.append("\n5. TOP 10 NODOS MÁS DIVERGENTES")
    report.append("-" * 40)
    
    if divergent_nodes:
        sorted_div = sorted(divergent_nodes, key=lambda x: x['delta_sim_pi'])[:10]
        for i, n in enumerate(sorted_div):
            report.append(f"   {i+1}. [{get_short_name(n['actor'])}] L{n['nivel']}: ΔsimΠ={n['delta_sim_pi']:.4f}")
            report.append(f"      \"{n['texto'][:70]}...\"")
    
    report.append("\n" + "=" * 70)
    
    return "\n".join(report)


# ============================================================================
# MAIN
# ============================================================================

def main():
    parser = argparse.ArgumentParser(description='Análisis de Divergencia - Principio de Naturalidad')
    parser.add_argument('--data-path', type=str, required=True,
                        help='Ruta a la carpeta con los archivos JSON')
    parser.add_argument('--output', type=str, default='divergence_report',
                        help='Prefijo para archivos de salida')
    
    args = parser.parse_args()
    data_path = Path(args.data_path)
    
    print("\n" + "=" * 60)
    print("ANÁLISIS DE DIVERGENCIA — PRINCIPIO DE NATURALIDAD")
    print("=" * 60 + "\n")
    
    # Cargar datos
    print("Cargando datos...")
    data = load_all_data(data_path)
    
    if not data.get('resultado'):
        print("ERROR: No se encontró resultado.json")
        return
    
    # Extraer nodos
    print("\nExtrayendo nodos...")
    nodes = extract_all_nodes(data)
    events_by_type = extract_events_by_type(data)
    
    # Análisis 1: Distancia inter-actor
    print("\nAnálisis 1: Distancia inter-actor...")
    actors, dist_matrix, overlap_matrix, nodes_by_actor = analyze_inter_actor_distance(nodes)
    fig1 = plot_inter_actor_heatmap(actors, dist_matrix, overlap_matrix)
    
    # Análisis 2: Rechazos
    print("\nAnálisis 2: Rechazos...")
    rejections = analyze_rejections(events_by_type)
    fig2 = plot_rejections_analysis(rejections)
    
    # Análisis 3: Inversiones de agencia
    print("\nAnálisis 3: Inversiones de agencia...")
    fig3 = plot_role_inversions(data)
    
    # Análisis 4: ΔsimΠ
    print("\nAnálisis 4: Distribución ΔsimΠ...")
    deltas = compute_delta_sim_pi(nodes)
    fig4 = plot_delta_distribution(deltas)
    
    # Análisis 5: Clusters divergencia
    print("\nAnálisis 5: Clusters de divergencia...")
    divergent_nodes, clusters = identify_divergence_clusters(nodes, deltas)
    fig5 = plot_divergence_clusters(divergent_nodes, clusters)
    
    # Análisis 6: Red de fusiones
    print("\nAnálisis 6: Red de fusiones...")
    fusion_data = analyze_fusion_network(data)
    fig6 = None
    if fusion_data:
        fusion_counts, fusion_sims = fusion_data
        fig6 = plot_fusion_network(fusion_counts, fusion_sims)
    
    # Generar reporte
    print("\nGenerando reporte...")
    report = generate_text_report(data, nodes, rejections, deltas, divergent_nodes)
    print("\n" + report)
    
    # Guardar outputs
    output_path = data_path / args.output
    output_path.mkdir(exist_ok=True)
    
    # Guardar HTML
    print(f"\nGuardando visualizaciones en {output_path}/...")
    
    fig1.write_html(str(output_path / 'inter_actor_distance.html'))
    fig2.write_html(str(output_path / 'rejections_analysis.html'))
    if fig3:
        fig3.write_html(str(output_path / 'role_inversions.html'))
    fig4.write_html(str(output_path / 'delta_sim_pi.html'))
    if fig5:
        fig5.write_html(str(output_path / 'divergence_clusters.html'))
    if fig6:
        fig6.write_html(str(output_path / 'fusion_network.html'))
    
    # Guardar reporte
    with open(output_path / 'divergence_report.txt', 'w', encoding='utf-8') as f:
        f.write(report)
    
    # Dashboard combinado
    print("\nGenerando dashboard combinado...")
    
    from plotly.subplots import make_subplots
    
    # Crear HTML combinado simple
    combined_html = f"""
    <!DOCTYPE html>
    <html>
    <head>
        <title>Análisis de Divergencia - Principio de Naturalidad</title>
        <style>
            body {{ font-family: Arial, sans-serif; margin: 20px; background: #1a1a2e; color: #eee; }}
            h1 {{ color: #2A9D8F; }}
            h2 {{ color: #E9C46A; border-bottom: 1px solid #E9C46A; padding-bottom: 5px; }}
            .section {{ margin: 30px 0; }}
            iframe {{ border: 1px solid #333; border-radius: 5px; }}
            pre {{ background: #0f0f23; padding: 15px; border-radius: 5px; overflow-x: auto; }}
        </style>
    </head>
    <body>
        <h1>🔍 Análisis de Divergencia — Principio de Naturalidad v9</h1>
        <p>Caso: {data.get('resultado', {}).get('case', 'N/A')}</p>
        
        <div class="section">
            <h2>1. Distancia Semántica Inter-Actor</h2>
            <iframe src="inter_actor_distance.html" width="100%" height="550" frameborder="0"></iframe>
        </div>
        
        <div class="section">
            <h2>2. Análisis de Rechazos</h2>
            <iframe src="rejections_analysis.html" width="100%" height="750" frameborder="0"></iframe>
        </div>
        
        <div class="section">
            <h2>3. Inversiones de Agencia</h2>
            <iframe src="role_inversions.html" width="100%" height="450" frameborder="0"></iframe>
        </div>
        
        <div class="section">
            <h2>4. Distribución ΔsimΠ (Convergencia/Divergencia)</h2>
            <iframe src="delta_sim_pi.html" width="100%" height="850" frameborder="0"></iframe>
        </div>
        
        <div class="section">
            <h2>5. Clusters de Divergencia</h2>
            <iframe src="divergence_clusters.html" width="100%" height="550" frameborder="0"></iframe>
        </div>
        
        <div class="section">
            <h2>6. Red de Fusiones</h2>
            <iframe src="fusion_network.html" width="100%" height="450" frameborder="0"></iframe>
        </div>
        
        <div class="section">
            <h2>📊 Reporte Estadístico</h2>
            <pre>{report}</pre>
        </div>
    </body>
    </html>
    """
    
    with open(output_path / 'dashboard.html', 'w', encoding='utf-8') as f:
        f.write(combined_html)
    
    print(f"\n✓ Dashboard guardado en: {output_path / 'dashboard.html'}")
    print("\nArchivos generados:")
    for f in output_path.glob('*.html'):
        print(f"  - {f.name}")
    print(f"  - divergence_report.txt")


if __name__ == '__main__':
    main()