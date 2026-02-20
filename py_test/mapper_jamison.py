"""
Mapper Visualization for Jamison Intrapersonal Conflict
========================================================
Generates a topological graph showing cognitive dissociation between
ManicState and DepressiveState as visually separated clusters.

Requirements:
    pip install kmapper scikit-learn numpy umap-learn sentence-transformers

Usage:
    python mapper_jamison.py

Output:
    - mapper_cognitive_dissociation.html (interactive visualization)
    - mapper_cognitive_dissociation.png (for paper)
"""

import numpy as np
import json
import hashlib
from pathlib import Path

# Mapper imports
import kmapper as km
from sklearn.cluster import DBSCAN
import umap

# For static image
import matplotlib.pyplot as plt
import networkx as nx

# ============================================================================
# CONFIGURATION - Adjust paths as needed
# ============================================================================
BASE_PATH = Path(r"D:\principio-naturalidad\v8.6\principio_naturalidad\casos\jamison\runs\2026-02-14_15-22-56")
EMBEDDINGS_FILE = BASE_PATH / "embeddings_cache.npz"
TREES_PATH = BASE_PATH / "arboles"
OUTPUT_PATH = BASE_PATH / "tda"

# Mapper parameters
N_CUBES = 10          # Fewer cubes = more points per cube
OVERLAP = 0.6         # More overlap = better connectivity
CLUSTERER = DBSCAN(eps=0.5, min_samples=2)  # More permissive clustering

# ============================================================================
# LOAD DATA
# ============================================================================
def extract_nodes_recursive(node, actor_name, nodes_list):
    """
    Recursively extract all nodes from tree structure.
    """
    nodes_list.append({
        'id': node['id'],
        'texto': node['texto'],
        'nivel': node['nivel'],
        'actor': actor_name,
        'es_hoja': node['es_hoja'],
        'sim_con_raiz': node.get('sim_con_raiz', 0),
        'sim_con_problema': node.get('sim_con_problema', 0)
    })
    
    for hijo in node.get('hijos', []):
        extract_nodes_recursive(hijo, actor_name, nodes_list)


def text_to_hash(text):
    """Convert text to MD5 hash (same as used in cache - lowercase)."""
    return hashlib.md5(text.lower().encode('utf-8')).hexdigest()


def load_embeddings_and_labels():
    """
    Load embeddings from cache using tree JSONs for structure.
    Returns embeddings array, actor labels, and texts.
    """
    # Load embeddings cache
    cache_data = np.load(EMBEDDINGS_FILE, allow_pickle=True)
    cache_keys = cache_data['keys']
    cache_embeddings = cache_data['embeddings']
    
    # Create hash -> embedding lookup
    hash_to_embedding = {k: cache_embeddings[i] for i, k in enumerate(cache_keys)}
    
    print(f"Loaded cache with {len(cache_keys)} embeddings")
    
    # Extract all nodes from both trees
    all_nodes = []
    
    for tree_file in TREES_PATH.glob("*.json"):
        with open(tree_file, 'r', encoding='utf-8') as f:
            tree_data = json.load(f)
        
        actor_name = tree_data['actor']
        print(f"Loading tree: {actor_name} ({tree_data['total_nodos']} nodes)")
        
        extract_nodes_recursive(tree_data['raiz'], actor_name, all_nodes)
    
    print(f"\nTotal nodes extracted: {len(all_nodes)}")
    
    # Match nodes to cached embeddings
    embeddings_list = []
    labels_list = []
    texts_list = []
    levels_list = []
    missing = 0
    
    for node in all_nodes:
        text_hash = text_to_hash(node['texto'])
        
        if text_hash in hash_to_embedding:
            embeddings_list.append(hash_to_embedding[text_hash])
            labels_list.append(node['actor'])
            texts_list.append(node['texto'])
            levels_list.append(node['nivel'])
        else:
            missing += 1
    
    if missing > 0:
        print(f"Warning: {missing} nodes not found in embedding cache")
    
    embeddings = np.array(embeddings_list)
    labels = np.array(labels_list)
    texts = texts_list
    levels = np.array(levels_list)
    
    return embeddings, labels, texts, levels


# ============================================================================
# MAPPER COMPUTATION
# ============================================================================
def compute_mapper(embeddings, labels, levels=None):
    """
    Compute Mapper graph using UMAP projection as filter function.
    """
    print("\n" + "="*60)
    print("COMPUTING MAPPER")
    print("="*60)
    
    # Initialize Mapper
    mapper = km.KeplerMapper(verbose=2)
    
    # Project to 2D using UMAP (filter function)
    print("\nStep 1: Computing UMAP projection (filter function)...")
    projected = mapper.fit_transform(
        embeddings,
        projection=umap.UMAP(
            n_components=2,
            n_neighbors=15,
            min_dist=0.1,
            metric='cosine',
            random_state=42
        )
    )
    
    # Create the Mapper graph
    print("\nStep 2: Building Mapper graph...")
    graph = mapper.map(
        projected,
        projected,  # Cluster in projected 2D space (not original 768D)
        cover=km.Cover(n_cubes=N_CUBES, perc_overlap=OVERLAP),
        clusterer=CLUSTERER
    )
    
    print(f"\nMapper graph statistics:")
    print(f"  Nodes: {len(graph['nodes'])}")
    print(f"  Edges: {len(graph['links'])}")
    
    return mapper, graph, projected


def compute_cluster_composition(graph, labels):
    """
    For each Mapper node, compute the proportion of each actor.
    """
    node_colors = {}
    node_compositions = {}
    
    unique_labels = np.unique(labels)
    
    for node_id, member_indices in graph['nodes'].items():
        member_labels = labels[member_indices]
        composition = {}
        for label in unique_labels:
            count = np.sum(member_labels == label)
            composition[label] = count / len(member_labels)
        node_compositions[node_id] = composition
        
        # Color based on dominant actor
        # ManicState = red, DepressiveState = blue, mixed = purple
        if 'Manic' in str(unique_labels):
            manic_key = [l for l in unique_labels if 'Manic' in str(l)][0]
            dep_key = [l for l in unique_labels if 'Depress' in str(l)][0]
        else:
            manic_key, dep_key = unique_labels[0], unique_labels[1]
        
        manic_ratio = composition.get(manic_key, 0)
        dep_ratio = composition.get(dep_key, 0)
        
        # RGB color interpolation
        r = manic_ratio
        b = dep_ratio
        g = 0.2 * min(manic_ratio, dep_ratio)  # Slight green for mixed
        
        node_colors[node_id] = f"rgb({int(r*255)}, {int(g*255)}, {int(b*255)})"
    
    return node_colors, node_compositions


# ============================================================================
# VISUALIZATION
# ============================================================================
def create_interactive_html(mapper, graph, embeddings, labels, texts, output_file):
    """
    Create interactive HTML visualization using KeplerMapper.
    """
    print("\nGenerating interactive HTML...")
    
    # Color values must be same size as original data points (not Mapper nodes)
    # Use 1 for ManicState, 0 for DepressiveState
    color_values = np.array([1.0 if 'Manic' in str(l) else 0.0 for l in labels])
    
    # Create tooltips with actual proposition texts (truncated)
    tooltips = []
    for i, (text, label) in enumerate(zip(texts, labels)):
        short_label = "Manic" if "Manic" in str(label) else "Depressive"
        truncated = text[:100] + "..." if len(text) > 100 else text
        tooltips.append(f"[{short_label}] {truncated}")
    
    html = mapper.visualize(
        graph,
        path_html=str(output_file),
        title="Mapper: Cognitive Dissociation in Bipolar Disorder (Jamison Case)",
        color_values=color_values,
        color_function_name="Cognitive State (Red=Manic, Blue=Depressive)",
        custom_tooltips=np.array(tooltips),
    )
    
    print(f"  Saved to: {output_file}")
    return html


def create_static_figure(graph, labels, projected, output_file):
    """
    Create publication-quality static figure.
    """
    print("\nGenerating static figure for paper...")
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    
    # --- Left panel: UMAP projection colored by actor ---
    ax1 = axes[0]
    unique_labels = np.unique(labels)
    colors = {'Manic': '#e74c3c', 'Depress': '#3498db'}  # Red, Blue
    
    for label in unique_labels:
        mask = labels == label
        color = colors.get('Manic', '#e74c3c') if 'Manic' in str(label) else colors.get('Depress', '#3498db')
        short_label = 'ManicState' if 'Manic' in str(label) else 'DepressiveState'
        ax1.scatter(projected[mask, 0], projected[mask, 1], 
                   c=color, alpha=0.6, s=20, label=short_label)
    
    ax1.set_xlabel('UMAP 1', fontsize=12)
    ax1.set_ylabel('UMAP 2', fontsize=12)
    ax1.set_title('A) Semantic Space (UMAP Projection)', fontsize=14, fontweight='bold')
    ax1.legend(loc='best', fontsize=10)
    ax1.grid(True, alpha=0.3)
    
    # --- Right panel: Mapper graph ---
    ax2 = axes[1]
    
    # Build NetworkX graph from Mapper output
    G = nx.Graph()
    
    # Add nodes
    node_colors_list = []
    node_sizes = []
    node_compositions, _ = {}, {}
    
    # Recompute compositions
    for node_id, member_indices in graph['nodes'].items():
        member_labels = labels[member_indices]
        manic_count = sum(1 for l in member_labels if 'Manic' in str(l))
        total = len(member_labels)
        manic_ratio = manic_count / total if total > 0 else 0.5
        
        G.add_node(node_id)
        
        # Color interpolation: red (manic) to blue (depressive)
        r = manic_ratio
        b = 1 - manic_ratio
        node_colors_list.append((r, 0.2, b))
        node_sizes.append(50 + total * 3)  # Size proportional to members
    
    # Add edges
    for source, targets in graph['links'].items():
        for target in targets:
            G.add_edge(source, target)
    
    # Layout
    pos = nx.spring_layout(G, k=2, iterations=100, seed=42)
    
    # Draw
    nx.draw_networkx_edges(G, pos, ax=ax2, alpha=0.4, edge_color='gray')
    nx.draw_networkx_nodes(G, pos, ax=ax2, 
                          node_color=node_colors_list,
                          node_size=node_sizes,
                          alpha=0.8)
    
    # Add legend manually
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor='#e74c3c', label='ManicState dominant'),
        Patch(facecolor='#9b59b6', label='Mixed'),
        Patch(facecolor='#3498db', label='DepressiveState dominant')
    ]
    ax2.legend(handles=legend_elements, loc='best', fontsize=10)
    
    ax2.set_title('B) Mapper Graph (Topological Skeleton)', fontsize=14, fontweight='bold')
    ax2.axis('off')
    
    plt.tight_layout()
    plt.savefig(output_file, dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()
    
    print(f"  Saved to: {output_file}")


# ============================================================================
# MAIN
# ============================================================================
def main():
    print("="*60)
    print("MAPPER VISUALIZATION - JAMISON CASE")
    print("Cognitive Dissociation in Bipolar I Disorder")
    print("="*60)
    
    # Ensure output directory exists
    OUTPUT_PATH.mkdir(parents=True, exist_ok=True)
    
    # Load data
    print("\nLoading embeddings and labels...")
    embeddings, labels, texts, levels = load_embeddings_and_labels()
    
    print(f"\nData loaded:")
    print(f"  Embeddings shape: {embeddings.shape}")
    print(f"  Unique actors: {np.unique(labels)}")
    print(f"  Samples per actor:")
    for label in np.unique(labels):
        count = np.sum(labels == label)
        print(f"    {label}: {count}")
    print(f"  Level distribution: min={levels.min()}, max={levels.max()}")
    
    # Compute Mapper
    mapper, graph, projected = compute_mapper(embeddings, labels, levels)
    
    # Generate visualizations
    html_file = OUTPUT_PATH / "mapper_cognitive_dissociation.html"
    png_file = OUTPUT_PATH / "mapper_cognitive_dissociation.png"
    
    create_interactive_html(mapper, graph, embeddings, labels, texts, html_file)
    create_static_figure(graph, labels, projected, png_file)
    
    # Summary statistics
    print("\n" + "="*60)
    print("ANALYSIS COMPLETE")
    print("="*60)
    print(f"\nFiles generated:")
    print(f"  Interactive: {html_file}")
    print(f"  Static (paper): {png_file}")
    
    # Compute separation metric
    node_purities = []
    for node_id, members in graph['nodes'].items():
        member_labels = labels[members]
        if len(member_labels) > 0:
            most_common = max(set(member_labels.tolist()), key=member_labels.tolist().count)
            purity = member_labels.tolist().count(most_common) / len(member_labels)
            node_purities.append(purity)
    
    avg_purity = np.mean(node_purities) if node_purities else 0
    
    print(f"\nSeparation metrics:")
    print(f"  Average node purity: {avg_purity:.3f}")
    print(f"  (1.0 = perfect separation, 0.5 = random mixing)")
    
    if avg_purity > 0.75:
        print("\n  → STRONG TOPOLOGICAL DISSOCIATION DETECTED")
        print("     ManicState and DepressiveState form distinct clusters")
    elif avg_purity > 0.6:
        print("\n  → MODERATE TOPOLOGICAL DISSOCIATION")
        print("     Partial separation with some overlap regions")
    else:
        print("\n  → WEAK TOPOLOGICAL DISSOCIATION")
        print("     States are intermingled in semantic space")
    dissociation_data = analyze_topological_dissociation(graph, labels, texts, OUTPUT_PATH)

# ============================================================================
# AÑADIR AL FINAL DEL main(), ANTES DEL if __name__
# ============================================================================

def analyze_topological_dissociation(graph, labels, texts, output_path):
    """
    Calcula β₀ y detecta bridges topológicos.
    ESTO ES LO QUE NECESITÁS PARA EL PAPER.
    """
    import networkx as nx
    
    print("\n" + "="*60)
    print("ANÁLISIS DE DISOCIACIÓN TOPOLÓGICA (H₀)")
    print("="*60)
    
    # Construir grafo NetworkX
    G = nx.Graph()
    
    # Agregar nodos con metadata
    node_data = {}
    for node_id, member_indices in graph['nodes'].items():
        member_labels = labels[member_indices]
        member_texts = [texts[i] for i in member_indices]
        
        manic_count = sum(1 for l in member_labels if 'Manic' in str(l))
        dep_count = len(member_labels) - manic_count
        total = len(member_labels)
        
        manic_ratio = manic_count / total if total > 0 else 0.5
        
        # Clasificar nodo
        if manic_ratio > 0.7:
            node_type = "ManicState"
        elif manic_ratio < 0.3:
            node_type = "DepressiveState"
        else:
            node_type = "Mixed"  # ESTOS SON LOS CANDIDATOS A BRIDGE/HUB
        
        G.add_node(node_id)
        node_data[node_id] = {
            'type': node_type,
            'manic_ratio': manic_ratio,
            'manic_count': manic_count,
            'depressive_count': dep_count,
            'total': total,
            'sample_texts': member_texts[:3]  # Primeros 3 textos como muestra
        }
    
    # Agregar edges
    for source, targets in graph['links'].items():
        for target in targets:
            G.add_edge(source, target)
    
    # ========================================
    # MÉTRICA CLAVE: β₀ (componentes conexos)
    # ========================================
    components = list(nx.connected_components(G))
    beta_0 = len(components)
    
    print(f"\n>>> β₀ (componentes conexos) = {beta_0}")
    
    # Analizar cada componente
    print(f"\nComponentes ({beta_0} total):")
    for i, comp in enumerate(components):
        comp_types = [node_data[n]['type'] for n in comp]
        manic_nodes = sum(1 for t in comp_types if t == 'ManicState')
        dep_nodes = sum(1 for t in comp_types if t == 'DepressiveState')
        mixed_nodes = sum(1 for t in comp_types if t == 'Mixed')
        
        print(f"\n  Componente {i+1}: {len(comp)} nodos")
        print(f"    ManicState: {manic_nodes}, DepressiveState: {dep_nodes}, Mixed: {mixed_nodes}")
        
        # Si es componente pequeño, mostrar textos
        if len(comp) <= 5:
            print(f"    Textos muestra:")
            for n in list(comp)[:3]:
                sample = node_data[n]['sample_texts'][0][:80] if node_data[n]['sample_texts'] else "N/A"
                print(f"      - [{node_data[n]['type']}] {sample}...")
    
    # ========================================
    # IDENTIFICAR BRIDGES TOPOLÓGICOS
    # ========================================
    print("\n" + "-"*60)
    print("BRIDGES TOPOLÓGICOS (nodos Mixed que conectan estados)")
    print("-"*60)
    
    mixed_nodes = [n for n, d in node_data.items() if d['type'] == 'Mixed']
    print(f"\nNodos Mixed encontrados: {len(mixed_nodes)}")
    
    bridges = []
    for node in mixed_nodes:
        # Verificar si conecta ManicState con DepressiveState
        neighbors = list(G.neighbors(node))
        neighbor_types = [node_data[n]['type'] for n in neighbors]
        
        connects_manic = any(t == 'ManicState' for t in neighbor_types)
        connects_dep = any(t == 'DepressiveState' for t in neighbor_types)
        
        if connects_manic and connects_dep:
            bridges.append(node)
            print(f"\n  BRIDGE: {node}")
            print(f"    Composición: {node_data[node]['manic_count']} manic, {node_data[node]['depressive_count']} dep")
            print(f"    Conecta a: {len([t for t in neighbor_types if t=='ManicState'])} manic, {len([t for t in neighbor_types if t=='DepressiveState'])} dep")
            print(f"    Texto muestra: {node_data[node]['sample_texts'][0][:100] if node_data[node]['sample_texts'] else 'N/A'}...")
    
    # ========================================
    # EXPERIMENTO: β₀ SIN BRIDGES
    # ========================================
    print("\n" + "-"*60)
    print("EXPERIMENTO: ¿Qué pasa si removemos los bridges?")
    print("-"*60)
    
    G_sin_bridges = G.copy()
    G_sin_bridges.remove_nodes_from(bridges)
    
    components_sin_bridges = list(nx.connected_components(G_sin_bridges))
    beta_0_sin_bridges = len(components_sin_bridges)
    
    print(f"\n  β₀ CON bridges:    {beta_0}")
    print(f"  β₀ SIN bridges:    {beta_0_sin_bridges}")
    print(f"  Δβ₀:               +{beta_0_sin_bridges - beta_0}")
    
    if beta_0_sin_bridges > beta_0:
        print(f"\n  >>> CONFIRMADO: Los bridges CAUSAN la integración topológica")
        print(f"  >>> Sin ellos, el sistema se DISOCIA en {beta_0_sin_bridges} componentes")
    
    # ========================================
    # EXPORTAR PARA PAPER
    # ========================================
    export_data = {
        'beta_0_con_bridges': beta_0,
        'beta_0_sin_bridges': beta_0_sin_bridges,
        'delta_beta_0': beta_0_sin_bridges - beta_0,
        'total_nodes': len(G.nodes()),
        'total_edges': len(G.edges()),
        'mixed_nodes_count': len(mixed_nodes),
        'bridges_count': len(bridges),
        'bridges': [
            {
                'node_id': b,
                'manic_ratio': node_data[b]['manic_ratio'],
                'sample_text': node_data[b]['sample_texts'][0] if node_data[b]['sample_texts'] else None
            }
            for b in bridges
        ],
        'components': [
            {
                'size': len(comp),
                'manic_nodes': sum(1 for n in comp if node_data[n]['type'] == 'ManicState'),
                'depressive_nodes': sum(1 for n in comp if node_data[n]['type'] == 'DepressiveState'),
                'mixed_nodes': sum(1 for n in comp if node_data[n]['type'] == 'Mixed')
            }
            for comp in components
        ]
    }
    
    export_file = output_path / "topological_dissociation_analysis.json"
    with open(export_file, 'w', encoding='utf-8') as f:
        json.dump(export_data, f, indent=2, ensure_ascii=False)
    
    print(f"\n  Exportado a: {export_file}")
    
    return export_data


# En main(), agregar antes del final:
# dissociation_data = analyze_topological_dissociation(graph, labels, texts, OUTPUT_PATH)

if __name__ == "__main__":
    main()