"""
Mapper Visualization for GERD Hydropolitical Conflict
======================================================
Generates a topological graph showing semantic structure between
Egypt, Ethiopia, and Sudan positions.

Requirements:
    pip install kmapper scikit-learn numpy umap-learn

Usage:
    python mapper_gerd.py

Output:
    - mapper_gerd.html (interactive visualization)
    - mapper_gerd.png (for paper)
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
# CONFIGURATION
# ============================================================================
BASE_PATH = Path(r"D:\principio-naturalidad\v8.6\principio_naturalidad\casos\gerd\runs\2026-02-05_15-29-23")
EMBEDDINGS_FILE = BASE_PATH / "embeddings_cache.npz"
TREES_PATH = BASE_PATH / "arboles"
OUTPUT_PATH = BASE_PATH / "tda"

# Mapper parameters
N_CUBES = 10
OVERLAP = 0.6
CLUSTERER = DBSCAN(eps=0.5, min_samples=2)

# Actor colors (3 actors)
ACTOR_COLORS = {
    'Egypt': '#e74c3c',      # Red
    'Ethiopia': '#2ecc71',   # Green
    'Sudan': '#3498db',      # Blue
}

# ============================================================================
# LOAD DATA
# ============================================================================
def extract_nodes_recursive(node, actor_name, nodes_list):
    """Recursively extract all nodes from tree structure."""
    nodes_list.append({
        'id': node['id'],
        'texto': node['texto'],
        'nivel': node['nivel'],
        'actor': actor_name,
        'es_hoja': node['es_hoja'],
    })
    for hijo in node.get('hijos', []):
        extract_nodes_recursive(hijo, actor_name, nodes_list)


def text_to_hash(text):
    """Convert text to MD5 hash (lowercase, same as cache)."""
    return hashlib.md5(text.lower().encode('utf-8')).hexdigest()


def get_short_actor_name(full_name):
    """Extract short name from full actor name."""
    if 'Egypt' in full_name:
        return 'Egypt'
    elif 'Ethiopia' in full_name:
        return 'Ethiopia'
    elif 'Sudan' in full_name:
        return 'Sudan'
    return full_name


def load_embeddings_and_labels():
    """Load embeddings from cache using tree JSONs for structure."""
    cache_data = np.load(EMBEDDINGS_FILE, allow_pickle=True)
    cache_keys = cache_data['keys']
    cache_embeddings = cache_data['embeddings']
    
    hash_to_embedding = {k: cache_embeddings[i] for i, k in enumerate(cache_keys)}
    print(f"Loaded cache with {len(cache_keys)} embeddings")
    
    all_nodes = []
    for tree_file in TREES_PATH.glob("*.json"):
        with open(tree_file, 'r', encoding='utf-8') as f:
            tree_data = json.load(f)
        
        actor_name = tree_data['actor']
        print(f"Loading tree: {actor_name} ({tree_data['total_nodos']} nodes)")
        extract_nodes_recursive(tree_data['raiz'], actor_name, all_nodes)
    
    print(f"\nTotal nodes extracted: {len(all_nodes)}")
    
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
    
    return np.array(embeddings_list), np.array(labels_list), texts_list, np.array(levels_list)


# ============================================================================
# MAPPER COMPUTATION
# ============================================================================
def compute_mapper(embeddings, labels, levels=None):
    """Compute Mapper graph using UMAP projection."""
    print("\n" + "="*60)
    print("COMPUTING MAPPER")
    print("="*60)
    
    mapper = km.KeplerMapper(verbose=2)
    
    print("\nStep 1: Computing UMAP projection...")
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
    
    print("\nStep 2: Building Mapper graph...")
    graph = mapper.map(
        projected,
        projected,
        cover=km.Cover(n_cubes=N_CUBES, perc_overlap=OVERLAP),
        clusterer=CLUSTERER
    )
    
    print(f"\nMapper graph statistics:")
    print(f"  Nodes: {len(graph['nodes'])}")
    print(f"  Edges: {len(graph['links'])}")
    
    return mapper, graph, projected


# ============================================================================
# VISUALIZATION
# ============================================================================
def create_interactive_html(mapper, graph, embeddings, labels, texts, output_file):
    """Create interactive HTML visualization."""
    print("\nGenerating interactive HTML...")
    
    # Color: Egypt=0, Ethiopia=0.5, Sudan=1
    color_values = []
    for l in labels:
        if 'Egypt' in str(l):
            color_values.append(0.0)
        elif 'Ethiopia' in str(l):
            color_values.append(0.5)
        else:  # Sudan
            color_values.append(1.0)
    color_values = np.array(color_values)
    
    # Tooltips
    tooltips = []
    for text, label in zip(texts, labels):
        short = get_short_actor_name(label)
        truncated = text[:100] + "..." if len(text) > 100 else text
        tooltips.append(f"[{short}] {truncated}")
    
    html = mapper.visualize(
        graph,
        path_html=str(output_file),
        title="Mapper: GERD Hydropolitical Conflict (Egypt-Ethiopia-Sudan)",
        color_values=color_values,
        color_function_name="Actor (Red=Egypt, Green=Ethiopia, Blue=Sudan)",
        custom_tooltips=np.array(tooltips),
    )
    
    print(f"  Saved to: {output_file}")
    return html


def create_static_figure(graph, labels, projected, output_file):
    """Create publication-quality static figure."""
    print("\nGenerating static figure for paper...")
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    
    # --- Panel A: UMAP projection ---
    ax1 = axes[0]
    for actor_key, color in ACTOR_COLORS.items():
        mask = np.array([actor_key in str(l) for l in labels])
        ax1.scatter(projected[mask, 0], projected[mask, 1],
                   c=color, alpha=0.6, s=20, label=actor_key)
    
    ax1.set_xlabel('UMAP 1', fontsize=12)
    ax1.set_ylabel('UMAP 2', fontsize=12)
    ax1.set_title('A) Semantic Space (UMAP Projection)', fontsize=14, fontweight='bold')
    ax1.legend(loc='best', fontsize=10)
    ax1.grid(True, alpha=0.3)
    
    # --- Panel B: Mapper graph ---
    ax2 = axes[1]
    G = nx.Graph()
    
    node_colors_list = []
    node_sizes = []
    
    for node_id, member_indices in graph['nodes'].items():
        member_labels = labels[member_indices]
        
        egypt_count = sum(1 for l in member_labels if 'Egypt' in str(l))
        ethiopia_count = sum(1 for l in member_labels if 'Ethiopia' in str(l))
        sudan_count = sum(1 for l in member_labels if 'Sudan' in str(l))
        total = len(member_labels)
        
        G.add_node(node_id)
        
        # RGB based on proportions
        r = egypt_count / total if total > 0 else 0
        g = ethiopia_count / total if total > 0 else 0
        b = sudan_count / total if total > 0 else 0
        
        node_colors_list.append((r, g, b))
        node_sizes.append(50 + total * 3)
    
    for source, targets in graph['links'].items():
        for target in targets:
            G.add_edge(source, target)
    
    pos = nx.spring_layout(G, k=2, iterations=100, seed=42)
    
    nx.draw_networkx_edges(G, pos, ax=ax2, alpha=0.4, edge_color='gray')
    nx.draw_networkx_nodes(G, pos, ax=ax2,
                          node_color=node_colors_list,
                          node_size=node_sizes,
                          alpha=0.8)
    
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor='#e74c3c', label='Egypt dominant'),
        Patch(facecolor='#2ecc71', label='Ethiopia dominant'),
        Patch(facecolor='#3498db', label='Sudan dominant'),
        Patch(facecolor='#7f8c8d', label='Mixed (convergence zones)'),
    ]
    ax2.legend(handles=legend_elements, loc='best', fontsize=9)
    
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
    print("MAPPER VISUALIZATION - GERD CASE")
    print("Hydropolitical Conflict: Egypt - Ethiopia - Sudan")
    print("="*60)
    
    OUTPUT_PATH.mkdir(parents=True, exist_ok=True)
    
    print("\nLoading embeddings and labels...")
    embeddings, labels, texts, levels = load_embeddings_and_labels()
    
    print(f"\nData loaded:")
    print(f"  Embeddings shape: {embeddings.shape}")
    print(f"  Unique actors: {np.unique(labels)}")
    print(f"  Samples per actor:")
    for label in np.unique(labels):
        count = np.sum(labels == label)
        short = get_short_actor_name(label)
        print(f"    {short}: {count}")
    print(f"  Level distribution: min={levels.min()}, max={levels.max()}")
    
    mapper, graph, projected = compute_mapper(embeddings, labels, levels)
    
    html_file = OUTPUT_PATH / "mapper_gerd.html"
    png_file = OUTPUT_PATH / "mapper_gerd.png"
    
    create_interactive_html(mapper, graph, embeddings, labels, texts, html_file)
    create_static_figure(graph, labels, projected, png_file)
    
    # Compute purity (for 3 actors)
    print("\n" + "="*60)
    print("ANALYSIS COMPLETE")
    print("="*60)
    print(f"\nFiles generated:")
    print(f"  Interactive: {html_file}")
    print(f"  Static (paper): {png_file}")
    
    node_purities = []
    for node_id, members in graph['nodes'].items():
        member_labels = labels[members]
        if len(member_labels) > 0:
            label_counts = {}
            for l in member_labels:
                short = get_short_actor_name(l)
                label_counts[short] = label_counts.get(short, 0) + 1
            max_count = max(label_counts.values())
            purity = max_count / len(member_labels)
            node_purities.append(purity)
    
    avg_purity = np.mean(node_purities) if node_purities else 0
    
    print(f"\nSeparation metrics:")
    print(f"  Average node purity: {avg_purity:.3f}")
    print(f"  (1.0 = perfect separation, 0.33 = random mixing for 3 actors)")
    
    if avg_purity > 0.70:
        print("\n  → STRONG TOPOLOGICAL SEPARATION")
        print("     Actors form distinct semantic clusters")
    elif avg_purity > 0.50:
        print("\n  → MODERATE TOPOLOGICAL SEPARATION")
        print("     Partial separation with overlap regions")
    else:
        print("\n  → WEAK TOPOLOGICAL SEPARATION")
        print("     Actors' positions are semantically intermingled")
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