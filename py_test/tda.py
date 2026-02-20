#!/usr/bin/env python3
"""
╔══════════════════════════════════════════════════════════════════╗
║  TDA Module — Topological Data Analysis for Principio de       ║
║  Naturalidad (PN)                                               ║
║                                                                  ║
║  Genérico: N actores, compara TODOS los pares.                 ║
║                                                                  ║
║  Computa persistencia homológica sobre embeddings semánticos    ║
║  en espacio ORIGINAL (768d, sin PCA). Sin pérdida de info.     ║
║                                                                  ║
║  Dependencias: ripser, persim, networkx, numpy, matplotlib,     ║
║                scikit-learn                                      ║
╚══════════════════════════════════════════════════════════════════╝

Uso:
    python tda.py --grafo grafo.graphml --embeddings embeddings_cache.npz

Salida:
    - tda_persistence_diagram.png    → Diagramas de persistencia por actor
    - tda_barcodes.png               → Barcodes comparativos
    - tda_betti_curves.png           → Curvas de Betti β₀ y β₁
    - tda_report.json                → Métricas topológicas numéricas
    - Consola: reporte legible
"""

import argparse
import hashlib
import json
import sys
from itertools import combinations
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import networkx as nx
from sklearn.metrics.pairwise import cosine_distances
from ripser import ripser
from persim import plot_diagrams, bottleneck, wasserstein


# ============================================================================
# 1. CARGA DE DATOS
# ============================================================================

def load_data(grafo_path, embeddings_path):
    """Carga grafo y embeddings, mapea embeddings a nodos."""
    print("=" * 65)
    print("  CARGA DE DATOS")
    print("=" * 65)
    
    G = nx.read_graphml(grafo_path)
    print(f"  Grafo: {G.number_of_nodes()} nodos, {G.number_of_edges()} aristas")
    
    data = np.load(embeddings_path, allow_pickle=True)
    keys = list(data['keys'])
    embeddings = data['embeddings']
    print(f"  Embeddings: {len(keys)} vectores × {embeddings.shape[1]}d")
    
    hash_to_emb = {k: embeddings[i] for i, k in enumerate(keys)}
    
    node_data = {}
    unmapped = 0
    for node_id, attrs in G.nodes(data=True):
        text = attrs.get('texto', '')
        actor = attrs.get('actor', 'unknown')
        nivel = int(attrs.get('nivel', 0))
        
        h = hashlib.md5(text.lower().encode('utf-8')).hexdigest()
        if h in hash_to_emb:
            node_data[node_id] = {
                'actor': actor,
                'nivel': nivel,
                'texto': text[:80],
                'embedding': hash_to_emb[h]
            }
        else:
            unmapped += 1
    
    print(f"  Mapeados: {len(node_data)} / {G.number_of_nodes()} nodos")
    if unmapped > 0:
        print(f"  Sin embedding: {unmapped} nodos")
    
    actors = sorted(set(d['actor'] for d in node_data.values()))
    print(f"  Actores ({len(actors)}): {actors}")
    
    return G, node_data, actors


def split_by_actor(node_data, actors):
    """Separa embeddings por actor."""
    actor_embeddings = {}
    actor_nodes = {}
    
    for actor in actors:
        nodes = [(nid, d) for nid, d in node_data.items() if d['actor'] == actor]
        if nodes:
            actor_embeddings[actor] = np.array([d['embedding'] for _, d in nodes])
            actor_nodes[actor] = [nid for nid, _ in nodes]
            print(f"  {actor}: {len(nodes)} nodos")
    
    return actor_embeddings, actor_nodes


# ============================================================================
# 2. PERSISTENT HOMOLOGY
# ============================================================================

def compute_persistence(embeddings_dict, max_dim=1, max_points=500):
    """Computa Vietoris-Rips persistent homology por actor."""
    print("\n" + "=" * 65)
    print("  PERSISTENT HOMOLOGY — Vietoris-Rips (distancia coseno, 768d)")
    print("=" * 65)
    
    results = {}
    
    for actor, embs in embeddings_dict.items():
        n = len(embs)
        
        if n > max_points:
            print(f"  {actor}: subsampling {n} → {max_points} puntos")
            idx = np.random.choice(n, max_points, replace=False)
            embs_use = embs[idx]
        else:
            embs_use = embs
        
        dist_matrix = cosine_distances(embs_use)
        
        print(f"  {actor}: computing Rips ({len(embs_use)} pts, dim≤{max_dim})...")
        print(f"    d_cos rango: [{dist_matrix.min():.4f}, {dist_matrix.max():.4f}], "
              f"media: {dist_matrix[np.triu_indices_from(dist_matrix, k=1)].mean():.4f}")
        
        result = ripser(dist_matrix, maxdim=max_dim, distance_matrix=True)
        results[actor] = result
        
        for dim, dgm in enumerate(result['dgms']):
            finite = dgm[dgm[:, 1] < np.inf]
            inf_count = np.sum(dgm[:, 1] == np.inf)
            if len(finite) > 0:
                lifetimes = finite[:, 1] - finite[:, 0]
                print(f"    H{dim}: {len(dgm)} features "
                      f"({inf_count} ∞, {len(finite)} finite)")
                print(f"      Lifetime — max: {lifetimes.max():.4f}, "
                      f"mean: {lifetimes.mean():.4f}, "
                      f"std: {lifetimes.std():.4f}")
                top_idx = np.argsort(lifetimes)[-5:][::-1]
                top_str = ", ".join(f"{lifetimes[i]:.4f}" for i in top_idx)
                print(f"      Top 5: [{top_str}]")
            else:
                print(f"    H{dim}: {inf_count} features (todas ∞)")
    
    return results


def compute_combined_persistence(embeddings_dict, max_dim=1, max_points=500):
    """Persistencia sobre la UNIÓN de todos los actores."""
    print("\n" + "-" * 65)
    print("  PERSISTENCIA COMBINADA (todos los actores unidos)")
    print("-" * 65)
    
    all_embs = np.vstack(list(embeddings_dict.values()))
    n = len(all_embs)
    
    if n > max_points:
        total = sum(len(e) for e in embeddings_dict.values())
        idx_list = []
        offset = 0
        for actor, embs in embeddings_dict.items():
            n_actor = int(max_points * len(embs) / total)
            idx = np.random.choice(len(embs), min(n_actor, len(embs)), replace=False)
            idx_list.extend(idx + offset)
            offset += len(embs)
        all_embs = all_embs[idx_list]
        print(f"  Subsampled: {n} → {len(all_embs)} puntos")
    
    dist_matrix = cosine_distances(all_embs)
    print(f"  Computing Rips ({len(all_embs)} pts, dim≤{max_dim})...")
    
    result = ripser(dist_matrix, maxdim=max_dim, distance_matrix=True)
    
    for dim, dgm in enumerate(result['dgms']):
        finite = dgm[dgm[:, 1] < np.inf]
        if len(finite) > 0:
            lifetimes = finite[:, 1] - finite[:, 0]
            print(f"  H{dim}: {len(dgm)} features, max lifetime: {lifetimes.max():.4f}")
    
    return result


# ============================================================================
# 3. MÉTRICAS COMPARATIVAS — TODOS LOS PARES
# ============================================================================

def compare_topology(results, actors):
    """Compara topología entre TODOS los pares de actores."""
    print("\n" + "=" * 65)
    print("  COMPARACIÓN TOPOLÓGICA — TODOS LOS PARES")
    print("=" * 65)
    
    if len(actors) < 2:
        print("  Se necesitan al menos 2 actores")
        return {}
    
    pairs = list(combinations(actors, 2))
    print(f"  Pares a comparar: {len(pairs)}")
    
    all_metrics = {}
    
    for a1, a2 in pairs:
        pair_key = f"{a1} vs {a2}"
        pair_metrics = {}
        
        print(f"\n  ── {pair_key} ──")
        
        n_dims = min(len(results[a1]['dgms']), len(results[a2]['dgms']))
        
        for dim in range(n_dims):
            dgm1 = results[a1]['dgms'][dim]
            dgm2 = results[a2]['dgms'][dim]
            
            fin1 = dgm1[dgm1[:, 1] < np.inf]
            fin2 = dgm2[dgm2[:, 1] < np.inf]
            
            if len(fin1) == 0 or len(fin2) == 0:
                print(f"    H{dim}: Sin features finitos")
                continue
            
            d_bottle = bottleneck(fin1, fin2)
            d_wasser = wasserstein(fin1, fin2)
            
            life1 = fin1[:, 1] - fin1[:, 0]
            life2 = fin2[:, 1] - fin2[:, 0]
            
            pair_metrics[f'H{dim}'] = {
                'bottleneck': float(d_bottle),
                'wasserstein': float(d_wasser),
                f'{a1}_features': int(len(dgm1)),
                f'{a2}_features': int(len(dgm2)),
                f'{a1}_max_lifetime': float(life1.max()),
                f'{a2}_max_lifetime': float(life2.max()),
                f'{a1}_mean_lifetime': float(life1.mean()),
                f'{a2}_mean_lifetime': float(life2.mean()),
                'mean_lifetime_diff': float(abs(life1.mean() - life2.mean())),
                'mean_lifetime_ratio': float(max(life1.mean(), life2.mean()) / 
                                             min(life1.mean(), life2.mean())) 
                                       if min(life1.mean(), life2.mean()) > 0 else 0,
            }
            
            print(f"    H{dim}:")
            print(f"      Bottleneck:  {d_bottle:.6f}")
            print(f"      Wasserstein: {d_wasser:.6f}")
            print(f"      Features: {a1}={len(dgm1)}, {a2}={len(dgm2)}")
            print(f"      Mean lifetime: {a1}={life1.mean():.4f}, {a2}={life2.mean():.4f}")
        
        all_metrics[pair_key] = pair_metrics
    
    return all_metrics


def compute_betti_curves(results, actors, n_steps=200):
    """Curvas de Betti para todos los actores."""
    curves = {}
    
    # Rango global (compartido para que sean comparables)
    global_max_death = 0
    for actor in actors:
        dgms = results[actor]['dgms']
        for d in dgms:
            finite_deaths = d[d[:, 1] < np.inf, 1]
            if len(finite_deaths) > 0:
                global_max_death = max(global_max_death, finite_deaths.max())
    
    eps_range = np.linspace(0, global_max_death * 1.1, n_steps)
    
    for actor in actors:
        dgms = results[actor]['dgms']
        actor_curves = {}
        for dim, dgm in enumerate(dgms):
            betti = np.zeros(n_steps)
            for birth, death in dgm:
                if death == np.inf:
                    death = eps_range[-1] + 1
                alive = (eps_range >= birth) & (eps_range < death)
                betti[alive] += 1
            actor_curves[f'H{dim}'] = betti
        curves[actor] = {'eps': eps_range, 'betti': actor_curves}
    
    return curves


# ============================================================================
# 4. INTERPRETACIÓN PN
# ============================================================================

def interpret_for_pn(all_metrics, results, actors, combined_result):
    """Interpreta resultados para todos los pares."""
    print("\n" + "=" * 65)
    print("  INTERPRETACIÓN PN — Topología del Conflicto")
    print("=" * 65)
    
    for pair_key, metrics in all_metrics.items():
        a1, a2 = pair_key.split(' vs ')
        print(f"\n  ━━━ {pair_key} ━━━")
        
        if 'H0' in metrics:
            m = metrics['H0']
            print(f"\n  ▶ H₀ — FRAGMENTACIÓN")
            print(f"    {a1}: {m[f'{a1}_features']} componentes, "
                  f"mean={m[f'{a1}_mean_lifetime']:.4f}")
            print(f"    {a2}: {m[f'{a2}_features']} componentes, "
                  f"mean={m[f'{a2}_mean_lifetime']:.4f}")
            print(f"    Bottleneck: {m['bottleneck']:.6f}")
        
        if 'H1' in metrics:
            m = metrics['H1']
            print(f"\n  ▶ H₁ — VACÍOS SEMÁNTICOS")
            print(f"    {a1}: {m[f'{a1}_features']} ciclos, "
                  f"mean={m[f'{a1}_mean_lifetime']:.4f}")
            print(f"    {a2}: {m[f'{a2}_features']} ciclos, "
                  f"mean={m[f'{a2}_mean_lifetime']:.4f}")
            print(f"    Ratio mean lifetime: {m['mean_lifetime_ratio']:.2f}x")
            print(f"    Wasserstein: {m['wasserstein']:.6f}")
            print(f"    Bottleneck:  {m['bottleneck']:.6f}")
    
    # Ranking de divergencia entre pares
    if len(all_metrics) > 1:
        print(f"\n  ▶ RANKING DE DIVERGENCIA TOPOLÓGICA (H₁ Wasserstein)")
        wasser_ranking = []
        for pair_key, metrics in all_metrics.items():
            if 'H1' in metrics:
                wasser_ranking.append((pair_key, metrics['H1']['wasserstein']))
        
        wasser_ranking.sort(key=lambda x: x[1], reverse=True)
        for i, (pair, w) in enumerate(wasser_ranking):
            print(f"    {i+1}. {pair}: {w:.6f}")
    
    # Espacio combinado
    print(f"\n  ▶ ESPACIO COMBINADO — Topología Global")
    for dim, dgm in enumerate(combined_result['dgms']):
        finite = dgm[dgm[:, 1] < np.inf]
        if len(finite) > 0:
            lifetimes = finite[:, 1] - finite[:, 0]
            print(f"    H{dim}: {len(dgm)} features, "
                  f"max lifetime: {lifetimes.max():.4f}")


# ============================================================================
# 5. VISUALIZACIONES
# ============================================================================

def plot_persistence_diagrams(results, actors, combined_result, output_dir, case_name=''):
    """Diagramas de persistencia — uno por actor + combinado."""
    n_plots = len(actors) + 1
    fig, axes = plt.subplots(1, n_plots, figsize=(5.5 * n_plots, 5))
    
    if n_plots == 1:
        axes = [axes]
    
    for i, actor in enumerate(actors):
        ax = axes[i]
        plot_diagrams(results[actor]['dgms'], ax=ax, show=False)
        ax.set_title(f'{actor}', fontsize=10, fontweight='bold')
        ax.set_xlabel('Birth (ε)', fontsize=9)
        ax.set_ylabel('Death (ε)', fontsize=9)
    
    ax = axes[-1]
    plot_diagrams(combined_result['dgms'], ax=ax, show=False)
    ax.set_title('COMBINADO', fontsize=10, fontweight='bold')
    ax.set_xlabel('Birth (ε)', fontsize=9)
    ax.set_ylabel('Death (ε)', fontsize=9)
    
    title = 'Diagramas de Persistencia\nDistancia coseno en 768d (sin pérdida dimensional)'
    if case_name:
        title = f'Diagramas de Persistencia — {case_name}\nDistancia coseno en 768d'
    fig.suptitle(title, fontsize=12, fontweight='bold')
    plt.tight_layout()
    
    path = Path(output_dir) / 'tda_persistence_diagram.png'
    fig.savefig(path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"  Saved: {path}")


def plot_barcodes(results, actors, output_dir, case_name=''):
    """Barcodes — N actores × 2 dimensiones."""
    n_actors = len(actors)
    fig, axes = plt.subplots(n_actors, 2, figsize=(14, 3.5 * n_actors),
                             gridspec_kw={'wspace': 0.3})
    
    if n_actors == 1:
        axes = axes.reshape(1, -1)
    
    colors_dim = {0: '#2196F3', 1: '#F44336'}
    
    for i, actor in enumerate(actors):
        for dim in range(min(2, len(results[actor]['dgms']))):
            ax = axes[i][dim]
            dgm = results[actor]['dgms'][dim]
            
            lifetimes = dgm[:, 1] - dgm[:, 0]
            max_finite = dgm[dgm[:, 1] < np.inf, 1].max() if np.any(dgm[:, 1] < np.inf) else 1
            order = np.argsort(lifetimes)[::-1][:50]
            
            for j, idx in enumerate(order):
                birth = dgm[idx, 0]
                death = dgm[idx, 1]
                if death == np.inf:
                    death = max_finite * 1.2
                    ax.barh(j, death - birth, left=birth,
                            color=colors_dim[dim], alpha=0.4,
                            edgecolor=colors_dim[dim], linewidth=0.5,
                            linestyle='--')
                else:
                    ax.barh(j, death - birth, left=birth,
                            color=colors_dim[dim], alpha=0.7,
                            edgecolor=colors_dim[dim], linewidth=0.5)
            
            ax.set_title(f'{actor} — H{dim}', fontsize=10, fontweight='bold')
            ax.set_xlabel('Radio ε (distancia coseno)', fontsize=8)
            ax.set_ylabel('Feature #', fontsize=8)
            ax.invert_yaxis()
    
    title = 'Barcodes Topológicos\nH₀: componentes conexas | H₁: ciclos/agujeros'
    if case_name:
        title = f'Barcodes Topológicos — {case_name}\nH₀: componentes | H₁: ciclos'
    fig.suptitle(title, fontsize=12, fontweight='bold')
    plt.tight_layout()
    
    path = Path(output_dir) / 'tda_barcodes.png'
    fig.savefig(path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"  Saved: {path}")


def plot_betti_curves(betti_data, actors, output_dir, case_name=''):
    """Curvas de Betti — N actores superpuestos."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    line_styles = ['-', '--', '-.', ':']
    actor_colors = ['#2196F3', '#F44336', '#4CAF50', '#FF9800', '#9C27B0']
    
    for dim_name, ax in zip(['H0', 'H1'], axes):
        for i, actor in enumerate(actors):
            if actor in betti_data and dim_name in betti_data[actor]['betti']:
                eps = betti_data[actor]['eps']
                beta = betti_data[actor]['betti'][dim_name]
                ax.plot(eps, beta,
                        color=actor_colors[i % len(actor_colors)],
                        linestyle=line_styles[i % len(line_styles)],
                        linewidth=2,
                        label=f'{actor}',
                        alpha=0.8)
        
        ax.set_title(f'Curva de Betti β{dim_name[-1]}(ε)', fontsize=12, fontweight='bold')
        ax.set_xlabel('Radio ε (distancia coseno)', fontsize=10)
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)
        
        if dim_name == 'H0':
            ax.set_ylabel('# Componentes conexas', fontsize=10)
        else:
            ax.set_ylabel('# Ciclos (agujeros)', fontsize=10)
    
    title = 'Curvas de Betti — Evolución topológica vs radio\ndistancia coseno 768d'
    if case_name:
        title = f'Curvas de Betti — {case_name}\ndistancia coseno 768d'
    fig.suptitle(title, fontsize=12, fontweight='bold')
    plt.tight_layout()
    
    path = Path(output_dir) / 'tda_betti_curves.png'
    fig.savefig(path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"  Saved: {path}")


# ============================================================================
# 6. REPORTE JSON
# ============================================================================

def save_report(all_metrics, results, actors, betti_data, output_dir, case_name=''):
    """Guarda reporte completo."""
    report = {
        'case': case_name or 'PN Case',
        'method': 'Vietoris-Rips Persistent Homology',
        'metric': 'Cosine distance in original 768d space (no PCA, no info loss)',
        'actors': actors,
        'n_actors': len(actors),
        'n_pairs': len(all_metrics),
        'pairwise_comparison': all_metrics,
        'per_actor': {},
    }
    
    for actor in actors:
        dgms = results[actor]['dgms']
        actor_report = {}
        
        for dim, dgm in enumerate(dgms):
            finite = dgm[dgm[:, 1] < np.inf]
            lifetimes = finite[:, 1] - finite[:, 0] if len(finite) > 0 else np.array([])
            
            actor_report[f'H{dim}'] = {
                'total_features': int(len(dgm)),
                'infinite_features': int(np.sum(dgm[:, 1] == np.inf)),
                'finite_features': int(len(finite)),
                'max_lifetime': float(lifetimes.max()) if len(lifetimes) > 0 else 0,
                'mean_lifetime': float(lifetimes.mean()) if len(lifetimes) > 0 else 0,
                'std_lifetime': float(lifetimes.std()) if len(lifetimes) > 0 else 0,
                'top_5_lifetimes': sorted(lifetimes.tolist(), reverse=True)[:5],
            }
        
        report['per_actor'][actor] = actor_report
    
    path = Path(output_dir) / 'tda_report.json'
    with open(path, 'w', encoding='utf-8') as f:
        json.dump(report, f, indent=2, ensure_ascii=False)
    print(f"  Saved: {path}")


# ============================================================================
# MAIN
# ============================================================================

def main():
    parser = argparse.ArgumentParser(
        description='TDA Module for Principio de Naturalidad — N actores')
    parser.add_argument('--grafo', required=True, help='Path to grafo.graphml')
    parser.add_argument('--embeddings', required=True, help='Path to embeddings_cache.npz')
    parser.add_argument('--output', default='.', help='Output directory')
    parser.add_argument('--case-name', default='', help='Case name for titles')
    parser.add_argument('--max-dim', type=int, default=1,
                        help='Max homological dimension (default: 1)')
    parser.add_argument('--max-points', type=int, default=500,
                        help='Max points per actor (default: 500)')
    parser.add_argument('--seed', type=int, default=42, help='Random seed')
    args = parser.parse_args()
    
    np.random.seed(args.seed)
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Detectar nombre del caso desde el path si no se provee
    case_name = args.case_name
    if not case_name:
        grafo_path = Path(args.grafo)
        for part in grafo_path.parts:
            if part.startswith('casos'):
                continue
            if part == 'runs' or part.startswith('20'):
                break
            case_name = part
    
    print("╔══════════════════════════════════════════════════════════════╗")
    print("║  TDA — Topological Data Analysis                           ║")
    print(f"║  Caso: {case_name:<52} ║")
    print(f"║  Actores: N (auto-detectados del grafo)                    ║")
    print("╚══════════════════════════════════════════════════════════════╝")
    print()
    
    # 1. Cargar
    G, node_data, actors = load_data(args.grafo, args.embeddings)
    actor_embeddings, actor_nodes = split_by_actor(node_data, actors)
    
    n_pairs = len(list(combinations(actors, 2)))
    print(f"\n  {len(actors)} actores → {n_pairs} pares a comparar")
    
    # 2. Persistencia por actor
    results = compute_persistence(actor_embeddings,
                                   max_dim=args.max_dim,
                                   max_points=args.max_points)
    
    # 3. Persistencia combinada
    combined = compute_combined_persistence(actor_embeddings,
                                             max_dim=args.max_dim,
                                             max_points=args.max_points)
    
    # 4. Comparación de todos los pares
    all_metrics = compare_topology(results, actors)
    
    # 5. Curvas de Betti
    betti_data = compute_betti_curves(results, actors)
    
    # 6. Interpretación
    interpret_for_pn(all_metrics, results, actors, combined)
    
    # 7. Visualizaciones
    print("\n" + "=" * 65)
    print("  GENERANDO VISUALIZACIONES")
    print("=" * 65)
    
    plot_persistence_diagrams(results, actors, combined, output_dir, case_name)
    plot_barcodes(results, actors, output_dir, case_name)
    plot_betti_curves(betti_data, actors, output_dir, case_name)
    
    # 8. Reporte
    save_report(all_metrics, results, actors, betti_data, output_dir, case_name)
    
    print("\n" + "=" * 65)
    print("  COMPLETADO")
    print("=" * 65)


if __name__ == '__main__':
    main()