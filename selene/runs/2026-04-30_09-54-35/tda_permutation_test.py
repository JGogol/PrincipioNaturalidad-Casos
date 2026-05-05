#!/usr/bin/env python3
"""
TDA Permutation Test — N actores, todos los pares.
"""

import argparse
import hashlib
import json
import sys
import time
from itertools import combinations
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import networkx as nx
from sklearn.metrics.pairwise import cosine_distances
from ripser import ripser
from persim import bottleneck, wasserstein


def load_and_split(grafo_path, embeddings_path):
    G = nx.read_graphml(grafo_path)
    data = np.load(embeddings_path, allow_pickle=True)
    keys = list(data['keys'])
    embeddings = data['embeddings']
    hash_to_emb = {k: embeddings[i] for i, k in enumerate(keys)}
    
    actor_embeddings = {}
    for node_id, attrs in G.nodes(data=True):
        text = attrs.get('texto', '')
        actor = attrs.get('actor', 'unknown')
        h = hashlib.md5(text.lower().encode('utf-8')).hexdigest()
        if h in hash_to_emb:
            if actor not in actor_embeddings:
                actor_embeddings[actor] = []
            actor_embeddings[actor].append(hash_to_emb[h])
    
    for actor in actor_embeddings:
        actor_embeddings[actor] = np.array(actor_embeddings[actor])
    
    actors = sorted(actor_embeddings.keys())
    print(f"  Grafo: {G.number_of_nodes()} nodos")
    for a in actors:
        print(f"  {a}: {len(actor_embeddings[a])} embeddings")
    return actor_embeddings, actors


def compute_pair_metrics(embs_a, embs_b, max_dim=1):
    results = {}
    for label, embs in [('A', embs_a), ('B', embs_b)]:
        dist_matrix = cosine_distances(embs)
        rips = ripser(dist_matrix, maxdim=max_dim, distance_matrix=True)
        results[label] = rips
    
    metrics = {}
    for dim in range(max_dim + 1):
        dgm_a = results['A']['dgms'][dim]
        dgm_b = results['B']['dgms'][dim]
        fin_a = dgm_a[dgm_a[:, 1] < np.inf]
        fin_b = dgm_b[dgm_b[:, 1] < np.inf]
        life_a = fin_a[:, 1] - fin_a[:, 0] if len(fin_a) > 0 else np.array([0])
        life_b = fin_b[:, 1] - fin_b[:, 0] if len(fin_b) > 0 else np.array([0])
        
        metrics[f'H{dim}_mean_lifetime_diff'] = abs(life_a.mean() - life_b.mean())
        metrics[f'H{dim}_max_lifetime_diff'] = abs(life_a.max() - life_b.max())
        metrics[f'H{dim}_feature_count_diff'] = abs(len(dgm_a) - len(dgm_b))
        metrics[f'H{dim}_std_lifetime_diff'] = abs(life_a.std() - life_b.std())
        
        if len(fin_a) > 0 and len(fin_b) > 0:
            metrics[f'H{dim}_wasserstein'] = wasserstein(fin_a, fin_b)
            metrics[f'H{dim}_bottleneck'] = bottleneck(fin_a, fin_b)
        else:
            metrics[f'H{dim}_wasserstein'] = 0.0
            metrics[f'H{dim}_bottleneck'] = 0.0
    return metrics


def run_pair_permutation(embs1, embs2, a1, a2, n_permutations=500,
                         max_points=300, max_dim=1, seed=42):
    np.random.seed(seed)
    n1, n2 = len(embs1), len(embs2)
    
    if n1 > max_points:
        embs1 = embs1[np.random.choice(n1, max_points, replace=False)]
        n1 = max_points
    if n2 > max_points:
        embs2 = embs2[np.random.choice(n2, max_points, replace=False)]
        n2 = max_points
    
    print(f"\n  -- {a1} vs {a2} --")
    print(f"  Puntos: {a1}={n1}, {a2}={n2}")
    
    pool = np.vstack([embs1, embs2])
    total = len(pool)
    
    print(f"  Calculando metricas REALES...")
    t0 = time.time()
    observed = compute_pair_metrics(embs1, embs2, max_dim=max_dim)
    t_real = time.time() - t0
    print(f"  -> {t_real:.1f}s")
    
    metric_names = list(observed.keys())
    null_distributions = {m: [] for m in metric_names}
    
    t_start = time.time()
    for i in range(n_permutations):
        perm = np.random.permutation(total)
        fake_a = pool[perm[:n1]]
        fake_b = pool[perm[n1:n1+n2]]
        try:
            perm_metrics = compute_pair_metrics(fake_a, fake_b, max_dim=max_dim)
            for m in metric_names:
                null_distributions[m].append(perm_metrics[m])
        except:
            continue
        
        if (i + 1) % 50 == 0 or i == 0:
            elapsed = time.time() - t_start
            rate = (i + 1) / elapsed
            remaining = (n_permutations - i - 1) / rate
            print(f"    [{i+1}/{n_permutations}] {elapsed:.0f}s, ~{remaining:.0f}s remaining")
    
    print(f"  Completado en {time.time() - t_start:.1f}s")
    
    results = {}
    for m in metric_names:
        null = np.array(null_distributions[m])
        obs = observed[m]
        p_value = float(np.mean(null >= obs))
        results[m] = {
            'observed': float(obs),
            'null_mean': float(null.mean()),
            'null_std': float(null.std()),
            'null_95th': float(np.percentile(null, 95)),
            'null_99th': float(np.percentile(null, 99)),
            'p_value': p_value,
            'significant_95': bool(p_value < 0.05),
            'significant_99': bool(p_value < 0.01),
            'effect_size_z': float((obs - null.mean()) / null.std()) if null.std() > 0 else 0,
        }
        sig = "***" if p_value < 0.01 else "** " if p_value < 0.05 else "   "
        print(f"  {sig} {m}: obs={obs:.6f}, p={p_value:.4f}, z={results[m]['effect_size_z']:.2f}")
    
    return observed, null_distributions, results


def plot_pair_distributions(observed, null_dists, results, pair_key, output_dir):
    key_metrics = [m for m in results.keys()
                   if 'H1' in m and ('mean_lifetime' in m or 'wasserstein' in m
                                      or 'bottleneck' in m or 'std_lifetime' in m)]
    key_metrics += [m for m in results.keys()
                    if 'H0' in m and 'mean_lifetime_diff' in m]
    
    n_plots = len(key_metrics)
    if n_plots == 0:
        return
    
    cols = min(3, n_plots)
    rows = (n_plots + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(6 * cols, 4.5 * rows))
    if n_plots == 1:
        axes = np.array([axes])
    axes = np.array(axes).flatten()
    
    for i, m in enumerate(key_metrics):
        ax = axes[i]
        null = np.array(null_dists[m])
        obs = observed[m]
        r = results[m]
        
        ax.hist(null, bins=40, color='#90CAF9', edgecolor='#42A5F5',
                alpha=0.7, density=True, label='Dist. nula')
        ax.axvline(obs, color='#F44336', linewidth=2.5, label=f'Obs={obs:.5f}')
        ax.axvline(np.percentile(null, 95), color='#FF9800', linewidth=1.5,
                   linestyle='--', label='95th')
        
        p = r['p_value']
        z = r['effect_size_z']
        sig_text = f'p={p:.4f} ***' if p < 0.01 else f'p={p:.4f} **' if p < 0.05 else f'p={p:.4f} n.s.'
        sig_color = '#F44336' if p < 0.01 else '#FF9800' if p < 0.05 else '#757575'
        
        ax.set_title(m.replace('_', ' '), fontsize=9, fontweight='bold')
        ax.text(0.98, 0.95, f'{sig_text}\nz={z:.2f}',
                transform=ax.transAxes, ha='right', va='top', fontsize=8,
                color=sig_color, fontweight='bold',
                bbox=dict(boxstyle='round,pad=0.3', facecolor='white',
                          alpha=0.8, edgecolor=sig_color))
        ax.legend(fontsize=6, loc='upper left')
        ax.set_xlabel('Valor', fontsize=8)
    
    for j in range(n_plots, len(axes)):
        axes[j].set_visible(False)
    
    fig.suptitle(f'Test de Permutacion -- {pair_key}', fontsize=12, fontweight='bold')
    plt.tight_layout()
    
    safe_name = pair_key.replace(' ', '_').replace('[', '').replace(']', '')
    path = Path(output_dir) / f'tda_perm_{safe_name}.png'
    fig.savefig(path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"  Saved: {path}")


def plot_global_summary(all_results, actors, output_dir):
    entries = []
    for pair_key, results in all_results.items():
        for m, r in results.items():
            if 'H1' in m and ('mean_lifetime' in m or 'wasserstein' in m or 'std_lifetime' in m):
                entries.append({'pair': pair_key, 'metric': m,
                               'z': r['effect_size_z'], 'p': r['p_value']})
    
    if not entries:
        return
    
    fig, ax = plt.subplots(figsize=(12, max(4, len(entries) * 0.6)))
    labels = [f"{e['pair']}\n{e['metric'].replace('_', ' ')}" for e in entries]
    z_scores = [e['z'] for e in entries]
    colors = ['#F44336' if e['p'] < 0.01 else '#FF9800' if e['p'] < 0.05
              else '#BDBDBD' for e in entries]
    
    ax.barh(range(len(entries)), z_scores, color=colors, height=0.6)
    ax.set_yticks(range(len(entries)))
    ax.set_yticklabels(labels, fontsize=7)
    ax.set_xlabel('Effect Size (z-score)', fontsize=11)
    ax.axvline(1.96, color='#FF9800', linestyle='--', alpha=0.5, label='z=1.96 (p=0.05)')
    ax.axvline(2.58, color='#F44336', linestyle='--', alpha=0.5, label='z=2.58 (p=0.01)')
    ax.axvline(0, color='black', linewidth=0.5)
    ax.legend(fontsize=8)
    ax.grid(axis='x', alpha=0.3)
    ax.set_title('Significancia Estadistica -- Todos los pares\nRojo: p<0.01 | Naranja: p<0.05 | Gris: n.s.',
                 fontsize=12, fontweight='bold')
    plt.tight_layout()
    
    path = Path(output_dir) / 'tda_permutation_summary.png'
    fig.savefig(path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"  Saved: {path}")


def save_report(all_results, actors, n_permutations, max_points, output_dir):
    pairs = list(combinations(actors, 2))
    report = {
        'test': 'Permutation Test for Topological Significance',
        'method': {
            'n_permutations': n_permutations,
            'max_points_per_actor': max_points,
            'metric_space': 'Cosine distance in 768d',
        },
        'actors': actors,
        'n_actors': len(actors),
        'n_pairs': len(pairs),
        'pairs': {},
        'summary': {},
    }
    
    for pair_key, results in all_results.items():
        pair_report = {}
        h1_sig = 0
        h1_total = 0
        for m, r in results.items():
            pair_report[m] = r
            if 'H1' in m:
                h1_total += 1
                if r['significant_95']:
                    h1_sig += 1
        
        report['pairs'][pair_key] = {
            'metrics': pair_report,
            'h1_significant': h1_sig,
            'h1_total': h1_total,
            'conclusion': (
                f'SIGNIFICATIVO ({h1_sig}/{h1_total} H1 p<0.05)'
                if h1_sig > h1_total / 2
                else f'Parcial ({h1_sig}/{h1_total})'
                if h1_sig > 0
                else 'No significativo'
            )
        }
    
    wasser_ranking = []
    for pair_key, results in all_results.items():
        if 'H1_wasserstein' in results:
            wasser_ranking.append({
                'pair': pair_key,
                'wasserstein_observed': results['H1_wasserstein']['observed'],
                'wasserstein_p': results['H1_wasserstein']['p_value'],
                'wasserstein_z': results['H1_wasserstein']['effect_size_z'],
            })
    wasser_ranking.sort(key=lambda x: x['wasserstein_observed'], reverse=True)
    report['summary']['divergence_ranking'] = wasser_ranking
    
    path = Path(output_dir) / 'tda_permutation_report.json'
    with open(path, 'w', encoding='utf-8') as f:
        json.dump(report, f, indent=2, ensure_ascii=False)
    print(f"  Saved: {path}")
    return report


def main():
    parser = argparse.ArgumentParser(description='TDA Permutation Test - N actores')
    parser.add_argument('--grafo', required=True)
    parser.add_argument('--embeddings', required=True)
    parser.add_argument('--output', default='.')
    parser.add_argument('--permutations', type=int, default=500)
    parser.add_argument('--max-points', type=int, default=300)
    parser.add_argument('--max-dim', type=int, default=1)
    parser.add_argument('--seed', type=int, default=42)
    args = parser.parse_args()
    
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print("=" * 65)
    print("  TDA Permutation Test -- N actores, todos los pares")
    print("=" * 65)
    
    print(f"\n  Cargando datos...")
    actor_embeddings, actors = load_and_split(args.grafo, args.embeddings)
    
    pairs = list(combinations(actors, 2))
    print(f"\n  {len(actors)} actores -> {len(pairs)} pares a testear")
    
    if len(actors) < 2:
        print("  ERROR: Se necesitan al menos 2 actores")
        sys.exit(1)
    
    all_observed, all_nulls, all_results = {}, {}, {}
    
    for pair_idx, (a1, a2) in enumerate(pairs):
        print(f"\n{'='*65}")
        print(f"  PAR {pair_idx+1}/{len(pairs)}: {a1} vs {a2}")
        print(f"{'='*65}")
        
        pair_seed = args.seed + pair_idx * 1000
        observed, nulls, results = run_pair_permutation(
            actor_embeddings[a1], actor_embeddings[a2], a1, a2,
            n_permutations=args.permutations, max_points=args.max_points,
            max_dim=args.max_dim, seed=pair_seed)
        
        pair_key = f"{a1} vs {a2}"
        all_observed[pair_key] = observed
        all_nulls[pair_key] = nulls
        all_results[pair_key] = results
    
    print(f"\n{'='*65}")
    print(f"  GENERANDO VISUALIZACIONES")
    print(f"{'='*65}")
    
    for pair_key in all_results:
        plot_pair_distributions(all_observed[pair_key], all_nulls[pair_key],
                               all_results[pair_key], pair_key, output_dir)
    
    plot_global_summary(all_results, actors, output_dir)
    report = save_report(all_results, actors, args.permutations, args.max_points, output_dir)
    
    print(f"\n{'='*65}")
    print(f"  RESUMEN FINAL")
    print(f"{'='*65}")
    for pair_key, pair_data in report['pairs'].items():
        print(f"  {pair_key}: {pair_data['conclusion']}")
    
    if report['summary'].get('divergence_ranking'):
        print(f"\n  Ranking divergencia (H1 Wasserstein):")
        for r in report['summary']['divergence_ranking']:
            print(f"    {r['pair']}: W={r['wasserstein_observed']:.4f}, p={r['wasserstein_p']:.4f}")
    
    print(f"\n  COMPLETADO")

if __name__ == '__main__':
    main()