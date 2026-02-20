"""
Diagnóstico: ¿El 99.7% →Π es señal real o artefacto dimensional?

Hipótesis a testear:
  H0: En 768d, cualquier paso padre→hijo tiene sesgo geométrico hacia Π
      (artefacto de la dimensionalidad + estructura del embedding space)
  H1: Hay convergencia semántica real hacia el problema central

Tests:
  1. Baseline aleatorio: barajar las aristas y recalcular θ_Π
  2. Baseline por actor cruzado: usar aristas de un actor con Π de otro
  3. Descomposición vectorial: ¿cuánto del paso es HACIA Π vs perpendicular?
  4. Tendencia por nivel: ¿θ disminuye (más convergencia) o aumenta?
  5. Magnitud del efecto: cos(θ) real vs cos(θ) baseline

Uso:
    python diagnostico_768d.py -r casos/gerd/runs/2026-02-04_11-13-40 -i casos/gerd/input.json

Autor: Javier Gogol Merletti / Claude
"""

import argparse
import hashlib
import json
from pathlib import Path
from collections import defaultdict

import networkx as nx
import numpy as np


# ============================================================================
# CARGA (copiado de wind_map)
# ============================================================================

def load_data(run_path, input_path=None):
    G = nx.read_graphml(str(run_path / 'grafo.graphml'))
    emb_data = np.load(str(run_path / 'embeddings_cache.npz'), allow_pickle=True)
    keys = list(emb_data['keys'])
    embeddings = emb_data['embeddings']
    
    input_data = None
    if input_path and input_path.exists():
        with open(input_path, 'r', encoding='utf-8') as f:
            input_data = json.load(f)
    
    return G, keys, embeddings, input_data


def get_pi_embedding(input_data, keys, embeddings):
    if not input_data:
        return None
    pi_text = input_data['problema'].get('decision_central_embedding') or \
              input_data['problema'].get('decision_central')
    if not pi_text:
        return None
    pi_hash = hashlib.md5(pi_text.lower().encode('utf-8')).hexdigest()
    if pi_hash in keys:
        return embeddings[keys.index(pi_hash)]
    return None


def map_embeddings(G, keys, embeddings):
    hash_to_emb = {k: embeddings[i] for i, k in enumerate(keys)}
    node_emb = {}
    for nid, attrs in G.nodes(data=True):
        h = hashlib.md5(attrs['texto'].lower().encode('utf-8')).hexdigest()
        if h in hash_to_emb:
            node_emb[nid] = hash_to_emb[h]
    return node_emb


def cosine_sim(a, b):
    dot = np.dot(a, b)
    na, nb = np.linalg.norm(a), np.linalg.norm(b)
    if na == 0 or nb == 0:
        return 0.0
    return float(np.clip(dot / (na * nb), -1, 1))


def angle_nd(v1, v2):
    n1, n2 = np.linalg.norm(v1), np.linalg.norm(v2)
    if n1 < 1e-10 or n2 < 1e-10:
        return 90.0
    cos_a = np.clip(np.dot(v1, v2) / (n1 * n2), -1, 1)
    return float(np.degrees(np.arccos(cos_a)))


# ============================================================================
# DIAGNÓSTICOS
# ============================================================================

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--run-path', '-r', type=str, required=True)
    parser.add_argument('--input-path', '-i', type=str, default=None)
    args = parser.parse_args()
    
    run_path = Path(args.run_path)
    input_path = Path(args.input_path) if args.input_path else None
    if not input_path:
        possible = run_path.parent.parent / 'input.json'
        if possible.exists():
            input_path = possible
    
    G, keys, embeddings, input_data = load_data(run_path, input_path)
    pi_emb = get_pi_embedding(input_data, keys, embeddings)
    node_emb = map_embeddings(G, keys, embeddings)
    
    dim = embeddings.shape[1]
    
    print(f"\n{'='*70}")
    print(f"  DIAGNÓSTICO: ¿99.7% →Π es real o artefacto?")
    print(f"  Espacio: {dim}d | Nodos: {G.number_of_nodes()} | Aristas: {G.number_of_edges()}")
    print(f"{'='*70}")
    
    # Recopilar aristas con embeddings
    edges = []
    for src, tgt in G.edges():
        if src in node_emb and tgt in node_emb:
            nivel_src = int(G.nodes[src].get('nivel', 0))
            actor = G.nodes[src]['actor']
            edges.append({
                'src': src, 'tgt': tgt,
                'src_emb': node_emb[src], 'tgt_emb': node_emb[tgt],
                'nivel': nivel_src, 'actor': actor,
            })
    
    print(f"  Aristas con embeddings: {len(edges)}")
    
    # ------------------------------------------------------------------
    # TEST 1: Distribución real de θ_Π
    # ------------------------------------------------------------------
    print(f"\n{'─'*70}")
    print(f"  TEST 1: Distribución real de θ_Π en {dim}d")
    print(f"{'─'*70}")
    
    thetas_real = []
    cos_thetas_real = []
    deltas_dist = []
    deltas_sim = []
    proj_toward = []  # componente hacia Π
    proj_perp = []    # componente perpendicular
    
    for e in edges:
        vec_edge = e['tgt_emb'] - e['src_emb']
        vec_to_pi = pi_emb - e['src_emb']
        
        theta = angle_nd(vec_edge, vec_to_pi)
        thetas_real.append(theta)
        cos_thetas_real.append(np.cos(np.radians(theta)))
        
        # Distancias y similitudes
        d_parent = np.linalg.norm(e['src_emb'] - pi_emb)
        d_child = np.linalg.norm(e['tgt_emb'] - pi_emb)
        deltas_dist.append(d_parent - d_child)
        
        sim_parent = cosine_sim(e['src_emb'], pi_emb)
        sim_child = cosine_sim(e['tgt_emb'], pi_emb)
        deltas_sim.append(sim_child - sim_parent)
        
        # Descomposición vectorial
        edge_norm = np.linalg.norm(vec_edge)
        if edge_norm > 0:
            proj_toward.append(edge_norm * np.cos(np.radians(theta)))
            proj_perp.append(edge_norm * np.sin(np.radians(theta)))
    
    thetas_real = np.array(thetas_real)
    cos_thetas_real = np.array(cos_thetas_real)
    
    pct_toward = 100 * np.sum(thetas_real < 90) / len(thetas_real)
    print(f"  →Π (θ<90°): {pct_toward:.1f}%")
    print(f"  θ̄ = {np.mean(thetas_real):.2f}° ± {np.std(thetas_real):.2f}°")
    print(f"  θ mediana = {np.median(thetas_real):.2f}°")
    print(f"  θ min/max = {np.min(thetas_real):.2f}° / {np.max(thetas_real):.2f}°")
    print(f"  cos(θ̄) = {np.mean(cos_thetas_real):.4f}")
    
    # Histograma en texto
    bins = [0, 15, 30, 45, 60, 75, 90, 105, 120, 135, 150, 165, 180]
    hist, _ = np.histogram(thetas_real, bins=bins)
    print(f"\n  Histograma θ_Π:")
    for i in range(len(hist)):
        bar = '█' * max(1, int(hist[i] / max(hist) * 40))
        pct = 100 * hist[i] / len(thetas_real)
        print(f"    {bins[i]:3d}°-{bins[i+1]:3d}°  {hist[i]:5d} ({pct:5.1f}%) {bar}")
    
    # ------------------------------------------------------------------
    # TEST 2: ¿Cuánto del paso es HACIA Π vs perpendicular?
    # ------------------------------------------------------------------
    print(f"\n{'─'*70}")
    print(f"  TEST 2: Descomposición vectorial")
    print(f"{'─'*70}")
    
    proj_toward = np.array(proj_toward)
    proj_perp = np.array(proj_perp)
    
    print(f"  Componente →Π:  media = {np.mean(proj_toward):.6f} ± {np.std(proj_toward):.6f}")
    print(f"  Componente ⊥Π:  media = {np.mean(proj_perp):.6f} ± {np.std(proj_perp):.6f}")
    print(f"  Ratio →Π/⊥Π:   {np.mean(proj_toward)/np.mean(proj_perp):.4f}")
    print(f"  (ratio=1 → 45°, ratio<1 → más perpendicular que convergente)")
    
    deltas_dist = np.array(deltas_dist)
    deltas_sim = np.array(deltas_sim)
    print(f"\n  Δd (eucl):    media = {np.mean(deltas_dist):+.6f}  {'ACERCA' if np.mean(deltas_dist) > 0 else 'ALEJA'}")
    print(f"  ΔsimΠ (cos):  media = {np.mean(deltas_sim):+.6f}  {'MÁS SIMILAR' if np.mean(deltas_sim) > 0 else 'MENOS SIMILAR'}")
    pct_acerca_d = 100 * np.sum(deltas_dist > 0) / len(deltas_dist)
    pct_acerca_s = 100 * np.sum(deltas_sim > 0) / len(deltas_sim)
    print(f"  % aristas que acercan (eucl): {pct_acerca_d:.1f}%")
    print(f"  % aristas que acercan (cos):  {pct_acerca_s:.1f}%")
    
    # ------------------------------------------------------------------
    # TEST 3: Baseline aleatorio — barajar destinos
    # ------------------------------------------------------------------
    print(f"\n{'─'*70}")
    print(f"  TEST 3: Baseline aleatorio (destinos barajados)")
    print(f"  Si θ<90% persiste → es artefacto geométrico, no convergencia")
    print(f"{'─'*70}")
    
    rng = np.random.default_rng(42)
    n_shuffles = 5
    
    tgt_embs = np.array([e['tgt_emb'] for e in edges])
    
    for trial in range(n_shuffles):
        # Barajar: cada padre recibe un hijo aleatorio
        idx_shuffle = rng.permutation(len(edges))
        thetas_shuf = []
        for i, e in enumerate(edges):
            vec_edge_shuf = tgt_embs[idx_shuffle[i]] - e['src_emb']
            vec_to_pi = pi_emb - e['src_emb']
            theta = angle_nd(vec_edge_shuf, vec_to_pi)
            thetas_shuf.append(theta)
        
        thetas_shuf = np.array(thetas_shuf)
        pct = 100 * np.sum(thetas_shuf < 90) / len(thetas_shuf)
        print(f"    Shuffle {trial+1}: →Π = {pct:.1f}%  θ̄ = {np.mean(thetas_shuf):.2f}°")
    
    # ------------------------------------------------------------------
    # TEST 4: Baseline con Π falso (vector aleatorio)
    # ------------------------------------------------------------------
    print(f"\n{'─'*70}")
    print(f"  TEST 4: Π falso (vector aleatorio en {dim}d)")
    print(f"  Si θ<90% persiste con Π falso → sesgo es del espacio, no de Π")
    print(f"{'─'*70}")
    
    for trial in range(3):
        pi_fake = rng.standard_normal(dim)
        pi_fake = pi_fake / np.linalg.norm(pi_fake)  # unitario
        # Escalar a magnitud similar a embeddings reales
        pi_fake = pi_fake * np.linalg.norm(pi_emb)
        
        thetas_fake = []
        for e in edges:
            vec_edge = e['tgt_emb'] - e['src_emb']
            vec_to_fake = pi_fake - e['src_emb']
            theta = angle_nd(vec_edge, vec_to_fake)
            thetas_fake.append(theta)
        
        thetas_fake = np.array(thetas_fake)
        pct = 100 * np.sum(thetas_fake < 90) / len(thetas_fake)
        print(f"    Π_random {trial+1}: →Π = {pct:.1f}%  θ̄ = {np.mean(thetas_fake):.2f}°")
    
    # ------------------------------------------------------------------
    # TEST 5: Baseline Π = centroide de todos los embeddings
    # ------------------------------------------------------------------
    print(f"\n{'─'*70}")
    print(f"  TEST 5: Π = centroide del espacio de embeddings")
    print(f"  Testa si el sesgo es hacia el centro del espacio en general")
    print(f"{'─'*70}")
    
    all_embs = np.array([node_emb[n] for n in node_emb])
    centroid = all_embs.mean(axis=0)
    
    sim_pi_centroid = cosine_sim(pi_emb, centroid)
    dist_pi_centroid = np.linalg.norm(pi_emb - centroid)
    print(f"  simCos(Π, centroide) = {sim_pi_centroid:.4f}")
    print(f"  dist(Π, centroide) = {dist_pi_centroid:.4f}")
    
    thetas_centroid = []
    for e in edges:
        vec_edge = e['tgt_emb'] - e['src_emb']
        vec_to_cent = centroid - e['src_emb']
        theta = angle_nd(vec_edge, vec_to_cent)
        thetas_centroid.append(theta)
    
    thetas_centroid = np.array(thetas_centroid)
    pct = 100 * np.sum(thetas_centroid < 90) / len(thetas_centroid)
    print(f"  →centroide (θ<90°): {pct:.1f}%  θ̄ = {np.mean(thetas_centroid):.2f}°")
    
    # ------------------------------------------------------------------
    # TEST 6: ¿Hay convergencia RELATIVA? Π real vs baseline
    # ------------------------------------------------------------------
    print(f"\n{'─'*70}")
    print(f"  TEST 6: Efecto neto — ¿Π real mejor que baselines?")
    print(f"{'─'*70}")
    
    # Π de otro actor como baseline
    actors = sorted(set(G.nodes[n]['actor'] for n in G.nodes()))
    roots = {G.nodes[n]['actor']: n for n in G.nodes() if int(G.nodes[n].get('nivel', 0)) == 0}
    
    for actor in actors:
        root_id = roots.get(actor)
        if not root_id or root_id not in node_emb:
            continue
        root_emb = node_emb[root_id]
        
        # θ hacia la raíz propia de cada actor
        actor_edges = [e for e in edges if e['actor'] == actor]
        thetas_root = []
        for e in actor_edges:
            vec_edge = e['tgt_emb'] - e['src_emb']
            vec_to_root = root_emb - e['src_emb']
            theta = angle_nd(vec_edge, vec_to_root)
            thetas_root.append(theta)
        
        thetas_root = np.array(thetas_root)
        pct_root = 100 * np.sum(thetas_root < 90) / max(len(thetas_root), 1)
        
        # θ hacia Π
        thetas_pi_actor = []
        for e in actor_edges:
            vec_edge = e['tgt_emb'] - e['src_emb']
            vec_to_pi = pi_emb - e['src_emb']
            theta = angle_nd(vec_edge, vec_to_pi)
            thetas_pi_actor.append(theta)
        
        thetas_pi_actor = np.array(thetas_pi_actor)
        pct_pi = 100 * np.sum(thetas_pi_actor < 90) / max(len(thetas_pi_actor), 1)
        
        short = actor.split()[-1]
        print(f"  {short:12s}  →Π: {pct_pi:.1f}% θ̄={np.mean(thetas_pi_actor):.1f}°  |  "
              f"→root propia: {pct_root:.1f}% θ̄={np.mean(thetas_root):.1f}°")
    
    # ------------------------------------------------------------------
    # TEST 7: Tendencia por nivel
    # ------------------------------------------------------------------
    print(f"\n{'─'*70}")
    print(f"  TEST 7: ¿θ_Π cambia con la profundidad?")
    print(f"{'─'*70}")
    
    by_level = defaultdict(list)
    by_level_dsim = defaultdict(list)
    for e in edges:
        vec_edge = e['tgt_emb'] - e['src_emb']
        vec_to_pi = pi_emb - e['src_emb']
        theta = angle_nd(vec_edge, vec_to_pi)
        by_level[e['nivel']].append(theta)
        
        dsim = cosine_sim(e['tgt_emb'], pi_emb) - cosine_sim(e['src_emb'], pi_emb)
        by_level_dsim[e['nivel']].append(dsim)
    
    for nivel in sorted(by_level.keys()):
        thetas = np.array(by_level[nivel])
        dsims = np.array(by_level_dsim[nivel])
        pct = 100 * np.sum(thetas < 90) / len(thetas)
        print(f"    L{nivel}→L{nivel+1}  θ̄={np.mean(thetas):.1f}°  →Π={pct:.1f}%  "
              f"Δ̄simΠ={np.mean(dsims):+.4f}  n={len(thetas)}")
    
    # ------------------------------------------------------------------
    # TEST 8: Magnitud de aristas — ¿los pasos son pequeños?
    # ------------------------------------------------------------------
    print(f"\n{'─'*70}")
    print(f"  TEST 8: Magnitud de aristas vs distancia a Π")
    print(f"{'─'*70}")
    
    edge_norms = []
    dist_to_pi = []
    for e in edges:
        en = np.linalg.norm(e['tgt_emb'] - e['src_emb'])
        dp = np.linalg.norm(e['src_emb'] - pi_emb)
        edge_norms.append(en)
        dist_to_pi.append(dp)
    
    edge_norms = np.array(edge_norms)
    dist_to_pi = np.array(dist_to_pi)
    ratio = edge_norms / dist_to_pi
    
    print(f"  ||arista||:    media = {np.mean(edge_norms):.4f} ± {np.std(edge_norms):.4f}")
    print(f"  ||padre→Π||:   media = {np.mean(dist_to_pi):.4f} ± {np.std(dist_to_pi):.4f}")
    print(f"  ratio ||a||/||→Π||: {np.mean(ratio):.4f}")
    print(f"  (si ratio << 1, los pasos son minúsculos vs la distancia a Π,")
    print(f"   y θ < 90° no implica convergencia práctica)")
    
    # ------------------------------------------------------------------
    # RESUMEN
    # ------------------------------------------------------------------
    print(f"\n{'='*70}")
    print(f"  RESUMEN DIAGNÓSTICO")
    print(f"{'='*70}")
    print(f"  θ̄ real →Π:         {np.mean(thetas_real):.2f}°")
    print(f"  cos(θ̄) real:        {np.mean(cos_thetas_real):.4f}")
    print(f"  Ratio →Π/⊥Π:       {np.mean(proj_toward)/np.mean(proj_perp):.4f}")
    print(f"  Δ̄d euclidiana:      {np.mean(deltas_dist):+.6f}  ({'ACERCA' if np.mean(deltas_dist) > 0 else 'ALEJA'})")
    print(f"  Δ̄simΠ coseno:       {np.mean(deltas_sim):+.6f}  ({'MÁS SIMILAR' if np.mean(deltas_sim) > 0 else 'MENOS SIMILAR'})")
    print(f"  % acerca (eucl):    {pct_acerca_d:.1f}%")
    print(f"  % acerca (cosSim):  {pct_acerca_s:.1f}%")
    print(f"  ||arista||/||→Π||:  {np.mean(ratio):.4f}")
    print(f"")
    print(f"  INTERPRETACIÓN:")
    print(f"  • Si baselines dan ~50% → el 99.7% es SEÑAL REAL")
    print(f"  • Si baselines dan >90% → es ARTEFACTO GEOMÉTRICO")
    print(f"  • Si Δd<0 y ΔsimΠ<0 → NO hay convergencia neta a Π")
    print(f"  • Si ratio <<1 → pasos demasiado pequeños para converger")
    print(f"{'='*70}")


if __name__ == '__main__':
    main()