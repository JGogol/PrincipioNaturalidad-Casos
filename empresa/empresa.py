#!/usr/bin/env python3
"""
Principio de Naturalidad - Ejemplo Ilustrativo: Estrategia Empresarial
=======================================================================

VERSIÓN CORREGIDA - Orden de etiquetas NLI correcto:
  idx 0 = contradiction
  idx 1 = entailment
  idx 2 = neutral

Ejecutar: python estrategia_fixed.py

Requisitos:
    pip install sentence-transformers torch numpy
"""

import numpy as np
from sentence_transformers import SentenceTransformer, CrossEncoder
import warnings
warnings.filterwarnings('ignore')

# =============================================================================
# CONFIGURACIÓN DE UMBRALES (del paper v8.6)
# =============================================================================
TAU_PRE = 0.40      # Umbral pre-filtro embedding
TAU_ENT = 0.55      # Umbral entailment bidireccional (promedio)
TAU_CONTR = 0.30    # Umbral veto por contradicción

# =============================================================================
# DEFINICIÓN DEL CONFLICTO
# =============================================================================

# Enunciado del problema (Π)
PI = "La empresa debe decidir su estrategia para el próximo año con recursos limitados."

# Actor A: Directivo que propone expansión
ACTOR_A_NAME = "Director A (expansión)"
ACTOR_A_ROOT = "La empresa debe expandirse a nuevos mercados."

# Actor B: Directivo que propone consolidación
ACTOR_B_NAME = "Director B (consolidación)"  
ACTOR_B_ROOT = "La empresa debe consolidar los mercados actuales."

# =============================================================================
# ÁRBOLES CAUSALES (4 niveles: L0 a L3)
# =============================================================================

TREE_A = {
    "L0": "La empresa debe expandirse a nuevos mercados.",
    "L1": "La empresa invierte recursos en captar nuevos clientes.",
    "L2": "Los ingresos por nuevos clientes aumentan.",
    "L3": "La empresa mejora su posición financiera."
}

TREE_B = {
    "L0": "La empresa debe consolidar los mercados actuales.",
    "L1": "La empresa invierte recursos en retener clientes actuales.",
    "L2": "Los ingresos por clientes recurrentes aumentan.",
    "L3": "La situación económica de la empresa se fortalece."
}

# =============================================================================
# FUNCIONES DE CÁLCULO
# =============================================================================

def cosine_similarity(v1, v2):
    """Calcula similitud coseno entre dos vectores."""
    dot = np.dot(v1, v2)
    norm1 = np.linalg.norm(v1)
    norm2 = np.linalg.norm(v2)
    return dot / (norm1 * norm2)

def calculate_edge_weight(node_embedding, root_embedding, pi_embedding):
    """
    Calcula el peso bicompuesto de arista (Definición 5.5 del paper):
    w(e) = (1 - sim(v, R)) + (1 - sim(v, Π))
    """
    sim_root = cosine_similarity(node_embedding, root_embedding)
    sim_pi = cosine_similarity(node_embedding, pi_embedding)
    weight = (1 - sim_root) + (1 - sim_pi)
    return weight, sim_root, sim_pi

def softmax(x):
    e_x = np.exp(x - np.max(x))
    return e_x / e_x.sum()

# =============================================================================
# EJECUCIÓN PRINCIPAL
# =============================================================================

def main():
    print("=" * 80)
    print("PRINCIPIO DE NATURALIDAD - EJEMPLO ESTRATEGIA EMPRESARIAL")
    print("VERSIÓN CORREGIDA - Orden NLI: idx0=CONTR, idx1=ENT, idx2=NEUT")
    print("=" * 80)
    
    # -------------------------------------------------------------------------
    # FASE 1: Cargar modelos
    # -------------------------------------------------------------------------
    print("\n[FASE 1] Cargando modelos...")
    
    print("  - Cargando modelo de embeddings (paraphrase-multilingual-mpnet-base-v2)...")
    embed_model = SentenceTransformer('paraphrase-multilingual-mpnet-base-v2')
    
    print("  - Cargando modelo NLI (cross-encoder/nli-deberta-v3-base)...")
    nli_model = CrossEncoder('cross-encoder/nli-deberta-v3-base')
    
    print("  ✓ Modelos cargados correctamente")
    
    # -------------------------------------------------------------------------
    # FASE 2: Calcular embeddings
    # -------------------------------------------------------------------------
    print("\n[FASE 2] Calculando embeddings (768 dimensiones)...")
    
    all_propositions = {
        "Π": PI,
        "A_L0": TREE_A["L0"],
        "A_L1": TREE_A["L1"],
        "A_L2": TREE_A["L2"],
        "A_L3": TREE_A["L3"],
        "B_L0": TREE_B["L0"],
        "B_L1": TREE_B["L1"],
        "B_L2": TREE_B["L2"],
        "B_L3": TREE_B["L3"],
    }
    
    embeddings = {}
    for key, text in all_propositions.items():
        embeddings[key] = embed_model.encode(text)
        print(f"  - {key}: \"{text[:55]}{'...' if len(text) > 55 else ''}\"")
    
    # -------------------------------------------------------------------------
    # FASE 3: Similitudes con Π y raíces
    # -------------------------------------------------------------------------
    print("\n[FASE 3] Calculando similitudes coseno...")
    
    print("\n  === Similitudes con Π (enunciado del problema) ===")
    for key in all_propositions:
        if key != "Π":
            sim = cosine_similarity(embeddings[key], embeddings["Π"])
            print(f"  sim({key}, Π) = {sim:.4f}")
    
    print("\n  === Similitudes con raíces ===")
    print("\n  Árbol A (similitud con A_L0):")
    for level in ["L1", "L2", "L3"]:
        key = f"A_{level}"
        sim = cosine_similarity(embeddings[key], embeddings["A_L0"])
        print(f"  sim({key}, A_L0) = {sim:.4f}")
    
    print("\n  Árbol B (similitud con B_L0):")
    for level in ["L1", "L2", "L3"]:
        key = f"B_{level}"
        sim = cosine_similarity(embeddings[key], embeddings["B_L0"])
        print(f"  sim({key}, B_L0) = {sim:.4f}")
    
    # -------------------------------------------------------------------------
    # FASE 4: Pesos bicompuestos de aristas
    # -------------------------------------------------------------------------
    print("\n[FASE 4] Calculando pesos bicompuestos de aristas...")
    print("  Fórmula: w(e) = (1 - sim(v, R)) + (1 - sim(v, Π))")
    
    weights_A = {}
    weights_B = {}
    
    print("\n  === Árbol A ===")
    cumulative_A = 0
    for level in ["L1", "L2", "L3"]:
        key = f"A_{level}"
        w, sim_r, sim_pi = calculate_edge_weight(
            embeddings[key], embeddings["A_L0"], embeddings["Π"]
        )
        weights_A[key] = w
        cumulative_A += w
        print(f"  {key}: w = (1-{sim_r:.4f}) + (1-{sim_pi:.4f}) = {w:.4f}  |  W_acum = {cumulative_A:.4f}")
    
    print("\n  === Árbol B ===")
    cumulative_B = 0
    for level in ["L1", "L2", "L3"]:
        key = f"B_{level}"
        w, sim_r, sim_pi = calculate_edge_weight(
            embeddings[key], embeddings["B_L0"], embeddings["Π"]
        )
        weights_B[key] = w
        cumulative_B += w
        print(f"  {key}: w = (1-{sim_r:.4f}) + (1-{sim_pi:.4f}) = {w:.4f}  |  W_acum = {cumulative_B:.4f}")
    
    W_A_L3 = cumulative_A
    W_B_L3 = cumulative_B
    
    # -------------------------------------------------------------------------
    # FASE 5: Detección de fusiones (pre-filtro por similitud)
    # -------------------------------------------------------------------------
    print("\n[FASE 5] Detección de fusiones - Pre-filtro por similitud coseno...")
    print(f"  Umbral τ_pre = {TAU_PRE}")
    
    candidates = []
    print("\n  === Matriz de similitudes inter-actor ===")
    print("  " + " " * 12 + "B_L0     B_L1     B_L2     B_L3")
    
    for a_level in ["L0", "L1", "L2", "L3"]:
        a_key = f"A_{a_level}"
        row = f"  {a_key:8}"
        for b_level in ["L0", "L1", "L2", "L3"]:
            b_key = f"B_{b_level}"
            sim = cosine_similarity(embeddings[a_key], embeddings[b_key])
            row += f"  {sim:.4f}"
            if sim >= TAU_PRE:
                candidates.append((a_key, b_key, sim))
        print(row)
    
    print(f"\n  Candidatos que pasan pre-filtro (sim ≥ {TAU_PRE}):")
    candidates = sorted(candidates, key=lambda x: -x[2])
    for a, b, sim in candidates:
        print(f"    ({a}, {b}): sim = {sim:.4f}")
    
    if not candidates:
        print("  ⚠ No hay candidatos que pasen el pre-filtro.")
        return
    
    # -------------------------------------------------------------------------
    # FASE 6: Validación NLI bidireccional
    # -------------------------------------------------------------------------
    print("\n[FASE 6] Validación NLI bidireccional...")
    print(f"  Umbral entailment τ_ent = {TAU_ENT} (promedio bidireccional)")
    print(f"  Umbral contradicción τ_contr = {TAU_CONTR} (máximo)")
    print("  ORDEN CORRECTO: idx0=CONTR, idx1=ENT, idx2=NEUT")
    
    approved_fusions = []
    all_nli_results = []
    
    for a_key, b_key, sim_cos in candidates:
        prop_a = all_propositions[a_key]
        prop_b = all_propositions[b_key]
        
        print(f"\n  --- Evaluando ({a_key}, {b_key}) ---")
        print(f"  A: \"{prop_a}\"")
        print(f"  B: \"{prop_b}\"")
        
        # NLI bidireccional usando CrossEncoder
        scores_a_b = nli_model.predict([(prop_a, prop_b)])[0]
        scores_b_a = nli_model.predict([(prop_b, prop_a)])[0]
        
        probs_a_b = softmax(scores_a_b)
        probs_b_a = softmax(scores_b_a)
        
        # ORDEN CORRECTO: idx0=contradiction, idx1=entailment, idx2=neutral
        contr_a_b = probs_a_b[0]
        ent_a_b = probs_a_b[1]
        neut_a_b = probs_a_b[2]
        
        contr_b_a = probs_b_a[0]
        ent_b_a = probs_b_a[1]
        neut_b_a = probs_b_a[2]
        
        ent_avg = (ent_a_b + ent_b_a) / 2
        contr_max = max(contr_a_b, contr_b_a)
        
        all_nli_results.append((a_key, b_key, sim_cos, ent_avg, contr_max))
        
        print(f"  NLI(A→B): [C={contr_a_b:.4f} E={ent_a_b:.4f} N={neut_a_b:.4f}]")
        print(f"  NLI(B→A): [C={contr_b_a:.4f} E={ent_b_a:.4f} N={neut_b_a:.4f}]")
        print(f"  ENT* = AVG({ent_a_b:.4f}, {ent_b_a:.4f}) = {ent_avg:.4f}")
        print(f"  CONTR* = MAX({contr_a_b:.4f}, {contr_b_a:.4f}) = {contr_max:.4f}")
        
        # Evaluar criterios
        passes_ent = ent_avg >= TAU_ENT
        passes_contr = contr_max < TAU_CONTR
        
        if passes_ent and passes_contr:
            print(f"  ✓ FUSIÓN APROBADA (ENT* ≥ {TAU_ENT} ∧ CONTR* < {TAU_CONTR})")
            approved_fusions.append((a_key, b_key, sim_cos, ent_avg, contr_max))
        else:
            reasons = []
            if not passes_ent:
                reasons.append(f"ENT* ({ent_avg:.4f}) < {TAU_ENT}")
            if not passes_contr:
                reasons.append(f"CONTR* ({contr_max:.4f}) ≥ {TAU_CONTR}")
            print(f"  ✗ FUSIÓN RECHAZADA: {', '.join(reasons)}")
    
    # -------------------------------------------------------------------------
    # FASE 7: Identificación de HUBs y cálculo de FTT
    # -------------------------------------------------------------------------
    print("\n[FASE 7] Identificación de HUBs...")
    
    if not approved_fusions:
        print("  ⚠ No se encontraron fusiones aprobadas con los umbrales estándar.")
        print("\n  === Análisis de sensibilidad ===")
        
        all_nli_results.sort(key=lambda x: -x[3])
        
        print("\n  Ranking por ENT* (mejores candidatos a fusión):")
        print("  " + "-" * 70)
        for a, b, sim, ent, contr in all_nli_results:
            contr_ok = "✓" if contr < TAU_CONTR else "✗"
            print(f"    ({a}, {b}): sim={sim:.4f}, ENT*={ent:.4f}, CONTR*={contr:.4f} {contr_ok}")
        
        best_valid = None
        for a, b, sim, ent, contr in all_nli_results:
            if contr < TAU_CONTR:
                best_valid = (a, b, sim, ent, contr)
                break
        
        if best_valid:
            a, b, sim, ent, contr = best_valid
            print(f"\n  → Mejor candidato válido (sin contradicción): ({a}, {b})")
            print(f"    Fusionaría con τ_ent ≤ {ent:.4f}")
    
    else:
        print(f"\n  ✓ Fusiones aprobadas: {len(approved_fusions)}")
        for a_key, b_key, sim, ent, contr in approved_fusions:
            print(f"    {a_key} ~ {b_key}: sim={sim:.4f}, ENT*={ent:.4f}, CONTR*={contr:.4f}")
        
        print("\n  === Cálculo de Fricción Topológica Total (FTT) ===")
        
        hub_metrics = []
        for a_key, b_key, sim, ent, contr in approved_fusions:
            level_a = int(a_key.split("_L")[1])
            level_b = int(b_key.split("_L")[1])
            
            ftt_a = sum(weights_A.get(f"A_L{i}", 0) for i in range(1, level_a + 1))
            ftt_b = sum(weights_B.get(f"B_L{i}", 0) for i in range(1, level_b + 1))
            
            ftt_sum = ftt_a + ftt_b
            ftt_max = max(ftt_a, ftt_b)
            
            hub_metrics.append((a_key, b_key, ftt_a, ftt_b, ftt_sum, ftt_max, sim, ent))
            
            print(f"\n  HUB: [{a_key} ~ {b_key}]")
            print(f"    Texto A: \"{all_propositions[a_key]}\"")
            print(f"    Texto B: \"{all_propositions[b_key]}\"")
            print(f"    FTT(A→HUB) = {ftt_a:.4f}")
            print(f"    FTT(B→HUB) = {ftt_b:.4f}")
            print(f"    FTT_sum = {ftt_sum:.4f}")
            print(f"    FTT_max = {ftt_max:.4f}")
        
        hub_metrics.sort(key=lambda x: (x[4], x[5]))
        best = hub_metrics[0]
        
        print(f"\n  ★ HUB ÓPTIMO: [{best[0]} ~ {best[1]}]")
        print(f"    FTT_sum = {best[4]:.4f}")
        print(f"    FTT_max = {best[5]:.4f}")
    
    # -------------------------------------------------------------------------
    # RESUMEN FINAL
    # -------------------------------------------------------------------------
    print("\n" + "=" * 80)
    print("RESUMEN COMPLETO PARA EL PAPER")
    print("=" * 80)
    
    print("\n[1] ENUNCIADO DEL PROBLEMA (Π):")
    print(f"    \"{PI}\"")
    
    print("\n[2] POSICIONES INICIALES (RAÍCES):")
    print(f"    Actor A (expansión): \"{ACTOR_A_ROOT}\"")
    print(f"    Actor B (consolidación): \"{ACTOR_B_ROOT}\"")
    
    print("\n[3] ÁRBOLES CAUSALES COMPLETOS:")
    print("\n    --- Árbol A (Director pro-expansión) ---")
    for level, text in TREE_A.items():
        print(f"    {level}: \"{text}\"")
    print("\n    --- Árbol B (Director pro-consolidación) ---")
    for level, text in TREE_B.items():
        print(f"    {level}: \"{text}\"")
    
    print("\n[4] CONFIGURACIÓN TÉCNICA:")
    print(f"    Modelo embeddings: paraphrase-multilingual-mpnet-base-v2 (768D)")
    print(f"    Modelo NLI: cross-encoder/nli-deberta-v3-base")
    print(f"    Orden etiquetas NLI: idx0=CONTR, idx1=ENT, idx2=NEUT")
    print(f"    τ_pre = {TAU_PRE}")
    print(f"    τ_ent = {TAU_ENT}")
    print(f"    τ_contr = {TAU_CONTR}")
    
    print("\n[5] MÉTRICAS CLAVE:")
    print(f"    W(A_L3) = {W_A_L3:.4f} (peso acumulado árbol A)")
    print(f"    W(B_L3) = {W_B_L3:.4f} (peso acumulado árbol B)")
    print(f"    Pares candidatos (pre-filtro): {len(candidates)}")
    print(f"    Fusiones aprobadas: {len(approved_fusions)}")
    
    print("\n" + "=" * 80)
    print("FIN DE EJECUCIÓN")
    print("=" * 80)

if __name__ == "__main__":
    main()