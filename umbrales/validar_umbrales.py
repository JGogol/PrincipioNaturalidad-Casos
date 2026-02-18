#!/usr/bin/env python3
"""
=============================================================================
VALIDACIÓN DE UMBRALES NLI - Principio de la Naturalidad v8
=============================================================================
Este script genera evidencia empírica para defender los umbrales:
- τ_ent (entailment): 0.55
- τ_contr (contradicción): 0.30
- τ_pre (pre-filtro coseno): 0.40

Ejecuta casos de prueba etiquetados contra dos modelos NLI y genera
métricas de precisión/recall para diferentes valores de umbral.

Autor: Validación para paper de Javier Gogol Merletti
Fecha: Febrero 2026
=============================================================================
"""

import json
import csv
import os
from datetime import datetime
from dataclasses import dataclass, asdict
from typing import List, Tuple, Dict, Optional
import warnings
warnings.filterwarnings('ignore')

# ===========================================================================
# CONFIGURACIÓN
# ===========================================================================

OUTPUT_DIR = "nli_validation_results"
TIMESTAMP = datetime.now().strftime("%Y%m%d_%H%M%S")

# Modelos a evaluar
MODELS = {
    "deberta": "cross-encoder/nli-deberta-v3-base",      # Recomendado (90% acc)
    "minilm": "cross-encoder/nli-MiniLM2-L6-H768"        # Usado en GERD (81% acc)
}

# Umbrales a evaluar
THRESHOLDS_ENT = [0.40, 0.45, 0.50, 0.55, 0.60, 0.65, 0.70, 0.75, 0.80]
THRESHOLDS_CONTR = [0.15, 0.20, 0.25, 0.30, 0.35, 0.40, 0.45, 0.50]

# ===========================================================================
# DEFINICIÓN DE CASOS DE PRUEBA
# ===========================================================================

@dataclass
class TestCase:
    """Caso de prueba con ground truth"""
    id: str
    category: str
    subcategory: str
    P: str
    Q: str
    expected_relation: str  # "ENTAILMENT", "CONTRADICTION", "NEUTRAL"
    expected_bidirectional: bool  # True si P↔Q, False si solo P→Q
    notes: str
    language: str  # "en" o "es"
    difficulty: str  # "easy", "medium", "hard"


# ---------------------------------------------------------------------------
# CATEGORÍA 1: EQUIVALENCIAS SEMÁNTICAS (deben fusionarse)
# ---------------------------------------------------------------------------
EQUIVALENCE_CASES = [
    # --- Sinónimos simples ---
    TestCase("EQ001", "equivalence", "synonyms", 
             "The car is red.", "The automobile is red.",
             "ENTAILMENT", True, "Sinónimo directo car/automobile", "en", "easy"),
    
    TestCase("EQ002", "equivalence", "synonyms",
             "The company must expand.", "The firm must expand.",
             "ENTAILMENT", True, "Sinónimo company/firm", "en", "easy"),
    
    TestCase("EQ003", "equivalence", "synonyms",
             "La empresa debe crecer.", "La compañía debe crecer.",
             "ENTAILMENT", True, "Sinónimo empresa/compañía (español)", "es", "easy"),
    
    # --- Reformulaciones ---
    TestCase("EQ004", "equivalence", "reformulation",
             "John is Mary's father.", "Mary is John's daughter.",
             "ENTAILMENT", True, "Reformulación relación familiar", "en", "easy"),
    
    TestCase("EQ005", "equivalence", "reformulation",
             "The door is open.", "The door is not closed.",
             "ENTAILMENT", True, "Doble negación", "en", "easy"),
    
    TestCase("EQ006", "equivalence", "reformulation",
             "Egypt loses bargaining power.", "Egypt's negotiating leverage decreases.",
             "ENTAILMENT", True, "Reformulación dominio GERD", "en", "medium"),
    
    # --- Voz activa/pasiva ---
    TestCase("EQ007", "equivalence", "voice",
             "The cat chases the mouse.", "The mouse is chased by the cat.",
             "ENTAILMENT", True, "Transformación activa/pasiva", "en", "easy"),
    
    TestCase("EQ008", "equivalence", "voice",
             "Ethiopia controls the dam.", "The dam is controlled by Ethiopia.",
             "ENTAILMENT", True, "Activa/pasiva dominio GERD", "en", "easy"),
    
    # --- Equivalencias del caso GERD ---
    TestCase("EQ009", "equivalence", "domain_gerd",
             "The company improves its financial position.",
             "The economic situation of the company strengthens.",
             "ENTAILMENT", True, "Caso empresarial del paper", "en", "medium"),
    
    TestCase("EQ010", "equivalence", "domain_gerd",
             "Egypt loses Ethiopia's cooperation in future Nile negotiations.",
             "Ethiopia reduces cooperation with Egypt in Nile talks.",
             "ENTAILMENT", True, "HUB #1 variante", "en", "medium"),
    
    TestCase("EQ011", "equivalence", "domain_gerd",
             "Sudan gains leverage in regional water negotiations.",
             "Sudan strengthens its position in regional water talks.",
             "ENTAILMENT", True, "HUB #2 componente Sudan", "en", "medium"),
    
    # --- Equivalencias complejas ---
    TestCase("EQ012", "equivalence", "complex",
             "Failure to reach agreement will harm all parties.",
             "All parties will suffer if no agreement is reached.",
             "ENTAILMENT", True, "Reformulación condicional", "en", "medium"),
    
    TestCase("EQ013", "equivalence", "complex",
             "Regional stability benefits all riparian states.",
             "All countries along the river benefit from regional stability.",
             "ENTAILMENT", True, "Reformulación con explicación", "en", "hard"),
]

# ---------------------------------------------------------------------------
# CATEGORÍA 2: CONTRADICCIONES (no deben fusionarse, veto)
# ---------------------------------------------------------------------------
CONTRADICTION_CASES = [
    # --- Negación directa ---
    TestCase("CT001", "contradiction", "negation",
             "The sky is blue.", "The sky is not blue.",
             "CONTRADICTION", True, "Negación directa", "en", "easy"),
    
    TestCase("CT002", "contradiction", "negation",
             "Egypt accepts the proposal.", "Egypt rejects the proposal.",
             "CONTRADICTION", True, "Aceptar/rechazar", "en", "easy"),
    
    # --- Valores opuestos ---
    TestCase("CT003", "contradiction", "opposites",
             "The temperature is high.", "The temperature is low.",
             "CONTRADICTION", True, "Alto/bajo", "en", "easy"),
    
    TestCase("CT004", "contradiction", "opposites",
             "The economy is growing.", "The economy is shrinking.",
             "CONTRADICTION", True, "Crecer/decrecer", "en", "easy"),
    
    TestCase("CT005", "contradiction", "opposites",
             "Water flows are increasing.", "Water flows are decreasing.",
             "CONTRADICTION", True, "Dominio GERD", "en", "easy"),
    
    # --- Estados mutuamente excluyentes ---
    TestCase("CT006", "contradiction", "exclusive_states",
             "John is alive.", "John is dead.",
             "CONTRADICTION", True, "Vivo/muerto", "en", "easy"),
    
    TestCase("CT007", "contradiction", "exclusive_states",
             "The treaty is binding.", "The treaty is non-binding.",
             "CONTRADICTION", True, "Vinculante/no vinculante", "en", "easy"),
    
    # --- Contradicciones del caso empresarial ---
    TestCase("CT008", "contradiction", "domain_business",
             "The company must expand to new markets.",
             "The company must consolidate current markets.",
             "CONTRADICTION", True, "Premisas raíz caso empresarial", "en", "medium"),
    
    TestCase("CT009", "contradiction", "domain_business",
             "The company invests in acquiring new clients.",
             "The company invests in retaining current clients.",
             "CONTRADICTION", True, "Estrategias opuestas", "en", "medium"),
    
    # --- Contradicciones del caso GERD ---
    TestCase("CT010", "contradiction", "domain_gerd",
             "Egypt has historical rights over the Nile.",
             "Ethiopia has sovereign rights over its water resources.",
             "CONTRADICTION", True, "Premisas GERD (implícita)", "en", "hard"),
    
    TestCase("CT011", "contradiction", "domain_gerd",
             "The dam should release minimum guaranteed flows.",
             "The dam should operate with flexible guidelines only.",
             "CONTRADICTION", True, "Posturas operativas opuestas", "en", "medium"),
    
    # --- Inversión de roles (crítico para el paper) ---
    TestCase("CT012", "contradiction", "role_inversion",
             "Egypt threatens to escalate armed conflict against Ethiopia.",
             "Ethiopia provokes military posturing from Egypt.",
             "CONTRADICTION", True, "Inversión de agencia - caso del paper", "en", "hard"),
    
    TestCase("CT013", "contradiction", "role_inversion",
             "Country A attacks Country B.", "Country B attacks Country A.",
             "CONTRADICTION", True, "Inversión de roles simple", "en", "medium"),
    
    # --- Contradicciones sutiles ---
    TestCase("CT014", "contradiction", "subtle",
             "The agreement prioritizes Egyptian water security.",
             "The agreement prioritizes Ethiopian development rights.",
             "CONTRADICTION", True, "Prioridades mutuamente excluyentes", "en", "hard"),
]

# ---------------------------------------------------------------------------
# CATEGORÍA 3: NEUTRALES (no deben fusionarse, pero no veto)
# ---------------------------------------------------------------------------
NEUTRAL_CASES = [
    # --- Temas no relacionados ---
    TestCase("NT001", "neutral", "unrelated",
             "The book is on the table.", "Tomorrow is Tuesday.",
             "NEUTRAL", True, "Completamente no relacionados", "en", "easy"),
    
    TestCase("NT002", "neutral", "unrelated",
             "The company has 100 employees.", "The building has 5 floors.",
             "NEUTRAL", True, "Mismo dominio, sin relación lógica", "en", "easy"),
    
    # --- Mismo tema, sin implicación ---
    TestCase("NT003", "neutral", "same_topic",
             "Egypt is concerned about water security.",
             "Ethiopia is building the GERD dam.",
             "NEUTRAL", True, "Mismo conflicto, hechos independientes", "en", "medium"),
    
    TestCase("NT004", "neutral", "same_topic",
             "The Nile River flows through multiple countries.",
             "The GERD dam has a capacity of 74 billion cubic meters.",
             "NEUTRAL", True, "Hechos relacionados pero independientes", "en", "medium"),
    
    TestCase("NT005", "neutral", "same_topic",
             "Sudan operates the Roseires Dam.",
             "Ethiopia exports electricity to neighboring countries.",
             "NEUTRAL", True, "Operaciones independientes", "en", "medium"),
    
    # --- Casos que parecen relacionados pero no lo son ---
    TestCase("NT006", "neutral", "misleading_similarity",
             "The company improved its market position.",
             "The company moved to a new office building.",
             "NEUTRAL", True, "Alta similitud léxica, sin relación", "en", "medium"),
    
    TestCase("NT007", "neutral", "misleading_similarity",
             "Egypt strengthens its legal position.",
             "Egypt strengthens its military position.",
             "NEUTRAL", True, "Estructura similar, dominios diferentes", "en", "hard"),
    
    # --- Casos del paper que NO deben fusionarse ---
    TestCase("NT008", "neutral", "paper_spurious",
             "The company invests in acquiring new clients.",
             "Revenue from recurring clients increases.",
             "NEUTRAL", True, "A_L1 vs B_L2 del caso empresarial", "en", "medium"),
    
    TestCase("NT009", "neutral", "paper_spurious",
             "The company improves its financial position.",
             "The company must consolidate current markets.",
             "NEUTRAL", True, "A_L3 vs B_L0 del caso empresarial", "en", "medium"),
    
    # --- Independencia causal ---
    TestCase("NT010", "neutral", "causal_independence",
             "It rained last night.", "The ground is wet.",
             "NEUTRAL", True, "Correlación sin implicación estricta", "en", "hard"),
    
    TestCase("NT011", "neutral", "causal_independence",
             "The employees are motivated.", "Sales increased.",
             "NEUTRAL", True, "Posible correlación, no implicación", "en", "hard"),
]

# ---------------------------------------------------------------------------
# CATEGORÍA 4: IMPLICACIÓN UNIDIRECCIONAL (P→Q pero no Q→P)
# NOTA: En el Principio de Naturalidad, estas NO deben fusionarse porque
#       el criterio requiere AVG(ENT_P→Q, ENT_Q→P) >= τ_ent (bidireccional)
# ---------------------------------------------------------------------------
UNIDIRECTIONAL_CASES = [
    # --- Específico a general ---
    TestCase("UD001", "unidirectional", "specific_to_general",
             "John ate a red apple.", "John ate a fruit.",
             "NEUTRAL", False, "Específico implica general - NO fusiona (unidireccional)", "en", "easy"),
    
    TestCase("UD002", "unidirectional", "specific_to_general",
             "Egypt demands legally binding minimum flows.",
             "Egypt makes demands about water flows.",
             "NEUTRAL", False, "Específico a general - GERD - NO fusiona", "en", "medium"),
    
    TestCase("UD003", "unidirectional", "specific_to_general",
             "The GERD dam generates 6,000 MW of electricity.",
             "The GERD dam generates electricity.",
             "NEUTRAL", False, "Detalle implica general - NO fusiona", "en", "easy"),
    
    # --- Causa a efecto (débil) ---
    TestCase("UD004", "unidirectional", "weak_causal",
             "Ethiopia fills the dam rapidly.",
             "Downstream water flows may decrease temporarily.",
             "NEUTRAL", False, "Causal probable - NO fusiona", "en", "hard"),
    
    # --- Hiperónimos ---
    TestCase("UD005", "unidirectional", "hypernym",
             "Egypt is an African country.",
             "Egypt is a country.",
             "NEUTRAL", False, "Hiperónimo - NO fusiona", "en", "easy"),
]

# ---------------------------------------------------------------------------
# CATEGORÍA 5: CASOS LÍMITE Y AMBIGUOS
# ---------------------------------------------------------------------------
EDGE_CASES = [
    # --- Casi equivalentes ---
    TestCase("ED001", "edge", "near_equivalent",
             "The situation may deteriorate.",
             "The situation will deteriorate.",
             "NEUTRAL", True, "Modal vs definitivo - ¿equivalente?", "en", "hard"),
    
    TestCase("ED002", "edge", "near_equivalent",
             "Egypt could lose bargaining power.",
             "Egypt might lose bargaining power.",
             "ENTAILMENT", True, "Modales similares", "en", "hard"),
    
    # --- Casi contradictorios ---
    TestCase("ED003", "edge", "near_contradiction",
             "The agreement is mostly beneficial.",
             "The agreement has significant drawbacks.",
             "NEUTRAL", True, "Tensión sin contradicción estricta", "en", "hard"),
    
    TestCase("ED004", "edge", "near_contradiction",
             "Progress has been made in negotiations.",
             "Negotiations have stalled.",
             "CONTRADICTION", True, "¿Contradicción o diferentes momentos?", "en", "hard"),
    
    # --- Ambigüedad de alcance ---
    TestCase("ED005", "edge", "scope_ambiguity",
             "All countries benefit from cooperation.",
             "Some countries benefit more than others.",
             "NEUTRAL", True, "Compatibles pero diferentes alcances", "en", "hard"),
    
    # --- Proposiciones genéricas/triviales ---
    TestCase("ED006", "edge", "trivial",
             "Regional prosperity is desirable.",
             "Economic growth benefits the region.",
             "ENTAILMENT", True, "¿Trivialmente similar?", "en", "hard"),
    
    TestCase("ED007", "edge", "trivial",
             "Peace is better than war.",
             "Conflict should be avoided.",
             "ENTAILMENT", True, "Genérico - ¿fusión válida?", "en", "hard"),
    
    # --- Negaciones parciales ---
    TestCase("ED008", "edge", "partial_negation",
             "Egypt is not entirely satisfied with the proposal.",
             "Egypt has reservations about the proposal.",
             "ENTAILMENT", True, "Negación parcial = reformulación", "en", "hard"),
    
    # --- Cuantificadores ---
    TestCase("ED009", "edge", "quantifiers",
             "Most experts agree on the risk.",
             "Some experts disagree about the risk.",
             "NEUTRAL", True, "Cuantificadores compatibles", "en", "hard"),
    
    TestCase("ED010", "edge", "quantifiers",
             "All parties must compromise.",
             "No party should have to compromise.",
             "CONTRADICTION", True, "Cuantificadores opuestos", "en", "medium"),
]

# ---------------------------------------------------------------------------
# COMPILAR TODOS LOS CASOS (solo inglés)
# ---------------------------------------------------------------------------
ALL_TEST_CASES: List[TestCase] = (
    EQUIVALENCE_CASES + 
    CONTRADICTION_CASES + 
    NEUTRAL_CASES + 
    UNIDIRECTIONAL_CASES + 
    EDGE_CASES
)

print(f"Total de casos de prueba: {len(ALL_TEST_CASES)}")
print(f"  - Equivalencias: {len(EQUIVALENCE_CASES)}")
print(f"  - Contradicciones: {len(CONTRADICTION_CASES)}")
print(f"  - Neutrales: {len(NEUTRAL_CASES)}")
print(f"  - Unidireccionales: {len(UNIDIRECTIONAL_CASES)}")
print(f"  - Casos límite: {len(EDGE_CASES)}")


# ===========================================================================
# FUNCIONES DE EVALUACIÓN
# ===========================================================================

def load_nli_model(model_name: str):
    """Carga un modelo NLI cross-encoder con soporte GPU si disponible"""
    from sentence_transformers import CrossEncoder
    import torch
    
    # Detectar dispositivo
    if torch.cuda.is_available():
        device = 'cuda'
        print(f"\n  GPU detectada: {torch.cuda.get_device_name(0)}")
    else:
        device = 'cpu'
        print(f"\n  Usando CPU (sin GPU disponible)")
    
    print(f"  Cargando modelo: {model_name}...")
    model = CrossEncoder(model_name, device=device)
    print(f"  ✓ Modelo cargado en {device}")
    return model


def evaluate_pair(model, P: str, Q: str) -> Dict:
    """
    Evalúa un par de proposiciones con NLI.
    Retorna scores para ambas direcciones.
    
    IMPORTANTE - Orden de etiquetas para cross-encoder NLI:
    Los modelos cross-encoder/nli-* de HuggingFace retornan:
      índice 0 = contradiction
      índice 1 = entailment  
      índice 2 = neutral
    
    Verificado empíricamente con casos triviales.
    Si usas otro modelo, verificar orden con:
      model.predict([("A dog is an animal", "A dog is an animal")])
      -> debería dar máximo en índice 1 (entailment)
    """
    # P → Q
    scores_pq = model.predict([(P, Q)], apply_softmax=True)[0]
    
    # Q → P
    scores_qp = model.predict([(Q, P)], apply_softmax=True)[0]
    
    # Índices: [0]=contradiction, [1]=entailment, [2]=neutral
    return {
        "P_to_Q": {
            "contradiction": float(scores_pq[0]),
            "entailment": float(scores_pq[1]),
            "neutral": float(scores_pq[2])
        },
        "Q_to_P": {
            "contradiction": float(scores_qp[0]),
            "entailment": float(scores_qp[1]),
            "neutral": float(scores_qp[2])
        },
        # Métricas agregadas (como en el paper)
        "ENT_avg": (float(scores_pq[1]) + float(scores_qp[1])) / 2,
        "CONTR_max": max(float(scores_pq[0]), float(scores_qp[0])),
        "ENT_min": min(float(scores_pq[1]), float(scores_qp[1])),
        "CONTR_avg": (float(scores_pq[0]) + float(scores_qp[0])) / 2,
    }


def classify_with_thresholds(scores: Dict, tau_ent: float, tau_contr: float) -> str:
    """
    Clasifica un par según los umbrales del paper.
    Retorna: "FUSION", "VETO", o "REJECT"
    """
    # Criterio del paper: AVG(ENT) >= τ_ent AND MAX(CONTR) < τ_contr
    if scores["CONTR_max"] >= tau_contr:
        return "VETO"  # Rechazado por contradicción
    elif scores["ENT_avg"] >= tau_ent:
        return "FUSION"  # Aprobado para fusión
    else:
        return "REJECT"  # Rechazado por entailment insuficiente


def compute_metrics(results: List[Dict], tau_ent: float, tau_contr: float) -> Dict:
    """
    Calcula métricas de precisión/recall para un conjunto de umbrales.
    """
    # Ground truth mapping
    gt_to_expected = {
        "ENTAILMENT": "FUSION",
        "CONTRADICTION": "VETO", 
        "NEUTRAL": "REJECT"
    }
    
    tp = fp = tn = fn = 0
    tp_contr = fp_contr = tn_contr = fn_contr = 0
    
    for r in results:
        expected = gt_to_expected[r["ground_truth"]]
        predicted = classify_with_thresholds(r["scores"], tau_ent, tau_contr)
        
        # Para fusiones (ENTAILMENT)
        if r["ground_truth"] == "ENTAILMENT":
            if predicted == "FUSION":
                tp += 1
            else:
                fn += 1
        else:
            if predicted == "FUSION":
                fp += 1
            else:
                tn += 1
        
        # Para contradicciones
        if r["ground_truth"] == "CONTRADICTION":
            if predicted == "VETO":
                tp_contr += 1
            else:
                fn_contr += 1
        else:
            if predicted == "VETO":
                fp_contr += 1
            else:
                tn_contr += 1
    
    # Calcular métricas
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
    
    precision_contr = tp_contr / (tp_contr + fp_contr) if (tp_contr + fp_contr) > 0 else 0
    recall_contr = tp_contr / (tp_contr + fn_contr) if (tp_contr + fn_contr) > 0 else 0
    f1_contr = 2 * precision_contr * recall_contr / (precision_contr + recall_contr) if (precision_contr + recall_contr) > 0 else 0
    
    return {
        "tau_ent": tau_ent,
        "tau_contr": tau_contr,
        "fusion": {
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "tp": tp, "fp": fp, "tn": tn, "fn": fn
        },
        "contradiction": {
            "precision": precision_contr,
            "recall": recall_contr,
            "f1": f1_contr,
            "tp": tp_contr, "fp": fp_contr, "tn": tn_contr, "fn": fn_contr
        }
    }


# ===========================================================================
# FUNCIÓN PRINCIPAL
# ===========================================================================

def run_validation():
    """Ejecuta la validación completa"""
    
    # Crear directorio de salida
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    results_by_model = {}
    
    for model_key, model_path in MODELS.items():
        print(f"\n{'='*60}")
        print(f"EVALUANDO MODELO: {model_key.upper()}")
        print(f"{'='*60}")
        
        # Cargar modelo
        model = load_nli_model(model_path)
        
        # Evaluar todos los casos
        results = []
        
        for i, tc in enumerate(ALL_TEST_CASES):
            print(f"\r  Evaluando caso {i+1}/{len(ALL_TEST_CASES)}: {tc.id}...", end="")
            
            scores = evaluate_pair(model, tc.P, tc.Q)
            
            result = {
                "id": tc.id,
                "category": tc.category,
                "subcategory": tc.subcategory,
                "P": tc.P,
                "Q": tc.Q,
                "ground_truth": tc.expected_relation,
                "bidirectional": tc.expected_bidirectional,
                "language": tc.language,
                "difficulty": tc.difficulty,
                "notes": tc.notes,
                "scores": scores
            }
            results.append(result)
        
        print(f"\n  ✓ {len(results)} casos evaluados")
        
        results_by_model[model_key] = results
        
        # Guardar resultados detallados
        output_file = os.path.join(OUTPUT_DIR, f"results_{model_key}_{TIMESTAMP}.json")
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=2, ensure_ascii=False)
        print(f"  ✓ Resultados guardados: {output_file}")
    
    # -----------------------------------------------------------------------
    # ANÁLISIS DE UMBRALES
    # -----------------------------------------------------------------------
    print(f"\n{'='*60}")
    print("ANÁLISIS DE UMBRALES")
    print(f"{'='*60}")
    
    threshold_analysis = {}
    
    for model_key, results in results_by_model.items():
        print(f"\n--- {model_key.upper()} ---")
        
        threshold_analysis[model_key] = []
        
        for tau_ent in THRESHOLDS_ENT:
            for tau_contr in THRESHOLDS_CONTR:
                metrics = compute_metrics(results, tau_ent, tau_contr)
                threshold_analysis[model_key].append(metrics)
        
        # Encontrar mejor combinación por F1 de fusión
        best = max(threshold_analysis[model_key], key=lambda x: x["fusion"]["f1"])
        print(f"  Mejor F1 fusión: τ_ent={best['tau_ent']}, τ_contr={best['tau_contr']}")
        print(f"    Precision={best['fusion']['precision']:.3f}, Recall={best['fusion']['recall']:.3f}, F1={best['fusion']['f1']:.3f}")
        
        # Métricas para umbrales del paper (τ_ent=0.55, τ_contr=0.30)
        paper_metrics = compute_metrics(results, 0.55, 0.30)
        print(f"\n  Con umbrales del paper (τ_ent=0.55, τ_contr=0.30):")
        print(f"    Fusión:       P={paper_metrics['fusion']['precision']:.3f}, R={paper_metrics['fusion']['recall']:.3f}, F1={paper_metrics['fusion']['f1']:.3f}")
        print(f"    Contradicción: P={paper_metrics['contradiction']['precision']:.3f}, R={paper_metrics['contradiction']['recall']:.3f}, F1={paper_metrics['contradiction']['f1']:.3f}")
    
    # Guardar análisis de umbrales
    threshold_file = os.path.join(OUTPUT_DIR, f"threshold_analysis_{TIMESTAMP}.json")
    with open(threshold_file, 'w', encoding='utf-8') as f:
        json.dump(threshold_analysis, f, indent=2)
    print(f"\n✓ Análisis de umbrales guardado: {threshold_file}")
    
    # -----------------------------------------------------------------------
    # GENERAR CSV RESUMEN
    # -----------------------------------------------------------------------
    csv_file = os.path.join(OUTPUT_DIR, f"summary_{TIMESTAMP}.csv")
    
    with open(csv_file, 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow([
            "model", "id", "category", "subcategory", "language", "difficulty",
            "ground_truth", "bidirectional",
            "ENT_P_to_Q", "ENT_Q_to_P", "ENT_avg",
            "CONTR_P_to_Q", "CONTR_Q_to_P", "CONTR_max",
            "classification_0.55_0.30",
            "P", "Q", "notes"
        ])
        
        for model_key, results in results_by_model.items():
            for r in results:
                classification = classify_with_thresholds(r["scores"], 0.55, 0.30)
                writer.writerow([
                    model_key,
                    r["id"],
                    r["category"],
                    r["subcategory"],
                    r["language"],
                    r["difficulty"],
                    r["ground_truth"],
                    r["bidirectional"],
                    f"{r['scores']['P_to_Q']['entailment']:.4f}",
                    f"{r['scores']['Q_to_P']['entailment']:.4f}",
                    f"{r['scores']['ENT_avg']:.4f}",
                    f"{r['scores']['P_to_Q']['contradiction']:.4f}",
                    f"{r['scores']['Q_to_P']['contradiction']:.4f}",
                    f"{r['scores']['CONTR_max']:.4f}",
                    classification,
                    r["P"],
                    r["Q"],
                    r["notes"]
                ])
    
    print(f"✓ Resumen CSV guardado: {csv_file}")
    
    # -----------------------------------------------------------------------
    # GENERAR REPORTE DE TEXTO
    # -----------------------------------------------------------------------
    report_file = os.path.join(OUTPUT_DIR, f"report_{TIMESTAMP}.txt")
    
    with open(report_file, 'w', encoding='utf-8') as f:
        f.write("="*70 + "\n")
        f.write("VALIDACIÓN DE UMBRALES NLI - Principio de la Naturalidad v8\n")
        f.write(f"Fecha: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write("="*70 + "\n\n")
        
        f.write(f"Total de casos evaluados: {len(ALL_TEST_CASES)}\n")
        f.write(f"  - Equivalencias: {len(EQUIVALENCE_CASES)}\n")
        f.write(f"  - Contradicciones: {len(CONTRADICTION_CASES)}\n")
        f.write(f"  - Neutrales: {len(NEUTRAL_CASES)}\n")
        f.write(f"  - Unidireccionales: {len(UNIDIRECTIONAL_CASES)} (esperado: REJECT - no fusionan)\n")
        f.write(f"  - Casos límite: {len(EDGE_CASES)}\n\n")
        
        for model_key, results in results_by_model.items():
            f.write(f"\n{'='*70}\n")
            f.write(f"MODELO: {model_key.upper()} ({MODELS[model_key]})\n")
            f.write(f"Casos evaluados: {len(results)}\n")
            f.write(f"{'='*70}\n\n")
            
            # Métricas con umbrales del paper
            paper_metrics = compute_metrics(results, 0.55, 0.30)
            f.write("MÉTRICAS CON UMBRALES DEL PAPER (τ_ent=0.55, τ_contr=0.30):\n")
            f.write("-"*50 + "\n")
            f.write(f"  Detección de FUSIONES:\n")
            f.write(f"    Precision: {paper_metrics['fusion']['precision']:.4f}\n")
            f.write(f"    Recall:    {paper_metrics['fusion']['recall']:.4f}\n")
            f.write(f"    F1-Score:  {paper_metrics['fusion']['f1']:.4f}\n")
            f.write(f"    TP={paper_metrics['fusion']['tp']}, FP={paper_metrics['fusion']['fp']}, ")
            f.write(f"TN={paper_metrics['fusion']['tn']}, FN={paper_metrics['fusion']['fn']}\n\n")
            
            f.write(f"  Detección de CONTRADICCIONES:\n")
            f.write(f"    Precision: {paper_metrics['contradiction']['precision']:.4f}\n")
            f.write(f"    Recall:    {paper_metrics['contradiction']['recall']:.4f}\n")
            f.write(f"    F1-Score:  {paper_metrics['contradiction']['f1']:.4f}\n")
            f.write(f"    TP={paper_metrics['contradiction']['tp']}, FP={paper_metrics['contradiction']['fp']}, ")
            f.write(f"TN={paper_metrics['contradiction']['tn']}, FN={paper_metrics['contradiction']['fn']}\n\n")
            
            # Casos problemáticos
            f.write("\nCASOS PROBLEMÁTICOS (clasificación incorrecta con τ_ent=0.55, τ_contr=0.30):\n")
            f.write("-"*50 + "\n")
            
            problem_count = 0
            for r in results:
                gt = r["ground_truth"]
                classification = classify_with_thresholds(r["scores"], 0.55, 0.30)
                
                # Mapear ground truth a clasificación esperada
                expected_map = {"ENTAILMENT": "FUSION", "CONTRADICTION": "VETO", "NEUTRAL": "REJECT"}
                expected = expected_map[gt]
                
                if classification != expected:
                    problem_count += 1
                    f.write(f"\n  [{r['id']}] {r['category']}/{r['subcategory']} ({r['difficulty']})\n")
                    f.write(f"    P: {r['P']}\n")
                    f.write(f"    Q: {r['Q']}\n")
                    f.write(f"    Ground Truth: {gt} → Esperado: {expected}\n")
                    f.write(f"    Clasificación: {classification}\n")
                    f.write(f"    Scores: ENT_avg={r['scores']['ENT_avg']:.4f}, CONTR_max={r['scores']['CONTR_max']:.4f}\n")
                    f.write(f"    Notas: {r['notes']}\n")
            
            if problem_count == 0:
                f.write("  ¡Ningún caso problemático!\n")
            else:
                f.write(f"\n  Total casos problemáticos: {problem_count}/{len(results)}\n")
            
            # Tabla de análisis por categoría
            f.write(f"\n\nANÁLISIS POR CATEGORÍA:\n")
            f.write("-"*50 + "\n")
            
            categories = {}
            for r in results:
                cat = r["category"]
                if cat not in categories:
                    categories[cat] = {"total": 0, "correct": 0}
                categories[cat]["total"] += 1
                
                gt = r["ground_truth"]
                classification = classify_with_thresholds(r["scores"], 0.55, 0.30)
                expected_map = {"ENTAILMENT": "FUSION", "CONTRADICTION": "VETO", "NEUTRAL": "REJECT"}
                
                if classification == expected_map[gt]:
                    categories[cat]["correct"] += 1
            
            for cat, stats in sorted(categories.items()):
                acc = stats["correct"] / stats["total"] * 100
                f.write(f"  {cat:20s}: {stats['correct']}/{stats['total']} ({acc:.1f}%)\n")
            
            # Análisis por dificultad
            f.write(f"\n\nANÁLISIS POR DIFICULTAD:\n")
            f.write("-"*50 + "\n")
            
            difficulties = {}
            for r in results:
                diff = r["difficulty"]
                if diff not in difficulties:
                    difficulties[diff] = {"total": 0, "correct": 0}
                difficulties[diff]["total"] += 1
                
                gt = r["ground_truth"]
                classification = classify_with_thresholds(r["scores"], 0.55, 0.30)
                expected_map = {"ENTAILMENT": "FUSION", "CONTRADICTION": "VETO", "NEUTRAL": "REJECT"}
                
                if classification == expected_map[gt]:
                    difficulties[diff]["correct"] += 1
            
            for diff in ["easy", "medium", "hard"]:
                if diff in difficulties:
                    stats = difficulties[diff]
                    acc = stats["correct"] / stats["total"] * 100
                    f.write(f"  {diff:10s}: {stats['correct']}/{stats['total']} ({acc:.1f}%)\n")
    
    print(f"✓ Reporte de texto guardado: {report_file}")
    
    # -----------------------------------------------------------------------
    # TABLA DE SENSIBILIDAD DE UMBRALES (para el paper)
    # -----------------------------------------------------------------------
    sensitivity_file = os.path.join(OUTPUT_DIR, f"sensitivity_table_{TIMESTAMP}.csv")
    
    with open(sensitivity_file, 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow([
            "model", "tau_ent", "tau_contr",
            "fusion_precision", "fusion_recall", "fusion_f1",
            "contr_precision", "contr_recall", "contr_f1"
        ])
        
        for model_key, analyses in threshold_analysis.items():
            for a in analyses:
                writer.writerow([
                    model_key,
                    a["tau_ent"],
                    a["tau_contr"],
                    f"{a['fusion']['precision']:.4f}",
                    f"{a['fusion']['recall']:.4f}",
                    f"{a['fusion']['f1']:.4f}",
                    f"{a['contradiction']['precision']:.4f}",
                    f"{a['contradiction']['recall']:.4f}",
                    f"{a['contradiction']['f1']:.4f}"
                ])
    
    print(f"✓ Tabla de sensibilidad guardada: {sensitivity_file}")
    
    # -----------------------------------------------------------------------
    # COMPARACIÓN ENTRE MODELOS
    # -----------------------------------------------------------------------
    comparison_file = os.path.join(OUTPUT_DIR, f"model_comparison_{TIMESTAMP}.txt")
    
    with open(comparison_file, 'w', encoding='utf-8') as f:
        f.write("="*70 + "\n")
        f.write("COMPARACIÓN DE MODELOS\n")
        f.write("="*70 + "\n\n")
        
        if "deberta" in results_by_model and "minilm" in results_by_model:
            deberta_results = results_by_model["deberta"]
            minilm_results = results_by_model["minilm"]
            
            f.write(f"Casos evaluados: {len(deberta_results)}\n\n")
            
            # Métricas con umbrales del paper
            f.write("Con umbrales del paper (τ_ent=0.55, τ_contr=0.30):\n")
            f.write("-"*50 + "\n")
            
            metrics_deberta = compute_metrics(deberta_results, 0.55, 0.30)
            metrics_minilm = compute_metrics(minilm_results, 0.55, 0.30)
            
            f.write(f"\n{'Métrica':<25} {'DeBERTa':>12} {'MiniLM':>12} {'Diferencia':>12}\n")
            f.write("-"*65 + "\n")
            
            comparisons = [
                ("Fusión Precision", metrics_deberta['fusion']['precision'], metrics_minilm['fusion']['precision']),
                ("Fusión Recall", metrics_deberta['fusion']['recall'], metrics_minilm['fusion']['recall']),
                ("Fusión F1", metrics_deberta['fusion']['f1'], metrics_minilm['fusion']['f1']),
                ("Contradicción Precision", metrics_deberta['contradiction']['precision'], metrics_minilm['contradiction']['precision']),
                ("Contradicción Recall", metrics_deberta['contradiction']['recall'], metrics_minilm['contradiction']['recall']),
                ("Contradicción F1", metrics_deberta['contradiction']['f1'], metrics_minilm['contradiction']['f1']),
            ]
            
            for name, val_d, val_m in comparisons:
                diff = val_d - val_m
                sign = "+" if diff > 0 else ""
                f.write(f"{name:<25} {val_d:>12.4f} {val_m:>12.4f} {sign}{diff:>11.4f}\n")
            
            # Casos donde difieren
            f.write(f"\n\nCASOS DONDE LOS MODELOS DIFIEREN:\n")
            f.write("-"*50 + "\n")
            
            differ_count = 0
            for rd, rm in zip(deberta_results, minilm_results):
                if rd["id"] != rm["id"]:
                    continue
                    
                class_d = classify_with_thresholds(rd["scores"], 0.55, 0.30)
                class_m = classify_with_thresholds(rm["scores"], 0.55, 0.30)
                
                if class_d != class_m:
                    differ_count += 1
                    f.write(f"\n  [{rd['id']}] {rd['category']}/{rd['subcategory']}\n")
                    f.write(f"    P: {rd['P'][:60]}...\n" if len(rd['P']) > 60 else f"    P: {rd['P']}\n")
                    f.write(f"    Q: {rd['Q'][:60]}...\n" if len(rd['Q']) > 60 else f"    Q: {rd['Q']}\n")
                    f.write(f"    Ground Truth: {rd['ground_truth']}\n")
                    f.write(f"    DeBERTa: {class_d} (ENT={rd['scores']['ENT_avg']:.3f}, CONTR={rd['scores']['CONTR_max']:.3f})\n")
                    f.write(f"    MiniLM:  {class_m} (ENT={rm['scores']['ENT_avg']:.3f}, CONTR={rm['scores']['CONTR_max']:.3f})\n")
            
            if differ_count == 0:
                f.write("  ¡Los modelos coinciden en todos los casos!\n")
            else:
                f.write(f"\n  Total diferencias: {differ_count}/{len(deberta_results)}\n")
    
    print(f"✓ Comparación de modelos guardada: {comparison_file}")
    
    print(f"\n{'='*60}")
    print("VALIDACIÓN COMPLETADA")
    print(f"{'='*60}")
    print(f"Todos los archivos en: {OUTPUT_DIR}/")
    
    return results_by_model, threshold_analysis


# ===========================================================================
# EJECUCIÓN
# ===========================================================================

if __name__ == "__main__":
    print("""
    ╔═══════════════════════════════════════════════════════════════╗
    ║  VALIDACIÓN DE UMBRALES NLI                                   ║
    ║  Principio de la Naturalidad v8                               ║
    ║                                                               ║
    ║  Este script evalúa los umbrales τ_ent y τ_contr contra       ║
    ║  un conjunto de casos de prueba con ground truth etiquetado.  ║
    ╚═══════════════════════════════════════════════════════════════╝
    """)
    
    try:
        results, analysis = run_validation()
    except ImportError as e:
        print(f"\n❌ Error de importación: {e}")
        print("\nAsegúrate de tener instaladas las dependencias:")
        print("  pip install sentence-transformers torch")
    except Exception as e:
        print(f"\n❌ Error: {e}")
        raise