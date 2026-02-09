#!/usr/bin/env python3
"""
Test Sistemático de NLI - VERSIÓN INGLÉS
=========================================

Los mismos 21 casos pero en inglés para comparar modelos.

Ejecutar: python nli_test_english_full.py
"""

import numpy as np
from sentence_transformers import CrossEncoder
import warnings
warnings.filterwarnings('ignore')

def softmax(x):
    e_x = np.exp(x - np.max(x))
    return e_x / e_x.sum()

# =============================================================================
# CASOS DE TEST EN INGLÉS
# =============================================================================

TEST_CASES = [
    # EQUIVALENCE
    {"category": "EQUIVALENCE", "description": "Direct synonyms",
     "P": "The car is red.", "Q": "The automobile is red.", "expected": "ENT bidir"},
    {"category": "EQUIVALENCE", "description": "Simple reformulation",
     "P": "John is Mary's father.", "Q": "Mary is John's daughter.", "expected": "ENT bidir"},
    {"category": "EQUIVALENCE", "description": "Double negation",
     "P": "The door is open.", "Q": "The door is not closed.", "expected": "ENT bidir"},
    {"category": "EQUIVALENCE", "description": "Passive/Active",
     "P": "The cat chases the mouse.", "Q": "The mouse is chased by the cat.", "expected": "ENT bidir"},
    
    # CONTRADICTION
    {"category": "CONTRADICTION", "description": "Direct negation",
     "P": "The sky is blue.", "Q": "The sky is not blue.", "expected": "CONTR bidir"},
    {"category": "CONTRADICTION", "description": "Opposite values",
     "P": "The temperature is high.", "Q": "The temperature is low.", "expected": "CONTR bidir"},
    {"category": "CONTRADICTION", "description": "Mutually exclusive states",
     "P": "John is alive.", "Q": "John is dead.", "expected": "CONTR bidir"},
    {"category": "CONTRADICTION", "description": "Opposite actions",
     "P": "The company should expand.", "Q": "The company should contract.", "expected": "CONTR bidir"},
    
    # UNIDIRECTIONAL IMPLICATION
    {"category": "IMPLICATION", "description": "Specific → General",
     "P": "John ate a red apple.", "Q": "John ate a fruit.", "expected": "P→Q yes, Q→P no"},
    {"category": "IMPLICATION", "description": "Cause → Effect",
     "P": "It rained all night.", "Q": "The ground is wet.", "expected": "P→Q yes, Q→P no"},
    
    # NEUTRAL
    {"category": "NEUTRAL", "description": "Unrelated topics",
     "P": "The book is on the table.", "Q": "Tomorrow is Tuesday.", "expected": "NEUTRAL bidir"},
    {"category": "NEUTRAL", "description": "Same domain, no relation",
     "P": "The company has 100 employees.", "Q": "The building has 5 floors.", "expected": "NEUTRAL bidir"},
    
    # PROBLEMATIC CASES FROM EXAMPLE
    {"category": "PROBLEMATIC", "description": "L3 vs L3 (expected equivalence)",
     "P": "The company improves its financial position.", "Q": "The company's economic situation strengthens.", "expected": "ENT bidir?"},
    {"category": "PROBLEMATIC", "description": "A_L3 vs B_L0 (should not approve)",
     "P": "The company improves its financial position.", "Q": "The company should consolidate current markets.", "expected": "NEUTRAL?"},
    {"category": "PROBLEMATIC", "description": "A_L1 vs B_L2 (should not approve)",
     "P": "The company invests resources in acquiring new customers.", "Q": "Revenue from recurring customers increases.", "expected": "NEUTRAL?"},
    {"category": "PROBLEMATIC", "description": "Contradictory roots",
     "P": "The company should expand to new markets.", "Q": "The company should consolidate current markets.", "expected": "CONTR bidir"},
    
    # POSITIVE BIAS TEST
    {"category": "BIAS", "description": "Two unrelated good things",
     "P": "Employees are motivated.", "Q": "Sales increased.", "expected": "NEUTRAL"},
    {"category": "BIAS", "description": "Two independent improvements",
     "P": "Product quality improved.", "Q": "Customer service improved.", "expected": "NEUTRAL"},
    {"category": "BIAS", "description": "Grow vs improve",
     "P": "The company grew.", "Q": "The company improved.", "expected": "NEUTRAL"},
    
    # TAUTOLOGY
    {"category": "TAUTOLOGY", "description": "Identical",
     "P": "The company reduces its capital.", "Q": "The company reduces its capital.", "expected": "ENT bidir"},
    {"category": "TAUTOLOGY", "description": "Near-identical (synonym)",
     "P": "The company reduces its available capital.", "Q": "The company decreases its available capital.", "expected": "ENT bidir"},
]

def test_model(model_name):
    print("=" * 90)
    print(f"TEST SISTEMÁTICO NLI (ENGLISH) - {model_name.split('/')[-1]}")
    print("=" * 90)
    
    print("\nLoading model...")
    nli_model = CrossEncoder(model_name)
    print("✓ Model loaded\n")
    
    current_category = None
    results = {"correct": 0, "partial": 0, "wrong": 0}
    
    for i, case in enumerate(TEST_CASES, 1):
        if case["category"] != current_category:
            current_category = case["category"]
            print("\n" + "=" * 90)
            print(f"CATEGORY: {current_category}")
            print("=" * 90)
        
        P = case["P"]
        Q = case["Q"]
        
        scores_pq = nli_model.predict([(P, Q)])[0]
        scores_qp = nli_model.predict([(Q, P)])[0]
        
        probs_pq = softmax(scores_pq)
        probs_qp = softmax(scores_qp)
        
        # Order: 0=contr, 1=ent, 2=neut
        contr_pq, ent_pq, neut_pq = probs_pq[0], probs_pq[1], probs_pq[2]
        contr_qp, ent_qp, neut_qp = probs_qp[0], probs_qp[1], probs_qp[2]
        
        def get_label(c, e, n):
            if c >= e and c >= n:
                return "CONTR"
            elif e >= c and e >= n:
                return "ENT"
            else:
                return "NEUT"
        
        label_pq = get_label(contr_pq, ent_pq, neut_pq)
        label_qp = get_label(contr_qp, ent_qp, neut_qp)
        
        print(f"\n[Case {i}] {case['description']}")
        print(f"  P: \"{P}\"")
        print(f"  Q: \"{Q}\"")
        print(f"  P→Q: {label_pq:5} [C={contr_pq:.3f} E={ent_pq:.3f} N={neut_pq:.3f}]")
        print(f"  Q→P: {label_qp:5} [C={contr_qp:.3f} E={ent_qp:.3f} N={neut_qp:.3f}]")
        print(f"  Expected: {case['expected']}")
        
        if label_pq == label_qp:
            print(f"  → {label_pq} bidirectional")
        else:
            print(f"  → Asymmetric ({label_pq} / {label_qp})")
    
    print("\n" + "=" * 90)
    print("END OF TEST")
    print("=" * 90)

def main():
    # Test both models
    test_model('cross-encoder/nli-deberta-v3-base')
    print("\n\n" + "#" * 90)
    print("#" * 90 + "\n\n")
    test_model('cross-encoder/nli-MiniLM2-L6-H768')

if __name__ == "__main__":
    main()