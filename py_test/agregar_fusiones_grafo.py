#!/usr/bin/env python3
"""
Agrega aristas de fusión al grafo existente.
Genera grafo_con_fusiones.graphml sin necesidad de re-correr el proceso.

Uso:
    python agregar_fusiones_grafo.py <run_dir>
    
Ejemplo:
    python agregar_fusiones_grafo.py casos/pizza/runs/2026-02-02_18-38-26
"""

import json
import sys
from pathlib import Path
import networkx as nx


def main():
    if len(sys.argv) < 2:
        print("Uso: python agregar_fusiones_grafo.py <run_dir>")
        sys.exit(1)
    
    run_dir = Path(sys.argv[1])
    
    grafo_path = run_dir / "grafo.graphml"
    fusiones_path = run_dir / "fusiones.json"
    output_path = run_dir / "grafo_con_fusiones.graphml"
    
    # Verificar archivos
    if not grafo_path.exists():
        print(f"ERROR: No existe {grafo_path}")
        sys.exit(1)
    if not fusiones_path.exists():
        print(f"ERROR: No existe {fusiones_path}")
        sys.exit(1)
    
    # Cargar grafo
    print(f"Cargando grafo: {grafo_path}")
    G = nx.read_graphml(grafo_path)
    print(f"  Nodos: {G.number_of_nodes()}, Aristas: {G.number_of_edges()}")
    
    # Cargar fusiones
    print(f"Cargando fusiones: {fusiones_path}")
    with open(fusiones_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    
    fusiones = data.get("fusiones", [])
    print(f"  Fusiones: {len(fusiones)}")
    
    # Agregar aristas de fusión
    fusion_count = 0
    missing = 0
    for fusion in fusiones:
        nodo_a = fusion["nodo_a_id"]
        nodo_b = fusion["nodo_b_id"]
        
        if nodo_a in G and nodo_b in G:
            G.add_edge(
                nodo_a, nodo_b,
                tipo="fusion",
                peso=0,
                score_nli=fusion.get("similitud", 0.0)
            )
            fusion_count += 1
        else:
            missing += 1
    
    print(f"\n  Aristas de fusión agregadas: {fusion_count}")
    if missing:
        print(f"  Fusiones con nodos faltantes: {missing}")
    
    # Guardar
    nx.write_graphml(G, output_path)
    print(f"\n✓ Guardado: {output_path}")
    print(f"  Nodos: {G.number_of_nodes()}, Aristas: {G.number_of_edges()}")


if __name__ == "__main__":
    main()