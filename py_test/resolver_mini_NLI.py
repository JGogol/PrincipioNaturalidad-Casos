#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Principle of Naturality - Main Orchestrator v8.5
=================================================

v8.5 CHANGES:
- Computes φ(Π) once and passes to tree constructor
- Bicomposite edge weight: w(e) = (1 - sim_raiz) + (1 - sim_problema)
- Best-first search with priority queue (replaces BFS)
- FTT uses Dijkstra with bicomposite weights exclusively
- Removed ftt_mode config ("geodesic" vs "idw" toggle)
- max_depth increased to 10 (paper Appendix D)
- umbral_admision: gate de calidad semántica en construcción de árboles

Faithful implementation of the algorithm according to Paper v8.5.

Paper v8.5, Section 6.2 (Main Flow Pseudocode):
- PHASE 1: Tree construction (best-first search with priority queue)
- PHASE 2: Fusion detection (S-BERT + NLI)
- PHASE 3: Graph construction (NO collapse, bicomposite weights)
- PHASE 4: HUB search (equivalence classes, bicomposite distances)
- PHASE 5: Solution generation

Author: Javier Gogol Merletti
"""

import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Any

import yaml

from core.estructuras import Caso, Arbol, Hub
from core.embeddings import MotorEmbeddings
from core.arboles import ConstructorArboles
from core.fusiones import DetectorFusiones
from core.grafo import ConstructorGrafo
from llm.cliente import ClienteLLM
from core.hubs import BuscadorHubs
from auditoria.exportador import GestorAuditoria
from visualizacion.progress import VisualizadorProgreso, crear_visualizador

LOG = logging.getLogger(__name__)


# ═══════════════════════════════════════════════════════════════════════════════
# DEFAULT CONFIGURATION v8.5
# ═══════════════════════════════════════════════════════════════════════════════

DEFAULT_CONFIG = {
    # ───────────────────────────────────────────────────────────────────────────
    # Embeddings model (Paper v8.5, Section 3.1)
    # Used for: bicomposite edge weights (sim with root + sim with Π)
    # NOT used for: fusion (NLI), cycles (removed since v7.1)
    # ───────────────────────────────────────────────────────────────────────────
    "embeddings_model": "paraphrase-multilingual-mpnet-base-v2",
    
    # ───────────────────────────────────────────────────────────────────────────
    # Claude model for consequence generation
    # ───────────────────────────────────────────────────────────────────────────
    "claude_model": "claude-sonnet-4-20250514",
    
    # ───────────────────────────────────────────────────────────────────────────
    # NLI Configuration (Natural Language Inference)
    # Paper v8.5, Section 3.4: NLI is SOLE criterion for fusion
    # ───────────────────────────────────────────────────────────────────────────
    
    # HuggingFace model for NLI
    "nli_model": "cross-encoder/nli-MiniLM2-L6-H768",
    # Minimum ENTAILMENT confidence to fuse (Paper Appendix D.1: τ_ent = 0.55)
    "nli_threshold": 0.55,
    "nli_criterion": "avg",             # "max", "min", or "avg"
    "normalize_actors": True,           # Normalize actors before NLI
    "contradiction_threshold": 0.30,    # Paper Appendix D.1: τ_contr = 0.30
    "nli_batch_size": 128, 
    "embedding_prefilter_threshold": 0.4,  # Paper Appendix D.1: τ_pre = 0.40
    # Device for NLI (-1 = CPU, 0+ = GPU)
    "nli_device": -1,
    
    # ───────────────────────────────────────────────────────────────────────────
    # Termination limits (Paper v8.5, Appendix D)
    # ───────────────────────────────────────────────────────────────────────────
    "max_depth": 10,                     # d_max = 10 (was 7 in v7.x)
    "max_nodes_per_tree": 500,           # n_max = 500
    "redundancy_threshold": 0.90,        # Cross-branch pruning: discard node if
                                         # sim_coseno > threshold with any existing node
    
    # ───────────────────────────────────────────────────────────────────────────
    # Admission gate for tree construction (Paper v8.5+)
    # Nodes with sim_con_problema < umbral_admision are rejected.
    # This ensures n_max budget is spent on conflict-relevant nodes only.
    # ───────────────────────────────────────────────────────────────────────────
    "umbral_admision": 0.15,
    
    # ───────────────────────────────────────────────────────────────────────────
    # Operation mode
    # v8.5: Removed ftt_mode (only bicomposite weighted Dijkstra)
    # ───────────────────────────────────────────────────────────────────────────
    "human_validation_mode": False,
    "strict_mode": True,        # Only HUBs connecting ALL actors
    
    # ───────────────────────────────────────────────────────────────────────────
    # LLM Fusion Verification (Role Inversion Detection)
    # ───────────────────────────────────────────────────────────────────────────
    "verify_fusions_llm": True,
    "verify_fusions_batch_size": 15,
    
    # ───────────────────────────────────────────────────────────────────────────
    # Visualization and audit
    # ───────────────────────────────────────────────────────────────────────────
    "visualization": "normal",  # silent, minimal, normal, verbose
    "save_audit": True,
}


class ResolverConflicto:
    """
    Main orchestrator of the Principle of Naturality v8.5.
    
    v8.5: Bicomposite edge weights, best-first search, no node collapse,
    HUBs as equivalence classes with bicomposite distance.
    Executes the 5 phases of the algorithm according to Paper v8.5.
    """
    
    def __init__(
        self,
        config: Optional[Dict] = None,
        caso_dir: Optional[Path] = None
    ):
        """
        Args:
            config: Configuration (merged with DEFAULT_CONFIG)
            caso_dir: Case directory for audit
        """
        self.config = {**DEFAULT_CONFIG, **(config or {})}
        self.caso_dir = caso_dir
        
        # Initialize components
        LOG.info("Initializing components (v8.5)...")
        
        # Embeddings engine
        self.motor_embeddings = MotorEmbeddings(
            self.config["embeddings_model"]
        )
        
        # LLM client
        self.cliente_llm = ClienteLLM(
            modelo=self.config["claude_model"],
            api_key=self.config.get("anthropic_api_key")
        )
        
        # Progress visualizer
        self.visualizador = crear_visualizador(
            modo=self.config.get("visualization", "normal")
        )
        
        # Algorithm components
        self.constructor_arboles = ConstructorArboles(
            self.cliente_llm,
            self.motor_embeddings,
            self.config,
            callback_progreso=self.visualizador.callback
        )
        
        self.detector_fusiones = DetectorFusiones(
            self.motor_embeddings,
            self.config,
            callback_progreso=self.visualizador.callback
        )
        
        self.constructor_grafo = ConstructorGrafo()
        
        # v8.5: No modo_ftt parameter (only bicomposite Dijkstra)
        self.buscador_hubs = BuscadorHubs(
            modo_estricto=self.config["strict_mode"],
            callback_progreso=self.visualizador.callback
        )
        
        # Audit manager
        self.auditoria = None
        if self.config.get("save_audit") and caso_dir:
            self.auditoria = GestorAuditoria(caso_dir)
    
    def cargar_caso(self, path: str) -> Caso:
        """Loads a case from JSON or YAML file."""
        path = Path(path)
        
        with open(path, "r", encoding="utf-8") as f:
            if path.suffix == ".json":
                data = json.load(f)
            else:
                data = yaml.safe_load(f)
        
        caso = Caso.from_dict(data)
        
        if caso.config:
            self.config.update(caso.config)
            LOG.info(f"Case config applied: {list(caso.config.keys())}")
        
        return caso
    
    def resolver(self, caso: Caso) -> Dict[str, Any]:
        """
        Executes the complete resolution algorithm.
        
        v8.5: Bicomposite weights, best-first search, complete path preservation.
        
        Returns:
            Dict with hub, solution, metrics and artifacts
        """
        inicio = datetime.now()
        
        # Banner
        self._mostrar_banner(caso)
        
        # Save audit metadata
        if self.auditoria:
            self.auditoria.guardar_metadata(self.config, caso)
        
        # ═══════════════════════════════════════════════════════════════
        # PRE-COMPUTATION: φ(Π) - Problem statement embedding
        # Paper v8.5, Definition 5.4: Π = decision_central
        # Uses vectorially neutral version if available (no actor names,
        # balanced register) to avoid lexical bias in sim_con_problema.
        # Computed once, passed to all tree constructors.
        # ═══════════════════════════════════════════════════════════════
        texto_pi = caso.decision_central_embedding or caso.decision_central
        LOG.info(f"Computing φ(Π) for: \"{texto_pi[:60]}...\"")
        emb_problema = self.motor_embeddings.obtener_embedding(texto_pi)
        LOG.info("φ(Π) computed successfully")
        
        # ═══════════════════════════════════════════════════════════════
        # PHASE 1: TREE CONSTRUCTION (best-first search)
        # ═══════════════════════════════════════════════════════════════
        self.visualizador.callback("fase_inicio", {
            "numero": 1,
            "fase": "Causal tree construction (best-first search)"
        })
        
        arboles: List[Arbol] = []
        for actor in caso.actores:
            # v8.5: Pass φ(Π) for bicomposite weight calculation
            arbol = self.constructor_arboles.construir_arbol(
                caso, actor, emb_problema
            )
            arboles.append(arbol)
            
            if self.auditoria:
                self.auditoria.guardar_arbol(arbol)
        
        self.visualizador.callback("fase_fin", {
            "mensaje": f"{len(arboles)} trees built"
        })
        
        # ═══════════════════════════════════════════════════════════════
        # PHASE 2: FUSION DETECTION (S-BERT + NLI)
        # ═══════════════════════════════════════════════════════════════
        self.visualizador.callback("fase_inicio", {
            "numero": 2,
            "fase": "Semantic fusion detection (S-BERT + NLI)"
        })
        
        fusiones = self.detector_fusiones.detectar_fusiones(arboles)
        
        # ═══════════════════════════════════════════════════════════════
        # PHASE 2b: LLM FUSION VERIFICATION (Optional)
        # ═══════════════════════════════════════════════════════════════
        rejected_role_inversions = []
        if self.config.get("verify_fusions_llm", True) and fusiones:
            self.visualizador.callback("fase_inicio", {
                "numero": "2b",
                "fase": "LLM fusion verification (role inversion detection)"
            })
            
            fusiones, rejected_role_inversions = (
                self.detector_fusiones.verificar_fusiones_lote(
                    fusiones,
                    self.cliente_llm,
                    batch_size=self.config.get(
                        "verify_fusions_batch_size", 15
                    )
                )
            )
            
            self.visualizador.callback("fase_fin", {
                "mensaje": (
                    f"{len(fusiones)} valid, "
                    f"{len(rejected_role_inversions)} rejected"
                )
            })
        
        # v7.5+: Build equivalence classes
        clases_equivalencia = (
            self.detector_fusiones.construir_clases_equivalencia(fusiones)
        )
        
        if self.auditoria:
            self.auditoria.guardar_fusiones(fusiones)
            self.auditoria.guardar_estadisticas_fusiones(
                self.detector_fusiones.stats,
                self.detector_fusiones.contradictions,
                self.detector_fusiones.rejected_fusions,
                self.detector_fusiones.all_nli_evaluations,
                self.detector_fusiones.skipped_pairs_sample,
                self.config
            )
            if rejected_role_inversions:
                self.auditoria.guardar_rechazos_inversion(
                    rejected_role_inversions
                )
            
            # Save equivalence classes
            self.auditoria.guardar_clases_equivalencia(clases_equivalencia)
        
        self.visualizador.callback("fase_fin", {
            "mensaje": (
                f"{len(fusiones)} fusions, "
                f"{len(clases_equivalencia)} equivalence classes"
            )
        })
        
        # ═══════════════════════════════════════════════════════════════
        # PHASE 3: GRAPH CONSTRUCTION (v8.5 - bicomposite weights)
        # ═══════════════════════════════════════════════════════════════
        self.visualizador.callback("fase_inicio", {
            "numero": 3,
            "fase": "Unified graph construction (v8.5 - bicomposite weights)"
        })
        
        grafo = self.constructor_grafo.construir(
            arboles, fusiones, clases_equivalencia
        )
        
        if self.auditoria:
            self.auditoria.guardar_grafo(grafo)
            self.constructor_grafo.exportar_graphml_con_fusiones(
                grafo, fusiones, 
                self.auditoria.run_dir / "grafo_con_fusiones.graphml"
            )
        
        stats_grafo = self.constructor_grafo.obtener_estadisticas(grafo)
        self.visualizador.callback("fase_fin", {
            "mensaje": (
                f"{stats_grafo['total_nodes']} nodes (ALL preserved), "
                f"{stats_grafo['total_edges']} edges"
            )
        })
        
        # ═══════════════════════════════════════════════════════════════
        # PHASE 4: HUB SEARCH (bicomposite weighted distances)
        # ═══════════════════════════════════════════════════════════════
        self.visualizador.callback("fase_inicio", {
            "numero": 4,
            "fase": "Convergence HUB search (v8.5 - bicomposite distances)"
        })
        
        hubs = self.buscador_hubs.buscar(
            grafo, arboles, clases_equivalencia
        )
        
        if self.auditoria:
            self.auditoria.guardar_hubs(hubs)
        
        if not hubs:
            self.visualizador.callback("fase_fin", {
                "mensaje": "⚠️ NO GLOBAL CONVERGENCE FOUND"
            })
            
            return self._resultado_sin_convergencia(caso, arboles, inicio)
        
        # Paper v8.5: "h* ← first candidate" (already sorted lexicographically)
        hub_optimo = hubs[0]
        
        self.visualizador.callback("fase_fin", {
            "mensaje": (
                f"Optimal HUB: FTT={hub_optimo.ftt_sum:.4f} "
                f"(max={hub_optimo.ftt_max:.4f})"
            )
        })
        
        # Show selection explanation
        print("\n" + self.buscador_hubs.explicar_seleccion(hubs))
        
        # Show detailed per-actor report
        print(self.buscador_hubs.generar_reporte_hub(hub_optimo))
        
        # ═══════════════════════════════════════════════════════════════
        # PHASE 5: SOLUTION GENERATION
        # ═══════════════════════════════════════════════════════════════
        self.visualizador.callback("fase_inicio", {
            "numero": 5,
            "fase": "Solution generation"
        })
        
        solucion = self.cliente_llm.generar_solucion(hub_optimo, caso)
        
        self.visualizador.callback("fase_fin", {
            "mensaje": "Solution generated"
        })
        
        # ═══════════════════════════════════════════════════════════════
        # FINAL RESULT
        # ═══════════════════════════════════════════════════════════════
        tiempo_total = (datetime.now() - inicio).total_seconds()
        
        resultado = {
            "success": True,
            "version": "8.5",
            "case": caso.nombre,
            "hub": hub_optimo.to_dict(),
            "solution": solucion,
            "metrics": {
                "total_time_seconds": tiempo_total,
                **self.cliente_llm.obtener_estadisticas(),
                **self.motor_embeddings.obtener_estadisticas(),
                "total_nodes": sum(
                    len(a.todos_los_nodos) for a in arboles
                ),
                "fusions_detected": len(fusiones),
                "equivalence_classes": len(clases_equivalencia),
                "hubs_found": len(hubs),
                "fusion_stats": self.detector_fusiones.stats
            },
            "trees": {a.actor: a.to_dict() for a in arboles},
            "graph_stats": stats_grafo
        }
        
        # Save result and events
        if self.auditoria:
            self.auditoria.guardar_eventos(
                self.visualizador.obtener_eventos()
            )
            self.auditoria.guardar_resultado(resultado)
            self.auditoria.guardar_embeddings_cache(
                self.motor_embeddings.cache
            )
        
        # Show summary
        self.visualizador.mostrar_resumen()
        
        return resultado
    
    def _mostrar_banner(self, caso: Caso):
        """Shows initial banner."""
        print("")
        print("╔" + "═" * 68 + "╗")
        print("║" + " PRINCIPLE OF NATURALITY v8.5 ".center(68) + "║")
        print("║" + " Semantic Convergence & Conflict Resolution ".center(68) + "║")
        print("║" + " (Bicomposite weights + Best-first search) ".center(68) + "║")
        print("╠" + "═" * 68 + "╣")
        print(f"║  Case: {caso.nombre[:56]:<56}  ║")
        print(f"║  Actors: {', '.join(a['nombre'] for a in caso.actores)[:54]:<54}  ║")
        print("╚" + "═" * 68 + "╝")
        print("")
    
    def _resultado_sin_convergencia(
        self,
        caso: Caso,
        arboles: List[Arbol],
        inicio: datetime
    ) -> Dict[str, Any]:
        """Builds result when no convergence is found."""
        tiempo_total = (datetime.now() - inicio).total_seconds()
        
        return {
            "success": False,
            "version": "8.5",
            "case": caso.nombre,
            "message": (
                "No global convergence HUB found connecting all actors"
            ),
            "metrics": {
                "total_time_seconds": tiempo_total,
                **self.cliente_llm.obtener_estadisticas(),
                **self.motor_embeddings.obtener_estadisticas(),
                "total_nodes": sum(
                    len(a.todos_los_nodos) for a in arboles
                ),
                "fusion_stats": self.detector_fusiones.stats
            },
            "trees": {a.actor: a.to_dict() for a in arboles}
        }
    
    def continuar_desde_run(
        self,
        run_dir: Path,
        caso: Caso,
        skip_nli: bool = False
    ) -> Dict[str, Any]:
        """
        Continue from existing run directory.
        
        v8.5: Recalculates bicomposite weights including sim_con_problema.
        """
        inicio = datetime.now()
        
        print("")
        print("╔" + "═" * 68 + "╗")
        print("║" + " CONTINUING FROM PREVIOUS RUN (v8.5) ".center(68) + "║")
        print(f"║  Run: {str(run_dir)[:56]:<56}  ║")
        print("╚" + "═" * 68 + "╝")
        print("")
        
        # Load trees
        arboles_dir = run_dir / "arboles"
        arboles: List[Arbol] = []
        
        for arbol_file in sorted(arboles_dir.glob("*.json")):
            with open(arbol_file, "r", encoding="utf-8") as f:
                data = json.load(f)
            arbol = Arbol.from_dict(data)
            arboles.append(arbol)
            print(f"  ✓ {arbol.actor}: {len(arbol.todos_los_nodos)} nodes")
        
        # Build node lookup
        nodos_por_id = {}
        for arbol in arboles:
            for nodo in arbol.todos_los_nodos:
                nodos_por_id[nodo.id] = nodo
        
        # Recalculate embeddings
        # v8.5: Root uses postura_embedding (neutral) if available
        print("\nRecalculating embeddings...")
        for arbol in arboles:
            for nodo in arbol.todos_los_nodos:
                if nodo == arbol.raiz:
                    actor = caso.obtener_actor(arbol.actor)
                    texto_emb = (
                        actor.get("postura_embedding", nodo.texto)
                        if actor else nodo.texto
                    )
                    nodo.embedding = self.motor_embeddings.obtener_embedding(
                        texto_emb
                    )
                else:
                    nodo.embedding = self.motor_embeddings.obtener_embedding(
                        nodo.texto
                    )
        
        # v8.5: Compute φ(Π) for bicomposite weights
        # Uses neutral version if available to avoid lexical bias
        texto_pi = caso.decision_central_embedding or caso.decision_central
        LOG.info(
            f"Computing φ(Π) for: "
            f"\"{texto_pi[:60]}...\""
        )
        emb_problema = self.motor_embeddings.obtener_embedding(texto_pi)
        
        # Calculate bicomposite weights (sim_con_raiz + sim_con_problema)
        print("Calculating bicomposite weights...")
        for arbol in arboles:
            raiz_embedding = arbol.raiz.embedding
            # Root properties
            arbol.raiz.sim_con_raiz = 1.0
            arbol.raiz.sim_con_problema = float(
                self.motor_embeddings.similitud(
                    arbol.raiz.embedding, emb_problema
                )
            )
            arbol.raiz.peso_arista = None
            arbol.raiz.peso_acumulado = 0.0
            
            # All other nodes
            for nodo in arbol.todos_los_nodos:
                if nodo == arbol.raiz:
                    continue
                nodo.sim_con_raiz = float(
                    self.motor_embeddings.similitud(
                        nodo.embedding, raiz_embedding
                    )
                )
                nodo.sim_con_problema = float(
                    self.motor_embeddings.similitud(
                        nodo.embedding, emb_problema
                    )
                )
                nodo.peso_arista = (
                    (1.0 - nodo.sim_con_raiz)
                    + (1.0 - nodo.sim_con_problema)
                )
                # Accumulated weight from parent
                padre_acum = (
                    nodo.padre.peso_acumulado
                    if nodo.padre and nodo.padre.peso_acumulado is not None
                    else 0.0
                )
                nodo.peso_acumulado = padre_acum + nodo.peso_arista
        
        # Save trees to new audit
        if self.auditoria:
            for arbol in arboles:
                self.auditoria.guardar_arbol(arbol)
        
        # ═══════════════════════════════════════════════════════════════
        # PHASE 2: FUSION DETECTION or LOAD FROM FILE
        # ═══════════════════════════════════════════════════════════════
        
        if skip_nli:
            # Load fusions from previous run
            fusiones_path = run_dir / "fusiones.json"
            if not fusiones_path.exists():
                raise FileNotFoundError(
                    f"fusiones.json not found in {run_dir}"
                )
            
            print(f"\nLoading fusions from: {fusiones_path}")
            
            with open(fusiones_path, "r", encoding="utf-8") as f:
                fusiones_data = json.load(f)
            
            from core.estructuras import Fusion
            fusiones = []
            for fd in fusiones_data.get("fusiones", []):
                nodo_a = nodos_por_id.get(fd["nodo_a_id"])
                nodo_b = nodos_por_id.get(fd["nodo_b_id"])
                if nodo_a and nodo_b:
                    fusion = Fusion(
                        nodo_a=nodo_a,
                        nodo_b=nodo_b,
                        similitud=fd.get("similitud", 0.0),
                        auto=fd.get("auto", True)
                    )
                    fusiones.append(fusion)
            
            print(f"  ✓ Loaded {len(fusiones)} fusions")
        else:
            self.visualizador.callback("fase_inicio", {
                "numero": 2,
                "fase": "Semantic fusion detection (S-BERT + NLI)"
            })
            
            fusiones = self.detector_fusiones.detectar_fusiones(arboles)
        
        # ═══════════════════════════════════════════════════════════════
        # PHASE 2b: LLM FUSION VERIFICATION
        # ═══════════════════════════════════════════════════════════════
        rejected_role_inversions = []
        if self.config.get("verify_fusions_llm", True) and fusiones:
            self.visualizador.callback("fase_inicio", {
                "numero": "2b",
                "fase": "LLM fusion verification"
            })
            
            fusiones, rejected_role_inversions = (
                self.detector_fusiones.verificar_fusiones_lote(
                    fusiones,
                    self.cliente_llm,
                    batch_size=self.config.get(
                        "verify_fusions_batch_size", 15
                    )
                )
            )
        
        # Build equivalence classes
        clases_equivalencia = (
            self.detector_fusiones.construir_clases_equivalencia(fusiones)
        )
        
        if self.auditoria:
            self.auditoria.guardar_fusiones(fusiones)
            if rejected_role_inversions:
                self.auditoria.guardar_rechazos_inversion(
                    rejected_role_inversions
                )
            self.auditoria.guardar_clases_equivalencia(clases_equivalencia)
        
        # ═══════════════════════════════════════════════════════════════
        # PHASE 3: GRAPH CONSTRUCTION (v8.5 - bicomposite)
        # ═══════════════════════════════════════════════════════════════
        self.visualizador.callback("fase_inicio", {
            "numero": 3,
            "fase": "Unified graph construction (v8.5 - bicomposite)"
        })
        
        grafo = self.constructor_grafo.construir(
            arboles, fusiones, clases_equivalencia
        )
        
        if self.auditoria:
            self.auditoria.guardar_grafo(grafo)
            self.constructor_grafo.exportar_graphml_con_fusiones(
                grafo, fusiones,
                self.auditoria.run_dir / "grafo_con_fusiones.graphml"
            )
        
        stats_grafo = self.constructor_grafo.obtener_estadisticas(grafo)
        
        # ═══════════════════════════════════════════════════════════════
        # PHASE 4: HUB SEARCH (v8.5 - bicomposite distances)
        # ═══════════════════════════════════════════════════════════════
        self.visualizador.callback("fase_inicio", {
            "numero": 4,
            "fase": "Convergence HUB search (v8.5 - bicomposite)"
        })
        
        hubs = self.buscador_hubs.buscar(
            grafo, arboles, clases_equivalencia
        )
        
        if self.auditoria:
            self.auditoria.guardar_hubs(hubs)
        
        if not hubs:
            return self._resultado_sin_convergencia(caso, arboles, inicio)
        
        hub_optimo = hubs[0]
        
        print("\n" + self.buscador_hubs.explicar_seleccion(hubs))
        print(self.buscador_hubs.generar_reporte_hub(hub_optimo))
        
        # ═══════════════════════════════════════════════════════════════
        # PHASE 5: SOLUTION GENERATION
        # ═══════════════════════════════════════════════════════════════
        self.visualizador.callback("fase_inicio", {
            "numero": 5,
            "fase": "Solution generation"
        })
        
        solucion = self.cliente_llm.generar_solucion(hub_optimo, caso)
        
        # ═══════════════════════════════════════════════════════════════
        # FINAL RESULT
        # ═══════════════════════════════════════════════════════════════
        tiempo_total = (datetime.now() - inicio).total_seconds()
        
        resultado = {
            "success": True,
            "version": "8.5",
            "case": caso.nombre,
            "continued_from": str(run_dir),
            "hub": hub_optimo.to_dict(),
            "solution": solucion,
            "metrics": {
                "total_time_seconds": tiempo_total,
                **self.cliente_llm.obtener_estadisticas(),
                **self.motor_embeddings.obtener_estadisticas(),
                "total_nodes": sum(
                    len(a.todos_los_nodos) for a in arboles
                ),
                "fusions_detected": len(fusiones),
                "equivalence_classes": len(clases_equivalencia),
                "hubs_found": len(hubs),
                "fusion_stats": self.detector_fusiones.stats
            },
            "trees": {a.actor: a.to_dict() for a in arboles},
            "graph_stats": stats_grafo
        }
        
        if self.auditoria:
            self.auditoria.guardar_eventos(
                self.visualizador.obtener_eventos()
            )
            self.auditoria.guardar_resultado(resultado)
        
        self.visualizador.mostrar_resumen()
        
        return resultado