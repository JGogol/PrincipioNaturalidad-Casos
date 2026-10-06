# PREREG — PN3 fase 1: certificación de estímulos E sobre el baseline de consenso
Fecha de congelamiento: 2026-09-02 (ANTES de correr). No se renegocia nada tras ver resultados.

## Pregunta
¿Qué estímulo E logra que los 5 actores respondan con nodos certificados (transitividad por actores), con NT representado?

## Baseline (congelado)
- F de consenso (doble juez, gate del paper): 904/1557 aristas → `F_consenso_parcial.json` (checkpoint completo 2026-09-02).
- 47 clases; UNA con los 5 actores: gigante de 294 nodos (NT: 11) → `baseline_fase1_clase_gigante.json`.
- Bosque congelado: `estado_nodos.json` (753 nodos), sin mutación en esta fase (solo medición).

## Familias E (textos congelados; ver `fase1_certificar_E.py`)
Familia = texto base + 4 paráfrasis (lección fase 0: sensibilidad a paráfrasis, Jaccard 0.181).
- **E_A** mapa de colores (institucional-visual) — incluye E1 y C4 ya corridos.
- **E_B** aviso anticipado del docente (institucional-procedimental).
- **E_C** rutina fija publicada (institucional-estructural).
- **CTRL** anti-E (transiciones sin aviso) — control de dirección; incluye C3.

## Instrumento (congelado)
- Jueces: base (nli-deberta-v3-base ONNX) y moritz (DeBERTa-v3-large-mnli-fever-anli-ling-wanli).
- Gate por par (mismo del paper): ent_avg = (fe+be)/2 ≥ 0.55 ∧ max(fc,bc) < 0.30.
- cert(texto, nodo) = gate bajo AMBOS jueces.
- Cert(familia) = nodos con cert en ≥3/5 miembros de la familia.
- Resta de baseline por nodo: se excluye todo nodo que pase el gate con C1 o C2 (off-topic ya corridos) bajo CUALQUIER juez.

## Loss (congelada — lexicográfica, definición de Javi)
- **PRIMARIO (binario)**: Cert(familia) contiene ≥1 nodo de CADA uno de los 5 actores.
- **Desempate** (solo entre familias que cumplen): 1) nº de nodos NT certificados; 2) balance por actor (min/max de conteos); 3) intersección con la clase gigante baseline.
- **NUNCA se optimiza la suma global de ENT/NLI** ("la suma premia la vaguedad").

## Predicciones (congeladas)
- P1: al menos una familia institucional (E_A, E_B, E_C) cumple el PRIMARIO.
- P2: CTRL NO cumple el PRIMARIO.
- P3: C1 y C2 certifican ≤3/753 nodos cada uno bajo doble juez (texto único, sin familia).

## Criterios de muerte
- P1 falla → la familia institucional no conecta a los 5 actores en este caso; se registra y el gate no se renegocia.
- P2 falla → el instrumento no discrimina la dirección del estímulo; P1 queda sin lectura (resultado de instrumento, no de contenido).

## Cómputo
20 textos × 753 nodos × 2 jueces; reutiliza scores existentes de E1, C4 y C3. Reanudable por texto.
Resultados → `fase1_resultado.json`; ningún archivo de estado se modifica.
