# PREREG — Rango efectivo del campo del juez (GERD, Selene)
Fecha de congelamiento: 2026-09-09, ANTES de correr `campo2_muestra.py` sobre GERD/Selene.

## Antecedente (medido, consenso, 753 nodos, campo J2 completo)
Contradicción: 2 / 19 / 74 componentes para 50 / 80 / 90 % de la energía; nulo permutado 90 / 254 / 354.
Implicación: 12 / 71 / 138; nulo 124 / 279 / 373. Primera componente = "NT contra todos".

## Pregunta
¿El campo del juez T[u,v,·,·,contr] tiene rango efectivo bajo también en los conflictos (GERD real, Selene ficticio)?

## Instrumento (congelado)
- J2 = MoritzLaurer/DeBERTa-v3-large-mnli-fever-anli-ling-wanli, textos con P~ (alias), bidireccional.
- Muestra de FILAS: 50 pivotes por actor, elegidos con semilla 20260909, contra TODOS los nodos de los otros actores
  (GERD: 150 pivotes × ~1.003 columnas ≈ 150k pares; Selene ídem). El espectro de un subconjunto de filas
  estima la dimensión del espacio columna; no es el campo completo y se declara así.
- Además, vista previa con J1 (juez base, campo prefiltrado por coseno ≥ 0.35; pares no evaluados = 0, declarado).

## Medida
k_p = mínimo número de componentes singulares que capturan la fracción p de la energía (suma de σ²), p ∈ {0.5, 0.8, 0.9},
sobre la matriz de contradicción (fc) y la de implicación (fe); nulo = misma matriz con entradas permutadas (semilla fija);
variante centrada (se resta la media) para descontar la componente constante.

## Predicciones (congeladas)
- P1 (rango bajo): k_0.8(observado) ≤ 0.25 · k_0.8(nulo) en contradicción, en GERD y en Selene.
- P2 (exploratoria, sin predicción): en qué actor se concentra la primera componente. Se reporta.

## Criterio de muerte
- P1 falla en un caso → "rango bajo no generalizable" para ese caso; se reporta tal cual.
- k_0.8(obs) > 0.5 · k_0.8(nulo) en ambos → hipótesis refutada; el tema no entra al paper salvo como negativo.
