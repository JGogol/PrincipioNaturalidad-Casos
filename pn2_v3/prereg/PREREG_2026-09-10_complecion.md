# PREREG — Compleción de bajo rango del campo del juez (corpus de consenso)
Congelado 2026-09-10, ANTES de correr `complecion.py`. No se renegocia tras ver resultados.

## Antecedente
Rango efectivo medido (2026-09-09, consenso, campo J2 completo): contradicción 2/19/74 componentes
para 50/80/90% de la energía, contra 90/254/353 del nulo permutado. Implicación 12/71/138 vs 124/278/373.

## Pregunta
Si el campo de contradicción de J2 es de rango bajo, ¿se puede RECONSTRUIR desde una fracción de las
entradas, y sobreviven los objetos del cálculo (kernels) computados sobre la reconstrucción?

## Datos (congelados)
`campo2_moritz.jsonl`: 753 nodos, 5 actores, 453.604 entradas inter-actor ordenadas, juez moritz, P~ aplicada.
Solo se reconstruye la matriz de contradicción FC de J2. J1 y los nulos se toman como dados
(no se reconstruyen): así la pregunta queda aislada.

## Método (congelado)
- Compleción: soft-impute (SVD iterada con relleno por la estimación corriente), rango r ∈ {1,2,5,10,20,50}.
- Dos regímenes de enmascarado, semilla 20260910:
  * **E (entradas)**: se ocultan al azar el 80% de las entradas observadas. Régimen fácil.
  * **F (filas)**: se ocultan TODAS las entradas de un 80% de los nodos como pivote. Régimen realista:
    equivale a certificar el 20% de los nodos y completar el resto.
- Selección de r: por error en un 10% de validación separado del test.

## Baselines (obligatorios)
media global; media por bloque actor×actor; modelo aditivo fila+columna.

## Medidas
RMSE sobre las entradas ocultas; y la prueba dura: el conjunto de kernels
K = {(u,w): min(FC[u,w],FC[w,u]) ≥ 0.50 en J2 ∧ max(fc,bc) ≥ 0.30 en J1 ∧ ninguno nulo}
recomputado sobre la matriz reconstruida, comparado contra los 12.356 kernels reales (precisión, recall, F1).

## Predicciones (congeladas)
- P1: en régimen E, RMSE de la compleción ≤ 0.80 × RMSE del mejor baseline, con r ≤ 20.
- P2 (decisiva): en régimen E, F1 de los kernels reconstruidos ≥ 0.80.
- P3 (realista): en régimen F, F1 de los kernels sobre pares con ambos extremos ocultos ≥ 0.70.

## Criterios de muerte
- P1 falla → el rango bajo no da compleción útil; se cierra la línea y se reporta como negativo.
- P2 falla → aunque el RMSE baje, los objetos del cálculo no sobreviven: no hay método. Decisivo.
- P3 falla con P2 cumplida → la compleción sirve para rellenar huecos, no para ahorrar certificación.
  Se reporta como alcance limitado, no como método de ahorro.
