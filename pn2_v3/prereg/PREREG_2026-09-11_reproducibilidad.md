# PREREG 2026-09-11 — ¿Se reproducen las proposiciones?

Escrito **antes** de regenerar. Nada de lo que sigue se decide después de ver el resultado.

## La pregunta

El mismo problema, la misma premisa, el mismo modelo y la misma configuración, ¿producen las
mismas proposiciones?

Hoy no se sabe. Todo número del paper que se cuente en proposiciones —terreno común 370,
puentes 304, base canónica 610, `|Ω|`— se reporta como medida sin saber si es una tirada.

## El diseño

Se regenera el árbol del GERD con `casos/gerd/input.json`, cuya descripción del problema
tiene `md5 = a8f663c8`: **exactamente la de la corrida 2026-05-01_01-11-03**, sobre la que
está construido el paper. Mismo `claude-sonnet-4-20250514`, misma config, misma premisa
(las raíces vienen del input, no del generador).

Se corta el pipeline cuando los árboles están en disco. **No se corre NLI**: la comparación es
texto contra texto y no necesita juez.

## La medida

Para cada proposición del árbol nuevo, su mejor par en el árbol viejo del mismo actor:

1. **exacta** — mismo texto tras minúsculas y recorte de espacios;
2. **TF-IDF** — coseno del mejor par, con `TfidfVectorizer(sublinear_tf=True,
   stop_words='english')` ajustado sobre la unión de los dos árboles de ese actor.

Es el mismo procedimiento con el que hoy se midió 28-04 contra 01-05 —problemas
distintos— y dio **0,3 % exacto** y **2,1 % con coseno ≥ 0,70**. Los números son comparables.

La raíz se excluye del cómputo: viene del input y coincide por construcción.

## Qué se concluye, fijado ahora

| resultado | conclusión |
|---|---|
| **≥ 50 %** con coseno ≥ 0,70 | las proposiciones se reproducen. El 0,3 % medido hoy es atribuible al cambio de problema, y los conteos del paper son medidas. |
| **< 20 %** | las proposiciones **no** se reproducen. Todo conteo de proposiciones del paper es una tirada, §9 tiene que decirlo, y la unidad de análisis sube a las estructuras. |
| entre 20 % y 50 % | reproducibilidad parcial: la unidad de análisis hay que argumentarla, no suponerla. |

En los tres casos el resultado se reporta. Ninguno se descarta por incómodo.

## Qué NO contesta

Un árbol, una tirada, un caso, un actor. Si el resultado cae en la zona baja, alcanza para
obligar a cambiar lo que se afirma; no alcanza para estimar cuánto varía.

Y no contesta la pregunta que sigue, que es la que importa más: **¿se reproducen las
estructuras aunque no se reproduzcan las proposiciones?** El terreno común, su reparto por
actor, el tamaño de la base canónica. Eso exige el pipeline completo con NLI y queda para
después de ver este resultado.

## Costo

30 minutos por árbol, solo API. El NLI —el cuello de botella sin GPU— no interviene.
