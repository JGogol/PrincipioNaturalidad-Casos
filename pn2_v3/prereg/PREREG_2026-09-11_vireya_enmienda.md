# Enmienda 2026-09-11 — Vireya sale de la mitad de inyección

Escrita **antes** de correr `lote_v4.bat`. Enmienda a `PREREG_2026-08-27_tercer_caso.md`.

## Criterio

Un corpus no es legible para la oposición si sus nodos contradicen premisas **irrelevantes**.
La medida es la contradicción máxima de cada nodo contra las tres premisas nulas, bajo el
juez 2 (el que no tiene el sesgo de negación), promediada por actor.

El criterio mira el corpus, **no** el resultado de P1. Vireya pasa P1 en un régimen de texto
y lo falla en el otro; esta exclusión no depende de cuál se elija, y por eso se escribe ahora.

## Medido

```
                       contradicción media contra premisas nulas (juez 2)
GERD      0,126  0,061  0,037
Selene    0,193  0,224  0,170
Vireya    0,450  0,459  0,448
```

Los nodos de Vireya contradicen a una premisa irrelevante diez veces más que los de GERD,
con el juez limpio. No se corrige cambiando de juez.

## Confirmación (no es el criterio)

- Ondrel contra Tessara: **53.876 de 59.182** pares evaluados contradicen (91,0 %). Las seis
  parejas de GERD y Selene están entre 27,9 % y 40,0 %.
- Inspeccionados los seis pares de contradicción 1,000/1,000 entre Ondrel y Tessara: son
  proposiciones sobre temas distintos que no se contradicen.
- La implicación dirigida sobre umbral tiene en Vireya una componente fuertemente conexa de
  **711 nodos**, el 47 % del corpus (GERD 123, Selene 342).
- Los nodos de Vireya tienen 13–14 palabras contra 8–9 en los otros dos. La tasa de negación
  no difiere (12–16 % contra 11–26 %), así que no es el sesgo de negación.

## Consecuencia

No es una decisión: la capa de contradicción de Vireya no mide contradicción, y todo lo que
se construye sobre ella no entra. Vireya queda fuera de **§6, §7 y §8** — inyección, costo de contracción, corte y rescate.

Su mitad estática se conserva y no depende del juez de contradicción: terreno común 772,
puentes 607, base canónica 679 = 72 + 607, verificados desde el objeto.

**P1 de `PREREG_2026-08-27_tercer_caso` no se evalúa.** No se reporta como pasado ni como
fallado: se reporta como **no evaluable**, con esta causa y estos números.

## Pre-registro para el próximo caso

Antes de certificar la inyección se miden las tres premisas nulas contra todos los nodos con
el juez 2. Si la contradicción media por actor alcanza **0,35**, el corpus se declara no
legible para la oposición y no se certifica.

El umbral se fija entre Selene (0,196, el caso legible de márgenes más finos) y Vireya
(0,452). Son tres puntos: es una elección, no una constante estimada, y es la única parte de
esta enmienda que no sale de la medición.

**Criterio de muerte de la regla.** La regla queda refutada si aparece un corpus con
contradicción nula media ≥ 0,35 cuyo perfil de oposición resulta legible —oposición
diferencial entre actores, estable bajo paráfrasis—, o uno por debajo de 0,35 que no lo sea.
En cualquiera de los dos casos el umbral se retira y se reporta la refutación, no se reajusta
para salvarlo.

## Costo evitado

4,75 h de cómputo sobre un corpus cuya capa de contradicción no es utilizable.
