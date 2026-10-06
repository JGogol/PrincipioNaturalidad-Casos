# pn2_v33 — mediciones nuevas para PN II v3.3 (24-sep-2026)

Segunda implementación del objeto, independiente de `dinamica/objeto.py`, hecha desde las
definiciones del paper (`objeto_min.py`). `validar.py` reproduce exactamente §2.2, §3.6, §3.7, §4.6 y
§4.7 en los dos casos (F 102/218, terreno común 370/551, puentes 304/442, base 610/673, regiones
distintas 1403/1271, reducto 94/183, regiones de raíz).

Correr desde esta carpeta (necesita numpy, scipy, pandas):

    python validar.py            # reproduce las cifras publicadas
    python nulo_F.py gerd 200    # modelo nulo de F  -> nulo_gerd.json
    python nulo_F.py selene 200  #                   -> nulo_selene.json
    python resumen_nulo.py       # tabla (RESULTADOS_nulo.txt)
    python clase_D.py            # quitar los pares clase D + triángulos (RESULTADOS_claseD.txt, clase_D.json)

## 1. Modelo nulo de F (`nulo_F.py`, 200 repeticiones, semilla 2026)

Tres nulos con |F| pares inter-actor: **campo** (al azar entre los pares que el juez vio),
**grados** (F reconectada preservando el grado de cada nodo en F), **uniforme** (pares inter-actor
cualesquiera). Ver RESULTADOS_nulo.txt. Lo que dice:

- **Terreno común**: Nilo 370 está en la mediana del nulo de campo (368, percentil 50). Selene 551
  está *por debajo* de todos los nulos (campo 1288, grados 694). En ningún caso es mayor que el
  nulo: su tamaño no es evidencia de convergencia; lo que puede serlo es su contenido.
- **Puentes y base**: por debajo del nulo de campo en los dos casos (304 vs 410; 442 vs 629): F se
  concentra en menos nodos que el azar. Bajo el nulo de grados son idénticos por construcción:
  dependen sólo de *qué* nodos tienen fusión, no de con quién.
- **Modularidad** (fracción de intersecciones vacías): en el nulo (0,973 vs 0,98; 0,920 vs 0,95).
  Es propiedad del bosque con F rala, no de F.
- **Regiones de los puentes**: mediana 29,5 y 538 contra 10 y 15 en el nulo de campo (percentil
  100) y contra 315 y 675 en el nulo de grados (percentil 0–1). F encadena más que pares al azar y
  mucho menos que la misma secuencia de grados reconectada: F es asortativa (agrupada por tema).
- **Triángulos inter-actor** (tres actores, tres pares): 3 y 8 contra 0 en el nulo de campo.

## 2. Propagación del error presente (`clase_D.py`)

Quitar los pares clase D de `paper/notas/clasificacion_F_2026-09-24.md` (26 de 102; 71 de 218) y
recalcular, contra quitar la misma cantidad al azar (200 rep.):

| | Nilo obs | sin D | azar (mediana, IC95) | Selene obs | sin D | azar |
|---|---:|---:|---|---:|---:|---|
| terreno común | 370 | **146** | 232 [123, 347] (pct 8) | 551 | **346** | 425 [323, 490] (pct 10) |
| reparto del terreno común | 106·70·194 | 19·9·118 | | 123·184·244 | 62·76·208 | |
| puentes | 304 | 265 | 266 [252, 279] | 442 | 363 | 377 [360, 392] |
| mediana \|Reg\| de puentes | 29,5 | 18 | 25 [17, 38] | 538 | 154 | 346 [65, 473] |
| triángulos | 3 | 3 | 1 [0, 3] | 8 | 2 | 2 [0, 5] |

Quedándose sólo con la clase S: terreno común 26 (Nilo) y 163 (Selene). Sin D ni P en Selene: 184.

Lectura: los pares que un lector rechaza sostienen el 61 % del terreno común del Nilo y el 37 % del
de Selene, más de lo que sostiene la misma cantidad de pares al azar (§3.8 medía eso último). En
Selene 6 de los 8 triángulos pasan por un par D.

## 3. Terreno común estricto (una fusión desde cada raíz) y el ejemplo de §3.1

| | F | sin D | sin D ni P | sólo S |
|---|---:|---:|---:|---:|
| Nilo, estricto | 80 | 32 | 32 | 4 |
| Selene, estricto | 283 | 215 | 75 | 68 |

El ejemplo de §3.1 (*Sudan requests African Union mediation intervention*): columna 161 → 75 sin D;
las tres premisas siguen alcanzándola sin D. Con sólo la clase S la alcanza únicamente Sudán: los
tres pares que la sostienen son clase C (misma medida, cada actor la suya).
