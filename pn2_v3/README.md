# Cálculo de razonamientos (PN II, v3.4) — depósito de datos

Todo lo necesario para recomputar cada cifra del paper *Cálculo de razonamientos — Un método para
comparar razonamientos que parten de premisas incompatibles* (El Principio de la Naturalidad, Parte
II, v3.4) **sin el código del pipeline**: los objetos de razonamiento de los dos casos y del corpus
excluido, las anotaciones, los prompts y una implementación independiente de las lecturas escrita
desde las definiciones del paper.

Licencia: CC-BY-4.0. Autor: Elio Javier Gogol Merletti (ORCID 0009-0004-4294-4816).

## Contenido

```
objeto/gerd/      el objeto (A, T, M) del caso del Nilo         1504 proposiciones
objeto/selene/    el objeto del caso Selene                      1503 proposiciones
objeto/vireya/    el tercer corpus, EXCLUIDO por el control de admisión (§4.2 del paper)
lectura/          segunda implementación de las lecturas + modelo nulo + pares rechazados
anotacion/        clasificación de los 320 pares certificados; anotación a ciegas de 36 pares del Nilo
prompts/          premisas nulas literales; prompt de la verificación Λ; la normalización P̃ (p_tilde.py) y su régimen
controles/        salidas de los controles del juez: premisas nulas, control cruzado, prefiltro
prereg/           los pre-registros (fase 1, rango, compleción, reproducibilidad, admisión del tercer corpus)
resultados/       salidas de todos los cómputos que el Apéndice D reporta (consolidación, terreno común, saltos, reducto, cobertura, rango bajo, …)
```

### objeto/<caso>/

| archivo | qué es |
|---|---|
| `nodos.csv` | una fila por proposición: `id`, `actor`, `padre`, `profundidad`, `texto`, pesos y similitudes de la expansión |
| `A.npz` | matriz de adyacencia del bosque (scipy CSR, booleana): `A[p, c] = 1` si `c` cuelga de `p` |
| `T.npz` | el tensor del juez, una matriz CSR `uint16` por capa: `j1_impl`, `j1_contr`, `j2_impl`, `j2_contr`, `sim`, `rechazo`, `intra_impl`, `intra_contr`, `j1_ph_*`, `j2_ph_*` |
| `emb.npy` | embeddings S-BERT (768 dim.) de cada proposición, en el orden de `nodos.csv` |
| `gluts.csv` | pares intra-actor que pasaron el criterio previo de la Parte I (capa intra-actor; no se lee en el paper) |
| `rechazos.csv` | los vetos de la verificación Λ, con la razón que dio el modelo |
| `meta.json` | actores, umbrales (τ, γ, δ, coseno), criterio de implicación, identificadores de los dos jueces, codificación, verificación de fidelidad (`F_datos = F_rearmado`) |

Codificación de los puntajes: `uint16`, `x = (q − 1) / 65534`, `q = 0` significa *no evaluado*.
El certificado `F` se rearma desde `T` con las cinco condiciones de §2.3 del paper; `lectura/validar.py`
lo hace y reproduce 102 / 218 pares.

### lectura/

`objeto_min.py` lee el objeto y calcula `Sub`, `F`, `Reg = Sub·(F·Sub)*`, terreno común, puentes,
base, regiones distintas, modularidad y reducto, desde las definiciones del paper y sin código del
pipeline. `validar.py` compara con las cifras publicadas. `nulo_F.py` corre el modelo nulo de `F`
(§4.3); `clase_D.py` quita los pares rechazados y cuenta triángulos. `nulo_*.json`, `clase_D.json` y
`RESULTADOS_*.txt` son las salidas que el paper y su Apéndice D reportan.

```
cd lectura
python validar.py            # reproduce §2.2, §3.3, D.1.6, D.1.7 (numpy, scipy, pandas)
python nulo_F.py gerd 200    # ~30 s
python nulo_F.py selene 200  # ~60 s
python clase_D.py
```

### anotacion/

`clasificacion_F_2026-09-24.md`: los 320 pares certificados con su clase (S / C / P / D), lectura
de un modelo de lenguaje declarada como tal en §4.2 del paper. `pares_gerd_36_CLAVE.txt`: los 36
pares de la anotación a ciegas del Nilo, con la clave (certificado / rechazado por poco) que los
anotadores no vieron; `pares_selene_24_CLAVE.txt`, ídem Selene; `prompt_anotacion.txt`, la
consigna; `anotacion_claude*.txt`, la anotación de uno de los modelos.

### prompts/

`premisas_nulas.txt`: las tres premisas nulas literales del control de §4.2. `verificacion_roles.txt`:
el prompt literal de la verificación Λ (modelo `claude-sonnet-4-20250514`, lotes de 15, temperatura 0).

### controles/

`premisas_nulas/cert_<caso>_null*.jsonl`: cada línea es una premisa nula (`E`) contra un actor
(`a`); `r` tiene una fila por proposición: `[id, j1_impl E→v, j1_impl v→E, j1_contr E→v, j1_contr
v→E, j2_impl E→v, j2_impl v→E, j2_contr E→v, j2_contr v→E]`. Tres regímenes de texto: sin sufijo
(normalización mínima, el que usa el control de admisión), `_ptilde` y `_v4` (P̃). Los `log_*.txt`
registran cada corrida. `control_cruzado/cruz_F.jsonl`: el juez 1 sobre los 45.000 pares cruzados
Nilo×Selene (`b` = lote; filas `[estrato, i_nilo, i_selene, coseno, impl→, impl←, contr→, contr←]`);
`cruz_muestra.json`, los pares muestreados por estrato; `cruz_cos.npy`, el coseno de los 2.260.512
pares cruzados. `prefiltro/`: la muestra de 132 000 pares descartados del Nilo evaluada por el juez 1
(`pasan` vacío en los 66 bloques).

### prereg/ y resultados/

Los pre-registros fechados, tal como se escribieron antes de cada corrida, y las salidas de los
cómputos que el Apéndice D del paper reporta, con los nombres que los comentarios de procedencia del
paper citan.

## Lo que este depósito no contiene

- El código del pipeline que construye el bosque y llena el tensor (registrado, licencia
  restrictiva). El Apéndice C del paper documenta el procedimiento al nivel que hace falta para
  repetirlo; el generador (`claude-sonnet-4-20250514`) fue retirado por su proveedor.
- Los scripts de análisis que produjeron `resultados/` (llaman a los jueces o al pipeline); se
  publican sus salidas y, en `lectura/`, una implementación independiente que reproduce las cifras
  centrales desde el objeto. `p_tilde.py` sí se incluye porque define la relación que `F` certifica.
- Las anotaciones del autor y de los modelos A, B y C sobre los 36 pares del Nilo (sólo está la de
  uno de los modelos). El paper ya no reporta esa anotación.

## Complementos (agregados el 6-oct-2026)

```
complementos/propuestas/       E_arbol_gerd.json, E_arbol_selene.json: las dos propuestas y sus consecuencias (V_E, 17 y 13 nodos)
complementos/juez_propuesta/   cert_<caso>_v4.jsonl: los dos jueces entre cada consecuencia de la propuesta y cada nodo del caso;
                               cert_cruzado_*: la propuesta de un caso contra los nodos del otro (control de la propuesta ajena)
complementos/contraccion_v4/   hansson_*.json, levi_*.json y sus logs: salidas de la corrida del 15-sep (régimen P̃ v4)
recalculo/                     scripts que recomputan desde el objeto tres cifras que antes no tenían archivo, con sus salidas
```

Formato de `cert_*.jsonl`: igual que `controles/premisas_nulas/` (una línea por consecuencia `E` y actor
`a`; `r` = `[id, j1_impl E→v, j1_impl v→E, j1_contr E→v, j1_contr v→E, j2_impl E→v, j2_impl v→E,
j2_contr E→v, j2_contr v→E]`).

`resultados/hansson_*.json` y `resultados/levi_*.json` son de una corrida anterior (9-sep); las de
`complementos/contraccion_v4/` son las del régimen v4. El paper no reporta tablas de contracción: se
publican como datos.

### recalculo/

```
cd recalculo
python asimetria.py         # §2.1: T[u,v] contra T[v,u], con varias reglas de conteo
python padre_hijo.py        # §2.1: mediana del juez sobre las aristas padre -> hijo (sólo Nilo)
python control_cruzado.py   # §4.2 / S4.2: observado vs. estimado, antes y después de Λ, por bandas
```

Requiere numpy, scipy y pandas. Las salidas están en `RESULTADOS_*.txt`.
