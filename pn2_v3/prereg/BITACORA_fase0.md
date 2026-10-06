# BITACORA FASE 0 - dinamica (30-ago-2026)

## Corrida 1: E1 mapa de colores (juez base, sin umbral fijo)
- 753 nodos, curva completa tau 0.05-0.95. Meseta 0.3-0.9 (distribucion
  bimodal, 205 nodos con contr>=0.99).
- Cascada D1: ~616/753 muertos (84%), F 1517/1557 muertas, gigante 467->5.
- Lectura preliminar (LUEGO CORREGIDA por controles, ver abajo).

## Corrida 2: bateria de controles C1-C4
- C1 pasta / C2 edificio (off-topic): matan 346 y 314 a tau=0.9 - MAS que
  la E real (260). EL NULL SUPERA A LA SENAL EN CRUDO.
- C4 parafrasis: Jaccard con E1 = 0.703 (estabilidad > contaminacion 0.52).
- C3 anti-E: espejo perfecto en refuerzo (6/6 reforzados por E1 mueren
  bajo C3; 22/36 al reves).

## DIAGNOSTICO (medido, no especulado): sesgo hipotesis-sola con negacion
- Bajo PASTA: nodos con negacion contr media 0.785 (70% >=0.9) vs 0.473
  (33%) sin negacion. Bajo EDIFICIO: 0.828/73% vs 0.412/25%.
- Correlacion contr(pasta) vs contr(edificio) por nodo: r=0.644 - la
  "contradiccion" es propiedad del NODO (su forma negada), no de la
  premisa. Clase de artefacto: poliak2018 (hypothesis-only bias), en
  escala: 264/753 nodos de este corpus estan negados.
- NO es bug de codigo: wrapper y labels verificados linea por linea.

## SENAL SOBRE NULL (el rescate del experimento)
- Muerte diferencial: contr(E1) alto Y contr(nulls) bajo -> 37 nodos.
  Composicion: NT 1, subtipos 36 (9/10/8/9). Validez de cara perfecta
  (lacks explicit rule / without explicit statement / ignores implicit
  cues / feedback too late).
- Poda corregida (37 semillas + cascada D1): 250/753 muertos
  (NT solo 2!), F vivas 568/1557, tasa endpoints 0.72->0.60.
- Estabilidad del residual bajo parafrasis: Jaccard 0.429 (n chicos;
  mejorar con correccion por baseline + doble juez).

## DISENO FASE 1 (emergente de fase 0)
1. Muerte por score corregido: contr(E) - max(contr(C1),contr(C2)) >= delta
   (correccion por linea de base por-nodo, estilo baseline de REINFORCE).
2. Doble juez: aplicar_E.py --modelo moritz (agregado hoy) sobre las 5
   premisas; muerte exige ambos jueces.
3. Cascada D1 sobre el conjunto corregido; recomputar objeto; comparar
   contra null de intervencion (C1/C2 con el mismo pipeline completo).
4. Todo con prereg antes de correr; los umbrales delta se congelan ahi.

## Corrida 3: los cinco _m bajo moritz (segundo juez)
- EL SESGO ERA DEL JUEZ 1, CONFIRMADO: bajo moritz, pasta mata 22 (era
  346) y edificio 5 (era 314); nodos negados bajo pasta caen de media
  0.785 a 0.135 (70% -> 6% sobre 0.9). Moritz es juez limpio en este
  corpus.
- Con juez limpio la senal existe: E1 mata 159 vs null 5-22 (ratio
  7-30x; con el juez base el null SUPERABA a la senal).
- HALLAZGO NUEVO (leccion 2 de fase 0): sensibilidad al refraseo.
  La parafrasis C4 mata solo 37 vs 159 de E1 (Jaccard 0.181). El
  veredicto de muerte depende de la redaccion exacta de E incluso con
  juez limpio. REGLA PARA FASE 1: E se presenta como FAMILIA de
  parafrasis; muerte = consenso entre parafrasis (ensemble), no una
  redaccion unica.
- Espejo bajo moritz: C3 anti-E mata 116 - activo y del lado correcto.

## MUERTE v2 (el instrumento que emerge de fase 0)
muerte(nodo) = contr(E) alto  AND  contr(nulls) bajo  AND  ambos jueces.
- Semillas doble-juez: 15 nodos (subtipos 14, NT 1), validez de cara
  perfecta (lacks explicit rule / implicit expectation not detected /
  feedback too late / prediction error persists without explicit
  statement).
- Ver muerte_v2_E1.json (semillas + cascada).

## EL PUNTAJE DE FASE 2 (definicion de Javi, congelada 30-ago — no ablandar)
Advertencia registrada: "sumar convergencia global (NLI agregado) NO es
mas fuerte que lograr transitividad global. Uno es un indicador numerico;
el otro es 'logre que razonen igual'." Precedentes medidos: Selene (max
convergencia agregada, estructura inutil) vs GERD (min agregada, el
triangulo UA); la propuesta vaga re-certificaba al maximo en tm_recert.
LA SUMA PREMIA LA VAGUEDAD.

puntaje(E) — orden lexicografico, dos niveles que NO se mezclan:
1. PRIMARIO (binario): el bosque re-expandido bajo E produce una clase
   de equivalencia certificada que contiene a los k actores, con:
   supervivencia al juez 2, ipvr >= umbral (prereg), y representacion
   minima por actor dentro de la clase (composicion lambda de la clase
   — sin actores token de 1 nodo).
2. DESEMPATE (solo entre los E que cumplen 1): calidad de esa clase —
   ipvr, supervivencia, balance entre actores, robustez a parafrasis
   (ensemble) y a nulls off-topic. El ENT como magnitud vive aca,
   ultimo, nunca como objetivo.
La suma global de NLI no aparece en ningun nivel: diagnostico, jamas
optimizacion.

## Correccion asentada (30-ago, noche)
El "colapso del gigante 467->5" de la manana queda RETIRADO: se computo
con el conjunto de muerte del juez sesgado (616 muertos). Con la muerte
v2 (36 muertos), F conserva 1413/1557 y el gigante casi no se mueve.

---
## 2026-09-02 — Cierre de campo + prereg fase 1
- campo2_moritz.jsonl COMPLETO (226.802 pares). F consenso: 904/1557 = 58,1%.
- 47 clases; UNA con los 5 actores (gigante 294, NT 11) → `baseline_fase1_clase_gigante.json`.
- Candidatas suprimidas (pasan gate moritz, no están en F): 1.171 (88 en pares NT vs 27 originales sobrevivientes) — el sesgo del juez base operó en ambas direcciones.
- PREREG fase 1 congelado ANTES de correr: `PREREG_2026-09-02_fase1.md` + `fase1_certificar_E.py` (4 familias × 5 textos, doble juez, consenso ≥3/5, resta de nulls, loss lexicográfica de Javi; P1–P3 y criterios de muerte fijados).

---
## 2026-09-02 (cont.) — Nivel 1: primer E real + capa direccional + campos a explorar

**Resultado nivel 1 (E = intervencion mapa de colores, subset NT<->ModerateAutism):**
- E expandido (12 nodos, prof 2) via generador. Certificacion doble juez contra 300 nodos = 3600 pares.
- Bajo el porton congelado (bidireccional): **0 fusiones**. Acuerdo aparente masivo (10/12 nodos g>=0.55), acuerdo real cero. El fenomeno central de PN2 sobre un E inyectado.
- Autopsia (nivel1_matriz.json): 109/112 pares con señal NO certifican por **unidireccionalidad** (fe~0.99, be~0). Contradiccion ~0 en entailment, jueces concuerdan -> REAL, no instrumento.
- **Capa direccional (nueva):** fe=ent(E->nodo), be=ent(nodo->E). Campanas: mutua / fuente(E->act) / sumidero(act->E) / inerte. E llega a NT por 16 nodos (15 fuente), a ModerateAutism por 1. E (intervencion positiva) forward-implica el razonamiento NT, casi no el autista (arbol de colapso, negado).
- **Ruptura marcada, NO leida:** 1041 (Moderate) / 370 (NT) contradicciones brutas confundidas por negacion (50% nodos negados, contradiccion direccionalmente asimetrica bc>>fc = firma del artefacto). Requiere control de negacion antes de contar. Se retira como hallazgo (precedente pasta).

**Insight estructural:** PN2 fundo el objeto SIMETRICO (tolerancia F mutua). La inyeccion de E revela una capa DIRIGIDA (fuente/sumidero) que el objeto simetrico no captura. La estamos tratando con herramientas simetricas.

**CAMPOS DE CIENCIA PARA EXPLORAR (colgar PN de teoria establecida -> credibilidad -> certificar todo):**
1. **Revision de creencias (AGM)** — inyectar E y ver que absorbe/muere/persiste = expansion/revision/contraccion. Ley D1 = cambio minimo de AGM. Da: entrenchment epistemico = costo/soporte; identidades Levi/Harper.
2. **Intervencion causal / do-calculus (Pearl)** — E es do(E) sobre un sistema de razonadores; T(S,e) = operador de intervencion. Asimetria fe/be = direccion causal.
3. **Analisis de Conceptos Formales (FCA)** — E⊨nodo unidireccional = implicacion de atributos (base Duquenne-Guigues). Formaliza fuente/sumidero que el reticulo simetrico no puede.
Objetivo: ver si PN cuelga de alguno -> paper perfecto, PN gana credibilidad, se certifica todo.

**Nota de escala:** autismo = arboles mas chicos del pipeline (150 nodos). Resto de casos = 500. Un autismo a 500 daria F mas densa / mas alcances. Escalado posterior.

---
## 2026-09-02 (cont. 2) — Control NT-propio + control de negacion de la ruptura

**Control positivo (E = inferencia social implicita, capacidad NT-propia, subset NT<->ModerateAutism):**
- E expandido (15 nodos). vs mismo subset (4500 pares, doble juez).
- **NT: 6 mutuas + 9 fusiones certificadas** (coincidencias semanticas reales: "detecta señales implicitas del cambio" <-> "NT infiere expectativas implicitas"). ModerateAutism: 0 cert, 0 mutua.
- Comparado con mapa_colores sobre el mismo subset: NT mutua 0->6, cert 0->9. **El mecanismo DISCRIMINA** una capacidad nativa (convergencia mutua) de una intervencion externa (una via). Validado sobre respuesta conocida en el BRAZO NT.

**Brazo autista (ausencia): CONFUNDIDO por el corpus.** El arbol autista es enteramente deficit/negado; cualquier E positivo lo sub-alcanza en entailment. "No llega a autismo" no aisla el deficit social.

**Control de negacion sobre la ruptura (cero computo, nulls C1/C2 de fase 0):**
- El juez base infla la contradiccion en nodos negados; moritz (sin sesgo) no contradice los nulls (null_moritz~0).
- Bajo moritz + control de null: ruptura real con autismo = **139** (no 1046; el resto era sesgo base).
- **Jaccard=1.000**: mapa_colores y nt_inferencia_social rompen con LOS MISMOS 139 nodos autistas -> ruptura **GENERICA** (estados-deficit vs cualquier positivo topico), NO especifica de E. NT: Jaccard 0.884.

**CERTIFICADO del mecanismo:**
- Campana de ENTAILMENT/convergencia = confiable, discrimina (mutua/fuente/sumidero).
- Campana de RUPTURA = en corpus de deficit es topica/generica, no se lee como especifica de E ni siquiera con control de negacion.
Scripts: nivel1_generar_E.py (registro de estimulos), nivel1_informe_E.py (campana integrada), campana_direccional.py, negacion_control.py.

---
## 2026-09-02 (cont. 3) — Enganche FCA / sistemas de clausura, computado

**ESTATICA = sistema de clausura (closure_fca.py, reducto_exacto.py):**
- R es OPERADOR DE CLAUSURA: extensivo/monotono/idempotente, 0 violaciones (200 pruebas).
- Definibles = sistema de clausura: R(u)∩R(v) definible, 0 violaciones (1500 pares).
- Fusiones = implicaciones. Reducto EXACTO = 341/904 (37.7%), reproduce las 753 regiones con 0 diferencia, irredundante (0 fallos) = base de implicaciones estilo Duquenne-Guigues, computada. (autismo denso: 62% redundante vs 7.8-24% en los 4 casos heterogeneos.)
- Conclusion: el objeto estatico de PN ES un sistema de clausura FCA, verificado con 0 violaciones. El certificado ES una base de implicaciones.

**DIRIGIDA = contexto formal (fca_dirigido.py, E=inferencia social, NT<->ModerateAutism):**
- Orientacion FUENTE (E=>actor): 11/15 nodos activos, 54 pares, 10 intents, 17 conceptos. NT=11, Moderate=3.
- Orientacion SUMIDERO (actor=>E): 8/15, 44 pares, 7 intents, 11 conceptos. NT=8, Moderate=2.
- E es mas FUENTE que sumidero (17 vs 11 conceptos): la capacidad inyectada es rio-arriba. NT>>Moderate en ambas direcciones (robusto). 10 intents/11 nodos = alcance estructurado, no bloque.
- FCA da a PN un objeto NUEVO que el paper no tiene: el reticulo dirigido de alcance con asimetria fuente/sumidero medible.

**ESPINA TEORICA (para el paper):** R es un operador de clausura -> estatica = reticulo de conceptos/definibles (FCA + rough-sets de tolerancia, base Duquenne-Guigues); dinamica = revision de bases (AGM/Hansson, D1=cambio minimo, costo=atrincheramiento) de ese sistema de clausura bajo inyeccion de E; capa dirigida = contexto formal con reticulo de alcance. do-calculus RECHAZADO como fundamento (falta substrato probabilistico), solo framing.

---
## 2026-09-02 (cont. 4) — Postulados AGM medidos (agm_postulados.py, agm_extensionalidad.py)

K = convergencias certificadas existentes; phi = E inyectado (actor virtual).
Mecanismo actual = EXPANSION (agrega E, no remueve; la muerte por ruptura es generica en este corpus).

| postulado | resultado | evidencia |
|---|---|---|
| P1 Clausura | FALLA | transitividad de F = 37.5% (4990 caminos, 1869 cierran) |
| P2 Exito | cumple | E se agrega; fusiones presentes |
| P3 Inclusion | cumple (trivial, expansion) | K*phi=K+phi |
| P4 Vacuidad | cumple (trivial, expansion) | sin remocion |
| P5 Consistencia | cumple | contr_max entre certificadas 0.201<0.30, 0 violaciones |
| P6 Extensionalidad | FALLA | parafrasis nt_inferencia_social: Jaccard 0.118 (9 vs 10 nodos-actor, 2 comunes) |

**VEREDICTO: 4/6.** Las dos que fallan (clausura + extensionalidad) son EXACTAMENTE las dos que la revision de BASES (Hansson) descarta respecto del AGM clasico:
- clausura: PN no cierra bajo consecuencia (F no-transitiva).
- extensionalidad: PN opera sobre bases sintacticas -> sensible a parafrasis.
=> PN es un operador de REVISION DE BASES de manual, confirmado midiendo que postulados cumple/falla. No es AGM roto.

**MATIZ (oro para el paper):** extensionalidad CUMPLE a nivel actor (ambas parafrasis revisan solo NT, mutuo), FALLA a nivel nodo (proposiciones distintas). PN es extensional en "que razonamiento converge", no en "que proposicion exacta". Granularidad a la que el paper puede afirmar: convergencia entre razonamientos, no entre frases.

---
## 2026-09-02 (cont. 5) — Bateria FCA/Hansson sobre GERD y Selene (casos/teoremas_caso.py)

Datos existentes del pipeline (arboles + fusiones), cero NLI.

| | autismo | GERD | Selene |
|---|---|---|---|
| actores/nodos/fusiones | 5/753/904 | 3/1504/102 | 3/1503/218 |
| region mediana | 684 (satura) | 1 | 1 |
| clausura (ext/idem/mono) | 0 | 0 | 0 |
| sist. clausura (∩) | 0 | 0 | 0 |
| base minima | 38% | 92.2% | 86.2% |
| reducto paper | - | 94/102=92% (7.8% red) EXACTO | 185/218=85% (~15%) ~ |
| nucleo comun (meet) | 684 | 370 | 551 |
| recovery colateral | 577 | 69 | 270 |

CONCLUSIONES:
1. Operador de clausura + sistema de clausura = 0 violaciones en los 3 corpus -> enganche FCA CORPUS-INDEPENDIENTE, probado.
2. Mi base minima reproduce el reducto del paper (GERD 7.8% EXACTO, calculado hace meses con otro codigo) -> "reducto = base de implicaciones FCA" PROBADO por dos caminos independientes.
3. Saturacion = artefacto del autismo denso (mediana region 684 vs GERD/Selene 1). En corpus ralos lambda/meet/recovery son limpios. Recovery/perseverancia = numero real que varia por caso (GERD 69, Selene 270), no constante de saturacion.

---
## 2026-09-02 (cont. 6) — INYECCION EN GERD (caso real, cert_gerd.jsonl 17x1504 doble juez)

E = "PropuestaOperacionSequia": acuerdo vinculante indexado a sequia, minimo garantizado aguas abajo.

RESULTADO:
- Convergencia: E certifica con 0 de 3 actores (0 mutuas). Esperado: una propuesta es rio-arriba (implica consecuencias, nadie la implica de vuelta). fuente: Egipto 1, Etiopia 2, Sudan 0.
- Ruptura REAL (simetrica, min moritz contr>=0.5, doble juez), por actor:
    Etiopia 942 (la MAS opuesta) > Sudan 824 > Egipto 563 (la MENOS opuesta).
- El orden es real-world-valido: la propuesta (constraint externo vinculante) contradice mas la postura de Etiopia (soberania/unilateral/decolonial - construyo el dam), menos la de Egipto (le garantiza agua = su interes). Verificado por muestra: contradicciones simetricas fuertes (0.97/0.99, 1.00/1.00) con "Ethiopia justifies unilateral decisions as decolonial", "intensifies unilateral decision-making".
- Limpio: GERD ralo (no satura), nodos son afirmaciones positivas (sin el confound de negacion del autismo). ~mitad de rupturas simetricas = contradiccion semantica real; la mitad asimetrica queda marcada.

REENCUADRE: el informe en caso real no da "converge/no" sino el PERFIL DIFERENCIAL DE OPOSICION: con que evidencia y cuanto una propuesta choca con cada parte, nodo por nodo. Ese es el instrumento unico.
Scripts: gerd_generar_E.py, cert_caso.py (resumable), informe_caso.py.

---
## 2026-09-02 (cont. 7) — Perfil diferencial de oposicion: FORMULA + control de nulls GERD

FORMULA (E inyectado, actor a, doble juez + control nulls):
  C(E,a) = #converge / (|N_E||V_a|)   [primal: mutua, rara]
  O(E,a) = #opone / (|N_E||V_a|)      [dual: contradiccion simetrica min(moritz fc,bc)>=0.5, base>=0.3, sin nodos-null]
  A(E,a) = C - O ;  perfil = vector O(E,a) ; dispersion Delta = maxO - minO
FIRMA DE TIPO: C>0 => capacidad nativa del actor; C=0 & O alta => propuesta externa; ambas bajas => inerte.

GERD (propuesta acuerdo sequia), null-controlado (90/1504 excluidos=6%):
  Etiopia O=96.75 > Sudan 88.53 > Egipto 52.61 (x10^-3). C=0 los tres. Delta=44.13.
  Sin null Delta=44.63 -> control cambia <1% => perfil REAL, no artefacto. Propuesta opone a Etiopia 1.8x mas que a Egipto.

ESTRUCTURA DEL DUAL (medida en GERD):
  - Ley #13 confirmada LOCALMENTE: si v contradice E, su padre tambien 26.5% vs base 9.7% = 2.7x (contradiccion sube por anc, estadistico no axiomatico).
  - Concentracion: densidad de oposicion por nivel 0%(raiz)/7.5/10.8/13.3(pico)/10.5/8.2(hojas) -> NO enraizada; vive en la BANDA DE COMPROMISO (nivel medio). Una propuesta choca donde el razonamiento se vuelve operativamente comprometido, no en la premisa abstracta ni en los detalles.

Cross-caso (A=C-O, autismo NT-vs-1tipo con null C1/C2; recordar: en autismo las cuentas son NT vs UN tipo, no todos-vs-todos):
  intervencion(mapa): NT -43, ModerateAut -259 | capacidad(inf.social): NT -9.8 (C=4>0!), ModerateAut -212 | propuesta(GERD): Eti -97, Egi -53.
  -> solo la CAPACIDAD da C>0 (con NT). A=C-O dominado por O (oposicion>>convergencia, coherente con tesis PN2). No colapsar: C y O son ejes distintos (primal/dual).
Scripts: perfil_oposicion.py, perfil_todos.py, E_arbol_gerd_null.json, cert_gerd_null.jsonl.

---
## 2026-09-03 — Los tres computos que ganan los teoremas (dg_canonica.py, levi_revision.py)

**#1 BASE CANONICA DUQUENNE-GUIGUES (exacta, unica):**
- La clausura de PN es de ALCANZABILIDAD (premisa unitaria) sobre granulos: cl(X u Y) = cl(X) u cl(Y), 0 violaciones (GERD y autismo, 300 pruebas c/u). [Bug cazado y corregido: la 1a corrida dio "no distribuye" por un salto en cl() con seeds sin granulo; no era PN.]
- => pseudo-intents = granulos no cerrados => base DG exacta sin enumeracion.
- GERD: 304 implicaciones canonicas (granulos cuya clausura excede su arbol) / 1200 cerrados. Autismo: 547 / 206.
- Puentes canonicos = nodos fusionados y sus ancestros (el granulo arrastra el puente). Unicos por DG: nadie los elige.

**#3 REVISION DE LEVI completa, GERD: K*E = (K - ¬E) + E**
- contraccion = O_a (contradiccion simetrica null-controlada) + cascada D1; expansion = arbol de E.
- Postulados de Hansson sobre la revision CORRIDA: exito ok, inclusion ok, consistencia ok, RELEVANCIA ok (3/3 actores). Uniformidad: no testeable (antecedente no realizable).
- Costo de Levi (fraccion del razonamiento a abandonar para absorber E):
    kernel (Hansson minimo, solo O):  Egipto 40% | Etiopia 52% | Sudan 58%
    D1 (con cascada de dependencia):  Egipto 86% | Etiopia 99.8% (retiene 1 nodo) | Sudan 92%
- Ranking preservado en ambos operadores. La diferencia kernel->D1 = DEPENDENCIA: cuanto del arbol cuelga de lo contradicho. Etiopia 52->99.8%: casi todo su razonamiento es carga sobre sus compromisos de soberania = exactamente lo que la propuesta niega. "Muerta al llegar" cuantificada.
- CONFIRMADO: O(E,a) ES el conjunto de contraccion de Levi; el perfil diferencial de oposicion ES el precio de revision por actor. Corrido, no solo identificado.

**#2 RELEVANCIA + UNIFORMIDAD:** relevancia verificada sobre Levi (arriba). Uniformidad honestamente N/A.

TEOREMAS AHORA INVOCABLES CON DERECHO en el paper: Birkhoff/Moore (meet unico), Wille (reticulo de conceptos), Duquenne-Guigues (base canonica exacta), Armstrong (derivacion completa), Dowling-Gallier (Horn lineal), Hansson (revision de bases: firma 4/6 + relevancia), Levi (O = contraccion = precio), recovery-failure (perseverancia medida).

---
## 2026-09-03 — PREREG inyeccion SELENE (congelado ANTES de correr)
E = "ProtocoloComandoRotativo": comando rotativo conjunto, soporte vital por triage medico, un voto por agencia, sin veto.
PREDICCION: perfil de oposicion (Levi) Aurelia (mas opuesta: pierde exclusividad+veto) > Cerulea > Borealis (menos: co-igualdad es su postura).
Si se cumple: 2 casos reales con orden predicho y confirmado para el Acto III.

---
## 2026-09-03 — SELENE: resultado vs prediccion congelada (cert_selene.jsonl 13x1503 + nulls)

PREDICCION (congelada antes): Aurelia > Cerulea > Borealis.
RESULTADO oposicion directa null-controlada (x10^-3): Aurelia 66.92 > Cerulea 63.77 > Borealis 62.15. **ORDEN CONFIRMADO** (margenes finos). Delta = 4.77.
Levi kernel (|O|/|K|): Aurelia 39.8% > Cerulea 36.4% > Borealis 33.0% (orden confirmado).
Levi D1 cascada: Cerulea 100% (retiene 0) >= Aurelia 99.6% (retiene 2) > Borealis 88.8% (retiene 56). Postulados Hansson ok 3/3 (exito, inclusion, consistencia, relevancia).
Null control: 258/1503 excluidos (17%; mas negadores de forma que GERD 6%).

HALLAZGO NUEVO — Delta caracteriza el tipo de propuesta/conflicto:
  GERD Delta=44 (propuesta ASIMETRICA: favorece a Egipto, un ganador claro).
  Selene Delta=4.8 (propuesta SIMETRICA: cada agencia pierde su reclamo exclusivo, todos ceden ~igual).
  => el perfil no solo ordena: su dispersion mide cuan UNILATERAL es una propuesta. Lectura de dos parametros: Delta (quien pierde mas) + amplificacion por dependencia (quien esta atrapado).

HALLAZGO (no anticipado): Cerulea colapsa 100% en cascada. Su reclamo (esencial por monopolio comms+medica) es exactamente lo que "triage medico independiente de contribucion" disuelve. Toda su dependencia cuelga de esa palanca. Mismo patron que Etiopia: no discrepa mas, DEPENDE mas.
Borealis: menos opuesto en ambos operadores, robusto (co-igualdad = su postura).

Acto III queda con 2 casos reales, 2 predicciones confirmadas, 2 firmas de conflicto (asimetrico/simetrico).
Scripts: selene_generar_E.py, levi_caso.py (generico), cert_selene*.jsonl.

---
## 2026-09-03 (noche) — Respuesta a referee #4: encadenamiento de F y replica de la ley de profundidad (salto_perfil.py, salto_ablacion.py, profundidad_caso.py)

**A. Saltos de F en las regiones de soporte** (Reg(v)=cl({v}), semillas singleton, todas):
- Masa de regiones que entra por >=2 fusiones encadenadas: GERD 70.6% (salto max 10), Selene 89.7% (max 15), autismo 90.3% (max 8). Masa propia (0 saltos): 19.4% / 4.7% / 1.1%.
- => la clausura es de ALCANZABILIDAD: la mayor parte de cada region cruzada se sostiene por cadenas de fusiones, no por fusion directa. Cada eslabon hereda el error del juez; NO hay cota. El referee tiene razon en que faltaba decirlo y medirlo.
- Semillas afectadas (region > propio arbol = puentes DG): GERD 304, Selene 442, autismo 547.
- Barrido tau (autismo, ent_avg moritz): 0.55->0.95 quita la mitad de F (904->451); Jaccard medio de regiones 1.00->0.73 (mediana 0.86); masa >=2 saltos estable ~90%.
- Ablacion aleatoria de F (20 reps), Jaccard medio SOLO en semillas puente: quitar 10%: GERD 0.893 / Selene 0.838 / autismo 0.969; 25%: 0.706 / 0.642 / 0.893; 50%: 0.506 / 0.309 / 0.781. Sobre todas las semillas (las hojas cerradas no cambian): 10% -> 0.978/0.952/0.977.
- Lectura: degradacion gradual, superlineal en F rala (Selene). Un 10% de fusiones falsas mueve ~11-16% de las regiones puente. Es la medida honesta de "propagacion de ruido"; se reporta como MEDIDO + limite, no como cota.

**B. Ley de profundidad — CORRECCION y replica:**
- Los numeros del paper (0/7.5/10.8/13.3/10.5/8.2; padre 26.5% vs 9.7%) se obtuvieron SIN control de nulls y truncando d6-d8 ("8.2 hojas" era d5). Reproducido exactamente con criterio moritz_min0.5+base0.3 sin nulls.
- Con el criterio de las tablas (null-controlado), GERD d0..d8: 0.0/7.5/9.0/12.3/9.2/7.3/6.6/7.5/5.9 (base 7.9%); pico d3; padre 24.1% vs 8.1% = x3.00.
- Selene d0..d7: 2.6/14.0/8.2/7.5/5.8/6.5/6.3/4.8 (base 6.4%); pico d1; padre 16.7% vs 6.5% = x2.56.
- REPLICA: (i) raiz minima (0.0 / 2.6, bajo la base), (ii) pico sobre la base por encima de la raiz, (iii) decae hacia hojas (5.9 / 4.8, bajo la base), (iv) propagacion al padre x3.0 / x2.6.  NO REPLICA: la posicion del pico ("banda intermedia" d3) — en Selene es d1. La lectura "banda de compromiso" queda como GERD-only; lo generalizable es raiz-minima + pico + decaimiento + propagacion.
Scripts: salto_perfil.py <run|autismo>, salto_ablacion.py <run|autismo>, profundidad_caso.py cert E run [null]. Salidas: saltos_*.json, profundidad_gerd.json, profundidad_selene.json.

---
## 2026-09-03 (noche, cont.) — Hansson: cinco postulados, no cuatro (resultado matematico, sin computo)

- La unica nocion de consistencia del sistema es la relacion de conflicto bajo J (Def. 6 del paper: "consistente con E" = ninguna proposicion conflictua con E). Con ella, K'+E es inconsistente sii K' ∩ A(E) ≠ ∅.
- UNIFORMIDAD se demuestra para toda funcion de contraccion: la hipotesis sobre singletons da A(E)=A(E'), y ctr es funcion del conjunto => K∩K*E = K∩K*E'. (Mi argumento anterior de "no demostrable" confundia "depende del juez" con "no es funcion de la relacion de consistencia": la relacion ES la del juez.)
- RELEVANCIA en la forma de Hansson (existe K'' consistente entre K∩K*E y K+E tal que K''∪{x} es inconsistente) vale para ctr=id y FALLA para D1: lo removido por dependencia no conflictua con E, asi que ningun K''∪{x} es inconsistente. La "relevancia relativa a la justificacion" de v2.3 no era el postulado de Hansson — corregido.
- Contraccion minima = revision partial-meet (= full-meet) de bases bajo la relacion de conflicto: el remanente maximal consistente es unico, K\A. Por el teorema de representacion cumple los 5. D1 no es partial-meet: viola relevancia por diseño (es lo que kappa_D1 mide).
- Paper v2.6: Prop. 3 reescrita (5 postulados en forma estandar; relevancia solo para minima; D1 4/5), Obs. "La contraccion minima es la revision partial-meet"; abstract, §1.2, §7.2, §8 (tablas: columna postulados = exito/inclusion/consistencia verificados 3/3), §10(v), §11, Tabla 3/4.

---
## 2026-09-03 (noche, cont. 2) — Estatica sobre Vireya y Selene (sin NLI): DG y encadenamiento en 4 corpus

BASE DG (singletons no cerrados, particion descenso/puentes), cl aditiva 0 viol:
  GERD   610 = 306 desc + 304 puentes (894 cerrados)  [reproduce el paper]
  Selene 673 = 231 desc + 442 puentes (830 cerrados)
  Vireya 679 =  72 desc + 607 puentes (825 cerrados)   <- 503 fusiones: casi todo puente
  autismo 592 = 45 desc + 547 puentes (161 cerrados)
ENCADENAMIENTO Vireya: masa >=2 saltos 96.0% (salto max 17); propia 1.7%. Ablacion (semillas puente, 607): 10% -> J 0.905; 25% -> 0.747; 50% -> 0.478.
=> §3.3 y §3.5 pueden reportar 4 corpus. Vireya (ficticio, denso en F) es el mas encadenado y el mas sensible a 50%.
Scripts: dg_canonica.py (cuenta puentes), particion inline via salto_perfil.py; salto_ablacion.py.

---
## 2026-09-03 — PREREG inyeccion VIREYA (tercer caso prospectivo; congelado ANTES de correr)
E = "InstrumentoCorredorHumedo": instrumento trilateral vinculante; tope de captura del Cirrus Array indexado a humedad del corredor con monitoreo independiente; paso minimo aguas abajo en estacion seca; Fase II solo bajo el tope; volumenes del conducto reducidos proporcionalmente en estacion seca con formula de compensacion ligada al caudal medido.
Analogo estructural de la propuesta GERD (constraint externo vinculante que favorece al actor aguas abajo). Corpus ficticio (sin memorizacion posible).
PREDICCIONES (Levi/perfil, doble juez, null-controlado, mismos umbrales tau=0.55 gamma=0.30 delta=0.50):
  P1 orden de oposicion O(E,a): Vireya (mas opuesta: pierde soberania y cronograma) > Ondrel > Tessara (menos: obtiene su demanda).
  P2 dispersion Delta grande, tipo GERD (propuesta asimetrica), no tipo Selene.
  P3 bajo D1, Vireya colapsa (retenido ~0: todo cuelga de "recurso soberano + Fase II"); Ondrel retiene parte (su reclamo de compensacion sobrevive).
  P4 replica de propagacion al padre (x2-3 sobre base) y de raiz minima + maximo sobre base + decaimiento hacia hojas (posicion del maximo: sin prediccion).
  C(E,a)=0 para los tres (propuesta rio arriba) — esperado, no es prediccion fuerte.
Criterio de muerte: P1 falla si el orden observado difiere en cualquier par; P2 falla si Delta < 15 (x10^-3); P3 falla si Vireya retiene >10% bajo D1 o si Ondrel retiene menos que Vireya.
PASOS manana: python vireya_generar_E.py -> E_arbol_vireya.json ; python cert_caso.py E_arbol_vireya.json ../../casos/vireya/runs/2026-08-28_21-26-39 vireya ; python cert_caso.py E_arbol_gerd_null.json ../../casos/vireya/runs/2026-08-28_21-26-39 vireya_null ; luego perfil_oposicion.py, levi_caso.py, profundidad_caso.py con cert_vireya.jsonl / cert_vireya_null.jsonl.

---
## 2026-09-03 (noche, cont. 3) — "Calcular implicaciones" en acto (consulta_implicacion.py) + Cn de Tarski

- cl es extensivo/monotono/idempotente = condiciones de Tarski => operador de consecuencia Cn_PN = cl (consecuencia estructural, no logica). En §6: Hansson se formula relativo a (Cn, inconsistencia); aqui Cn_PN = cl y la inconsistencia = relacion de conflicto del juez; dos primitivas independientes.
- Consulta hacia atras en GERD (0-1 BFS, minimiza fusiones): q = "Sudan requests African Union mediation intervention" (Sudan, prof 5). Sostenedores {x: q in cl({x})} = 160: Egipto 69, Etiopia 50, Sudan 41; saltos 0..8. Las DOS premisas ajenas sostienen q a 1 fusion: Egipto en 6 descensos + 1 fusion (crisis -> internacionalizar -> apelar a la UA ~F~ q); Etiopia en 4 descensos + 1 fusion (marco ilegitimo -> trabada -> marcos alternativos -> mediacion UA ~F~ q). Consenso inadvertido con derivacion nombrada.
- Tratabilidad: NO viene de DG (base exponencial en general, pseudo-intent coNP-completo, Kuznetsov 2004) ni de Horn (todo conjunto de implicaciones es Horn definido); viene de la aditividad = alcanzabilidad O(|V|+|F|) por consulta. Base compilada sum|cl({x})|: GERD 4.5e4, Selene 1.9e5, autismo 3.5e5, Vireya 4.0e5 literales vs |V|^2 ~ 2.3e6.
- Paper v2.7: §3.2 (Tarski), §3.3 (4 corpus DG), §3.4 (consulta en acto + parrafo de tratabilidad), §3.5 (4 corpus), §6 (Cn/conflicto), Tabla 3/4, bib kuznetsov2004.
Script: consulta_implicacion.py <run_dir> [id_q] -> consulta_<run>.json

---
## 2026-09-04 — Geometria lambda recuperada (lambda_geometria.py) + Obs. "Cuatro lecturas"

- lambda(v) = composicion por actor de Reg(v) = cl({v}); punto del simplex de actores. Computada sobre certificados congelados (sin NLI).
- GERD: 1200 regiones puras (un actor) + 304 mixtas (= puentes DG). Selene: 1061 puras + 442 mixtas (= puentes). Las mixtas son exactamente los nodos con clausura fuera de su arbol.
- Nucleo: regiones grandes (>=50% de la mayor) con composicion casi fija: Selene 338 regiones, centroide (0.219, 0.341, 0.441), desvio 0.02 -> las que alcanzan la clase gigante. GERD: solo 8 (centroide (0.34,0.24,0.42), desvio 0.18): F rala, sin clase gigante dominante.
- La consulta hacia atras (sostenedores de q por actor) computa Lambda(Reg): la geometria estaba escondida en §3.4.
- Paper v2.8: Obs. "Cuatro lecturas de un mismo objeto" en §3.2 (orden / certificado-vs-campo / composicion por actor / consecuencia iterada; mutuamente olvidadizas), Fig. simplex_lambda.png (3D real, GERD y Selene), nota en §3.4. Recupera el Cor. "tres vistas" de PN2 viejo y agrega la cuarta.
Salidas: lambda_2026-05-01_01-11-03.json (GERD), lambda_2026-04-30_09-54-35.json (Selene).

---
## 2026-09-05 — Implicacion padre->hijo VERIFICADA (padre_hijo_nli.py, GERD, 1.501 aristas, doble juez)

Origen del numero heredado: CATEGORIAS_2026-08-29 F17, "E(padre->hijo) mediana 0.001 (60 aristas GERD)". Muestra exploratoria; el paper lo citaba como dato firme. Se midio completo.
RESULTADO (1.501 aristas, dos jueces, dos direcciones, umbral tau=0.55):
  padre->hijo (fe): base mediana 0.001 media 0.050 P(>=tau)=4.5% | moritz mediana 0.003 media 0.077 P(>=tau)=6.1%
  hijo->padre (be): base mediana 0.001 media 0.097 P(>=tau)=9.7% | moritz mediana 0.004 media 0.112 P(>=tau)=9.9%
  contradiccion max (base): mediana 0.001, P(>=0.55)=10.7%
LECTURA: (1) la mediana 0.001 se sostiene sobre el bosque entero y con el juez corregido; (2) no es "ninguna": 4.5-6.1% de las aristas son implicacion textual (parafrasis/especificacion trivial) — el paper dice la fraccion, no solo la mediana; (3) ASIMETRIA: hijo->padre supera tau el doble de veces que padre->hijo (9.7-9.9% vs 4.5-6.1%): lo especifico implica lo general. Misma asimetria "rio arriba" de la inyeccion, ahora medida dentro de los arboles sin E.
Paper v2.9: §2.1, §5.2, Obs. homonimo (§3.3), Tabla 4. Cuaderno: seccion 1, tabla seccion 3, lamina 11.
Salida: padre_hijo_gerd.jsonl. Pendiente opcional: misma medida en Selene/Vireya (barato, ~20 min c/u).

## 2026-09-05 — Propagacion de la oposicion: test de direccion y control de similitud (propagacion_direccion.py) — RESULTADO NEGATIVO
Script: propagacion_direccion.py cert E run [null]  (mismo criterio y nulos que profundidad_caso.py). Salidas: propagacion_gerd.json, propagacion_selene.json. Runs: GERD casos/gerd/runs/2026-05-01_01-11-03, Selene casos/selene/runs/2026-04-30_09-54-35.
- Lift padre (P(padre opone|hijo opone)/P(padre opone)): GERD 3.00 IC95 bootstrap [2.75,3.26]; Selene 2.56 [2.22,2.91]. Replica.
- PERO: hermano (sin arista) GERD 3.01 / Selene 1.94; abuelo=nieto 2.36 / 1.75. En GERD el hermano opone igual que el padre.
- Control por similitud lexica (par no adyacente mismo actor, Jaccard emparejado al par padre-hijo): GERD padre 24.1% vs control 16.3% vs base 7.9%; Selene 16.7% vs 11.9% vs 6.4%. La similitud de texto explica ~52-53% del exceso.
- Direccion (nodos con padre e hijos, cada lado normalizado por su base): GERD hacia padre 2.92 vs hacia hijos 2.88; Selene 1.98 vs 2.61. NO hay direccion; "sube por ancestros" refutado.
- Decision (Javi): la seccion 7.3 "Propagacion de la oposicion" se ELIMINA del paper (v2.13) junto con Fig. 6; queda como limite (x) en §10 con los numeros. No es propiedad del objeto: es agrupamiento en vecindarios textuales.
- Tambien eliminado como relleno: cajas "Lo que esta seccion establecio", §3.3 Reticulo de conceptos, §9.4 Auditoria del instrumento, parrafo duplicado sobre "calculo" en intro/§2.

## 2026-09-05 — Terreno comun, puentes y compromiso cruzado CON CONTENIDO (terreno_comun.py) — v2.14 reestructurada por preguntas
Script: terreno_comun.py <run_dir> -> terreno_<run>.json. Runs: GERD 2026-05-01_01-11-03, Selene 2026-04-30_09-54-35.
- GERD: R(raiz) Egipto 502+320, Etiopia 501+322, Sudan 501+200. Meet 370 (Egipto 106, Etiopia 70, Sudan 194). 64 cabezas; mayores: "Sudan threatens to withdraw from trilateral frameworks" (71), "Sudan seeks alternative water security partnerships" (70), "Regional powers align..." (30), "Egypt increases military posturing" (20). Puentes 304 = 137 fusionados + 167 ancestros; mayor alcance "Ethiopia reduces coordination with downstream monitoring systems" (300 ajenos). Compromiso cruzado (fila compromete columna): Egipto->Sudan 48%, Etiopia->Sudan 40%, Sudan->Etiopia 15%.
- Selene: R(raiz) Cerulea 503+544, Borealis 500+440, Aurelia 500+336. Meet 551 (Aurelia 244, Borealis 184, Cerulea 123). 112 cabezas; mayores: "Other crews must request permission for basic life support adjustments" (108), "Resource allocation debates intensify..." (78). Puentes 442 = 243+199; mayor alcance "Cerulea's communication array experiences power allocation conflicts" (459 ajenos). Cerulea->Aurelia 60%.
- Lectura: el terreno comun esta hecho de consecuencias de la ruptura, no de soluciones, en ambos casos.
- Paper v2.14: abstract, intro y conclusion reescritos por cuatro preguntas (P1 que compromete cada proposicion / P2 terreno comun y frontera / P3 puentes y compromiso cruzado / P4 costo de una propuesta) con las garantias debajo; §3.4-3.5 comprimidas; nueva §3.5 "Las tres respuestas sobre los dos conflictos" con Tabla 2.

## 2026-09-05 — Consulta hacia atras sobre TODOS los nodos (sostenedores_top.py) — v2.15 (ronda final de precision)
Script: sostenedores_top.py <run_dir> [k] -> sostenedores_<run>.json. sost(q)={x: q in cl({x})} con salto minimo (BFS 0-1).
- GERD top-5 por sostenedores ajenos: "Sudan requests AU mediation intervention" 119 (Eg 69, Et 50; premisas ajenas a salto 1/1); "Sudan gains potential allies..." 116; "Sudan shifts diplomatic focus toward non-Nile partnerships" 116; "Sudan's negotiating position ... externally constrained" 113; "Ethiopia engages AU mediation mechanisms" 111 (Eg 69, Su 42).
- Selene top-5: cinco consecuencias de Borealis d5 (control unilateral, restriccion de informacion, sospecha) con 239 ajenos cada una (Au 121, Ce 118), premisas ajenas a salto 1.
- Chequeo: nodos comprometidos por TODAS las premisas ajenas = 370 (GERD) / 551 (Selene) = meet. Consistente con terreno_comun.py.
- Paper v2.15: Tabla 2 (sostenedores); "consulta lineal" -> O(|V|+|F|) en abstract/intro/conclusion; puente definido como antecedente (no literal de conclusion); §6.3 retitulada "Relacion con la revision AGM de teorias"; garantias separadas en matematicas vs relativas al procedimiento de conflicto (abstract, intro, conclusion).

## 2026-09-05 — Reordenamiento de la carpeta principio_naturalidad
pn3 -> pn_dinamica (esta carpeta; copia identica del core v1, sandbox de dinamica; lo multipremisa vive en principio_naturalidad_autismo). pn2 -> analisis/pn2_estatica; analisis_posthoc -> analisis/posthoc; gephi_exploracion -> analisis/gephi; tests_y_exploracion -> analisis/exploracion. PAPER -> paper (PNII_v2_ES, PNII_v2_EN, zenodo, notas, _archivo). documentacion -> docs; CONCURSOS -> concursos; key.txt -> config/. Scripts de analisis/ con ruta relativa a la raiz: +1 nivel (18 archivos). Mapa completo en README.md de la raiz.

## 2026-09-05 (cierre) — Tres operaciones sobre consenso y Jamison; criterio NLI de Vireya; v2.20
- terreno_comun.py y sostenedores_top.py aceptan 'autismo' (estado_nodos.json + F_consenso_parcial.json consenso). Salidas: terreno_autismo.json, terreno_2026-05-01_16-30-33.json (Jamison), sostenedores_*.json.
- Consenso (5 arquitecturas, 753 nodos, 904 fusiones): meet 684/753 (91%); c(NT->subtipo) 97-99%, c(subtipo->NT) 61%; 58 nodos NT fuera de M; puente mayor "NT rapidly updates its contextual expectations" (592 ajenos). Puentes 547 = 458 fusionados + 89 ancestros.
- Jamison (2 estados, 1004 nodos, 38 fusiones): meet 136 (14%), 73% manico; cabezas: "decreased perceived need for psychiatric supervision" (27), "decreased insight" (16), "loses lithium's effects" (15); c(dep->man) 20%, c(man->dep) 7%; puente mayor "[Depressive] avoids discussing treatment options with her doctor" (73). Puentes 139 = 47+92.
- HALLAZGO: metadata.json de los runs: gerd/selene/jamison/lee/autismo nli_criterion=avg; vireya (2 runs) nli_criterion=min. Declarado en §2.2 y limite (ix); Vireya solo entra en base canonica y encadenamiento. Pendiente decidir si recertificar Vireya con avg.
- Paper v2.20 (ES+EN): §3.5 "Las tres operaciones sobre cuatro corpus" (Tabla con GERD, Selene, consenso, Jamison); P2/P3 de §1.1 y cajas Aportes con los tres tipos de razonamiento (conflicto, arquitecturas cognitivas, dos estados de una mente); lenguaje "partes/mediador" reducido a §8.
