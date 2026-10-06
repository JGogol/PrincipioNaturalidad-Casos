#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
p_tilde: UNA sola definicion de la normalizacion pragmatica P~, para todo el proyecto.

Existe porque el mismo error aparecio tres veces en tres lugares distintos: cada script
resolvia por su cuenta "con que string reemplazo al actor" y cada uno eligio distinto.
  - cert_caso.py    pasaba el NOMBRE FORMAL ("Arab Republic of Egypt") -> no-op en el 66% de gerd
  - cert_caso_v3.py pasaba el ALIAS de input.json ("RAurelia" en selene)  -> no-op en el 100%
  - fase0_intra_actor.py usaba .replace() literal con el alias -> mismo problema, ademas sin
    limites de palabra
Ninguna de las tres limpia el texto en general. Este modulo aplica TODAS las claves conocidas,
de la mas larga a la mas corta, y despues las formas derivadas.

Medido sobre los tres corpus (4.508 nodos): residuo 0. Con cualquiera de las claves sueltas,
entre 0% y 66% segun el caso.

    from p_tilde import normalizar
    normalizar("Highland Republic of Vireya expects...", "Highland Republic of Vireya", "Vireya")
      -> "ACTOR expects..."
    normalizar("Aurelian personnel resist", "Republic of Aurelia", "RAurelia")
      -> "ACTOR personnel resist"
"""
import re

PLACEHOLDER = 'ACTOR'
# sufijos demonimicos observados en los corpus: Ethiopia+n, Egypt+ian, Sudan+ese,
# Aurelia+n, Tessara+n, Vireya+n
SUFIJOS = r"(?:n|an|ian|ese|s)?(?:'s)?"


def claves(actor, alias=None):
    """Las formas con que un actor puede aparecer en su propio texto, de la mas larga
    a la mas corta: nombre formal, alias declarado en input.json, y la ultima palabra
    del nombre formal (la forma corta que el generador escribe de hecho)."""
    c = [actor, (alias or '').strip(), actor.split()[-1] if actor else '']
    return sorted({x for x in c if x}, key=len, reverse=True)


def _rep(texto, clave, placeholder=PLACEHOLDER):
    e = re.escape(clave)
    texto = re.sub(rf"\b{e}'s\b", placeholder + "'s", texto, flags=re.IGNORECASE)
    return re.sub(rf"\b{e}\b", placeholder, texto, flags=re.IGNORECASE)


def normalizar(texto, actor, alias=None, derivadas=True, placeholder=PLACEHOLDER):
    """Reemplaza toda mencion del actor en su propio texto por el marcador generico.
    derivadas=True alcanza tambien los gentilicios ("Aurelian", "Egyptian", "Sudanese"),
    que identifican al actor igual que su nombre y aparecen de forma muy desigual entre
    actores: dejarlos fuera sesga cualquier cantidad que se mida por actor."""
    if not texto or not actor:
        return texto
    for k in claves(actor, alias):
        texto = _rep(texto, k, placeholder)
    if derivadas:
        corta = actor.split()[-1]
        texto = re.sub(rf"\b{re.escape(corta)}{SUFIJOS}\b", placeholder, texto, flags=re.IGNORECASE)
    return texto


def residuo(textos, actor, alias=None):
    """Cuantos textos siguen nombrando al actor despues de normalizar. Sirve como control:
    si no da 0, el regimen de texto no es el que se declara."""
    corta = re.escape(actor.split()[-1])
    n = 0
    for t in textos:
        s = normalizar(t, actor, alias)
        if re.search(rf"\b{corta}\w*", s, re.IGNORECASE) or actor.lower() in s.lower():
            n += 1
    return n


def alias_de(input_caso, actor):
    """Alias declarado en el input.json del caso; cadena vacia si no hay."""
    for a in (input_caso or {}).get('actores', []):
        if a.get('nombre') == actor:
            return (a.get('alias') or '').strip()
    return ''
