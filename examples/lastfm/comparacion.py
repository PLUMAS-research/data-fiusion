"""Familias mezcladas contra monofamilia en Last.fm.

La pregunta que este experimento aisla: cuando una relacion de conteos
sobredispersos (reproducciones artista-usuario) convive con una relacion
de etiquetas (tags por artista), ¿modelar cada una con su familia
(Poisson + gaussiana) predice mejor que forzar todo por la cuadratica
con los conteos transformados?

Protocolo: los artistas etiquetados se parten 60/20/20. La relacion de
etiquetas entra al ajuste con mascara de filas sobre el entrenamiento;
validacion y prueba se predicen con predict_proba desde el factor
compartido. Cada brazo elige sus hiperparametros en validacion y se
evalua en prueba con 4 semillas, apareadas entre brazos.

- Brazo A (monofamilia): escuchas gaussianas con la mejor transformacion
  de {crudo, sqrt, log1p}, mas etiquetas gaussianas enmascaradas.
- Brazo B (mezclado): escuchas con family="poisson" sobre los conteos
  crudos, mas las mismas etiquetas gaussianas enmascaradas.

Criterio de exito, declarado antes de correr: B le gana a A en AP media
de prueba por 2 errores estandar de la diferencia apareada.

AP es la precision promedio por artista (average_precision_score sobre
los 100 tags), promediada sobre los artistas evaluados.

Nota de protocolo: la grilla de pesos original (0.3 a 10) se extendio
tras una primera corrida de validacion, porque el brazo A elegia el
borde con la curva todavia subiendo y el brazo B no respondia al peso
(el gradiente KL sin calibrar dominaba numericamente al gaussiano; la calibracion
por desviacion nula corrige eso en la libreria). La seleccion sigue
siendo solo en validacion.

Ejecucion: uv run python examples/lastfm/comparacion.py
"""

# %%
import os
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score

from datafiusion import Relation, fuse

from datos import cargar


DIR_SALIDA = Path(os.environ.get("DIR_SALIDA", Path(__file__).parent / "output"))

SEMILLAS = (0, 1, 2, 3)
RANKS = {"artista": 30, "usuario": 20, "tag": 20}
MAX_ITER = 150
PESOS_ETIQUETAS = (0.3, 1.0, 3.0, 10.0, 30.0, 100.0)
TRANSFORMACIONES = ("crudo", "sqrt", "log1p")


# %%
d = cargar()
escuchas, etiquetas, etiquetados = d["escuchas"], d["etiquetas"], d["etiquetados"]

rng = np.random.default_rng(0)
orden = rng.permutation(etiquetados)
n_ent = int(0.6 * len(orden))
n_val = int(0.2 * len(orden))
entrenamiento = np.sort(orden[:n_ent])
validacion = np.sort(orden[n_ent:n_ent + n_val])
prueba = np.sort(orden[n_ent + n_val:])
print(f"{len(entrenamiento)} artistas de entrenamiento, {len(validacion)} de "
      f"validacion, {len(prueba)} de prueba")

Y = etiquetas.toarray()


def transformar(M, como):
    salida = M.copy().astype(np.float64)
    if como == "sqrt":
        salida.data = np.sqrt(salida.data)
    elif como == "log1p":
        salida.data = np.log1p(salida.data)
    elif como != "crudo":
        raise ValueError(como)
    return salida


def ap_media(proba, artistas):
    puntajes = [average_precision_score(Y[a], proba[a]) for a in artistas]
    return float(np.mean(puntajes))


def ajustar(matriz_escuchas, familia, peso, semilla):
    R = {"escuchas": Relation(src="artista", dst="usuario",
                              matrix=matriz_escuchas, family=familia),
         "etiquetas": Relation(src="artista", dst="tag", matrix=etiquetas,
                               rows=entrenamiento)}
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        modelo = fuse(R, RANKS, weights={"etiquetas": peso}, max_iter=MAX_ITER,
                     tol=1e-6, init="random", random_state=semilla)
    return modelo.predict_proba(target="tag", views=["etiquetas"])


# %%
popularidad = np.asarray(etiquetas[entrenamiento].sum(axis=0)).ravel()
ap_baseline = float(np.mean([average_precision_score(Y[a], popularidad)
                             for a in prueba]))
print(f"baseline de popularidad en prueba: AP = {ap_baseline:.3f}")


# %%
print("\nbrazo A (monofamilia gaussiana), seleccion en validacion, semilla 0")
mejor_a, registro_val = None, []
for como in TRANSFORMACIONES:
    V = transformar(escuchas, como)
    for peso in PESOS_ETIQUETAS:
        ap = ap_media(ajustar(V, "gaussian", peso, semilla=0), validacion)
        registro_val.append({"brazo": "A", "config": f"{como}, peso {peso}", "AP val": ap})
        print(f"  {como:<6} peso {peso:>4}: AP val = {ap:.3f}")
        if mejor_a is None or ap > mejor_a[0]:
            mejor_a = (ap, como, peso)
print(f"  elegido: {mejor_a[1]} con peso {mejor_a[2]}")

print("\nbrazo B (mezclado, escuchas poisson crudas), seleccion en validacion, semilla 0")
mejor_b = None
for peso in PESOS_ETIQUETAS:
    ap = ap_media(ajustar(escuchas, "poisson", peso, semilla=0), validacion)
    registro_val.append({"brazo": "B", "config": f"poisson, peso {peso}", "AP val": ap})
    print(f"  poisson peso {peso:>4}: AP val = {ap:.3f}")
    if mejor_b is None or ap > mejor_b[0]:
        mejor_b = (ap, peso)
print(f"  elegido: peso {mejor_b[1]}")


# %%
print("\nevaluacion final en prueba, 4 semillas apareadas")
filas = []
V_a = transformar(escuchas, mejor_a[1])
for semilla in SEMILLAS:
    ap_a = ap_media(ajustar(V_a, "gaussian", mejor_a[2], semilla), prueba)
    ap_b = ap_media(ajustar(escuchas, "poisson", mejor_b[1], semilla), prueba)
    filas.append({"semilla": semilla, "AP A": ap_a, "AP B": ap_b,
                  "diferencia": ap_b - ap_a})
    print(f"  semilla {semilla}: A = {ap_a:.3f}, B = {ap_b:.3f}, "
          f"B - A = {ap_b - ap_a:+.3f}")

tabla = pd.DataFrame(filas)
media_a, media_b = tabla["AP A"].mean(), tabla["AP B"].mean()
dif = tabla["diferencia"]
sem = dif.std(ddof=1) / np.sqrt(len(dif))
veces = dif.mean() / max(sem, 1e-12)

print("\n" + "=" * 64)
print(f"brazo A (gaussiana, {mejor_a[1]}, peso {mejor_a[2]}): "
      f"AP = {media_a:.3f} +- {tabla['AP A'].sem():.3f}")
print(f"brazo B (mezclado poisson, peso {mejor_b[1]}):        "
      f"AP = {media_b:.3f} +- {tabla['AP B'].sem():.3f}")
print(f"baseline de popularidad:                    AP = {ap_baseline:.3f}")
print(f"diferencia apareada B - A: {dif.mean():+.3f} +- {sem:.3f} ({veces:+.1f} SE)")
veredicto = "SUPERA el criterio" if veces > 2 else \
    ("empata" if abs(veces) <= 2 else "PIERDE")
print(f"criterio declarado (ganar por 2 SE): {veredicto}")

DIR_SALIDA.mkdir(parents=True, exist_ok=True)
pd.DataFrame(registro_val).to_csv(DIR_SALIDA / "validacion.csv", index=False)
tabla.to_csv(DIR_SALIDA / "prueba.csv", index=False)
print(f"\ntablas guardadas en {DIR_SALIDA}")
