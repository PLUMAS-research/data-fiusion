"""Como elegir el rango y los pesos, y sobre todo como evaluar la eleccion.

El error de reconstruccion es lo que el ajuste minimiza, asi que baja
siempre al subir el rango. Elegir por esa curva lleva al rango mas alto
que uno tenga paciencia de correr, que casi nunca es el que conviene.

Este script barre rango y peso, y mide por configuracion:

  error de reconstruccion   lo que el modelo optimiza (train). Va de 0
                            (exacto) a 1 (tan malo como predecir cero)
  AP held-out               average precision al predecir el genero de
                            peliculas reservadas. De 0 a 1; el baseline
                            marginal da 0.550
  NMI, ARI                  parecido entre la particion del modelo y la
                            de los generos. De 0 (azar) a 1 (identicas);
                            ARI puede ser negativo. NMI sube al pedir mas
                            grupos aunque no haya mas estructura, ARI no
  grupos usados             cuantos de los grupos pedidos quedan ocupados

El glosario completo, con ejemplos numericos, esta en README.md.

Las peliculas de validacion y test nunca aportan su genero al ajuste: la
relacion de generos entra enmascarada a las filas de entrenamiento, asi
que su etiqueta no puede filtrarse.

Genera dos figuras en output/ y una tabla con todo lo medido.

Ejecucion: uv run python examples/movielens/curvas.py
Toma alrededor de diez minutos.
"""

# %%
import os
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy.sparse as sp
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score

from datafiusion import Relation, fit

import datos as datos_movielens
from datos import CLAVE_GENERO


DIR_SALIDA = Path(os.environ.get("DIR_SALIDA", Path(__file__).parent / "output"))

SEMILLAS = (0, 1, 2, 3)
MAX_ITER = 200
RANKS_FIJOS = {"usuario": 25, "actor": 40, "genero": 15}
GRILLA_RANGO = (3, 5, 8, 12, 19, 30, 50, 80, 120)
GRILLA_PESO = (0.003, 0.01, 0.03, 0.1, 0.3, 1.0, 3.0, 10.0, 30.0, 100.0, 300.0)
RANGO_BASE, PESO_BASE = 19, 3.0

VISTAS = ("usuario", "actor")
NOMBRES = {"usuario": "vistas_usuario", "actor": "vistas_actor"}
ETIQUETAS = "etiquetas"

# Paleta Okabe-Ito: separable con las formas comunes de daltonismo.
AZUL, NARANJO, VERDE, BERMELLON = "#0072B2", "#E69F00", "#009E73", "#D55E00"
GRIS = "#666666"


# %%
def cargar():
    datos = datos_movielens.cargar(semilla=0, verbose=False)
    piezas = {c: sp.vstack([datos.train.R[c][0], datos.valid.R[c][0],
                            datos.test.R[c][0]], format="csr")
              for c in [("pelicula", "usuario"), CLAVE_GENERO, ("pelicula", "actor")]}
    n_tr, n_va = len(datos.train), len(datos.valid)
    indices = {"train": np.arange(n_tr + n_va),
               "test": np.arange(n_tr + n_va, piezas[CLAVE_GENERO].shape[0])}
    return datos, piezas, indices


datos, piezas, indices = cargar()
Y = np.asarray(piezas[CLAVE_GENERO].todense())
frecuencia = Y.sum(axis=0)
dominante = np.array([np.flatnonzero(f)[np.argmin(frecuencia[np.flatnonzero(f)])]
                      for f in Y])
print(f"{Y.shape[0]} peliculas, {Y.shape[1]} generos. "
      f"{len(indices['train'])} con etiqueta en el ajuste, "
      f"{len(indices['test'])} reservadas")


# %%
def ap_media(scores, Y_verdad):
    orden = np.argsort(-scores, axis=1)
    Yo = np.take_along_axis(Y_verdad, orden, axis=1)
    k = np.maximum(Y_verdad.sum(axis=1).astype(int), 1)
    prec = np.cumsum(Yo, axis=1) / np.arange(1, Y_verdad.shape[1] + 1)
    return float(((prec * Yo).sum(axis=1) / k).mean())


def medir(rango, peso, semilla):
    R = {ETIQUETAS: Relation(src="pelicula", dst="genero",
                             matrix=piezas[CLAVE_GENERO], rows=indices["train"])}
    for tipo in VISTAS:
        R[NOMBRES[tipo]] = Relation(src="pelicula", dst=tipo,
                                    matrix=piezas[("pelicula", tipo)])
    ranks = {"pelicula": rango, "genero": min(RANKS_FIJOS["genero"], rango)}
    ranks.update({t: RANKS_FIJOS[t] for t in VISTAS})

    modelo = fit(R, ranks, weights={ETIQUETAS: peso}, max_iter=MAX_ITER,
                 tol=1e-7, init="random", random_state=semilla)

    scores = modelo.predict_proba(target="genero", views=[ETIQUETAS],
                                  known={"pelicula": np.arange(Y.shape[0])})
    grupos = modelo.factor("pelicula").argmax(axis=1)
    return {
        "rango": rango, "peso": peso, "semilla": semilla,
        "error etiquetas": modelo.rel_error[ETIQUETAS],
        "error vistas": np.mean([modelo.rel_error[NOMBRES[t]] for t in VISTAS]),
        "AP train": ap_media(scores[indices["train"]], Y[indices["train"]]),
        "AP held-out": ap_media(scores[indices["test"]], Y[indices["test"]]),
        "NMI": normalized_mutual_info_score(dominante, grupos),
        # Sobre las peliculas reservadas el agrupamiento no vio su etiqueta,
        # asi que medir contra ella no es circular. La NMI de arriba si lo es
        # cuando lo que se barre es justamente el peso de esa relacion.
        "NMI held-out": normalized_mutual_info_score(dominante[indices["test"]],
                                                     grupos[indices["test"]]),
        # NMI sube casi siempre al partir en mas grupos, sesgo conocido. ARI
        # corrige por azar, asi que es la que decide cuando se barre el rango.
        "ARI held-out": adjusted_rand_score(dominante[indices["test"]],
                                            grupos[indices["test"]]),
        "grupos usados": len(np.unique(grupos)),
    }


# %%
print("\nbarriendo el rango...")
filas = [medir(rango, PESO_BASE, semilla)
         for rango in GRILLA_RANGO for semilla in SEMILLAS]
por_rango = pd.DataFrame(filas)
print(por_rango.groupby("rango")[["error etiquetas", "AP held-out",
                                  "NMI held-out", "ARI held-out"]].mean().round(3).to_string())

print("\nbarriendo el peso...")
filas = [medir(RANGO_BASE, peso, semilla)
         for peso in GRILLA_PESO for semilla in SEMILLAS]
por_peso = pd.DataFrame(filas)
print(por_peso.groupby("peso")[["error etiquetas", "error vistas", "AP held-out",
                                "NMI", "NMI held-out"]].mean().round(3).to_string())

DIR_SALIDA.mkdir(parents=True, exist_ok=True)
pd.concat([por_rango.assign(barrido="rango"),
           por_peso.assign(barrido="peso")]).to_csv(DIR_SALIDA / "curvas.csv", index=False)


# %%
def banda(ax, tabla, x, y, color, etiqueta=None, marcador="o"):
    """Media por valor de x, con una banda de un error estandar."""
    agrupado = tabla.groupby(x)[y].agg(["mean", "sem"])
    ax.plot(agrupado.index, agrupado["mean"], color=color, linewidth=2,
            marker=marcador, markersize=5, label=etiqueta, zorder=3)
    ax.fill_between(agrupado.index, agrupado["mean"] - agrupado["sem"],
                    agrupado["mean"] + agrupado["sem"], color=color, alpha=0.18,
                    linewidth=0, zorder=2)
    return agrupado


def marcar_maximo(ax, agrupado, color):
    """Senala el maximo de la curva, que es el valor a elegir."""
    mejor = agrupado["mean"].idxmax()
    ax.axvline(mejor, color=color, linestyle=":", linewidth=1.5, zorder=1)
    ax.annotate(f"{mejor:g}", xy=(mejor, agrupado["mean"].max()),
                xytext=(4, -12), textcoords="offset points",
                color=color, fontsize=9, fontweight="bold")
    return mejor


def estilo(ax, titulo, xlabel, ylabel, ticks=None):
    """Ticks explicitos: en escala log matplotlib solo rotula las potencias."""
    ax.set_title(titulo, fontsize=10.5, loc="left")
    if xlabel:
        ax.set_xlabel(xlabel, fontsize=9)
    ax.set_ylabel(ylabel, fontsize=9)
    ax.set_xscale("log")
    if ticks is not None:
        ax.set_xticks(list(ticks))
        ax.set_xticklabels([f"{v:g}" for v in ticks], fontsize=7.5)
        ax.minorticks_off()
    ax.grid(True, alpha=0.25, linewidth=0.6)
    ax.set_axisbelow(True)
    for lado in ("top", "right"):
        ax.spines[lado].set_visible(False)
    ax.tick_params(labelsize=8)


# %%
# Figura 1: el rango. Cada panel responde una pregunta distinta y las escalas
# no son comparables entre si, asi que van en paneles separados y nunca en un
# eje doble.
fig, axes = plt.subplots(2, 2, figsize=(11.5, 8.5))

ax = axes[0, 0]
banda(ax, por_rango, "rango", "error etiquetas", GRIS)
estilo(ax, "1. Lo que el ajuste minimiza baja siempre", None,
       "error de reconstruccion", GRILLA_RANGO)

ax = axes[0, 1]
agrupado = banda(ax, por_rango, "rango", "AP train", NARANJO,
                 "peliculas que aportaron su etiqueta", marcador="s")
agrupado_out = banda(ax, por_rango, "rango", "AP held-out", AZUL,
                     "peliculas reservadas")
marcar_maximo(ax, agrupado_out, AZUL)
estilo(ax, "2. Sobre datos reservados aparece un maximo", None, "AP media",
       GRILLA_RANGO)
ax.legend(fontsize=7.5, frameon=False, loc="center right")

ax = axes[1, 0]
banda(ax, por_rango, "rango", "NMI held-out", GRIS, "NMI", marcador="s")
agrupado = banda(ax, por_rango, "rango", "ARI held-out", VERDE, "ARI")
marcar_maximo(ax, agrupado, VERDE)
estilo(ax, "3. Agrupar tiene su propio optimo, y no es el mismo",
       "rango del tipo pelicula", "sobre peliculas reservadas", GRILLA_RANGO)
ax.legend(fontsize=7.5, frameon=False, loc="upper right")

ax = axes[1, 1]
brecha = por_rango.assign(brecha=por_rango["AP train"] - por_rango["AP held-out"])
banda(ax, brecha, "rango", "brecha", BERMELLON)
estilo(ax, "4. Y la distancia entre ambas mide cuanto memoriza",
       "rango del tipo pelicula", "AP del ajuste menos AP reservada", GRILLA_RANGO)

fig.suptitle("Elegir el rango: el error de entrenamiento no sirve como criterio, "
             "los datos reservados si", fontsize=12.5, x=0.02, ha="left")
fig.subplots_adjust(hspace=0.35, top=0.90)
fig.savefig(DIR_SALIDA / "eleccion_rango.png", dpi=150, bbox_inches="tight")
print(f"\nfigura: {DIR_SALIDA / 'eleccion_rango.png'}")


# %%
# Figura 2: el peso de la relacion de etiquetas.
fig, axes = plt.subplots(1, 3, figsize=(13.5, 4.4))

ax = axes[0]
banda(ax, por_peso, "peso", "error etiquetas", AZUL, "relacion de etiquetas")
banda(ax, por_peso, "peso", "error vistas", NARANJO, "las otras relaciones",
      marcador="s")
estilo(ax, "1. Subir el peso mejora una relacion\ny empeora las demas",
       "peso de la relacion de etiquetas", "error de reconstruccion", GRILLA_PESO)
ax.legend(fontsize=7.5, frameon=False)

ax = axes[1]
agrupado = banda(ax, por_peso, "peso", "AP held-out", AZUL)
marcar_maximo(ax, agrupado, AZUL)
estilo(ax, "2. Para predecir conviene un peso bajo",
       "peso de la relacion de etiquetas", "AP media held-out", GRILLA_PESO)

ax = axes[2]
banda(ax, por_peso, "peso", "NMI", GRIS, "todas las peliculas", marcador="s")
agrupado = banda(ax, por_peso, "peso", "NMI held-out", VERDE,
                 "solo las reservadas")
marcar_maximo(ax, agrupado, VERDE)
estilo(ax, "3. Los grupos se parecen a la relacion\nque mas pesa",
       "peso de la relacion de etiquetas", "NMI contra los generos", GRILLA_PESO)
ax.legend(fontsize=7.5, frameon=False, loc="upper left")

fig.suptitle("Elegir el peso: cuanto cuenta cada relacion en la perdida",
             fontsize=12.5, x=0.02, ha="left")
fig.subplots_adjust(top=0.82, wspace=0.28)
fig.savefig(DIR_SALIDA / "eleccion_peso.png", dpi=150, bbox_inches="tight")
print(f"figura: {DIR_SALIDA / 'eleccion_peso.png'}")


# %%
def reportar(tabla, columna, metrica):
    """Valor optimo, y si la grilla se queda corta.

    Estar en el borde solo obliga a extender si la curva todavia se mueve
    ahi. Si ya se aplano, el borde no es una limitacion: es una asintota.
    """
    agrupado = tabla.groupby(columna)[metrica].agg(["mean", "sem"])
    mejor = agrupado["mean"].idxmax()
    valores = list(agrupado.index)
    if mejor not in (valores[0], valores[-1]):
        return mejor, "ok"
    vecino = valores[1] if mejor == valores[0] else valores[-2]
    salto = abs(agrupado["mean"][mejor] - agrupado["mean"][vecino])
    ruido = max(agrupado["sem"][mejor], 1e-12)
    return mejor, "extender" if salto > 2 * ruido else "aplanado"


print("\n" + "=" * 74)
print("que elegir")
print("=" * 74)
for columna, tabla in (("rango", por_rango), ("peso", por_peso)):
    for metrica, tarea in (("AP held-out", "predecir"), ("ARI held-out", "agrupar")):
        mejor, estado = reportar(tabla, columna, metrica)
        aviso = {"ok": "",
                 "extender": "  <- en el borde y todavia subiendo, extender la grilla",
                 "aplanado": "  <- en el borde, pero la curva ya se aplano ahi"}[estado]
        print(f"  para {tarea:<9} {columna:<6} = {mejor:g}{aviso}")

print("\nCuando los dos optimos no coinciden hay que decidir para que se va a")
print("usar el modelo. No existe un valor que sea el mejor para las dos cosas.")
print("\nSe reporta ARI y no NMI porque NMI sube casi siempre al pedir mas")
print("grupos, aunque no haya mas estructura. ARI corrige por azar.")
print("\nAviso de circularidad: medir el agrupamiento contra la misma relacion")
print("cuyo peso se esta barriendo siempre premia subir ese peso. Por eso la")
print("curva que decide es la de las peliculas reservadas, que no aportaron su")
print("etiqueta al ajuste.")
