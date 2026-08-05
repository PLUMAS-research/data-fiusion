"""Publication-oriented diagram of a fusion setup.

`fusion_diagram` draws the schema of a fusion problem: entity types as
boxes (with their sizes and ranks) and relations as arrows from source
to destination, annotated with dimensions, density, preprocess chain and
row masks, and colored by family. It accepts the named relations that
`fit` takes, the legacy dict keyed by (src, dst), or a fitted
`FusionModel`.

Optional module: it needs matplotlib, which the rest of the library does
not.
"""

import numpy as np
import matplotlib.pyplot as plt
import scipy.sparse as sp
from matplotlib.lines import Line2D
from matplotlib.patches import FancyArrowPatch

from .model import FusionModel, _as_relations, _infer_sizes


# Okabe-Ito, seguros para daltonismo.
COLOR_FAMILIA = {"gaussian": "#0072B2", "poisson": "#D55E00"}
TINTA = "#333333"


def _miles(n):
    return f"{n:,}".replace(",", " ")


def _esquema(source, ranks):
    """(tipos ordenados, tamaños, aristas) desde cualquiera de las entradas."""
    if isinstance(source, FusionModel):
        sizes = dict(source.sizes)
        ranks = dict(source.ranks) if ranks is None else ranks
        prep = source.params.get("preprocess") or {}
        aristas = []
        for nombre, (src, dst) in source.rel.items():
            detalle = [", ".join(prep[nombre])] if nombre in prep else []
            aristas.append(dict(nombre=nombre, src=src, dst=dst,
                                dimensiones=(sizes[src], sizes[dst]),
                                detalle=detalle, familia=None))
    else:
        relaciones = _as_relations(source)
        sizes = _infer_sizes(relaciones)
        aristas = []
        for nombre, relacion in relaciones.items():
            n_i, n_j = relacion.shape
            if sp.issparse(relacion.matrix):
                densidad = relacion.matrix.nnz / (n_i * n_j)
            else:
                densidad = float(np.mean(np.asarray(relacion.matrix) != 0))
            partes = [f"{densidad:.2%}"
                      + (" · " + ", ".join(relacion.preprocess)
                         if relacion.preprocess else "")]
            if relacion.rows is not None:
                partes.append(f"{_miles(relacion.rows.size)} observed rows")
            if relacion.weighted:
                partes.append("entry weights")
            aristas.append(dict(nombre=nombre, src=relacion.src, dst=relacion.dst,
                                dimensiones=(n_i, n_j), detalle=partes,
                                familia=relacion.family))
    tipos = []
    for arista in aristas:
        for tipo in (arista["src"], arista["dst"]):
            if tipo not in tipos:
                tipos.append(tipo)
    return tipos, sizes, ranks, aristas


def _en_circulo(tipos):
    """Deterministic positions on a unit circle, first type on top."""
    angulos = np.pi / 2 - 2 * np.pi * np.arange(len(tipos)) / max(len(tipos), 1)
    if len(tipos) == 1:
        return {tipos[0]: (0.0, 0.0)}
    if len(tipos) == 2:
        return {tipos[0]: (-1.0, 0.0), tipos[1]: (1.0, 0.0)}
    return {t: (float(np.cos(a)), float(np.sin(a))) for t, a in zip(tipos, angulos)}


def fusion_diagram(source, ranks=None, positions=None, figsize=(6.0, 5.0), ax=None):
    """Draw the schema of a fusion setup.

    Parameters
    ----------
    source : dict[str, Relation], legacy dict[(src, dst), list], or FusionModel
        What to draw. With Relation objects the arrows carry density,
        preprocess, row masks and entry weights, and their color encodes
        the family; with a FusionModel the annotations come from what
        the model stores (sizes, ranks, preprocess).
    ranks : dict[str, int], optional
        Rank per type, annotated inside each box as c. Taken from the
        model when a FusionModel is given.
    positions : dict[str, (x, y)], optional
        Position per type. Default is a circle, first type on top.
    figsize : tuple
        Ignored when `ax` is given.
    ax : matplotlib Axes, optional

    Returns
    -------
    (figure, ax)
    """
    tipos, sizes, ranks, aristas = _esquema(source, ranks)
    if positions is None:
        positions = _en_circulo(tipos)
    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)
    else:
        fig = ax.figure

    # Curvatura por par no orientado: la primera arista va recta y las
    # siguientes se abren en arcos alternados.
    centroide = np.mean([np.asarray(p, dtype=float)
                         for p in positions.values()], axis=0)
    vistos = {}
    for arista in aristas:
        par = frozenset((arista["src"], arista["dst"]))
        k = vistos.get(par, 0)
        vistos[par] = k + 1
        arista["curva"] = 0.0 if k == 0 else 0.3 * ((k + 1) // 2) * (-1) ** k

    for arista in aristas:
        p1 = np.asarray(positions[arista["src"]], dtype=float)
        p2 = np.asarray(positions[arista["dst"]], dtype=float)
        color = COLOR_FAMILIA.get(arista["familia"], TINTA)
        ax.add_patch(FancyArrowPatch(
            p1, p2, arrowstyle="-|>", mutation_scale=13,
            connectionstyle=f"arc3,rad={arista['curva']}",
            shrinkA=42, shrinkB=42, linewidth=1.4, color=color, zorder=1))

        d = p2 - p1
        largo = float(np.hypot(*d))
        perpendicular = (np.array([-d[1], d[0]]) / largo if largo > 0
                         else np.array([0.0, 1.0]))
        # La etiqueta se corre perpendicular a la arista, alejandose del
        # centroide del diagrama para que aristas vecinas no choquen; en
        # los arcos ademas sigue al apice.
        desvio = arista["curva"] * largo / 2
        if arista["curva"] != 0:
            desvio += 0.10 * np.sign(arista["curva"])
        medio = (p1 + p2) / 2 + perpendicular * desvio
        hacia_afuera = medio - centroide
        signo = np.sign(float(hacia_afuera @ perpendicular)) or 1.0
        medio = medio + perpendicular * (signo * 0.24)

        n_i, n_j = arista["dimensiones"]
        lineas = [arista["nombre"],
                  f"{_miles(n_i)} $\\times$ {_miles(n_j)}"]
        lineas += list(arista["detalle"])
        ax.annotate("\n".join(lineas), medio, ha="center", va="center",
                    fontsize=8.5, color=TINTA, zorder=3, linespacing=1.4,
                    bbox=dict(boxstyle="round,pad=0.35", facecolor="white",
                              edgecolor="none"))

    for tipo in tipos:
        lineas = [tipo, f"n = {_miles(sizes[tipo])}"]
        if ranks and tipo in ranks:
            lineas.append(f"c = {ranks[tipo]}")
        ax.annotate("\n".join(lineas), positions[tipo], ha="center", va="center",
                    fontsize=10.5, color="black", zorder=4, linespacing=1.5,
                    bbox=dict(boxstyle="round,pad=0.55", facecolor="#F5F5F5",
                              edgecolor=TINTA, linewidth=1.1))

    familias = {a["familia"] for a in aristas if a["familia"]}
    if len(familias) > 1:
        ax.legend(handles=[Line2D([], [], color=COLOR_FAMILIA[f], lw=1.6, label=f)
                           for f in sorted(familias)],
                  loc="lower center", frameon=False, ncols=len(familias),
                  fontsize=9, bbox_to_anchor=(0.5, -0.04))

    xs = [p[0] for p in positions.values()]
    ys = [p[1] for p in positions.values()]
    ax.set_xlim(min(xs) - 0.85, max(xs) + 0.85)
    ax.set_ylim(min(ys) - 0.7, max(ys) + 0.7)
    ax.set_aspect("equal")
    ax.set_axis_off()
    return fig, ax
