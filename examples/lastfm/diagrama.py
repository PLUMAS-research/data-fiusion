"""Diagrama del esquema de fusion de Last.fm.

Produce la figura del esquema usado en la comparacion: escuchas con
log1p y etiquetas con mascara de filas sobre el entrenamiento, con el
artista como tipo compartido. Sale en PDF (vectorial, para el paper) y
PNG (para mirar).

Ejecucion: uv run python examples/lastfm/diagrama.py
"""

# %%
import os
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import numpy as np

from datafiusion import Relation
from datafiusion.diagram import fusion_diagram

from datos import cargar


DIR_SALIDA = Path(os.environ.get("DIR_SALIDA", Path(__file__).parent / "output"))

# %%
d = cargar()
rng = np.random.default_rng(0)
orden = rng.permutation(d["etiquetados"])
entrenamiento = np.sort(orden[:int(0.6 * len(orden))])

relaciones = {
    "escuchas": Relation(src="artista", dst="usuario", matrix=d["escuchas"],
                         preprocess="log1p"),
    "etiquetas": Relation(src="artista", dst="tag", matrix=d["etiquetas"],
                          rows=entrenamiento),
}
fig, ax = fusion_diagram(relaciones,
                         ranks={"artista": 30, "usuario": 20, "tag": 20})

DIR_SALIDA.mkdir(parents=True, exist_ok=True)
fig.savefig(DIR_SALIDA / "diagrama.pdf", bbox_inches="tight")
fig.savefig(DIR_SALIDA / "diagrama.png", bbox_inches="tight", dpi=200)
print(f"figura en {DIR_SALIDA / 'diagrama.pdf'} y .png")
