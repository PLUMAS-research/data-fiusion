"""Row observation masks: what they guarantee and what they do not.

A fraction of the users have a known class label, and the labels are
encoded as one more relation. The unlabeled rows of that relation are the
problem: the matrix has to contain something, and without a mask whatever
sits there is learned as an observation. A zero row reads as "this user
belongs to no class"; a uniform fill reads as "this user belongs equally
to all of them". Neither was observed.

`Relation.rows` removes those rows from the domain of the loss instead of
filling them.

What that guarantees, and this script verifies it: the content of a masked
row cannot affect the fit at all. Two fits whose hidden rows hold
completely different values produce identical factors.

What it does NOT guarantee, and this script also shows it: better
accuracy. Masking only removes a false claim. If the observed relations do
not carry the class signal, removing the false claim leaves nothing in its
place, and a uniform fill can even score better by acting as a crude
regularizer. On this synthetic instance every variant sits near chance.

For evidence that masks pay off on real data, see
examples/movielens/prediccion_fit.py, where the masked protocol reaches AP
0.716 against 0.550 for the marginal baseline.

Ejecucion: uv run python examples/semi_supervised.py
"""

# %%
import numpy as np
import scipy.sparse as sp

from datafiusion import Relation, fit


SEMILLA = 0
N_USUARIOS, N_ITEMS, N_CLASES = 400, 120, 3
FRACCION_ETIQUETADA = 0.25
RANKS = {"usuario": 8, "item": 6, "clase": 2}


# %%
def generar():
    """User factor with a class part plus an interest part orthogonal to it."""
    rng = np.random.default_rng(SEMILLA)
    medias = np.abs(rng.standard_normal((N_CLASES, 3)))
    clase_real = rng.integers(0, N_CLASES, N_USUARIOS)
    G_usuario = np.maximum(np.hstack([
        medias[clase_real] * 2.0,
        np.abs(rng.standard_normal((N_USUARIOS, 2))),
    ]), 0.0)
    G_item = np.abs(rng.standard_normal((N_ITEMS, 5)))

    ratings = np.maximum(G_usuario @ G_item.T
                         + 0.1 * rng.standard_normal((N_USUARIOS, N_ITEMS)), 0.0)
    R_ui = sp.csr_matrix(np.where(rng.random(ratings.shape) < 0.25, ratings, 0.0))

    etiquetados = np.sort(rng.choice(N_USUARIOS,
                                     size=int(FRACCION_ETIQUETADA * N_USUARIOS),
                                     replace=False))
    sin_etiqueta = np.setdiff1d(np.arange(N_USUARIOS), etiquetados)
    Y = np.zeros((N_USUARIOS, N_CLASES))
    Y[np.arange(N_USUARIOS), clase_real] = 1.0
    return R_ui, Y, clase_real, etiquetados, sin_etiqueta


R_ui, Y, clase_real, etiquetados, sin_etiqueta = generar()
print(f"usuarios: {N_USUARIOS}, items: {N_ITEMS}, clases: {N_CLASES}")
print(f"etiquetados: {len(etiquetados)}, sin etiqueta: {len(sin_etiqueta)}")


# %%
def ajustar(Y_matriz, filas_observadas):
    """El patron: la mascara va en la relacion, no en los datos."""
    relaciones = {
        "ratings": Relation(src="usuario", dst="item", matrix=R_ui),
        "etiquetas": Relation(src="usuario", dst="clase",
                              matrix=sp.csr_matrix(Y_matriz), rows=filas_observadas),
    }
    return fit(relaciones, RANKS, max_iter=300, tol=1e-7, init="nndsvd",
               random_state=SEMILLA)


def predecir(modelo):
    return modelo.predict_proba(target="clase", views=["etiquetas"],
                                known={"usuario": np.arange(N_USUARIOS)})


def acierto(scores, indices):
    return float((scores[indices].argmax(axis=1) == clase_real[indices]).mean())


# %%
# Lo que la mascara SI garantiza: el contenido de las filas ocultas no
# entra en el ajuste. Se comprueba poniendo basura distinta en ellas.
rng = np.random.default_rng(1)
basura_a, basura_b = Y.copy(), Y.copy()
basura_a[sin_etiqueta] = rng.exponential(50.0, (len(sin_etiqueta), N_CLASES))
basura_b[sin_etiqueta] = rng.exponential(0.01, (len(sin_etiqueta), N_CLASES))

modelo_a = ajustar(basura_a, etiquetados)
modelo_b = ajustar(basura_b, etiquetados)
desvio = max(np.abs(modelo_a.G[t] - modelo_b.G[t]).max() for t in modelo_a.G)
print(f"\nmaximo desvio entre dos ajustes cuyas filas ocultas difieren por completo: "
      f"{desvio:.2e}")
print("Sin mascara ese numero seria grande: cada relleno se aprende como observacion.")


# %%
# Lo que la mascara NO garantiza: mejor acierto.
Y_cero = Y.copy()
Y_cero[sin_etiqueta] = 0.0
Y_uniforme = Y.copy()
Y_uniforme[sin_etiqueta] = 1.0 / N_CLASES

variantes = {
    "sin mascara, relleno cero": predecir(ajustar(Y_cero, None)),
    "sin mascara, relleno uniforme": predecir(ajustar(Y_uniforme, None)),
    "con mascara por fila": predecir(ajustar(Y, etiquetados)),
}

print(f"\n{'variante':<32} {'sin etiqueta':>13} {'etiquetados':>12}")
for nombre, scores in variantes.items():
    print(f"{nombre:<32} {acierto(scores, sin_etiqueta):>13.3f} "
          f"{acierto(scores, etiquetados):>12.3f}")
print(f"{'azar':<32} {1 / N_CLASES:>13.3f}")

print("\nLas tres quedan cerca del azar sobre los usuarios sin etiqueta. En esta")
print("instancia sintetica la clase no es recuperable desde los ratings, asi que")
print("quitar la afirmacion falsa no deja nada mejor en su lugar. La mascara")
print("corrige la semantica del modelo, no crea senal donde no la hay.")
