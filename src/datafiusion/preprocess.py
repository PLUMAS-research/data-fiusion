"""Named value transforms that travel with the model.

A `Relation(preprocess=...)` names one or more transforms from a closed
registry. `fit` learns any per-transform state (the idf vector) from the
training data and stores it in the model, and `FusionModel.transform`
and `loss` reapply the SAME transform with the SAME state to incoming
data. This closes the same silent-units bug that the stored Frobenius
scale closes: fitting on log1p(X) and folding in raw X, or recomputing
idf on the new batch, produces wrong factors with no error.

The registry is closed on purpose: names persist in meta.json and state
arrays in the saved model, which a callable could not.

Every transform maps 0 to 0, so the sparsity pattern is untouched and
`entry_weights` stay aligned with `matrix.data`.
"""

import numpy as np
import scipy.sparse as sp


ELEMENTALES = ("log1p", "sqrt", "anscombe")
CON_ESTADO = ("idf",)
REGISTRO = ELEMENTALES + CON_ESTADO

_RAIZ_ANSCOMBE = np.sqrt(0.375)


def validate_names(preprocess):
    """Normalize `preprocess` to a tuple of registered names."""
    nombres = (preprocess,) if isinstance(preprocess, str) else tuple(preprocess)
    for nombre in nombres:
        if nombre not in REGISTRO:
            raise ValueError(
                f"unknown preprocess {nombre!r}; registered: {', '.join(REGISTRO)}")
    return nombres


def apply_chain(matrix, nombres, estado=None):
    """Apply the transforms in order to the stored entries of `matrix`.

    With estado=None the stateful transforms LEARN from `matrix` (fit
    time) and the learned state is returned. With a dict (possibly
    empty) they REUSE it (fold-in time) and raise if a needed key is
    missing. The result shares the index arrays of `matrix`; only the
    data array is new, so the caller's matrix is never modified.

    Returns (transformed csr_matrix, state dict).
    """
    matrix = matrix.tocsr()
    datos = matrix.data.astype(np.float64, copy=True)
    aprender = estado is None
    estado = {} if aprender else dict(estado)
    for nombre in nombres:
        if nombre in ELEMENTALES:
            if datos.size and datos.min() < 0:
                raise ValueError(
                    f"preprocess {nombre!r} needs non-negative data")
            if nombre == "log1p":
                datos = np.log1p(datos)
            elif nombre == "sqrt":
                datos = np.sqrt(datos)
            else:
                datos = 2.0 * (np.sqrt(datos + 0.375) - _RAIZ_ANSCOMBE)
        elif nombre == "idf":
            if aprender:
                df = matrix.getnnz(axis=0)
                idf = np.log((1.0 + matrix.shape[0]) / (1.0 + df)) + 1.0
                estado["idf"] = idf
            elif "idf" not in estado:
                raise ValueError(
                    "the fitted model carries no idf state for this relation")
            else:
                idf = np.asarray(estado["idf"])
                if idf.shape[0] != matrix.shape[1]:
                    raise ValueError(
                        f"stored idf has {idf.shape[0]} columns but the matrix "
                        f"has {matrix.shape[1]}")
            datos = datos * idf[matrix.indices]
    salida = sp.csr_matrix((datos, matrix.indices, matrix.indptr),
                           shape=matrix.shape)
    return salida, estado


def invert_chain(valores, nombres, estado, columnas):
    """Undo the chain on reconstructed values at given column indices.

    Every registered transform is invertible, but note that inverting a
    reconstruction does not give the conditional mean in original units
    (the transforms are non-linear); it is a readable approximation.
    """
    v = np.asarray(valores, dtype=np.float64).copy()
    columnas = np.asarray(columnas)
    for nombre in reversed(tuple(nombres)):
        if nombre == "log1p":
            v = np.expm1(v)
        elif nombre == "sqrt":
            v = np.square(v)
        elif nombre == "anscombe":
            v = np.square(v / 2.0 + _RAIZ_ANSCOMBE) - 0.375
        elif nombre == "idf":
            v = v / np.asarray(estado["idf"])[columnas]
    return v
