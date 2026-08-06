"""Optional GPU backend through CuPy.

``fuse(..., device="gpu")`` runs the iteration loop on a CUDA GPU: the
relation matrices are uploaded once, the sparse passes go through
cuSPARSE, the factor algebra through cuBLAS, and the fitted model comes
back as plain numpy, so everything downstream (``save``, ``transform``,
``predict_proba``) is unchanged. The default ``device="cpu"`` never
imports this module, so systems without a GPU need no extra dependency
and no configuration.

Install with the ``gpu`` extra (``pip install 'data-fiusion[gpu]'`` or
``uv sync --extra gpu``), which resolves to ``cupy-cuda13x[ctk]``. An
NVIDIA driver with CUDA 13 support is required; on a CUDA 12 driver
install ``cupy-cuda12x[ctk]`` instead. The unsuffixed NVIDIA wheels for
CUDA 13 place their libraries under ``nvidia/cu13/lib``, a layout that
``cuda-pathfinder`` 1.6 does not search; ``_cuda_path()`` points
``CUDA_PATH`` there before importing cupy, so no manual configuration is
needed.

The transposed sparse pass uses ``cusparse.spmm(..., transa=True)`` on
the original CSR: it scatters into the small output instead of gathering
from the large factor, measured 12x faster than multiplying by a
materialized transpose, and it avoids a second copy of the relation in
device memory.
"""

import os
import pathlib

import numpy as np
import scipy.sparse as sp


_INSTALL = (
    "device='gpu' needs CuPy and a CUDA GPU. Install the optional "
    "dependency with `pip install 'data-fiusion[gpu]'` or "
    "`uv sync --extra gpu` (NVIDIA driver with CUDA 13; on a CUDA 12 "
    "driver install 'cupy-cuda12x[ctk]' instead). Systems without a GPU "
    "need nothing: the default device='cpu' never imports cupy."
)


def _cuda_path():
    """Point CUDA_PATH at the NVIDIA wheel libraries if nothing set it."""
    if os.environ.get("CUDA_PATH"):
        return
    try:
        import nvidia
    except ImportError:
        return
    for base in nvidia.__path__:
        cu13 = pathlib.Path(base) / "cu13"
        if (cu13 / "lib").is_dir():
            os.environ["CUDA_PATH"] = str(cu13)
            return


def _cupy():
    """Import cupy with a usable CUDA device, or raise with instructions."""
    _cuda_path()
    try:
        import cupy
    except ImportError as exc:
        raise ImportError(_INSTALL) from exc
    try:
        n = cupy.cuda.runtime.getDeviceCount()
    except Exception as exc:
        raise RuntimeError(
            f"cupy is installed but CUDA is not usable ({exc}). " + _INSTALL
        ) from exc
    if n == 0:
        raise RuntimeError(
            "cupy is installed but no CUDA GPU was detected. " + _INSTALL)
    return cupy


def available():
    """Whether device='gpu' would work here, without raising."""
    try:
        _cupy()
    except Exception:
        return False
    return True


def to_device(a):
    """Copy an ndarray or scipy sparse matrix to the GPU."""
    cupy = _cupy()
    if sp.issparse(a):
        import cupyx.scipy.sparse as csp
        return csp.csr_matrix(sp.csr_matrix(a))
    return cupy.asarray(a)


def to_host(a):
    """Copy back to numpy; numpy input passes through."""
    if isinstance(a, np.ndarray):
        return a
    import cupy
    return cupy.asnumpy(a)


def spmm_transpose(M, B):
    """M.T @ B on the GPU without materializing the transpose."""
    if not hasattr(M, "indptr"):
        return M.T @ B
    from cupyx import cusparse
    cupy = _cupy()
    return cusparse.spmm(M, cupy.asfortranarray(B), transa=True)


class _DeviceRelation:
    """The slice of a Relation that the fit loop reads, on the GPU.

    Only the observed matrix travels: the row mask is applied once on
    the CPU and the sliced matrix uploaded, so `observed()` costs
    nothing per call.
    """

    __slots__ = ("src", "dst", "weighted", "_observed")

    def __init__(self, relacion):
        cupy = _cupy()
        self.src, self.dst = relacion.src, relacion.dst
        self.weighted = False
        M, filas = relacion.observed()
        self._observed = (to_device(M),
                          None if filas is None else cupy.asarray(filas))

    def observed(self):
        return self._observed


def convert_relations(relations):
    """GPU-resident views of the relations, rejecting what has no GPU path."""
    for nombre, relacion in relations.items():
        if relacion.weighted:
            raise ValueError(
                f"relation {nombre!r} carries entry weights, which are not "
                "supported with device='gpu' yet; fit it on device='cpu'")
    return {nombre: _DeviceRelation(r) for nombre, r in relations.items()}
