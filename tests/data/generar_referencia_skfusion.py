"""Regenera las trazas de referencia de scikit-fusion.

Este es el unico lugar del repo que importa scikit-fusion. Corre Dfmf
sobre las instancias sinteticas que usa `tests/test_compare_skfusion.py`
y guarda las trazas de error en `referencia_skfusion.json`; el test
compara contra ese archivo y no necesita la dependencia.

Requiere un checkout de scikit-fusion instalado en el venv:

    uv pip install -e <ruta a scikit-fusion> --python .venv/bin/python
    uv run python tests/data/generar_referencia_skfusion.py
"""

import json
import pathlib
import sys

import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
from test_compare_skfusion import INSTANCIAS, make_low_rank_data, relative_frobenius


def _parchar_par_bdot():
    """El filtro `entry != []` de _par_bdot revienta con numpy >= 1.25."""
    from joblib import Parallel, delayed
    import skfusion.fusion.decomposition._dfmf as _dfmf

    bdot = None
    for name in dir(_dfmf):
        if name.endswith("__bdot"):
            bdot = getattr(_dfmf, name)
            break
    if bdot is None:
        raise RuntimeError("no encuentro __bdot dentro de skfusion._dfmf")

    def _par_bdot(A, B, obj_types, verbose, n_jobs):
        paralelo = Parallel(n_jobs=n_jobs, max_nbytes=1e3, verbose=verbose,
                            backend="multiprocessing")
        tareas = (delayed(bdot)(A, B, i, j, obj_types)
                  for i in obj_types for j in obj_types)
        entradas = paralelo(tareas)
        return {(i, j): entrada for i, j, entrada in entradas
                if not (isinstance(entrada, list) and len(entrada) == 0)}

    _dfmf._par_bdot = _par_bdot


def _importar_skfusion():
    """Shims de compatibilidad (np.float, collections.Iterable) e import."""
    if not hasattr(np, "float"):
        np.float = float
    if not hasattr(np, "int"):
        np.int = int
    import collections
    import collections.abc
    if not hasattr(collections, "Iterable"):
        collections.Iterable = collections.abc.Iterable
    from skfusion import fusion as skf
    _parchar_par_bdot()
    return skf


def traza_skfusion(skf, R12, R13, R23, ranks, max_iter, seed, paso=10):
    """Error de reconstruccion cada `paso` iteraciones, refiteando desde cero.

    Dfmf no expone reanudacion, asi que cada punto de la traza es un fit
    completo hasta ese numero de iteraciones con la misma semilla.
    """
    t1 = skf.ObjectType("t1", ranks[0])
    t2 = skf.ObjectType("t2", ranks[1])
    t3 = skf.ObjectType("t3", ranks[2])
    relaciones = [skf.Relation(R12, t1, t2), skf.Relation(R13, t1, t3),
                  skf.Relation(R23, t2, t3)]
    fg = skf.FusionGraph(relaciones)

    traza = []
    for hasta in range(paso, max_iter + 1, paso):
        fuser = skf.Dfmf(max_iter=hasta, init_type="random",
                         random_state=np.random.RandomState(seed)).fuse(fg)
        G = {"t1": fuser.factor(t1), "t2": fuser.factor(t2),
             "t3": fuser.factor(t3)}
        S12, S13, S23 = (fuser.backbone(r) for r in relaciones)
        err = (relative_frobenius(R12, G["t1"] @ S12 @ G["t2"].T)
               + relative_frobenius(R13, G["t1"] @ S13 @ G["t3"].T)
               + relative_frobenius(R23, G["t2"] @ S23 @ G["t3"].T))
        traza.append(float(err))
        print(f"  {hasta:4d} iteraciones: error {err:.4f}")
    return traza


def main():
    skf = _importar_skfusion()
    import skfusion

    salida = {"generado_con": {"skfusion": skfusion.__version__,
                               "numpy": np.__version__},
              "instancias": {}}
    for nombre, params in INSTANCIAS.items():
        print(f"instancia {nombre}: n={params['n']}, c={params['c']}, "
              f"max_iter={params['max_iter']}")
        R12, R13, R23, piso = make_low_rank_data(
            *params["n"], *params["c"], noise=params["noise"],
            seed=params["seed"])
        traza = traza_skfusion(skf, R12, R13, R23, params["c"],
                               params["max_iter"], params["seed"],
                               paso=params["paso"])
        salida["instancias"][nombre] = dict(params, traza=traza,
                                            noise_floor=float(piso))

    destino = pathlib.Path(__file__).with_name("referencia_skfusion.json")
    destino.write_text(json.dumps(salida, indent=2) + "\n")
    print(f"guardado en {destino}")


if __name__ == "__main__":
    main()
