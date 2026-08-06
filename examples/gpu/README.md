# Ajuste en GPU

`comparacion.py` corre `fuse` con los mismos datos, semilla e hiperparámetros
en CPU y en GPU (`device="gpu"`), y reporta tiempo por ajuste, aceleración y
el desvío relativo de la trayectoria de pérdida.

```
uv sync --extra gpu
uv run python examples/gpu/comparacion.py
```

Medido sobre una RTX 2080 (8 GB, CUDA 13, CuPy 14.1.1) contra un hilo de CPU,
todo en `float32`, 10 iteraciones, rangos 20 y 15:

| Instancia | nnz | CPU | GPU | Aceleración |
|---|---|---|---|---|
| 200k por 5k | 5M | 2.2 s | 0.38 s | 5.8x |
| 2M por 10k | 50M | 25.2 s | 3.3 s | 7.6x |

El desvío relativo de la pérdida entre dispositivos queda bajo $10^{-6}$, y el
modelo devuelto es numpy en ambos casos. Los tiempos de GPU incluyen la
transferencia de datos; la primera corrida de un proceso paga además la
compilación de kernels (el script la separa con un calentamiento).

`primitivas.py` mide las mismas instancias operación por operación (SpMM
directo y transpuesto, grams, GEMMs de factores, update). Es la vista que
ubica el cuello: la pasada transpuesta con la transpuesta materializada rinde
4x, y la variante `cusparse.spmm(transa=True)` que usa `fuse` la baja de 344 a
29 ms en la instancia grande.
