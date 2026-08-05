# Guía de uso

Esta guía recorre los flujos de trabajo de `data-fiusion` por tarea, con código
ejecutable y con las decisiones respaldadas por mediciones. El `README.md` describe
el modelo y la API; acá el orden es el del trabajo: preparar datos, ajustar, decidir
hiperparámetros, evaluar, persistir y escalar. Cada recomendación indica dónde está
medida.

## El modelo

Cada relación entre dos tipos de entidad se factoriza como
$R_{ij} \approx G_i S_{ij} G_j^T$, con factores no negativos $G$ (uno por tipo de
entidad, compartido entre todas las relaciones donde el tipo aparece) y un backbone
$S$ por relación. Que el factor sea compartido es lo que hace que las fuentes se
informen entre sí. La fila de $G_i$ es el embedding de la entidad; su argmax es su
grupo (co-clustering); el producto $G_i S_{ij} G_j^T$ reconstruye la relación.

## Primer ajuste

```python
import numpy as np
import scipy.sparse as sp
from datafiusion import Relation, fuse

relaciones = {
    "viajes": Relation(src="usuario", dst="zona", matrix=matriz_viajes),
    "horarios": Relation(src="usuario", dst="hora", matrix=matriz_horarios),
}
modelo = fuse(relaciones, ranks={"usuario": 30, "zona": 20, "hora": 10})

modelo.stop_reason          # "tol" convergio, "max_iter" se quedo corto
modelo.rel_error            # error relativo por relacion; cerca de 1 = no aprendio
modelo.factor("usuario")    # embedding (n_usuarios, 30)
grupos = modelo.factor("usuario").argmax(axis=1)
```

Las matrices son `scipy.sparse` (CSR idealmente) y los nombres de relaciones y
tipos se usan como nombres de archivo al guardar, así que sin `/` ni `\`.

## Preparar los datos

### De un DataFrame de coordenadas a una relación

El patrón para construir la matriz sin pasar por una grilla densa:

```python
import pandas as pd

usuarios = pd.Categorical(viajes_df["id_usuario"])
zonas = pd.Categorical(viajes_df["zona"])
matriz = sp.csr_matrix(
    (viajes_df["n_viajes"].to_numpy(dtype="float64"),
     (usuarios.codes, zonas.codes)),
    shape=(len(usuarios.categories), len(zonas.categories)))

Relation(src="usuario", dst="zona", matrix=matriz,
         row_labels=usuarios.categories.to_numpy(),
         col_labels=zonas.categories.to_numpy())
```

Cuando un mismo tipo aparece en varias relaciones, las filas deben referirse a las
mismas entidades en el mismo orden. Con `row_labels` y `col_labels` puestos, el
fold-in valida ese orden y una permutación de columnas falla con error en vez de
puntuar sinsentidos.

No usar `DataFusionModel` (el wrapper de `base.py`) con datos grandes: densifica
para alinear índices. Está en el repo por compatibilidad.

### Transformar los conteos antes de ajustar

Con conteos sobredispersos (viajes, reproducciones, palabras), la pérdida
cuadrática sobre los valores crudos es la peor opción medida, porque las entradas
de mayor magnitud dominan el ajuste. La recomendación con evidencia es
comprimirlos:

```python
Relation(src="usuario", dst="zona", matrix=matriz, preprocess="log1p")
```

`preprocess` es parte del modelo: la cadena y su estado (por ejemplo el vector idf)
se guardan con él, y `transform` y `loss` la reaplican a datos nuevos crudos. Eso
elimina el error silencioso de ajustar sobre `log1p(X)` y proyectar `X` crudo.
Registro disponible: `log1p`, `sqrt`, `anscombe` desplazada, `idf`, componibles
(`preprocess=("log1p", "idf")`).

Evidencia: en Last.fm, log1p ganó a los conteos crudos por 0.314 contra 0.156 de AP
(`examples/lastfm/README.md`); en 20 Newsgroups, TF-IDF ganó a todo
(`examples/newsgroups/README.md`). Existe `family="poisson"` para modelar conteos
con su likelihood, y perdió o empató contra estas transformaciones en todos los
protocolos medidos: úsala solo si el modelo de conteo se necesita por sus
propiedades, no por accuracy.

### Qué significa un cero

Sin indicación en contrario, cada cero es una observación ("esta entidad no se
relaciona con esto"). Cuando eso es falso hay tres mecanismos, por granularidad:

| mecanismo | granularidad | caso típico |
|---|---|---|
| `Relation(rows=...)` | fila completa | entidades sin etiqueta en una relación de etiquetas |
| `Relation(entry_weights=..., background=...)` | entrada | entradas retenidas para validar; ceros como negativos débiles |
| `fuse` avisa y registra `empty_rows` | entidad | entidades que quedaron sin ninguna observación |

El régimen semi-supervisado es el primero: la relación de etiquetas entra con
máscara sobre las filas etiquetadas y las demás entidades no aportan su fila, pero
sí participan del ajuste por las otras relaciones. Verificado que el contenido de
una fila enmascarada no afecta el ajuste (test a $10^{-12}$).

`background` controla los ceros no almacenados: `1.0` los observa, `0.0` los
ignora, intermedio los lee como negativos débiles (retroalimentación implícita).

## Elegir hiperparámetros

### El rango es la palanca principal

El rango del tipo fuente controla el sobreajuste; `lambda_G` no sirve (está medido
y explicado en el README: sin punto fijo o inerte). Para elegirlo con datos, la
curva de validación por entradas retenidas:

```python
from datafiusion import holdout_entries

pesos, (filas, columnas) = holdout_entries(matriz, fraction=0.1, random_state=0)
rel = Relation(src="usuario", dst="zona", matrix=matriz, entry_weights=pesos)

for rango in (10, 20, 40, 80):
    modelo = fuse({"viajes": rel}, ranks={"usuario": rango, "zona": 20})
    pred = modelo.reconstruct_entries("viajes", filas, columnas)
    verdad = np.asarray(matriz[filas, columnas]).ravel()
    print(rango, float(np.sqrt(np.mean((pred - verdad) ** 2))))
```

Las entradas retenidas no influyen en el ajuste por ninguna vía (pérdida, escala,
inicialización); esa invariancia tiene test propio.

### Pesos entre relaciones

`weights` fija cuánto pesa cada relación en la pérdida. Con la normalización
Frobenius de serie (y la calibración por desviación nula en relaciones Poisson), el
peso lee como fracción del presupuesto de pérdida y es comparable entre relaciones.
Se elige en validación; en los casos medidos el óptimo estuvo entre 3 y 100 para la
relación de etiquetas. Si el óptimo cae en el borde de la grilla, extender la
grilla antes de concluir.

### Inicialización y reinicios

`init="nndsvd"` (determinista) es el default y en MovieLens vale +0.25 de AP sobre
la aleatoria. Con `init="random"`, `n_runs=k` corre k reinicios y devuelve el de
menor pérdida final:

```python
modelo = fuse(relaciones, ranks, init="random", n_runs=5, random_state=0)
modelo.params["run_losses"], modelo.params["best_run"]
```

No comparar configuraciones con `init` distinto: la inicialización domina por sobre
casi todo lo demás (medido en `examples/movielens/`).

### Clustering contra predicción

Para **agrupar**, `fuse` con el gauge por columnas (default): en MovieLens sube el
ARI de 0.464 a 0.741 contra la ruta anterior, y el gauge es lo que hace legible el
argmax. Para **predecir** un atributo retenido las dos rutas quedan parejas (la
anterior 0.010 de AP arriba). Los números y protocolos están en
`examples/movielens/README.md`.

## Evaluar sin engañarse

Tres reglas que salieron de errores reales, documentados en los ejemplos:

- **No evaluar in-sample**: predecir una relación que entró al ajuste da AP cercano a
  1 por construcción. Evaluar con entidades cuya fila de etiquetas estuvo
  enmascarada, o retenidas del ajuste.
- **No leer NMI solo**: premia partir grupos. Decidir por ARI (o por la métrica de
  la tarea) y reportar NMI como complemento.
- **Declarar el criterio antes de correr**: los scripts de `examples/` imprimen su
  criterio (ganar por 2 errores estándar de la diferencia apareada) y las
  conclusiones del repo se atienen a eso, incluidos los resultados negativos.

Para predecir un atributo desde varias vistas:

```python
proba = modelo.predict_proba(target="proposito", views=["od", "horarios"])
```

Combina las vistas por media geométrica (Naive Bayes suavizado) y corre por lotes
de filas; con `top_k` devuelve solo los mejores puntajes cuando la matriz completa
no cabe.

## Persistir, reanudar, escalar

```python
modelo.save("modelo_viajes")
modelo = FusionModel.load("modelo_viajes", mmap=True)   # factores quedan en disco
modelo = modelo.resume(relaciones, max_iter=50)         # continua el ajuste
```

`save` escribe un directorio (factores como `.npy` mapeables, configuración
completa: supervisión, máscaras, grafos, preprocesamiento y su estado). `resume`
tras `load` reaplica esa configuración sin repetirla a mano. Guardar sobre un
directorio ya usado lo limpia primero.

El patrón para millones de entidades: ajustar sobre una muestra y proyectar el
resto sin reajustar.

```python
nuevas = {"viajes": Relation(src="usuario", dst="zona", matrix=matriz_nueva)}
derivado = modelo.transform(nuevas, target="usuario")
derivado.predict_proba(target="proposito", views=["od"])
```

`transform` reaplica la escala, los pesos y el preprocesamiento del entrenamiento
(aplicar los propios del lote nuevo produce factores incorrectos sin aviso; está
medido), honra máscaras de fila, devuelve factores no negativos como el ajuste, y
reporta en `empty_rows` las entidades nuevas sin observaciones. Un modelo derivado
por `transform` no se puede reanudar: es una proyección, no un checkpoint del
ajuste.

## Señales de problema y qué significan

| señal | lectura |
|---|---|
| `UserWarning` de entidades sin observación | un filtro aguas arriba dejó filas vacías; su factor es cero y argmax las manda todas al grupo 0. Ver `modelo.empty_rows` |
| `stop_reason == "max_iter"` | no convergió; subir `max_iter` o reanudar con `resume` |
| `rel_error` cercano a 1 en una relación | esa relación no se está aprendiendo: revisar su peso y su escala |
| `dead_columns` no vacío | componentes colapsadas; suele indicar rango excesivo |
| `ValueError` con nombres de relación o tipo | los nombres se usan como archivos en `save`; sin `/` ni `\` |
| `ValueError` por declarar `preprocess` distinto al del ajuste | el modelo aplica su propia cadena a datos crudos; no pre-transformar |

## Rendimiento

- El costo por iteración es O(nnz por rango) y la pérdida nunca materializa una
  matriz del tamaño de la relación. Contra la implementación de referencia
  densificada: 2 a 5 veces menos tiempo y 7 a 10 veces menos memoria.
- `block_rows` acota la memoria de las pasadas por bloques (default 200 mil filas).
- Los pesos por entrada cuestan 1.9x con 10% de entradas retenidas y 5.2x en el
  régimen implícito denso (medido; `docs/oportunidades.md` anota las vías de
  optimización).
- Con datos de más de ~30 minutos de cómputo, decidir qué columnas intermedias
  persistir antes de correr, no después.

## Dónde profundizar

- `examples/movielens/`: predicción y clustering con verdad de referencia, la
  comparación entre rutas y contra scikit-fusion.
- `examples/newsgroups/`: semi-supervisión (anclaje TS-NMF) y likelihoods sobre
  texto.
- `examples/lastfm/`: familias mezcladas sobre conteos reales.
- `docs/oportunidades.md`: lo que falta, con lo medido de cada cosa, y el registro
  de lo que se probó y no funcionó.
- `docs/diseno-regularizacion.md`: por qué la API es la que es.
