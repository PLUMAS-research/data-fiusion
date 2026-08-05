# data-fiusion

Factorización tri-no negativa de matrices dispersas para fusión de datos, siguiendo el
modelo de Zitnik y Zupan (2015). Las matrices de relación se mantienen en
`scipy.sparse` y los factores latentes como arreglos densos de rango bajo.

El objetivo es que el modelo corra sobre relaciones con millones de filas y densidad de
pocos por ciento, donde una implementación que densifica no cabe en memoria. Sobre el
mismo problema, la implementación de referencia densificada usa entre 7 y 10 veces más
memoria y entre 2 y 5 veces más tiempo.

## Modelo

Una relación $R_{ij}$ de forma $(n_i, n_j)$ se factoriza como

$$R_{ij} \approx G_i\, S_{ij}\, G_j^T$$

con factores no negativos $G_i \in \mathbb{R}_{+}^{n_i \times c_i}$ y backbones
$S_{ij} \in \mathbb{R}^{c_i \times c_j}$ sin restricción de signo. Cada tipo de entidad
aparece en una o más relaciones y su embedding $G_i$ es compartido entre todas ellas,
que es lo que hace que las fuentes se informen entre sí.

## Instalación

```bash
git clone <repo-url> data-fiusion
cd data-fiusion
uv venv
uv pip install -e .
```

Opcional: `uv pip install -e ".[viz]"` para el diagrama de relaciones.

## Uso

```python
import numpy as np
import scipy.sparse as sp
from datafiusion import Relation, fuse

rng = np.random.default_rng(0)
relaciones = {
    "ratings": Relation(src="usuario", dst="pelicula",
                        matrix=sp.random(1000, 500, density=0.05, format="csr")),
    "generos": Relation(src="pelicula", dst="genero",
                        matrix=sp.random(500, 19, density=0.15, format="csr")),
}
modelo = fuse(relaciones, ranks={"usuario": 20, "pelicula": 15, "genero": 10},
             max_iter=200, tol=1e-6)

print(modelo.n_iter, modelo.stop_reason, modelo.rel_error)
```

`fuse` devuelve un `FusionModel` que lleva consigo los factores, los backbones, el
escalado aplicado durante el ajuste y la traza de la pérdida. Ese estado es lo que
permite proyectar entidades nuevas sin que quien llama tenga que recordar en qué
unidades quedó el ajuste.

### Proyectar entidades nuevas

```python
nuevas = {"ratings": Relation(src="usuario", dst="pelicula", matrix=matriz_nueva)}
derivado = modelo.transform(nuevas, target="usuario")
proba = derivado.predict_proba(target="genero", views=["generos"])
```

`transform` devuelve un modelo derivado que comparte todo salvo el factor proyectado,
así que compone directamente con `predict_proba`. La solución es no negativa, como los
factores que produce el ajuste. Honra `Relation.rows` con la misma semántica que `fuse`:
cada entidad nueva se resuelve solo desde las relaciones donde fue observada. Las
entidades nuevas sin ninguna observación quedan con factor cero, generan un aviso y
aparecen en `empty_rows` del modelo derivado.

### Observaciones faltantes

Sin máscara, cada cero de una relación es una observación: el modelo aprende que esa
entidad no tiene relación con nada. `Relation.rows` marca qué filas fueron observadas,
y las demás no entran a la pérdida.

```python
Relation(src="pelicula", dst="genero", matrix=Y, rows=indices_etiquetados)
```

Esto es lo que hace posible un régimen semi-supervisado: se ajusta con todas las
entidades presentes, pero solo las etiquetadas aportan su etiqueta.

### Pesos por entrada y entradas retenidas

`Relation.rows` opera por fila completa. `entry_weights` baja a la entrada: cada
entrada almacenada lleva un peso (alineado con `matrix.data`) y las no almacenadas
un peso de fondo `background`. Peso cero oculta la entrada del ajuste por todas las
vías (pérdida, escala e inicialización), lo que da validación con entradas retenidas
sin reservar entidades completas:

```python
from datafiusion import holdout_entries

pesos, (filas, columnas) = holdout_entries(M, fraction=0.1, random_state=0)
modelo = fuse({"ratings": Relation(src="usuario", dst="pelicula", matrix=M,
                                  entry_weights=pesos)}, ranks)
pred = modelo.reconstruct_entries("ratings", filas, columnas)
```

`background` controla qué son los ceros no almacenados: `1.0` mantiene la semántica
usual (cada cero es una observación), `0.0` los ignora, y valores intermedios los
leen como negativos débiles, que es el régimen de retroalimentación implícita de Hu,
Koren y Volinsky. Pesos mayores que uno marcan entradas más importantes.

### Transformaciones que viajan con el modelo

Las transformaciones de valores que ganaron en los experimentos (log1p, idf) se
declaran en la relación y pasan a ser parte del modelo:

```python
Relation(src="artista", dst="usuario", matrix=escuchas, preprocess="log1p")
Relation(src="documento", dst="termino", matrix=conteos, preprocess=("log1p", "idf"))
```

El registro es cerrado (`log1p`, `sqrt`, `anscombe` desplazada, `idf`) porque los
nombres y el estado persisten con `save`. La diferencia con transformar afuera está
en los datos no vistos: `FusionModel.transform` y `loss` reaplican la misma cadena a
datos crudos entrantes, y el idf que usan es el aprendido en el entrenamiento, no
uno recalculado sobre el lote nuevo. Es la misma clase de error silencioso de
unidades que el modelo ya cierra con la escala Frobenius, cerrada ahora para las
transformaciones. La matriz del usuario nunca se modifica, y
`reconstruct_entries(..., original=True)` invierte la cadena para leer
reconstrucciones en unidades originales (aproximación legible, no la media
condicional).

### Relaciones de conteo

`Relation(..., family="poisson")` cambia la pérdida de esa relación a la divergencia
KL generalizada, la likelihood de Poisson para conteos. El costo sigue siendo O(nnz
por rango): la masa total de la reconstrucción colapsa a tamaño rango por las sumas
de columna, y el término logarítmico solo se evalúa en las entradas almacenadas.

Dos consecuencias del cambio de pérdida: el backbone $S$ pasa a ser no negativo (la
reconstrucción debe ser positiva) y se actualiza de forma multiplicativa, y los datos
no se normalizan por Frobenius (el balance entre relaciones se controla con
`weights`). La pérdida reportada es la razón de desviación
KL(X||R) / KL(X||tasa constante): 0 es ajuste perfecto y 1 es el modelo nulo.

Las familias se pueden mezclar en un ajuste: una relación de conteos Poisson junto a
relaciones gaussianas (con sus máscaras y pesos por entrada), compartiendo factores.
El paso conjunto minimiza la suma de los majorizantes de ambas familias en forma
cerrada, y se reduce a la regla clásica sin Poisson y a la regla KL sin gaussianas.
El balance entre familias no tiene unidad natural, así que el peso relativo entre
una relación Poisson y una gaussiana se valida, no se hereda.

Con Poisson no están disponibles todavía `transform`, las máscaras de fila ni los
pesos por entrada sobre la relación Poisson misma. Pesos por fila o columna (por
ejemplo idf) se expresan escalando los datos: la KL ponderada por columna equivale a
la KL sobre la matriz con las columnas escaladas.

Medido en dos datasets (20 Newsgroups y Last.fm), modelar los conteos con su
likelihood no superó a transformarlos (log1p, idf) bajo la cuadrática; los números y
los protocolos están en `examples/newsgroups/README.md` y `examples/lastfm/README.md`.
La familia existe para cuando el modelo de conteo se necesite por sus propiedades
(tasas, interpretación generativa), no como palanca de accuracy.

### Entidades que se quedan sin datos

`fuse` avisa cuando una entidad no tiene ninguna observación en ninguna relación, y las
deja registradas en `modelo.empty_rows`. Sin ese aviso el problema es invisible: el
factor de esas entidades queda donde lo dejó la inicialización, despreciable frente a
cualquier fila ajustada, y con `nndsvd` idéntico para todas, así que `argmax` las manda
a todas al mismo grupo e infla ese grupo sin ninguna señal.

Pasa en la práctica cada vez que un filtro aguas arriba deja entidades sin filas.

```python
modelo = fuse(relaciones, ranks)      # UserWarning si las hay
modelo.empty_rows                    # {"usuario": array([12, 87, ...])}
```

### Anclar componentes a etiquetas conocidas

`Relation.rows` decide qué observaciones entran a la pérdida. Una entidad sin etiqueta
no aporta a esa relación, y sus componentes quedan libres. Pero cuando la etiqueta sí se
conoce, esa información dice algo más: **en qué componente debe cargar la entidad**, no
solo que la fila existe.

`supervision` usa eso. Es una matriz booleana de (entidades x componentes) por tipo, que
declara qué componentes puede activar cada entidad. La reconstrucción pasa a ser
$(G_i \circ L_i)\, S_{ij}\, G_j^T$, siguiendo TS-NMF (MacMillan y Wilson, 2017).

```python
permitido = np.ones((n_peliculas, 19), dtype=bool)
permitido[etiquetadas] = Y[etiquetadas] > 0      # solo sus generos

modelo = fuse(relaciones, ranks={"pelicula": 19, ...},
             supervision={"pelicula": permitido})
```

Con esto la componente $j$ **es** el género $j$, en vez de un grupo latente que el
backbone traduce. Medido en MovieLens, la fracción de películas reservadas cuyo grupo
coincide directamente con su género sube de 0.061 a 0.242.

Cuándo conviene depende de si la relación supervisada es la principal fuente de
estructura, y eso se midió en dos casos opuestos:

| caso | efecto del anclaje |
|---|---|
| MovieLens, tres relaciones | predecir -0.060 de AP, agrupar -0.036 de ARI |
| 20 Newsgroups, una relación | agrupar +0.042 de ARI con 10% supervisado, +0.108 con 50% |

En MovieLens, ratings y elenco traen estructura que no se alinea con el género, y forzar
las componentes al género la destruye. En texto, la relación documento-término es la
única y las categorías sí corresponden al vocabulario, así que el anclaje agrega
información. Ver `examples/newsgroups/README.md`.

### Los mecanismos son independientes

Se pueden combinar (salvo `rows` con `entry_weights` en la misma relación, todavía),
y responden a preguntas distintas:

| Mecanismo | Pregunta que responde |
|---|---|
| `Relation.rows` | ¿esta fila fue observada en esta relación? |
| `Relation.entry_weights` | ¿cuánto pesa esta entrada, y qué son los ceros? |
| `supervision` | ¿qué componentes puede activar esta entidad? |
| `empty_rows` | ¿quedó alguna entidad sin observación en ninguna parte? |

Una entidad puede tener datos en una relación y no en otra, tener etiqueta conocida o
no, y las dos cosas se declaran por separado. `supervision` con una fila de puros `True`
significa "no sé en qué componentes carga", que no es lo mismo que "no hay datos".

## Qué regulariza y qué no

**No se ofrece penalización L2 sobre $G$**, y es deliberado. Tras el solve cerrado de
$S$ vale la identidad $\langle G_t, N_t \rangle = \langle G_t, D_t \rangle$, así que una
penalización que solo alimenta el denominador no tiene punto fijo y lleva $G$ a cero.
Bajo el gauge de columnas, $\|G_t\|_F^2 = c_t$ es constante y la penalización es inerte.

Las palancas que sí cambian el resultado:

| Palanca | Qué hace |
|---|---|
| `ranks` | Controla cuánto puede memorizar el ajuste. Es la palanca principal. |
| `weights` | Cuánto pesa cada relación en la pérdida. |
| `masks` / `Relation.rows` | Qué filas de datos fueron observadas. |
| `supervision` | Qué componentes puede activar cada entidad. |
| `alpha_graph` + `graphs` | Suavizado sobre un grafo de vecindad por tipo. |

Cada `alpha` es adimensional y se calibra contra la energía del gradiente de datos de su
propio tipo, así que es comparable entre tipos y no depende de la escala de los datos.

## Los factores como clusters

La tri-factorización es un método de co-clustering: $G_i$ asigna filas a grupos, $G_j$
columnas a grupos, y $S_{ij}$ describe cómo se relacionan esos grupos. Leer el argmax de
una fila de $G_i$ como su grupo es una de las razones para usar el modelo, y es una
pregunta distinta de si predice bien un atributo retenido.

El gauge de columnas existe sobre todo por esto. El grupo de una fila es el argmax sobre
las columnas de $G$, así que si las columnas quedan con escalas arbitrarias ese argmax
lo decide la columna que más creció durante el ajuste. Fijando $\|G_t\|_F^2 = c_t$ las
columnas quedan comparables.

Medido en `examples/movielens/clustering.py`, agrupando películas en 19 grupos con
inicialización aleatoria y cuatro semillas, contra los géneros como referencia:

| método | NMI | ARI |
|---|---|---|
| k-means sobre las mismas matrices | 0.083 | 0.015 |
| `dfmf_sparse` | 0.579 | 0.464 |
| `fuse` sin gauge | 0.620 | 0.509 |
| `fuse` con gauge por columnas | **0.756** | **0.741** |

Con `nndsvd`, que ya parte de un punto bien escalado, la diferencia se reduce a 0.764
contra 0.745. El gauge sirve sobre todo para que el resultado no dependa de eso.

## Dos rutas de ajuste

`fuse` es la API actual (`fit` se mantiene como alias). `dfmf_sparse` es la
anterior, congelada: su comportamiento
numérico está fijado por `tests/test_golden.py` y no cambia.

Cuál conviene depende de para qué. Para **agrupar**, `fuse` con el gauge por columnas, por
un margen amplio. Para **predecir** un atributo retenido, las dos quedan parejas y la
anterior sale 0.010 de AP arriba en MovieLens. `fuse` agrega además máscaras de
observación, parada por tolerancia, y reanudación de ajustes largos.

## Guía de uso

`docs/guia-de-uso.md` recorre los flujos por tarea: preparar datos desde
DataFrames, elegir transformaciones e hiperparámetros, evaluar sin engañarse,
persistir y escalar a millones de entidades. Cada recomendación indica dónde está
medida, y sus fragmentos de código corren tal como aparecen.

## Referencia de API

| Función | Módulo | Qué hace |
|---|---|---|
| `fuse(relations, ranks, ...)` | `model` | Ajusta y devuelve un `FusionModel`. |
| `fuse(..., supervision={tipo: L})` | `model` | Ancla componentes a etiquetas conocidas (TS-NMF). |
| `fuse(..., n_runs=k)` | `model` | Reinicios aleatorios; devuelve la corrida de menor pérdida. |
| `Relation(..., family="poisson")` | `model` | Relación de conteos por KL generalizada. |
| `Relation(..., preprocess="log1p")` | `model` | Transformación de valores que viaja con el modelo. |
| `FusionModel.transform(relations, target)` | `model` | Proyecta entidades nuevas, sin refit. |
| `FusionModel.predict_proba(target, views)` | `model` | Distribución sobre un tipo, combinando vistas. |
| `FusionModel.loss(relations)` | `model` | Pérdida por la identidad de traza. |
| `FusionModel.reconstruct_entries(name, rows, cols)` | `model` | Reconstrucción en coordenadas, unidades originales. |
| `FusionModel.save/load/resume` | `model` | Persistencia y reanudación de un ajuste. |
| `holdout_entries(matrix, fraction)` | `utils` | Retiene entradas para validación. |
| `fusion_diagram(relations_or_model)` | `diagram` | Esquema de la fusión, orientado a publicación. |
| `sddmm(pattern, A, B)` | `ops` | Producto de rango bajo muestreado en un patrón disperso. |
| `dfmf_sparse(R, ranks, ...)` | `core` | Ruta anterior, congelada. |
| `reconstruction_error(R, G, S)` | `core` | Suma de errores Frobenius relativos. |
| `fold_in_entities(R_new, G, S, target_type)` | `core` | Fold-in de la ruta anterior. |
| `predict_attribute(known, G, S, ...)` | `core` | Predicción de la ruta anterior. |
| `merge_relations`, `normalize_relations` | `utils` | Preparación de relaciones. |
| `init_random`, `init_nndsvd` | `init` | Inicialización de factores. |

## Ejemplo completo

`examples/movielens/` contiene un caso end to end con ground truth: predecir los géneros
de películas que el modelo nunca vio, a partir de quién las calificó y quién actúa en
ellas. Incluye la comparación contra baselines, la comparación entre rutas de ajuste,
la comparación contra la implementación de referencia y un benchmark de escala.
Ver `examples/movielens/README.md`.

Los datos vienen incluidos con `scikit-fusion` (6.5 MB, sin descarga). Con
`scikit-fusion` instalado se encuentran solos; si no, se apunta `MOVIELENS_DIR` al
directorio del dataset.

`examples/lastfm/` agrega el caso de familias mezcladas sobre conteos reales
(Last.fm HetRec 2011, descarga de 2.5 MB, uso no comercial), y
`examples/newsgroups/` el de semi-supervisión y likelihoods sobre texto.

## Tests

```bash
uv run pytest
```

115 tests. Cubren la identidad de traza contra el cálculo denso, la familia Poisson
(descenso monótono de la desviación, recuperación de bloques plantados, fusión
mezclada con gaussianas), que el preprocesamiento declarado equivale al manual y
reusa el idf del entrenamiento en datos nuevos, la invariancia de la
calibración a la escala de los datos, que el contenido de una fila enmascarada o de
una entrada con peso cero no afecte el ajuste, que el fold-in quede dentro del 0.5%
del óptimo de `scipy.optimize.nnls`, la conservación de masa del término de grafo, el
SDDMM contra el producto denso, la reducción exacta de los pesos uniformes a la ruta
clásica, el roundtrip de guardado con supervisión, máscaras y grafos, las máscaras de
fila en `transform`, y que la ruta congelada siga dando los mismos números.

La comparación contra `scikit-fusion` está en `tests/test_compare_skfusion.py`. Como
test corre una versión reducida y se salta si `scikit-fusion` no está instalado (no
está en PyPI; se instala desde un checkout). Como script imprime el reporte completo.

## Oportunidades de mejora

`docs/oportunidades.md` documenta lo que falta y por qué, con lo que se sabe medido de
cada cosa.

## Licencia

MIT. Ver `LICENSE`.

## Referencias

- Zitnik, M. y Zupan, B. (2015). Data fusion by matrix factorization. IEEE Transactions on Pattern Analysis and Machine Intelligence, 37(1), 41 a 53.
- MacMillan, K. y Wilson, J. D. (2017). Topic supervised non-negative matrix factorization. arXiv:1706.05084. (Datos tomados del preprint; verificar si hubo publicación posterior en revista antes de citarlo en un paper.)
- Boutsidis, C. y Gallopoulos, E. (2008). SVD based initialization: A head start for nonnegative matrix factorization. Pattern Recognition, 41(4), 1350 a 1362.
