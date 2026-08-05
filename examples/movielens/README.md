# Caso de uso: MovieLens

Prueba de la implementación sobre un problema chico con ground truth completo. Un caso
de uso real mezcla la evaluación del algoritmo con la complejidad del dominio, así que
un resultado raro es ambiguo entre error de código y error de datos. Acá la etiqueta
existe para todas las entidades y el experimento corre en minutos.

## Datos

MovieLens en la copia que trae `scikit-fusion`, 6.5 MB en disco, sin descarga. Con
`scikit-fusion` instalado se encuentra solo; si no, se apunta `MOVIELENS_DIR` a un
directorio con `ratings.csv.gz`, `movies.csv.gz` y `actors.csv.gz`.

Universo tras filtrar películas con al menos 5 ratings, género y elenco:

| Relación | Shape | nnz | Densidad | Densificada |
|---|---|---|---|---|
| `(pelicula, usuario)` | 3310 x 706 | 90 587 | 3.88% | 18.7 MB |
| `(pelicula, genero)` | 3310 x 19 | 8 212 | 13.06% | 0.5 MB |
| `(pelicula, actor)` | 3310 x 5 842 | 48 539 | 0.25% | 154.7 MB |

Las tres relaciones tienen `pelicula` como tipo fuente. Esa orientación permite
proyectar películas nuevas, porque el fold-in exige que el tipo objetivo sea la fuente
de todas las relaciones nuevas.

## Tarea

Predecir los géneros de una película que el modelo nunca vio, a partir de quién la
calificó y quién actúa en ella. El 20% de las películas va a validación y otro 20% a
test; ninguna participa del ajuste en ninguna relación, así que su género no puede
filtrarse al modelo.

El género es multi-etiqueta (2.44 géneros por película), así que se miden tres cosas:
`hit@1` (el género de mayor score está entre los verdaderos), R-precision (aciertos
entre los k primeros, con k = géneros verdaderos) y AP media.

## Scripts

| Script | Qué hace | Tiempo |
|---|---|---|
| `datos.py` | Construye las relaciones y los tres splits. Imprime un resumen. | segundos |
| `clustering.py` | El lado de co-clustering: qué agrupan los factores, contra k-means. | ~15 min |
| `curvas.py` | Barre rango y peso, y genera las figuras de cómo elegirlos. | ~10 min |
| `supervision.py` | Anclar componentes a los géneros conocidos, al estilo TS-NMF. | ~10 min |
| `prediccion.py` | Baselines, barrido en validación y evaluación en test, por la ruta anterior. | ~3 min |
| `prediccion_fit.py` | La misma tarea con `fit`, comparando máscaras contra fold-in. | ~20 min |
| `comparacion.py` | Las dos rutas cara a cara, misma grilla y mismas semillas, con test pareado. | ~30 min |
| `equivalencia.py` | Costo del init, y `dfmf_sparse` contra scikit-fusion con el mismo evaluador. | ~5 min |
| `escala.py` | Tiempo y memoria contra scikit-fusion, variando filas y columnas. | ~5 min |

`escala.py` y `equivalencia.py` corren cada configuración en su propio proceso, con
estimación previa de huella y un guardián que la termina si `MemAvailable` baja del
umbral (`UMBRAL_MEMORIA_GB`, 6 GB por defecto). Los resultados quedan en `output/`,
configurable con `DIR_SALIDA`.

## Resultados

Predicción de género, hiperparámetros elegidos en validación por configuración:

| | hit@1 | R-precision | AP media |
|---|---|---|---|
| marginal | 0.462 | 0.434 | 0.550 |
| kNN solo ratings | 0.746 | 0.615 | 0.732 |
| kNN solo elenco | 0.630 | 0.532 | 0.647 |
| kNN fusión | 0.764 | 0.630 | 0.749 |
| solo ratings | 0.692 | 0.569 | 0.678 |
| solo elenco | 0.628 | 0.513 | 0.621 |
| fusión | 0.751 | 0.606 | 0.718 |

Fusionar aporta sobre cada fuente por separado: +0.040 de AP sobre la mejor de las dos
y +0.168 sobre el marginal. La factorización queda 0.031 por debajo de un kNN por
coseno sobre las mismas matrices.

Las dos rutas de ajuste, comparación pareada sobre cuatro semillas:

| protocolo | AP media | contra la ruta anterior |
|---|---|---|
| anterior (`dfmf_sparse`) | 0.732 $\pm$ 0.003 | |
| `fit` + `transform` | 0.722 $\pm$ 0.006 | -0.010 $\pm$ 0.004 |
| `fit` con máscaras | 0.716 $\pm$ 0.005 | -0.017 $\pm$ 0.002 |

La ruta nueva no generaliza mejor **para predecir**. Para agrupar sí, y por un margen
grande: ver la sección de co-clustering. Las causas de la diferencia en predicción que
se descartaron por ablación están en `docs/oportunidades.md`.

### Co-clustering

La tri-factorización agrupa filas y columnas a la vez, y esa lectura es una razón
principal para usarla. El cluster de una película es el argmax de su fila en
`G[pelicula]`. Agrupación en 19 grupos, init aleatorio, cuatro semillas:

| método | NMI | ARI | coherencia |
|---|---|---|---|
| azar | 0.017 | -0.000 | 1.000 |
| k-means sobre ratings | 0.083 | 0.015 | 0.970 |
| k-means sobre elenco | 0.030 | 0.002 | 1.000 |
| `fit` sin ver los géneros | 0.140 | 0.051 | 1.190 |
| `dfmf_sparse` | 0.579 | 0.464 | 1.870 |
| `fit` sin gauge | 0.620 | 0.509 | 1.934 |
| `fit` con gauge por columnas | **0.756** | **0.741** | 1.992 |

Coherencia es cuántas veces más probable es que dos películas del mismo grupo compartan
un género, comparado con dos al azar. No necesita elegir un género dominante, así que no
arrastra esa convención.

Dos lecturas:

**El gauge por columnas es lo que hace los factores legibles como clusters.** Contra la
ruta anterior, pareado por semilla, da +0.177 de NMI (15.7 SE) y +0.277 de ARI (14.7
SE); sin gauge la ventaja cae a +0.041. Tiene una explicación directa: el cluster de una
fila es el argmax sobre las columnas de `G`, y si las columnas quedan con escalas
arbitrarias, ese argmax lo decide la columna que más creció. El gauge las deja
comparables. Con `nndsvd`, que ya parte de un punto bien escalado, la diferencia se
reduce a 0.764 contra 0.745: el gauge sirve sobre todo para no depender de eso.

**El grupo se puede nombrar leyendo el backbone.** En el 79% de los grupos, el género
con mayor peso en `S["etiquetas"] @ G["genero"].T` coincide con el género más frecuente
entre sus miembros. Los que fallan cargan hacia géneros raros como IMAX o Film-Noir.

Advertencia para no sobreleer la primera tabla: la relación de géneros entra al ajuste,
así que recuperar géneros es en parte tautológico. La fila que mide agrupación no
supervisada es `fit` sin ver los géneros, con NMI 0.140. Es modesto, pero sigue estando
por encima de k-means sobre las mismas matrices (0.083).

Mismo modelo, dos motores, con el mismo evaluador aguas abajo:

| motor | init | AP | segundos | memoria |
|---|---|---|---|---|
| `dfmf_sparse` | random | 0.476 | 3.2 | 35 MB |
| scikit-fusion | random | 0.442 | 7.8 | 249 MB |
| scikit-fusion | random_vcol | 0.519 | 8.9 | |
| `dfmf_sparse` | nndsvd | 0.730 | 3.9 | 35 MB |

Con la misma inicialización los dos coinciden, y la diferencia grande que aparece si se
comparan inits distintos es del inicializador, no del motor.

Costo al crecer las relaciones, 20 iteraciones:

| Configuración | `dfmf_sparse` | scikit-fusion | Razón |
|---|---|---|---|
| 3310 x 6 567 | 0.5 s, 33 MB | 1.4 s, 319 MB | 2.7x tiempo, 9.8x memoria |
| 3310 x 29 776 | 1.1 s, 123 MB | 5.8 s, 1 247 MB | 5.2x tiempo, 10.2x memoria |
| 3310 x 108 536 | 5.8 s, 386 MB | 23.5 s, 4 031 MB | 4.0x tiempo, 10.4x memoria |

## Glosario de las métricas

### Para predecir un atributo

**AP media** (*average precision*). El modelo ordena los 19 géneros por score y AP mira
en qué posiciones quedaron los verdaderos. Para una película cuyos géneros reales son
Comedy y Drama, con Comedy primero y Drama tercero en el ranking:

| posición | acertó | precisión hasta ahí |
|---|---|---|
| 1 (Comedy) | sí | 1 de 1 = 1.00 |
| 2 | no | |
| 3 (Drama) | sí | 2 de 3 = 0.67 |

AP de esa película = (1.00 + 0.67) / 2 = 0.83. Se promedia sobre todas. Va de 0 a 1 y
premia poner los verdaderos arriba. Referencia: predecir siempre los géneros más
frecuentes, sin mirar la película, da 0.550.

**hit@1.** Fracción de películas donde el género de mayor score es uno de los
verdaderos. Fácil de leer, pero ignora el resto del ranking.

**R-precision.** Si la película tiene k géneros verdaderos, cuántos de los k primeros
del ranking son correctos, dividido por k.

### Para agrupar

Las dos comparan la partición que produjo el modelo contra una de referencia, acá los
géneros. Los nombres de los grupos no importan, solo qué queda junto con qué.

**NMI**, *normalized mutual information*, información mutua normalizada. Si te digo en
qué grupo del modelo cayó una película, ¿cuánta incertidumbre te quita sobre su género?
Se calcula con entropías y se divide por la entropía promedio de las dos particiones,
para dejarla entre 0 (el grupo no dice nada del género) y 1 (lo determina).

**ARI**, *adjusted rand index*, índice de Rand ajustado. Mira todos los pares de
películas y cuenta en cuántos el modelo y la referencia coinciden, ya sea poniéndolas
juntas o separándolas. El "ajustado" descuenta los aciertos que saldrían por azar, así
que una partición al azar da 0 en vez de un número positivo. Puede ser negativa.

| la partición del modelo es | NMI | ARI |
|---|---|---|
| idéntica a la referencia, con otros nombres | 1.000 | 1.000 |
| la referencia, pero cada grupo partido en dos | 0.775 | 0.429 |
| independiente de la referencia (azar) | 0.000 | -0.333 |

La fila del medio sale de las definiciones. Partir un grupo en dos no destruye
información, porque sabiendo el subgrupo se sigue sabiendo el grupo original, así que la
información mutua casi no baja. Pero sí rompe pares que antes estaban juntos, y eso es
lo que ARI cuenta. Por eso al barrer el rango decide ARI: pedir más grupos infla el NMI
sin que haya más estructura.

**Coherencia.** Cuántas veces más probable es que dos películas del mismo grupo
compartan un género, comparado con dos al azar. 1.0 es el azar. No necesita elegir una
etiqueta verdadera por película.

### Para leer las comparaciones

**Error de reconstrucción relativo.** `||M - G S G^T||_F / ||M||_F`. Vale 0 si la
reconstrucción es exacta y 1 si es tan mala como predecir cero en todas partes. Es lo
que el ajuste minimiza, así que no sirve como criterio de calidad: siempre baja al subir
el rango.

**SE** (error estándar). Cuánto varía el promedio entre repeticiones con distinta
semilla. Una diferencia menor a 2 SE no se declara como diferencia.

## Advertencias de método que salieron de acá

**La evaluación in-sample no sirve para validar.** Predecir el género de las películas
que sí entraron al ajuste da AP 0.98, y 1.00 exacto cuando el rango del tipo `genero`
iguala el número de clases. Con rango suficiente, `G[pelicula]` puede satisfacer la
relación de etiquetas y las de las vistas por separado, sin que nada obligue a que la
etiqueta sea derivable de las vistas. Validar siempre con entidades fuera del ajuste.

**La regularización L2 sobre `G` no ayuda.** En validación, `lambda_G` de 0.0 da AP
0.722 y 1.0 cae a 0.541, el nivel del baseline marginal. La palanca que sí controla el
sobreajuste es el rango del tipo fuente.

**Un óptimo en el borde de la grilla no es un resultado.** El primer barrido eligió los
extremos inferiores de ambas grillas; con la grilla centrada el óptimo quedó adentro y
la diferencia medida se redujo a la mitad.

**Cuidado al medir memoria.** `ru_maxrss` es un máximo histórico del proceso, así que
restarle el RSS actual atribuye a la etapa medida todo lo reservado antes. Medir como
`pico_despues - pico_antes` y generar los datos de prueba en otro proceso.
