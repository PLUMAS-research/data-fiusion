# Oportunidades de mejora

Cada entrada dice qué es, qué se sabe hoy y qué costaría. Se distingue lo que está
**medido** de lo que es **conjetura**, porque varias de estas ideas suenan razonables y
ya hay al menos una que se probó y no funcionó.

## Semi-supervisión

El régimen semi-supervisado es la línea con más recorrido, y está a medio camino.

### Lo que ya está

**Máscaras de observación por fila** (`Relation.rows`). Sin ellas, cada cero de una
relación se aprende como una observación real: la entidad no tiene relación con nada.
Con ellas, se ajusta con todas las entidades presentes y solo las etiquetadas aportan
su etiqueta.

Medido: ocultar poniendo ceros da AP 0.468 sobre las filas ocultas contra 0.982 sobre
las etiquetadas. Con máscara, el contenido de una fila oculta no afecta el ajuste
(verificado a $10^{-12}$ en `tests/test_fit.py`, y a $0$ exacto en
`examples/semi_supervised.py`).

**Lo que las máscaras no hacen, medido**: por sí solas no mejoran el acierto. En una
instancia sintética donde la clase no es recuperable desde la relación observada, las
tres variantes (relleno cero, relleno uniforme, máscara) quedan cerca del azar, y el
relleno uniforme a veces gana porque actúa como regularizador crudo. La máscara corrige
la semántica del modelo, no crea señal donde no la hay. El aporte aparece cuando la
señal existe: en MovieLens el protocolo enmascarado llega a AP 0.716 contra 0.550 del
baseline marginal.

Eso deja el trabajo real de semi-supervisión en los tres puntos que siguen, no en la
máscara por fila, que ya está.

### Anclaje de componentes (implementado)

`supervision` restringe qué componentes latentes puede activar cada entidad, siguiendo
TS-NMF (MacMillan y Wilson, 2017): la reconstrucción pasa a ser $(G_i \circ L_i) S G_j^T$.
Es distinto de la máscara por fila, que solo decide qué observaciones entran a la
pérdida: acá se usa la información de *columna* que la etiqueta aporta.

Medido en MovieLens, anclando 19 componentes a los 19 géneros: la fracción de películas
reservadas cuyo grupo coincide con su género sube de 0.061 a 0.242 (10.9 SE), mientras
predecir baja 0.060 de AP y agrupar 0.036 de ARI, ambas significativas.

En texto pasa lo contrario, y eso aclara cuándo conviene. Sobre 20 Newsgroups, con una
sola relación documento-término, el anclaje **mejora** de forma monótona: +0.042 de ARI
con 10% de documentos supervisados (3.8 SE) y +0.108 con 50% (8.5 SE), sobre los
documentos no supervisados. Es el resultado que reporta el paper.

La diferencia entre los dos casos es la regla práctica: `supervision` conviene cuando la
relación supervisada es la principal fuente de estructura, y no cuando compite con otras
que aportan estructura distinta. En MovieLens, ratings y elenco traen señal que no se
alinea con el género, y forzar las componentes al género la destruye.

Como referencia, sin supervisar la tri-factorización da ARI 0.185 en 20 Newsgroups y el
NMF de `sklearn` da 0.184, así que agrupar los términos no cuesta calidad ahí.

Queda pendiente barrer el peso de la relación de etiquetas con anclaje activado, porque
el óptimo puede no ser el mismo que sin él.

### Máscaras por entrada (implementado)

`Relation(entry_weights=..., background=...)` da la granularidad por entrada: cada
entrada almacenada lleva un peso $w_{ab}$ y las no almacenadas un peso de fondo $w_0$
(retroalimentación implícita, Hu, Koren y Volinsky). Peso cero oculta la entrada del
ajuste por todas las vías (pérdida, escala e inicialización, verificado a $10^{-12}$
en `tests/test_pesos.py`). La primitiva SDDMM que esto exige está en `ops.py`.

Dos decisiones de la implementación que conviene conocer:

- La actualización de $G$ usa la fórmula registrada por el análisis (separar el signo
  de la matriz cuadrática, no del producto), que mantiene el piso positivo del
  denominador; la versión ingenua diverge a NaN con $w_0 = 0$ y hay un test que cubre
  ese régimen.
- $S$ pierde el solve cerrado. Se actualiza con un paso precondicionado amortiguado
  cuyo punto fijo satisface las ecuaciones normales ponderadas y que se reduce
  exactamente al solve cerrado con pesos uniformes. Los pesos se normalizan al máximo
  (la magnitud se pliega en el peso de la relación) para que la iteración quede
  contraída.

La reconstrucción solo se evalúa donde $w_{ab} \neq w_0$, así que el sobrecosto
depende del régimen. Medido sobre una relación de 200k por 5k con 5M de entradas
(8 iteraciones, rangos 20 y 15): ruta clásica 5.1 s; entradas retenidas al 10%,
1.9x; régimen implícito (delta denso en el patrón), 5.2x, dominado por tres pasadas
de recolección por iteración (actualización de S, de G y pérdida) a ~0.5 Gflop/s
contra productos BLAS.

Pendiente de esta línea: soporte en `transform`, combinar `entry_weights` con `rows`
en la misma relación, cachear entre pasadas la reconstrucción sobre el soporte de
delta, y un kernel compilado para el SDDMM si el régimen implícito se vuelve el
caso dominante.

### Consistencia entre el ajuste y el fold-in (`alpha_consistency`)

La mitad de la brecha entre el rendimiento in-sample y el held-out no es sobreajuste
sino desajuste de estimador: durante el ajuste, $G_t$ se obtiene minimizando la pérdida
completa, mientras que en el fold-in se obtiene resolviendo desde las vistas. Son dos
estimadores distintos para la misma cantidad.

Medido: proyectar las mismas entidades del entrenamiento desde sus vistas da 0.859,
contra 0.981 del ajuste directo. Al subir el término de consistencia de 0 a 10, la
brecha baja de 0.258 a 0.096.

El término penaliza la distancia entre $G_t$ y lo que el fold-in devolvería:
$E_t = [B_t Q_t^{-1}]_+$, con $Q_t$ y $B_t$ ya disponibles dentro de la pasada por
bloques, así que no agrega ninguna pasada sparse.

Conjetura, no medida: debería ayudar de verdad cuando la suma de $n_{dst}$ es mucho
menor que $n_t$, es decir cuando hay muchas entidades descritas por pocos atributos.
En MovieLens esa condición no se cumple y el held-out quedó plano.

### Validación interna con entradas retenidas (implementado)

`holdout_entries(matrix, fraction)` retiene un subconjunto aleatorio de las entradas
almacenadas (peso cero), y `FusionModel.reconstruct_entries(nombre, filas, columnas)`
las puntúa tras el ajuste, en unidades originales y en O(k por rango). Con eso el
error de validación por entradas queda disponible sin reservar entidades completas.

Pendiente: parada temprana por ese error durante el ajuste. El `callback` actual
recibe (iteración, pérdida, G) pero no S, así que no puede puntuar entradas; hay que
pasarle S o incorporar la validación al loop.

## Rendimiento del bucle de ajuste

**Medido**: alrededor del 70% del tiempo de una iteración se va en productos
`scipy.sparse`, que corren en un solo hilo. Los productos van a unos 1.7 GB/s contra
33.8 GB/s de un `memcpy`, porque están limitados por acceso aleatorio a memoria y no por
cómputo. El techo real de paralelizarlos que midió el análisis es 3.1x y 2.1x según la
operación, no el 15x que sugerían las estimaciones ingenuas.

**Pasadas redundantes**: la estructura actual hace tres pasadas sparse por matriz donde
bastan dos, ordenando el cálculo para que `middle` salga del primer producto en vez de
recalcularse.

**Actualización en sitio por bloques**: acumular numerador y denominador completos por
tipo cuesta tres veces el tamaño de $G$. Calcular por bloques de filas y escribir en
sitio baja el pico. El análisis midió 11.28 GB a 6.71 GB en un caso de 5 millones de
filas. La complicación es que solo es equivalente cuando el tipo aparece en un solo lado
de sus relaciones, así que hay que detectar el caso y avisar cuando no se cumple.

## La diferencia de 1.6% entre las dos rutas

`fuse` queda 0.010 de AP por debajo de `dfmf_sparse` en MovieLens, con comparación
pareada sobre cuatro semillas. Se descartaron por ablación: el gauge de columnas, el
paso `eta`, `lambda_S`, el fold-in no negativo, el rango de la grilla de pesos (barrida
de 0.01 a 1000), el solve final de $S$ y la semántica del peso.

**Conjetura**: el orden en que se concatenan los bloques dentro de la inicialización
NNDSVD, que cambia la SVD y por tanto el punto de partida. Es una propiedad de cómo se
acomodan los datos, no del método. Verificarlo es barato: fijar el orden de los bloques
y repetir la comparación.

## Ingesta desde DataFrames

`relation_from_frame`: construir una `Relation` desde un DataFrame de coordenadas
(entidad, atributo, valor) usando `pd.Categorical` para el mapeo a índices, sin pasar
nunca por una grilla densa.

El wrapper anterior (`base.py`) hacía la alineación de índices con `reindex` sobre un
DataFrame denso de $n_i \times n_j$, que es la razón por la que no sirve a escala. Sigue
en el repo por compatibilidad, pero no debería usarse para datos grandes.

## Precisión de los factores

`float32` bajaría el pico de memoria de los factores a la mitad. El análisis midió que
con factores en `float32` el producto que calcula `middle` da un error relativo de
$4 \times 10^{-5}$, suficiente para que la pérdida salga negativa en régimen convergido
y para dejar una tolerancia de $10^{-4}$ en el piso del ruido.

Viable, pero exige acumular la pérdida en `float64` y subir el piso de la tolerancia.
La conversión debe hacerse aguas arriba: convertir dentro de `fuse` duplicaría el nnz en
memoria, que es justamente lo que se quiere evitar.

## Selección de rango

El rango es la palanca principal de regularización y hoy se elige por barrido manual.
Con `holdout_entries` más `reconstruct_entries` la curva de error de validación contra
rango ya es medible por corrida; falta el helper que haga el barrido y reporte la
curva, y la parada temprana anotada arriba.

## Likelihoods de conteo (Poisson, negative binomial)

Las relaciones típicas del dominio objetivo son conteos sobredispersos, y la pérdida
cuadrática es la likelihood gaussiana. El orden de trabajo, con criterio de éxito
declarado antes de cada paso:

1. **Baseline medido** (`examples/newsgroups/conteos.py`): sobre 20 Newsgroups con
   conteos crudos (índice de dispersión 283), la cuadrática da ARI 0.026;
   estabilizar varianza sube a 0.103 con sqrt y 0.114 con log1p; TF-IDF llega a
   0.185. La referencia a ganar para una likelihood de conteo es TF-IDF, no los
   conteos crudos, y TF-IDF no es estabilización de varianza sino reponderado por
   especificidad: un modelo Poisson que no incorpore ese reponderado parte en
   desventaja. Las transformaciones ganadoras quedaron después declarables en el
   modelo (`Relation(preprocess=...)`), con el estado (idf) persistido y reaplicado
   a datos no vistos por `transform` y `loss`.
2. **Poisson/KL por relación (implementado)**: `Relation(family="poisson")` despacha
   a un loop KL propio (`poisson.py`), con actualizaciones multiplicativas cuyo
   numerador y denominador se acumulan entre las relaciones de cada tipo (el paso
   conjunto desciende la pérdida conjunta), $S$ no negativo, el gauge compensado en
   $S$ y la pérdida como razón de desviación contra el modelo nulo de tasa
   constante. El experimento contra la referencia cuadrática está en
   `examples/newsgroups/poisson_vs_cuadratica.py`.
   **Medido, y el criterio no se cumplió**: Poisson sobre conteos crudos da ARI
   0.113 (el nivel de log1p bajo la cuadrática, 0.114); con columnas escaladas por
   idf sube a 0.151 y empata estadísticamente con la referencia (-1.3 SE); sobre TF-IDF
   normalizado degenera. La likelihood logra lo mismo que la transformación, el
   reponderado por idf es lo que mejora bajo ambas pérdidas, y no
   hay razón medida para preferir Poisson en agrupamiento de texto. El paso
   siguiente (familias mezcladas) se midió aparte y tampoco superó a la
   cuadrática; ver abajo.
   **Familias mezcladas: implementado y medido.** Un ajuste puede combinar relaciones
   Poisson y gaussianas compartiendo factores (la línea de collective matrix
   factorization de Singh y Gordon; verificar la referencia antes de citarla). El
   paso conjunto minimiza la suma de los majorizantes de ambas familias en forma
   cerrada (raíz positiva de una cuadrática por entrada) y se reduce exactamente a
   cada regla pura en los casos extremos. Medido en Last.fm (`examples/lastfm/`),
   con criterio declarado (ganar por 2 SE en AP de tags retenidos): el brazo
   mezclado (Poisson crudo más etiquetas gaussianas) pierde contra la monofamilia
   con log1p por $-0.050 \pm 0.017$ ($-2.9$ SE), con ambos brazos sobre el
   baseline de popularidad. La corrida encontró un defecto real, corregido: sin
   calibrar el gradiente KL por la desviación nula de su relación, el término de
   conteos domina numéricamente a cualquier gaussiana y el peso entre familias no tiene efecto.
   Conclusión de la línea de conteos completa: en dos datasets y tres protocolos,
   modelar conteos con su likelihood nunca superó a transformarlos bajo la
   cuadrática. La infraestructura queda (familia por relación, calibración entre
   familias) para cuando un caso la exija por razones de modelo (tasas,
   interpretación generativa), no de accuracy.
   Pendiente de esta línea: `transform` bajo KL, y máscaras o pesos por entrada
   sobre la relación Poisson misma (el peso de fondo $w_0$ daría el régimen
   implícito también aquí).
3. **Negative binomial solo con evidencia**: medir la sobredispersión condicional
   bajo el modelo Poisson ya ajustado (varianza contra media) y agregar el parámetro
   de dispersión por relación solo si Poisson falla de forma visible. Con Poisson
   empatando o perdiendo contra la cuadrática transformada en todos los protocolos
   medidos, esta línea queda en baja prioridad.

## Distribuciones por entidad a través de cadenas

Para leer la distribución de un atributo a nivel del grano más fino (por ejemplo,
el modo de cada viaje cuando la relación etiquetada vive en otro tipo), hace falta
reconstruir una relación que no existe en el grafo, encadenando backbones a través
de los tipos compartidos: $G_{viaje} S_{v,a} (G_a^T G_a) S_{a,m} G_m^T$, con todos
los intermedios de tamaño rango. El wrapper antiguo lo hacía (`relation_profiles`
en `base.py`, densificado); falta la versión O(n por rango) en la API actual. Es
la pieza que habilita rebanar una actualización por hora o por segmento al agregar
distribuciones por entidad fina, en vez de multiplicar el tipo objetivo. El
protocolo de reconstrucción-como-actualización que la motiva vive en el análisis
que consume la librería, no en este repo.

## Criterio de parada por fila en el refinado no negativo

`nonneg_refine` detiene la iteración por el cambio máximo del bloque completo, así que
las filas fáciles quedan acopladas a la fila que converge más lento. En `transform` con
máscaras cada patrón de observación se refina por separado, y el número de iteraciones
de un grupo de filas puede depender de junto a quiénes se refinó. Con un criterio por
fila (congelar las filas ya convergidas) el resultado sería independiente de cómo se
agrupan. Costo: tocar `core.py`, que hoy comparte esta función entre las dos rutas.

## Integración continua

Un workflow de GitHub Actions que corra `pytest` en cada push. La suite corre sin
descargas ni credenciales, y la comparación con scikit-fusion se salta sola si no está
instalado, así que no hay bloqueo técnico. Pendiente por decisión del mantenedor.

## Lo que se probó y no funcionó

Registrado para no volver a intentarlo sin una razón nueva.

- **Penalización L2 sobre $G$**: no tiene punto fijo sin gauge, y es inerte con gauge.
- **Ortogonalidad de columnas**: tiene punto fijo, pero es cuártica y no convexa, así
  que rompe la garantía que permite el paso completo y obligaría a duplicar iteraciones.
- **Filas cuasi estocásticas**: incompatible con el gauge por columnas salvo que el
  número de entidades sea menor que el cuadrado del rango.
- **Laplaciano $I - D^{-1/2} W D^{-1/2}$**: no conserva masa, encoge los nodos de grado
  bajo un 17.5% medido sobre una grilla con borde. Se usa
  $\operatorname{diag}(W_{sym} \mathbf{1}) - W_{sym}$, que sí lo conserva.
- **Parametrización por codificador** ($G_i = Z W$ con $Z$ las vistas concatenadas):
  elimina la brecha de estimador por construcción, pero empeoró en MovieLens (0.601
  contra 0.718) porque $Z$ tenía más columnas que filas.
- **`LinearOperator` para abaratar la inicialización**: no redujo la memoria, porque el
  consumo estaba en el subespacio de Krylov de ARPACK y no en la concatenación. La
  solución que sí funcionó fue resolver por el lado chico.
- **KL sobre TF-IDF normalizado por fila**: la KL modela masa y la normalización L2
  la elimina; el ajuste degenera en 2 iteraciones con ARI 0.000. Los pesos por
  columna (idf) sí son compatibles con KL, escalando la matriz.
