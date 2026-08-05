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

### Máscaras por entrada, no por fila

Hoy la granularidad es la fila completa. Falta el caso de una fila parcialmente
observada: se conocen algunas entradas y el resto es desconocido, no cero.

La formulación es la de retroalimentación implícita (Hu, Koren y Volinsky): cada entrada
tiene un peso $w_{ab}$, con $w_0$ de fondo para las no observadas. La actualización
correcta separa el término de fondo, que sigue siendo un producto denso barato, del
término de corrección sobre las entradas observadas, que es un SDDMM (producto
denso muestreado en el patrón disperso).

Medido: el análisis encontró que una versión ingenua de esta actualización parte el
signo del producto y pierde el piso positivo del denominador, divergiendo a NaN cuando
$w_0 = 0$. La fórmula correcta quedó registrada.

Costo: `scipy` no tiene primitiva de SDDMM, así que hay que escribirla. La estimación
del análisis fue +50% de tiempo por iteración sobre la relación grande. Se difirió por
eso, no porque esté mal.

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

### Validación interna con entradas retenidas

Hoy la única forma de validar es reservar entidades completas y proyectarlas. No hay
manera de retener *entradas sueltas* dentro de una relación para medir el error de
reconstrucción sobre ellas durante el ajuste.

Con máscaras por entrada, esto sale casi gratis: se retiene un porcentaje de las
entradas observadas, se ajusta sin ellas y se evalúa ahí. Habilita parada temprana por
error de validación en vez de por tolerancia sobre la pérdida de entrenamiento, que es
lo que hacen las librerías de factorización maduras.

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

`fit` queda 0.010 de AP por debajo de `dfmf_sparse` en MovieLens, con comparación
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
La conversión debe hacerse aguas arriba: convertir dentro de `fit` duplicaría el nnz en
memoria, que es justamente lo que se quiere evitar.

## Selección de rango

El rango es la palanca principal de regularización y hoy se elige por barrido manual.
Con validación por entradas retenidas se podría automatizar, o al menos reportar una
curva de error de validación contra rango en una sola corrida.

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
