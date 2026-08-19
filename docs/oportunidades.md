# Oportunidades de mejora

Lo que falta, en orden de utilidad, y lo que ya se probó sin resultado. Cada entrada
distingue lo **medido** de la **conjetura**. Lo que sí está implementado se documenta
donde se usa: `README.md` para la API, `docs/guia-de-uso.md` para los flujos, los
`README.md` de `examples/` para las cifras y sus protocolos, y los docstrings de
`_accumulate_weighted`, `solve_backbone` y del módulo `poisson.py` para las
derivaciones de las actualizaciones.

## Semi-supervisión y pesos

**Parada temprana por error de validación.** `holdout_entries` retiene entradas y
`FusionModel.reconstruct_entries` las puntúa tras el ajuste, así que la curva de
validación ya es medible por corrida. Falta engancharla al bucle: el `callback` recibe
`(iteración, pérdida, G)` y no $S$, así que no puede puntuar entradas. Hay que pasarle
$S$ o incorporar la validación al loop.

**Selección de rango.** El rango es la palanca principal de regularización y hoy se
elige por barrido manual. Con la parada temprana anterior, falta el helper que corra el
barrido y reporte la curva de error de validación contra rango.

**`rows` junto a `entry_weights` en la misma relación.** Hoy declarar las dos levanta
`ValueError`. Son granularidades distintas de la misma pregunta (qué observaciones
entran a la pérdida) y no hay razón de fondo para que se excluyan.

**`transform` con pesos por entrada.** La proyección de entidades nuevas no soporta
todavía relaciones con `entry_weights`.

**Barrido del peso de etiquetas con anclaje activado.** El óptimo del peso de la
relación de etiquetas se midió sin `supervision`; con anclaje puede no ser el mismo.

**Consistencia entre el ajuste y el fold-in.** La mitad de la brecha entre el
rendimiento in-sample y el held-out no es sobreajuste sino desajuste de estimador:
durante el ajuste, $G_t$ se obtiene minimizando la pérdida completa, mientras que en el
fold-in se obtiene resolviendo desde las vistas. Son dos estimadores distintos de la
misma cantidad. Medido sobre el prototipo del análisis: proyectar las mismas entidades
del entrenamiento desde sus vistas da 0.859 contra 0.981 del ajuste directo, y subir el
término de consistencia de 0 a 10 baja la brecha de 0.258 a 0.096. El término penaliza
la distancia entre $G_t$ y lo que el fold-in devolvería, $E_t = [B_t Q_t^{-1}]_+$, con
$Q_t$ y $B_t$ ya disponibles dentro de la pasada por bloques, así que no agrega ninguna
pasada sparse. No está implementado en la librería. Conjetura sobre cuándo rendiría:
cuando la suma de $n_{dst}$ es mucho menor que $n_t$, es decir con muchas entidades
descritas por pocos atributos. En MovieLens esa condición no se cumple y el held-out
quedó plano.

## Conteos

**`transform` bajo KL**, y máscaras o pesos por entrada sobre la relación Poisson misma.
El peso de fondo $w_0$ daría el régimen implícito también en esa familia.

**Negative binomial, solo con evidencia.** El paso declarado es medir la sobredispersión
condicional bajo el modelo Poisson ya ajustado (varianza contra media) y agregar el
parámetro de dispersión por relación solo si Poisson falla de forma visible. Con Poisson
empatando o perdiendo contra la cuadrática transformada en los tres protocolos medidos
(`examples/newsgroups/README.md` y `examples/lastfm/README.md`), la línea queda en baja
prioridad.

## Rendimiento

**Sincronizaciones del bucle en GPU.** La suma de los kernels de una iteración proyecta
38x en la instancia grande y el ajuste completo da 7.6x, así que cerca de dos tercios
del tiempo se va en costos por iteración fuera de los kernels: lanzamientos,
sincronizaciones por `float()` y solves chicos en cuSOLVER. El paso siguiente es
acumular la pérdida en el dispositivo y bajarla cada $k$ iteraciones. Las cifras y el
detalle de las primitivas están en `examples/gpu/README.md`.

**Poisson y pesos por entrada en GPU.** Hoy se rechazan con error y el ajuste corre en
CPU.

**Actualización en sitio por bloques.** El bucle recorre las filas por bloques, pero
acumula numerador y denominador completos por tipo, lo que cuesta tres veces el tamaño
de $G$. Escribir en sitio dentro del bloque baja el pico: el análisis midió 11.28 GB a
6.71 GB en un caso de 5 millones de filas. Solo es equivalente cuando el tipo aparece en
un solo lado de sus relaciones, así que hay que detectar el caso y avisar cuando no se
cumple.

**Costo del régimen implícito.** Con delta denso en el patrón, los pesos por entrada
cuestan 5.2x contra la ruta clásica, dominados por tres pasadas de recolección por
iteración (actualización de $S$, de $G$ y pérdida) a unos 0.5 Gflop/s contra productos
BLAS. Las dos vías son cachear entre pasadas la reconstrucción sobre el soporte de delta
y un kernel compilado para el SDDMM, y se justifican si ese régimen se vuelve el caso
dominante.

### ROCm y APUs AMD (no probado)

El mismo backend debería correr sobre GPUs AMD, porque CuPy soporta ROCm mapeando
cuSPARSE a hipSPARSE, incluido el `spmm` con `transa` que usa la pasada transpuesta.
Nada de esta entrada está medido: es análisis previo, y ninguno de los números de CUDA
se traslada.

Qué esperar según el hardware, como conjetura. El loop está limitado por ancho de banda,
y en una APU la GPU integrada comparte el bus de memoria con la CPU:

- En una APU de escritorio (DDR5 de dos canales, 90 a 100 GB/s compartidos) la iGPU no
  tiene ventaja de ancho de banda sobre la CPU; su aporte sería el paralelismo para el
  gather/scatter irregular del SpMM. La ganancia esperable es del orden del techo medido
  de paralelizar la CPU (3x), no el 10x a 14x de una GPU discreta.
- En un Strix Halo / Ryzen AI Max (256 GB/s de LPDDR5X, hasta 128 GB unificados) el caso
  es la capacidad: la mitad del ancho de banda de una RTX 2080, pero sin techo de VRAM ni
  costo de transferencia. La memoria unificada también la ve la CPU, así que la capacidad
  pura ya la aprovecha `device="cpu"`; la iGPU agrega el paralelismo sobre esa misma
  memoria.

Receta para probarlo desde una máquina AMD:

1. ROCm 7.2 o superior. Las APUs no están en la matriz oficial de soporte pero funcionan
   en la práctica; en gfx1151 (Strix Halo) hace falta `HSA_OVERRIDE_GFX_VERSION=11.5.1`.
2. CuPy para ROCm en el venv del proyecto, en lugar del extra `gpu` (que fija
   `cupy-cuda13x` y no sirve en AMD): los wheels upstream se descontinuaron desde CuPy
   13.4, pero AMD publica los suyos con
   `pip install amd-cupy --extra-index-url=https://pypi.amd.com/simple` (ROCm 7), y
   existe `cupy-rocm-7-0` en PyPI.
3. Correr `uv run pytest tests/test_gpu.py -v` y `uv run python examples/gpu/primitivas.py`.
   Los tests validan que `device="gpu"` produce el mismo modelo que la CPU;
   `primitivas.py` mide las operaciones una a una. El kernel a vigilar es el `spmm` con
   `transa`: en CUDA es lo que evita que la pasada transpuesta domine, y su rendimiento
   en hipSPARSE sobre una iGPU es la incógnita principal.
4. El código no necesita cambios: el apuntador de `CUDA_PATH` no hace nada sin wheels
   NVIDIA y el chequeo de disponibilidad usa la API que el build HIP también expone. Si
   los tests pasan, la integración que faltaría es un extra `gpu-rocm` en el `pyproject`
   y la nota de instalación en el README.

Si CuPy sobre ROCm resulta frágil (es de las rutas menos mantenidas del ecosistema AMD),
el plan B es un backend PyTorch: torch-rocm está mantenido y su CSR por denso va por
hipSPARSE, pero es un port del loop, no una recompilación, así que solo se justifica con
una necesidad concreta.

## Diagnóstico abierto de la diferencia entre las dos rutas

`fuse` queda 0.010 de AP por debajo de `dfmf_sparse` en MovieLens, con comparación
pareada sobre cuatro semillas (`examples/movielens/README.md`). Se descartaron por
ablación: el gauge de columnas, el paso `eta`, `lambda_S`, el fold-in no negativo, el
rango de la grilla de pesos (barrida de 0.01 a 1000), el solve final de $S$ y la
semántica del peso.

**Conjetura**: el orden en que se concatenan los bloques dentro de la inicialización
NNDSVD, que cambia la SVD y por tanto el punto de partida. Es una propiedad de cómo se
acomodan los datos, no del método. Verificarlo es barato: fijar el orden de los bloques
y repetir la comparación.

## Otros pendientes

**Distribuciones por entidad a través de cadenas.** Para leer la distribución de un
atributo al grano más fino (por ejemplo, el modo de cada viaje cuando la relación
etiquetada vive en otro tipo) hace falta reconstruir una relación que no existe en el
grafo, encadenando backbones a través de los tipos compartidos:
$G_{viaje} S_{v,a} (G_a^T G_a) S_{a,m} G_m^T$, con todos los intermedios de tamaño
rango. No existe en ninguna de las dos rutas. Es la pieza que habilita rebanar una
actualización por hora o por segmento agregando distribuciones por entidad fina, en vez
de multiplicar el tipo objetivo.

**Criterio de parada por fila en el refinado no negativo.** `nonneg_refine` detiene la
iteración por el cambio máximo del bloque completo, así que las filas fáciles quedan
acopladas a la que converge más lento. En `transform` con máscaras cada patrón de
observación se refina por separado, y el número de iteraciones de un grupo de filas
puede depender de junto a quiénes se refinó. Con un criterio por fila (congelar las
filas ya convergidas) el resultado sería independiente de cómo se agrupan. Costo: tocar
`core.py`, que hoy comparte esta función entre las dos rutas.

**Integración continua.** Un workflow de GitHub Actions que corra `pytest` en cada push.
La suite corre sin descargas ni credenciales, la comparación con scikit-fusion usa
trazas guardadas en el repo y los tests de GPU se saltan solos sin tarjeta, así que no
hay bloqueo técnico. Pendiente por decisión del mantenedor.

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
- **KL sobre TF-IDF normalizado por fila**: la KL modela masa y la normalización L2 la
  elimina; el ajuste degenera en 2 iteraciones con ARI 0.000. Los pesos por columna (idf)
  sí son compatibles con KL, escalando la matriz.
- **Una jerarquía de clases para la configuración** (tipos `FusionData`, `Scale`,
  `Block`, `Trace` además de `Relation`): sube el costo de entrada sin resolver ningún
  defecto medido. La entrada mínima desde un dict de matrices dispersas es el patrón
  principal y desaparecía.
- **Paralelismo agresivo de las pasadas sparse**: el techo medido es 3.1x y 2.1x según
  la operación, no el 15x que sugerían las estimaciones ingenuas. Los productos corren a
  1.7 GB/s contra 33.8 GB/s de un `memcpy` porque están limitados por acceso aleatorio a
  memoria, y a rango alto el término dominante ni siquiera es el sparse sino el
  elementwise sobre arreglos de $(n_t, c_t)$.
