# Diseno de regularizacion y API para datasets grandes

Documento de referencia producido por el analisis multiagente del 2026-08-03.
Registra las decisiones, sus razones y el plan. No es documentacion de uso.

## Resumen

La base es la propuesta "minimo", que fue la mejor puntuada por los tres jueces y la unica compatible con el presupuesto de codigo pedido, con tres injertos: de "estimador", las relaciones con nombre propio, el modelo serializable con resume y el contrato de validacion del fold-in; de "escala", el split del Laplaciano que conserva masa y la formulacion por bloques; y de los jueces, las correcciones que ninguna propuesta traia. La decision central es admitir que el encogimiento no es la palanca: tras el solve cerrado de S vale <G_t, N_t^datos> = <G_t, D_t^datos>, asi que cualquier penalizacion que solo alimente el denominador carece de punto fijo y colapsa G, y bajo la normalizacion de columnas ||G_t||_F^2 = c_t es constante, asi que la L2 es literalmente inerte. Se eliminan L2, L1 y las filas cuasi estocasticas, la escala la fija el gauge por columnas, y quedan como regularizacion las que alimentan numerador y denominador (grafo y consistencia con el estimador no negativo del fold-in), mas las dos cosas que si mueven el resultado medido: el peso por relacion, que pasa a ser hiperparametro de `fit` y multiplica tanto el termino lineal como el cuadratico, y las mascaras de observacion por fila, que arreglan el regimen semi supervisado. Todos los alpha son adimensionales y se calibran contra la energia del gradiente de datos del tipo tomada de la iteracion anterior y congelada dentro de la iteracion, lo que preserva el streaming por bloques y deja un objetivo fijo por paso. En costo, se prohibe materializar cualquier cantidad de tamano n_i x n_j: la perdida sale de la identidad de traza en cada iteracion y por relacion, el bucle baja a dos pasadas sparse con el orden Z primero (el unico que cierra), los acumuladores viven en el bloque de filas con actualizacion en sitio, y eta=1.0 se justifica por teorema, no por medicion, lo que elimina la maquinaria de retroceso. `dfmf_sparse` queda congelado como ruta de compatibilidad para que los ocho notebooks reproduzcan sus resultados publicados sin editar una linea, y la API nueva vive en `fit`, con `transform` devolviendo un modelo derivado que compone con `predict_proba` y reaplica por si mismo la transformacion del ajuste.

## Decisiones

### 1. La base es la propuesta "minimo": un solo bucle en core.py, el resultado como dataclass, base.py eliminado. Se injertan de "estimador" las relaciones con nombre, save/load/resume y el contrato de validacion de transform; de "escala", la correccion del split del Laplaciano y la formulacion por bloques del fold-in.

**Por que.** Es la propuesta mejor puntuada por los tres jueces (7/5/5 contra 6/4/6 y 6/4/4), la unica que cabe en el presupuesto de codigo que pide el usuario (el repo queda cerca de su tamano actual) y la unica que no obliga a aprender siete conceptos nuevos para ajustar un modelo chico. Los aportes de las otras dos son locales y se injertan sin arrastrar su jerarquia.

**Defecto que evita.** El juez de ergonomia sobre "estimador" conto 27 parametros en __init__ y siete conceptos donde hoy hay dos dicts; el de "escala" registro que desaparece la entrada minima desde un dict de sparse.

### 2. `dfmf_sparse`, `fold_in_entities`, `predict_attribute` y `reconstruction_error` quedan congelados en semantica numerica (sin gauge, eta=0.5, sin normalizacion, lambda_G absoluto) como ruta de compatibilidad. La entrada nueva es `fit`, con los defaults buenos. Lo unico que cambia en la ruta vieja es la implementacion de `reconstruction_error`.

**Por que.** Los ocho notebooks que ajustan importan solo `dfmf_sparse` y `reconstruction_error`; congelarlos cuesta un wrapper de 20 lineas y deja intacto trabajo real ya hecho, incluidas las cifras de report/main.md. Cambiar `reconstruction_error` a la identidad de traza da el mismo valor y elimina el bloqueo sin editar una linea de los notebooks.

**Defecto que evita.** El juez de ergonomia sobre "minimo" mostro que `normalize="frobenius"` por default borra en silencio los pesos aplicados por fuera (comparacion_variantes_filtro.py y perfil_usuarios_filtrados.py). Con la ruta vieja sin normalizacion, ese pisado no puede ocurrir en codigo existente.

### 3. Toda perdida se calcula por la identidad de traza, en cada iteracion y por relacion, y alimenta `history`, `rel_error`, la parada por `tol` y el aviso de no monotonia. Se exige `has_canonical_format` (o `sum_duplicates()`) al validar la entrada y `middle` y los grams se acumulan en float64.

**Por que.** La version densa mide 38.1 bytes por celda de n_i x n_j, 10.2 GB y 8.97 s para 200k x 1341, y se invoca dentro del bucle con verbose>0, que es lo que pasa el uso interactivo; a la escala de un caso real con millones de filas pide 81 GB. La identidad da el mismo numero con cero bytes de pico y sus tres ingredientes ya se calculan.

**Defecto que evita.** Dos defectos medidos: con duplicados en el CSR, sum(data**2) da 1.19e-5 de error relativo en la perdida, justo la escala donde vive el corte por tol; y con factores float32 el gemm de `middle` da 3.98e-5 de error relativo y perdida negativa en regimen convergido, lo que deja tol=1e-4 en el piso de ruido.

### 4. El barrido de una iteracion es: (a) pasada sparse Z_r = s * M_r^T G_src, de la que sale `middle = Z_r^T G_dst`; (b) solve de S; (c) lado destino desde Z_r; (d) segunda pasada sparse por bloques de filas Y_b = s * M_r[bloque] G_dst, con N, D locales al bloque y actualizacion de G_src en sitio. Grams cacheados por (tipo, mascara), buffers preasignados fuera del bucle.

**Por que.** Son dos pasadas sparse por matriz en vez de tres y un gram por tipo por iteracion en vez de 3*grado(t). El orden Z primero es el unico que cierra: `middle` sale de Z a costo (n_dst, c_src), medido 0.0001 s contra 0.335 s, y no exige retener Y completo.

**Defecto que evita.** El juez marco como serio que la fusion de "minimo" no cierra en el orden (S se resuelve DESDE middle, asi que Y ya no existe cuando se necesita tmp1), lo que obligaba a retener 1.6 GB no presupuestados o a volver a tres pasadas. Y la variante bloque-outer con escritura en sitio bajo el pico medido de 11.28 a 6.71 GB y la iteracion de 11.72 a 8.56 s.

### 5. No se ofrece penalizacion L2 ni L1 sobre G. La escala la fija el gauge por columnas. Las penalizaciones disponibles son las que alimentan numerador y denominador: grafo y consistencia.

**Por que.** Tras el solve cerrado de S vale <G_t, N_t^datos> = <G_t, D_t^datos> de forma exacta, asi que cualquier penalizacion que solo alimente el denominador crea un desbalance que ningun G equilibra: no tiene punto fijo y G colapsa a tasa (1+alpha)^{-eta} hasta que el solve de S revienta con LinAlgError. Y bajo el gauge ||G_t||_F^2 = c_t identicamente, asi que la L2 es una constante del problema factible y no puede mover el minimizador. Eso explica el 0.721 contra 0.722 que las tres propuestas reportaban como sorpresa empirica.

**Defecto que evita.** Dos fatales: alpha_l2 inerte bajo el gauge (verificado ||G_a||_F^2 = 10.000000 identico para alpha en {0, 0.1, 1, 10, 100}, con la perdida de datos BAJANDO al penalizar y cond(G^T G) subiendo de 27 a 570), y colapso geometrico sin gauge.

### 6. Cada alpha se calibra como fraccion de la energia del gradiente de datos del tipo, e_t = <G_t, D_t^datos>, tomada de la iteracion ANTERIOR y congelada dentro de la iteracion.

**Por que.** Con lambda absoluto la energia de datos varia 1100x entre tipos y el mismo lambda regulariza 620 veces mas fuerte a uno que a otro, hasta llevar G[actor] a 0.000e+00 exacto en la iteracion 75, punto del que las actualizaciones multiplicativas no vuelven. La calibracion relativa es invariante a R -> cR, a G -> aG con S compensando, y a n_t y c_t.

**Defecto que evita.** Serio de escala-real sobre "estimador": e_t es una reduccion global sobre el denominador completo del tipo, asi que calcularlo en la iteracion en curso obliga a materializar D_t entero y anula la promesa de memoria de block_rows. Y serio de correctitud: recalibrar dentro de la iteracion deja sin definir el objetivo que tol y el guardian comparan.

### 7. eta=1.0 por default, justificado por teorema y no por medicion: el paso es descenso de gradiente escalado con P = diag(G / (2 D)) y desciende cuando el Hessiano es PSD. Sin maquinaria de retroceso: si `history` sube, se avisa una vez y se sugiere eta=0.5.

**Por que.** Es un factor 2 en el costo dominante (eta=1 a 100 iteraciones iguala exactamente a eta=0.5 a 200) y la hipotesis se cumple siempre aca porque B = S (G_j^T G_j) S^T es PSD, el termino del grafo es PSD y el de consistencia tambien. El maximo de lambda_max(P B) sobre 20000 matrices PSD aleatorias dio 1.000000 exacto, nunca por encima.

**Defecto que evita.** El guardian de monotonia que las tres propuestas vendian como gratis no lo es: la perdida barata es la del punto PREVIO, asi que verificar el paso exige otra pasada sparse (unos 10 s de los 12.8 s de la iteracion) o una copia de G mas la iteracion descartada. Con el teorema el guardian sobra.

### 8. `weights` (beta_r) es hiperparametro de `fit` y multiplica el termino de perdida completo, es decir tanto A = beta*s*(M G_j) S^T como B = beta * S (G_j^T G_j) S^T. `scale` (1 / ||M||_F) es una propiedad de transporte que el modelo guarda. Ninguno de los dos se premultiplica en los datos: son escalares plegados en los productos.

**Por que.** El peso por relacion es la unica palanca que movio el held-out medido (AP 0.710 con 0.1 y 0.731 con 10.0) y hoy vive en `normalize_relations`, indistinguible de la normalizacion; barrerlo obliga a reconstruir los datos. Separarlo de la escala hace que `transform` reaplique la transformacion del ajuste sin ambiguedad.

**Defecto que evita.** Fatal: poner beta solo en el termino lineal optimiza otra funcion que la declarada (error relativo 0.64 contra diferencias finitas con beta=3). Y el bug de transporte: en una replica del patron de fold-in a gran escala el coseno medio entre el fold-in crudo y el correcto es 0.44 y cambia el 38% del top 50% seleccionado.

### 9. Las relaciones se identifican por nombre en un `dict[str, Relation]`, y S queda indexado por nombre. Un par (src, dst) con mas de una matriz exige nombre explicito. `views`, `weights` y `transform` referencian ese nombre.

**Por que.** Hoy `S[('usuario','celda')][0]` y `[1]` son origen y destino solo por el orden de construccion, que no queda registrado en ninguna estructura; intercambiarlos corre sin error y da otro resultado. Un caso real llego a meter cuatro matrices en una sola clave sin registrar cual era cada una.

**Defecto que evita.** El juez de ergonomia marco como fatal el autonombrado posicional f"{src}~{dst}#{k}", porque reordenar la lista entre dos corridas reetiqueta todo y desalinea un modelo guardado, y ambos nombres pasan la validacion.

### 10. `transform` devuelve un FusionModel derivado (mismo S, misma escala, G[target] reemplazado), valida nombres, formas y etiquetas de columna contra el ajuste, y proyecta al cono no negativo con actualizaciones multiplicativas desde un piso estrictamente positivo, con tolerancia relativa y el ridge contado una sola vez.

**Por que.** El caso central del repo es proyectar y despues predecir; con el ndarray suelto hay que desarmar el modelo y volver a los dicts crudos. La formulacion acumulada (Q y rhs) elimina las tres copias del decodificador denso y deja la memoria independiente de n_dst.

**Defecto que evita.** Tres fatales de golpe. (a) `fold_in` y `predict` no componian, y el ejemplo de la propia propuesta calculaba G_nuevo y no lo usaba. (b) Permutar las columnas del tipo destino corria sin aviso y bajaba el AP de 0.714 a 0.459, bajo el baseline marginal. (c) La MU no negativa partiendo del ridge crudo da 43% de NaN a 20 iteraciones, y partiendo del clamp duro congela el soporte (55.9% en cero, +0.911% de objetivo incluso a 2000 iteraciones); ademas n_iter=20 dejaba 12.79% de brecha contra el NNLS exacto.

### 11. Mascaras de observacion por FILA y por relacion, con el gram cacheado por (tipo, mascara). Las mascaras por entrada con peso de fondo w0 quedan fuera de esta iteracion, con su formula correcta registrada para cuando se necesiten.

**Por que.** Hoy cada cero es una observacion: poner en cero la fila de etiquetas del 25% de las peliculas da AP 0.468 sobre esas filas contra 0.982 sobre las que si entraron. La granularidad que arregla el caso documentado como roto (examples/semi_supervised.py) es la fila completa, y ahi el truco del gram sobrevive intacto y el costo baja en proporcion a las filas ocultas.

**Defecto que evita.** Fatal de correctitud sobre "escala": la actualizacion con mascara por entrada tal como estaba escrita parte el signo del producto y pierde el piso positivo del denominador, con divergencia a NaN con background=0.0, que ademas era su default. Y serio de escala-real: el SDDMM que exige no tiene primitiva en scipy y cuesta +50% por iteracion sobre la relacion grande. Serio de correctitud sobre "minimo": con mascara el gram depende de la matriz, no solo del tipo, y cachearlo por tipo cambia el factor resultante en 70%.

### 12. Guardas de escala explicitas: `init` sigue en "random" por default con nndsvd reescrito por LinearOperator y dtype propagado; `predict_proba` exige `top_k` o un presupuesto en bytes; `save` escribe un directorio con un .npy por arreglo; `fit` levanta ValueError cuando dos tipos mayores que block_rows se relacionan entre si.

**Por que.** Son los cuatro puntos donde el diseno se rompia fuera del bucle. init_nndsvd mide 424 s y 16.5 GB de exceso a 5M filas, mas que el pico del ajuste completo y mas que el max_memory_gb que el propio ejemplo pasaba. `np.load(archivo.npz, mmap_mode='r')` devuelve ndarrays ordinarios, verificado, asi que la promesa de proyectar sin residentes no era realizable.

**Defecto que evita.** Un fatal (init como bomba de memoria mayor que el ajuste) y tres serios (mmap imposible sobre .npz, predict devolviendo 53.6 GB por default con objetivo grande, y degradacion silenciosa cuando la descomposicion por bloques deja de ser exacta).

## Regularizaciones

### gauge por columnas (proyeccion, no penalizacion)

- **Default:** "column" (None reproduce la trayectoria legacy)
- **Ataca:** La deriva de escala bajo (G_i, G_j, S) -> (a G_i, a G_j, S / a^2), que deja el termino de datos invariante y hace que ninguna penalizacion sobre G este bien definida. Sin gauge, ||G||_F medido llega a 401 con eta=0.5 y a 7587 con eta=1.0, con riesgo de desborde en float32 y con los denominadores de calibracion moviendose sin control.
- **Escala:** Deja ||G_t||_F^2 = c_t exacto y diag(G_t^T G_t) = 1, con lo que lambda_S se lee como fraccion de la energia por columna y los alpha quedan comparables entre tipos. Con lambda_S=0 es un cambio de variables exacto (misma perdida a 1e-12, verificado); con lambda_S>0 mueve levemente el punto fijo porque el termino de Tikhonov no es homogeneo en escala, y eso se documenta en vez de venderse como mejora.
- **Actualizacion:** Despues de la actualizacion multiplicativa: d_k = max(||G_t[:, k]||_2, eps_rel) para cada columna k, G_t[:, k] <- G_t[:, k] / d_k. S no se corrige a mano, se re-resuelve en forma cerrada al inicio de la iteracion siguiente. Las columnas con d_k == eps_rel se registran en `dead_columns` y se avisa una vez.

### weights: peso beta_r por relacion, hiperparametro del ajuste

- **Default:** 1.0 por relacion, normalize="frobenius" en `fit` y None en la ruta legacy
- **Ataca:** Es la unica palanca que movio el held-out en las mediciones (AP 0.710 con 0.1, 0.707 con 1.0, 0.722 con 3.0, 0.731 con 10.0), y hoy vive fuera del algoritmo, indistinguible de la normalizacion. Esa confusion produce el bug de transporte del fold-in: coseno medio 0.44 contra el fold-in correcto y 38% del top 50% seleccionado que cambia, sin ninguna senal de error.
- **Escala:** beta_r es el peso de esa relacion en la perdida, con `normalize="frobenius"` cada matriz entra con norma 1, asi que beta_r es directamente su cuota del presupuesto de energia. s_r y beta_r se pliegan como escalares en los productos: cero copias del nnz, contra las dos copias completas que hace hoy normalize_relations. Ambos quedan guardados en el modelo y transform los reaplica.
- **Actualizacion:** Lado fuente: A = beta_r * s_r * (M_r G_j) S_r^T y B = beta_r * S_r (G_j^T G_j) S_r^T; N_i += [A]_+ + G_i [B]_-, D_i += [A]_- + G_i [B]_+. Lado destino simetrico con A' = beta_r * s_r * (M_r^T G_i) S_r y B' = beta_r * S_r^T (G_i^T G_i) S_r. beta multiplica los DOS terminos. En la perdida, beta_r * (s_r^2 ||M_r||_F^2 - 2 s_r <G_i^T M_r G_j, S_r> + <(G_i^T G_i) S_r (G_j^T G_j), S_r>).

### masks: mascara de observacion por fila y por relacion

- **Default:** None (todas las filas observadas)
- **Ataca:** Hoy ocultar equivale a ensenar el cero: poner en cero la fila de etiquetas del 25% de las peliculas del train da AP 0.468 sobre esas filas contra 0.982 sobre las que si entraron. Sin mascara no hay regimen semi supervisado (examples/semi_supervised.py lo documenta como contraejemplo) ni held-out interno. Usar el gram sin enmascarar en el lado destino cambia el factor resultante en 70% de norma relativa, en silencio.
- **Escala:** No tiene escala que fijar: restringe el dominio de la suma, no penaliza. El costo por iteracion baja en proporcion a las filas ocultas y el truco del gram sobrevive intacto.
- **Actualizacion:** Para la matriz r con filas observadas `rows`, se sustituye M_r por M_r[rows] y G_src por G_src[rows] en los tres lugares donde aparecen: el gram del tipo fuente, `middle` y la acumulacion del lado fuente. El lado destino usa el gram ENMASCARADO S_r^T (G_src[rows]^T G_src[rows]) S_r, y por eso el cache de grams se indexa por (tipo, mascara) y no solo por tipo. ||M_r||_F^2 se cachea sobre las filas observadas. El slicing se hace una vez al inicio del ajuste.

### alpha_graph: suavizado por grafo, calibrado contra la energia de datos

- **Default:** 0.0 (pasar `graphs` con alpha_graph=0 levanta ValueError)
- **Ataca:** El caso extremo de escala absoluta no era lambda_G sino theta: con el Laplaciano crudo de contigueidad H3 sobre relaciones Frobenius normalizadas, el termino pesa 130x el de los datos, o sea que el notebook insignia ajusta principalmente el suavizado geografico. Ademas I - W_sym, que las otras propuestas usaban, encoge las celdas de grado bajo un 17.5% a 200 iteraciones, un artefacto sistematico en todo el borde del area de estudio.
- **Escala:** gamma_t fija la contribucion del grafo en exactamente alpha_graph veces la energia de datos del tipo, cualquiera sea el grado del grafo o la normalizacion de las relaciones. El split conserva masa exactamente (el vector constante esta en el nucleo de diag(W_sym 1) - W_sym para cualquier distribucion de grados), asi que suaviza sin encoger, y es PSD, asi que preserva la garantia de eta=1.0. `graphs` acepta lista de matrices por tipo.
- **Actualizacion:** Con W_sym = D^{-1/2} W D^{-1/2} y L = diag(W_sym 1) - W_sym: gamma_t = alpha_graph * e_t / <G_t, diag(W_sym 1) G_t>, con e_t = <G_t, D_t^datos> de la iteracion anterior. Numerador: N_t += gamma_t * (W_sym @ G_t). Denominador: D_t += gamma_t * (diag(W_sym 1) * G_t). Costo O(nnz(W) * c_t).

### alpha_consistency: acoplamiento entre el estimador del ajuste y el del fold-in

- **Default:** 0.0 (con `views` obligatorio cuando es mayor que cero, y viceversa)
- **Ataca:** La mitad de la brecha entre 0.98 in-sample y 0.72 held-out no es sobreajuste sino desajuste de estimador: proyectar las MISMAS peliculas del train desde las vistas da 0.859, no 0.981. Ningun regularizador de encogimiento toca esa mitad. Medido, alpha de 0 a 10 lleva la brecha de 0.258 a 0.096 con el held-out plano en MovieLens; la condicion bajo la que deberia ayudar de verdad, suma de n_dst mucho menor que n_t, MovieLens no la cumple y si la cumple un caso con muchas entidades descritas por pocos atributos.
- **Escala:** Misma unidad que los demas alpha. El clamp a la parte positiva no es cosmetico: alinea el objetivo con lo que `transform` realmente devuelve (la solucion no negativa), y sin el, el termino persigue el ridge de signo libre, que difiere en 0.51 a 0.63 de norma relativa y tiene entre 24% y 38% de entradas negativas. Calculado bloque a bloque, no agrega ninguna pasada sparse ni ningun arreglo de (n_t, c_t): el costo es O(n_t c_t^2 + c^3), no O(c^3).
- **Actualizacion:** Con Q_t = sum_v beta_v S_v (G_v^T G_v) S_v^T + lambda_S I calculado una vez por iteracion (c x c) y B_t[bloque] = sum_v beta_v s_v (M_v[bloque] G_v) S_v^T, que es exactamente el A que la pasada por bloques ya acumula: E_t[bloque] = [solve(Q_t^T, B_t[bloque]^T)^T]_+. mu_t = alpha_consistency * e_t / ||G_t||_F^2, es decir alpha_consistency * e_t / c_t bajo el gauge. Numerador: N_t[bloque] += mu_t * E_t[bloque]. Denominador: D_t[bloque] += mu_t * G_t[bloque]. Como E_t >= 0 por construccion, el termino [E_t]_- desaparece.

### lambda_S: Tikhonov del solve de S, con la penalizacion documentada correctamente

- **Default:** 1e-2 en `fit`, 0.0 en la ruta legacy
- **Ataca:** Tres cosas. La penalizacion documentada no era la implementada. lambda_S era ademas lo unico que frenaba por accidente la degeneracion de escala, rol que ahora cumple el gauge de forma explicita. Y S se resolvia al inicio de la iteracion mientras G se actualizaba despues, asi que la S devuelta correspondia a la G anterior (medido 1.0e-2 de diferencia relativa) y tanto el fold-in como la prediccion suponian una optimalidad que esa S no cumplia.
- **Escala:** Bajo el gauge, G^T G tiene diagonal exactamente 1, asi que lambda_S se lee como fraccion de la energia por columna y ya no depende de cuanto haya crecido G. beta_r no entra al solve de S, porque se cancela en el subproblema; eso se documenta para que lambda_S siga siendo comparable entre relaciones con pesos distintos.
- **Actualizacion:** Sin cambio de forma: A_i = G_i^T G_i + lambda_S I, A_j = G_j^T G_j + lambda_S I, S_r = solve(A_i, middle_r) y despues por la derecha contra A_j. Se agrega un solve final de S despues del ultimo paso de G. El docstring pasa a decir lo que penaliza de verdad: lambda_S (||G_i S||_F^2 + ||S G_j^T||_F^2) + lambda_S^2 ||S||_F^2, no lambda_S ||S||_F^2 (verificado: el minimizador de la primera coincide con lo que devuelve _solve_S a 1.5e-6, el de la segunda difiere en 4.6e-2).

## API

```python
# =====================================================================
# src/datafiusion/model.py  (new)
# =====================================================================
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Literal, Mapping, Sequence

import numpy as np
import pandas as pd
import scipy.sparse as sp

Matrix = sp.csr_matrix | np.ndarray   # dense is allowed and faster when n_dst is small
Alpha = float | Mapping[str, float]   # global value, or one value per type


@dataclass
class Relation:
    """One relation matrix, its two endpoints and its row observation mask."""

    src: str
    dst: str
    matrix: Matrix
    rows: np.ndarray | None = None          # observed row indices; None means every row
    row_labels: np.ndarray | None = None    # optional entity labels, checked at fold-in
    col_labels: np.ndarray | None = None

    @property
    def shape(self) -> tuple[int, int]:
        """Shape of the underlying matrix."""


@dataclass
class FusionModel:
    """A fitted model: factors, backbones, the transform applied at fit time, and the trace."""

    G: dict[str, np.ndarray]                # (n_t, c_t) per type, non negative
    S: dict[str, np.ndarray]                # (c_src, c_dst) per relation NAME, never a list
    rel: dict[str, tuple[str, str]]         # relation name -> (src, dst)
    ranks: dict[str, int]
    sizes: dict[str, int]
    scale: dict[str, float]                 # 1 / ||M||_F applied at fit, per relation
    weight: dict[str, float]                # beta_r used at fit, per relation
    index: dict[str, np.ndarray | None]     # entity labels per type, when they were given
    history: np.ndarray                     # (n_iter,) data loss, measured at each iteration start
    rel_error: dict[str, float]             # relative Frobenius error per relation
    n_iter: int
    converged: bool
    stop_reason: Literal["tol", "max_iter", "callback"]
    dead_columns: dict[str, np.ndarray]     # columns of G_t that collapsed to zero
    params: dict                            # every hyperparameter of the call that produced it

    def __iter__(self):
        """Yield (G, S) so `G, S = model` works. S is keyed by relation name."""

    def factor(self, type_name: str, as_frame: bool = False) -> np.ndarray | pd.DataFrame:
        """Return G_t, optionally indexed by the entity labels of that type."""

    def backbone(self, name: str, as_frame: bool = False) -> np.ndarray | pd.DataFrame:
        """Return S_r for relation `name`."""

    def loss(
        self,
        relations: Mapping[str, Relation] | None = None,
        *,
        per_relation: bool = False,
        relative: bool = True,
    ) -> float | dict[str, float]:
        """Loss by the trace identity, never materializing G_i S G_j^T.

        With relative=True each term is divided by ||scale * M||_F^2 and the
        aggregate is the MEAN over relations, so two models with a different
        number of relations are comparable.
        """

    def transform(
        self,
        relations: Mapping[str, Relation],
        target: str,
        *,
        nonneg: bool = True,
        alpha: float = 1e-3,
        max_iter: int = 100,
        tol: float = 1e-4,
        block_rows: int | None = None,
        out: np.ndarray | str | Path | None = None,
        n_threads: int = 1,
    ) -> "FusionModel":
        """Fold new entities of type `target` in, returning a DERIVED model.

        The returned model shares S, scale, weight and every other factor, and
        replaces G[target]; that is what makes `model.transform(...).predict_proba(...)`
        the natural path. `target` may be the source or the destination of each
        relation. Every relation name must exist in `self.rel`, the fixed side
        must match `sizes`, and when both `self.index[dst]` and `col_labels` are
        present they must be equal, otherwise ValueError.

        The fit-time scale and weight are reapplied by the model, never by the
        caller. Accumulates, over row blocks and without building S G_dst^T:

            Q  = sum_r beta_r * S_r (G_dst^T G_dst) S_r^T          (c, c)
            b  = sum_r beta_r * s_r * (M_r @ G_dst) @ S_r^T        (n_new, c)
            X  = solve(Q + lam I, b)                               lam = alpha * trace(Q) / c

        With nonneg=True, X is floored at a strictly positive value
        (tau = 1e-6 * rowmax) and refined with
            X <- X * ((num) / (den))^eta,  num = [b]_+ + X [Q]_-,
                                           den = [b]_- + X [Q]_+ + lam * X
        until the maximum relative change drops below `tol`. The floor is
        mandatory: a multiplicative update cannot change a sign, so starting
        from the raw ridge gives NaN and starting from a hard clamp freezes the
        support at zero.

        Memory is O(n_new * c) for the output plus O(block_rows * c) of work,
        independent of n_dst. Pass `out` to write G_new into a memmap.
        """

    def predict_proba(
        self,
        target: str,
        views: Sequence[str],
        known: Mapping[str, np.ndarray] | None = None,
        *,
        combine: Literal["geometric_mean", "product", "sum"] = "geometric_mean",
        eps: float = 1e-3,
        top_k: int | None = None,
        max_bytes: int = 256 << 20,
    ) -> np.ndarray | tuple[np.ndarray, np.ndarray]:
        """Distribution over `target` combining several views, by row batches.

        Views are named relations and may point in either direction: when
        `target` is the source of the relation, S^T is used. `known` accepts
        positions or entity labels and defaults to every row of the non-target
        side. Batches are sized as max_bytes // (2 * n_target * itemsize) and
        the log of the views is accumulated in a single buffer, without
        np.stack. Without `top_k`, raises ValueError when the full output
        would exceed `max_bytes`; with `top_k` returns (indices, scores) of
        shape (n, top_k).
        """

    def resume(
        self, relations: Mapping[str, Relation], *, max_iter: int, **overrides
    ) -> "FusionModel":
        """Continue the fit from the current G, reusing `params` and the stored
        scale and weight (never recomputed from the data). `history` and
        `n_iter` accumulate."""

    def save(self, path: str | Path) -> None:
        """Write a directory: factors/<type>.npy, backbones/<name>.npy,
        index/<type>.npy and meta.json. One .npy per array so `load(mmap=True)`
        really memory-maps; written to a temporary directory and renamed."""

    @classmethod
    def load(cls, path: str | Path, *, mmap: bool = False) -> "FusionModel":
        """Read back a model saved with `save`."""


def fit(
    relations: Mapping[str, Relation],
    ranks: Mapping[str, int],
    *,
    # --- balance between relations (the lever that moves held-out) --------
    weights: Mapping[str, float] | None = None,
    normalize: Literal["frobenius"] | None = "frobenius",
    # --- regularization, dimensionless except lambda_S -------------------
    graphs: Mapping[str, sp.spmatrix] | None = None,
    alpha_graph: Alpha = 0.0,
    alpha_consistency: Alpha = 0.0,
    views: Mapping[str, Sequence[str]] | None = None,
    lambda_S: float = 1e-2,
    gauge: Literal["column"] | None = "column",
    # --- loop ------------------------------------------------------------
    max_iter: int = 200,
    tol: float = 1e-5,
    eta: float = 1.0,
    init: Literal["random", "nndsvd"] | Mapping[str, np.ndarray] = "random",
    random_state: int | np.random.Generator | None = None,
    # --- cost ------------------------------------------------------------
    dtype: np.dtype = np.float64,
    block_rows: int | None = None,
    n_threads: int = 1,
    # --- observability ---------------------------------------------------
    eval_every: int = 1,
    callback: Callable[["FusionModel"], bool | None] | None = None,
    verbose: int = 0,
) -> FusionModel:
    """Sparse non-negative tri-factorization for data fusion, by row blocks.

    R_r ~ G_i S_r G_j^T with G_i >= 0 dense and S_r of free sign. The loss is

        sum_r beta_r * || s_r * M_r - G_i S_r G_j^T ||_F^2   (over observed rows)

    with s_r = 1 / ||M_r||_F when normalize="frobenius" and beta_r = weights[r].
    Neither s_r nor beta_r ever touches the data: both are scalars folded into
    the products, so the relations are never copied, and both are stored in the
    model so `transform` reapplies exactly the fit-time transform. beta_r
    multiplies BOTH the linear term A and the quadratic term B; putting it only
    on A optimizes a different function than the one documented.

    Every loss value comes from the trace identity

        ||R||^2 - 2 * <G_i^T R G_j, S> + <(G_i^T G_i) S (G_j^T G_j), S>

    whose three ingredients the update already computes, so the trace, the
    tolerance test and the per-relation error are free and no quantity of size
    n_i x n_j is ever materialized. `history[k]` is measured at the START of
    iteration k, that is at (G^(k-1), S^(k)); a rise in that sequence emits a
    warning once.

    Regularization contract. Any penalty that only feeds the DENOMINATOR has no
    fixed point here, because the closed-form S solve makes <G_t, N_t> equal to
    <G_t, D_t> on the data terms; and under the column gauge ||G_t||_F^2 equals
    c_t identically, so an L2 penalty is constant on the feasible set. L2 and L1
    shrinkage are therefore not offered. Scale is fixed by the gauge, and the
    penalties that are offered (graph, consistency) feed numerator and
    denominator. Each alpha is the fraction of the data-gradient energy
    e_t = <G_t, D_t^data> that the penalty contributes; e_t is taken from the
    previous iteration and frozen inside the iteration, so the objective is
    fixed within a step and the row-block streaming stays exact.

    eta=1.0 is not a heuristic: the update is scaled gradient descent with
    P = diag(G / (2 D)) and descends whenever the Hessian is PSD, which holds
    for the data terms (B = S (G_j^T G_j) S^T is PSD), the graph term and the
    consistency term. Use eta=0.5 only to reproduce the legacy trajectory.

    Row blocks are exact, not approximate, as long as every large type appears
    on a single side of its relations. `fit` raises ValueError naming the pair
    when two types larger than `block_rows` relate to each other.

    Parameters
    ----------
    relations : Mapping[str, Relation]
        Keyed by relation NAME. Names carry into S, weights, views and
        transform, so `S["usr_origen"]` replaces the positional S[key][0].
    weights : Mapping[str, float]
        beta_r per relation, applied after `normalize`. Missing keys are 1.0.
        This is a hyperparameter of the fit, not a property of the data, so it
        can be swept without rebuilding the relations.
    graphs, alpha_graph : Mapping[str, sparse], float or Mapping[str, float]
        Adjacency W per type. The Laplacian is built as diag(W_sym 1) - W_sym
        with W_sym = D^-1/2 W D^-1/2: that split conserves mass exactly for any
        degree distribution, while I - W_sym does not and shrinks low-degree
        cells (borders of the H3 grid) by about 17% over 200 iterations.
        Passing `graphs` with alpha_graph == 0 raises ValueError.
    alpha_consistency, views : float or Mapping[str, float], Mapping[str, list[str]]
        Couples G_t to its own fold-in projection computed over `views[t]`, the
        relations available at inference. One without the other raises ValueError.
    lambda_S : float
        Tikhonov of the S solve. It penalizes
        lambda_S * (||G_i S||^2 + ||S G_j^T||^2) + lambda_S^2 * ||S||^2,
        not ||S||^2 alone. Under the column gauge diag(G^T G) is 1, so
        lambda_S reads as a fraction of the per-column energy.
    gauge : {"column", None}
        Normalizes each column of G to unit L2 norm after every update; S
        absorbs the scale in the next closed-form solve. With lambda_S=0 it is
        an exact reparametrization; with lambda_S>0 it slightly moves the fixed
        point, because the Tikhonov term is not scale homogeneous. Its purpose
        is to keep ||G_t||_F^2 = c_t so lambda_S and every alpha are readable,
        and to stop the drift that took ||G||_F to 7.6e3 without it.
    dtype : np.dtype
        Governs the dense factors. Raises TypeError when the relation data does
        not already match, since converting them here would duplicate the nnz;
        use `datafiusion.utils.as_dtype` upstream.
    block_rows : int or None
        Memory knob, not a result knob. None derives it from a 256 MB working
        budget.
    callback : callable
        Called every `eval_every` iterations with the live model; returning
        False stops the fit. Checkpointing is `lambda m: m.save(path)`.
    """


# =====================================================================
# src/datafiusion/utils.py  (add)
# =====================================================================

def relation_from_frame(
    df: pd.DataFrame,
    src: str,
    dst: str,
    *,
    src_col: str,
    dst_col: str,
    value_col: str | None = None,
    src_labels: Sequence | None = None,
    dst_labels: Sequence | None = None,
    aggregate: Literal["sum", "count", "mean"] = "sum",
    rows: np.ndarray | None = None,
    dtype: np.dtype = np.float64,
) -> Relation:
    """Long table (entity, entity, value) to a Relation, without a dense grid.

    Label to position mapping uses pd.Categorical codes, which is vectorized in
    C; the dict plus .map() pattern of the notebooks costs 27 s of the 58 s of
    ingestion on 67M rows. Pass `dst_labels=model.index[dst]` when building the
    relations for a fold-in, so the new cohort shares the fit-time numbering.
    """


def as_dtype(M: Matrix, dtype: np.dtype) -> Matrix:
    """Cast the data of a sparse matrix in place, leaving the indices alone."""


# =====================================================================
# End to end: MovieLens (fit, project held-out movies, predict genre)
# =====================================================================
import numpy as np

from datafiusion import Relation, fit

import datos as dm            # examples/movielens/datos.py

datos = dm.cargar(semilla=0)


def relaciones(split, con_genero: bool) -> dict[str, Relation]:
    R = {
        "usuarios": Relation("pelicula", "usuario", split.R[("pelicula", "usuario")][0]),
        "actores": Relation("pelicula", "actor", split.R[("pelicula", "actor")][0]),
    }
    if con_genero:
        R["generos"] = Relation("pelicula", "genero", split.R[("pelicula", "genero")][0])
    return R


modelo = fit(
    relaciones(datos.train, con_genero=True),
    ranks={"pelicula": 20, "genero": 15, "usuario": 25, "actor": 40},
    weights={"generos": 3.0},                       # the lever that moves held-out AP
    alpha_consistency={"pelicula": 0.3},            # couples fit and fold-in estimators
    views={"pelicula": ["usuarios", "actores"]},    # what inference will actually see
    lambda_S=1e-2, gauge="column",
    max_iter=400, tol=1e-5, eta=1.0, init="nndsvd", verbose=25,
)
print(modelo.n_iter, modelo.stop_reason, modelo.rel_error, modelo.dead_columns)

# Fold-in returns a derived model, so prediction composes with no dict surgery
# and no manual rescaling: scale and weight travel inside the model.
proyectado = modelo.transform(relaciones(datos.test, con_genero=False), target="pelicula")
assert (proyectado.G["pelicula"] >= 0).all()

scores = proyectado.predict_proba(target="genero", views=["generos"])

# Sweeping the relation weight no longer rebuilds anything nor refits from zero.
for beta in (0.1, 1.0, 3.0, 10.0, 30.0):
    m = fit(relaciones(datos.train, True), ranks=modelo.ranks, weights={"generos": beta},
            init=modelo.G, max_iter=100, **{k: v for k, v in modelo.params.items()
                                            if k not in ("weights", "init", "max_iter")})
    print(beta, m.transform(relaciones(datos.valid, False), "pelicula")
                 .predict_proba(target="genero", views=["generos"]).shape)
```

## Plan de implementacion

1. **tests/test_golden.py y pyproject.toml** Declara pytest y congela la salida actual de dfmf_sparse (G y S, init random y nndsvd, semilla fija, 100 iteraciones) en un .npz de referencia. Es la red antes de tocar nada.
   - Verificacion: uv run pytest tests/test_golden.py pasa contra el codigo sin modificar. El .npz queda versionado.

2. **src/datafiusion/core.py** Reimplementa reconstruction_error por la identidad de traza, con validacion de formato canonico (sum_duplicates) y acumulacion en float64. Firma intacta.
   - Verificacion: Test nuevo: coincide con el calculo denso a 1e-12 relativo en tres instancias chicas, incluida una con duplicados. Medicion de pico RSS y tiempo en una relacion de 200k x 1341: se espera 0 MB de delta contra los 10.2 GB actuales.

3. **src/datafiusion/core.py** Reestructura el bucle sin cambiar semantica: orden Z primero (middle desde Z), gram cacheado por tipo, buffers preasignados, _split_signs materializando solo la parte negativa, bloques de filas con N y D locales y actualizacion en sitio.
   - Verificacion: El test dorado del paso 1 sigue pasando a 1e-10. Test de bloques: ajuste por bloques de 50k filas contra ajuste completo a 1e-14. Medicion de segundos por iteracion y pico RSS sobre el caso sintetico de 5M filas.

4. **src/datafiusion/model.py (nuevo) y core.py** Introduce Relation y FusionModel, la funcion fit con relaciones por nombre, scale y weight plegados como escalares, history, tol, stop_reason y rel_error. dfmf_sparse pasa a ser un wrapper delgado que arma relaciones anonimas y fija los defaults legacy.
   - Verificacion: Test dorado sobre la ruta legacy. Test nuevo sobre la ruta nueva: mismo resultado que la legacy cuando se pasan gauge=None, eta=0.5, normalize=None. Test de que weights beta y scale s dan exactamente la perdida declarada, contra diferencias finitas, para beta en {0.3, 1, 10}.

5. **src/datafiusion/model.py** Agrega el gauge por columnas, dead_columns y la calibracion e_t de la iteracion anterior congelada dentro de la iteracion.
   - Verificacion: Con lambda_S=0 el gauge es un cambio de variables exacto: la perdida debe coincidir a 1e-12 con y sin gauge. Test de invariancia: ajustar con R y con 3.7*R da la misma trayectoria de G a 1e-12. ||G_t||_F^2 == c_t en cada iteracion.

6. **src/datafiusion/model.py** Mascaras por fila (Relation.rows), con grams cacheados por (tipo, mascara) y ||R||^2 restringido a las filas observadas.
   - Verificacion: Test: con mascara que cubre todas las filas, resultado identico al caso sin mascara. Con mascara parcial, el gradiente coincide con diferencias finitas de la perdida restringida. examples/semi_supervised.py reescrito: AP sobre filas sin etiqueta contra el baseline de relleno cero (hoy 0.468 contra 0.982 de las filas etiquetadas).

7. **src/datafiusion/model.py** FusionModel.transform: Q y rhs acumulados por bloques, validacion de nombres, formas y col_labels, reaplicacion de scale y weight, MU no negativa desde piso positivo con tolerancia relativa y ridge contado una vez, retorno de un modelo derivado.
   - Verificacion: assert (G_new >= 0).all(). Objetivo a menos de 0.5% del optimo de scipy.optimize.nnls en una instancia de 400 filas y c=12. Test de que permutar col_labels levanta ValueError. Test de composicion: transform(...).predict_proba(...) corre sin tocar dicts.

8. **src/datafiusion/model.py** Termino de grafo con el split diag(W_sym 1) - W_sym y calibracion alpha_graph, mas ValueError cuando se pasa graphs con alpha_graph=0 o alpha_consistency sin views.
   - Verificacion: Conservacion de masa: sum(N_grafo) / sum(D_grafo) = 1 a 1e-12 para G aleatoria y para G constante. Grilla hexagonal de 1350 celdas con borde: razon norma_borde sobre norma_interior dentro de 1.00 +- 0.01 a 200 iteraciones.

9. **src/datafiusion/model.py** alpha_consistency calculado bloque a bloque dentro de la pasada de filas, con E_t = [B_t Q_t^{-1}]_+, o sea el mismo estimador no negativo que devuelve transform.
   - Verificacion: Costo declarado O(n_t c_t^2 + c^3) y cero pasadas sparse extra, medido contra la iteracion sin el termino. Diagnostico sobre MovieLens: brecha entre AP in-sample y AP proyectando el propio train baja de 0.121 hacia 0.05 al subir alpha.

10. **src/datafiusion/model.py** predict_proba por lotes con top_k, presupuesto en bytes, vistas por nombre en cualquier direccion y known por etiquetas o posiciones. Mas save, load y resume como directorio de .npy con meta.json.
   - Verificacion: Test: mismos valores que predict_attribute en una instancia chica. Test de que sin top_k y con objetivo grande levanta ValueError. Test de que 200 iteraciones de corrido coinciden bit a bit con 100 mas save, load y resume(100). Test de que load(mmap=True) devuelve np.memmap.

11. **src/datafiusion/init.py y utils.py** nndsvd por LinearOperator sobre los bloques con la escala de Frobenius al vuelo, dtype propagado, y descomposicion propia de M M^T cuando el rango pedido iguala el numero de niveles. Agrega relation_from_frame con pd.Categorical y as_dtype.
   - Verificacion: Los factores coinciden con los actuales a 1e-8 en instancias chicas y el sign pinning se conserva. Pico RSS del init medido a 250k y 1M filas: se espera que deje de crecer con el nnz copiado (hoy 0.86 GB y 3.3 GB). relation_from_frame contra el patron manual de los notebooks: mismas matrices, tiempo de ingesta medido.

12. **examples/ y src/datafiusion/base.py** Elimina base.py y su reexport, reescribe toy_sparse.py y semi_supervised.py sobre fit, adapta diagram.py, y reescribe examples/movielens/prediccion.py con el barrido de pesos ampliado, cuatro semillas y error estandar, mas la extension de escala.py con el caso sintetico de 5M filas bajo guardian de memoria.
   - Verificacion: El protocolo completo de MovieLens corre y reporta media y error estandar contra la libreria actual y contra el kNN por coseno. escala.py reporta pico medido contra la formula declarada, con desvio menor al 20%.

## Como se valida

La demostracion de que la version nueva funciona mejor se separa en tres capas, porque son tres afirmaciones distintas.

1. NO ROMPE LO EXISTENTE (equivalencia numerica). Test dorado: se congela hoy la salida de `dfmf_sparse` sobre la instancia sintetica de `tests/test_compare_skfusion.py` (semilla fija, init nndsvd y random, 100 iteraciones) en un .npz, y despues de cada paso del plan se exige `max|G_nuevo - G_viejo| / max|G_viejo| < 1e-10` y lo mismo para S. La ruta legacy conserva eta=0.5, sin gauge, sin normalizacion y lambda_G absoluto, asi que los ocho notebooks reproducen sus numeros publicados. Unico cambio de valor esperado: `reconstruction_error` pasa de suma densa a identidad de traza, que se valida contra el calculo denso a 1e-12 relativo en instancias chicas.

2. ES CORRECTO DONDE ANTES NO LO ERA (tests con verdad conocida, no metricas de opinion).
   - `transform`: `(G_new >= 0).all()` y objetivo a menos de 0.5% del optimo de `scipy.optimize.nnls` sobre una instancia de 400 filas y c=12. Hoy hay entre 24% y 38% de entradas negativas y el clamp del ejemplo queda 2.09% peor en objetivo.
   - Bloques: ajuste por bloques de 50k filas contra ajuste completo, error relativo esperado menor a 1e-14 en G, S y perdida.
   - Invariancia de la calibracion: ajustar con R y con 3.7*R debe dar la misma trayectoria de G a 1e-12; ajustar con G inicial escalado por a con S compensando, idem.
   - Grafo: sobre una grilla hexagonal de 1350 celdas con borde, razon norma_borde / norma_interior a 200 iteraciones dentro de 1.00 +- 0.01 con el split diag(W_sym 1) - W_sym (con I - W_sym da 0.825, sesgo sistematico contra el perimetro del area de estudio).
   - Escala transportada: fold-in de la misma cohorte con y sin `model.transform` debe coincidir; permutar `col_labels` debe levantar ValueError en vez de correr (hoy corre y el AP cae de 0.714 a 0.459, bajo el baseline marginal 0.550).
   - save/load/resume: ajustar 200 iteraciones de corrido contra 100 + save + load + resume(100) debe coincidir bit a bit.

3. GENERALIZA MEJOR (el unico criterio que decide, con incertidumbre). Dataset: MovieLens con el protocolo ya escrito en `examples/movielens/`, peliculas completas reservadas y proyectadas con fold-in, metrica AP media sobre los 19 generos. El error estandar medido es 0.0102 sobre 662 peliculas, asi que se corren 4 particiones (semillas 0 a 3) y se reporta media y error estandar; diferencias menores a 0.020 no se declaran. Referencias fijas en la misma particion: baseline marginal 0.550, kNN por coseno sobre las mismas matrices (hoy 0.031 de AP por encima del modelo), y la libreria actual en su mejor configuracion (0.722).
   Barrido: peso de la relacion de etiquetas en {0.1, 1, 3, 10, 30} (la grilla actual, 1.0 y 3.0, esta truncada por ambos lados: 0.731 en 10.0), rango de `pelicula` en {10, 19, 20, 40}, alpha_consistency en {0, 0.3, 1.0}, mascara por fila activada y desactivada en la relacion de etiquetas. El eje `max_iter` desaparece del barrido: sale de `history` de una sola corrida de 400.
   Regla de decision: gana el modelo mas chico dentro de un error estandar del mejor. Criterio de exito declarado antes de correr: el mejor de la version nueva supera al mejor de la actual por al menos 2 errores estandar (0.020), y la brecha contra el kNN se cierra o se invierte. Si no se cumple, se reporta asi, porque el diseno se justifica igual por costo y correctitud, pero no se afirmara que generaliza mejor.
   Diagnostico obligatorio en el mismo reporte: AP in-sample, AP proyectando el propio train (separa desajuste de estimador de generalizacion: hoy 0.981 contra 0.859 contra 0.722) y AP sobre filas enmascaradas, para saber cual de las tres causas se movio.

4. COSTO (regresion medida, no proyectada). Se extiende `examples/movielens/escala.py` con un caso sintetico de 5e6 usuarios por 1341 celdas al 1% mas 5e6 por 48, rangos 50/30/10, corrido bajo un guardian de memoria (subprocess mas sondeo de MemAvailable). Se reportan pico RSS y segundos por iteracion contra la formula declarada, mas el pico del init y el pico de ingesta, que son las dos etapas que hoy nadie mide. Criterios: ninguna asignacion mayor a O(nnz + sum_t n_t c_t), pico dentro del 20% de la formula, y `verbose > 0` sin costo de memoria medible. Referencias medidas por los jueces sobre ese mismo caso: estructura actual 18.2 s y 15.25 GB por iteracion, estructura por bloques con actualizacion en sitio 8.6 s y 6.7 GB.

## Descartado

- alpha_G / alpha_l2 (L2 sobre G), de las tres propuestas. Bajo el gauge por columnas ||G_t||_F^2 = c_t identicamente, asi que la penalizacion es constante en el conjunto factible y no puede mover el minimizador; sin gauge no tiene punto fijo, porque tras el solve cerrado de S vale <G_t, N_t^datos> = <G_t, D_t^datos> y un termino que solo alimenta el denominador colapsa G a tasa (1+alpha)^{-eta} hasta que el solve revienta. Verificado que la perdida de datos BAJA al penalizar y que cond(G^T G) sube de 27 a 570, dos sintomas que una penalizacion genuina no puede tener.
- alpha_ortho (ortogonalidad de columnas), de "estimador". Alimenta ambos lados y por eso tiene punto fijo, pero es cuartico y no convexo, asi que rompe la hipotesis PSD del teorema que justifica eta=1.0 y obligaria a volver a eta=0.5, es decir a duplicar el costo del ajuste. Cuesta ademas O(n_t c_t^2) por iteracion y un hiperparametro mas, sin ninguna ganancia medida. Bajo el gauge su unico efecto restante es decorrelacionar columnas, que se puede diagnosticar mirando cond(G^T G) en vez de penalizarlo.
- alpha_stochastic / alpha_rowsum (filas cuasi estocasticas), de "estimador" y "escala". Es infactible junto al gauge por columnas que ambas propuestas imponen: filas que suman 1 exigen ||G_t||_F^2 >= n_t / c_t y columnas unitarias fijan ||G_t||_F^2 = c_t, compatibles solo si n_t <= c_t^2, o sea 256 filas para el caso de 1.6e6 usuarios con rango 16. Ademas el calibrador se indefine (0/0) justo cuando el termino se satisface. La calibracion de la prediccion se resuelve donde corresponde, normalizando la salida de predict_proba.
- Mascaras por entrada con peso de fondo w0 (regimen implicito estilo Hu, Koren y Volinsky), de "estimador" y "escala". Se difieren, no se descartan por incorrectas: el caso documentado como roto es semi supervisado, o sea mascara por fila completa, y ahi el truco del gram sobrevive sin SDDMM. El SDDMM no tiene primitiva en scipy, la via natural corre a 0.56 Gflop/s y cuesta +50% sobre una iteracion de 15 s en la relacion grande. Queda registrada la formula correcta para cuando se necesite, porque la version publicada diverge: con v = (1-w0)*(G_i B) - C hay que hacer N_i += [A]_+ + G_i[B]_- + [v]_+ y D_i += [A]_- + G_i[B]_+ + [v]_-, es decir separar el signo de la matriz cuadratica y no el del producto.
- Laplaciano simetrico L_sym = I - D^{-1/2} W D^{-1/2}, de "estimador" y "escala". No conserva masa: su nucleo es D^{1/2}1, no 1, asi que encoge de forma sistematica las celdas de grado bajo. Medido sobre una grilla hexagonal de 1350 celdas con borde, la razon numerador sobre denominador es 1.026 para grado 6 y 0.661 para grado 2, y a 200 iteraciones las celdas de borde quedan 17.5% mas chicas en norma. Se usa en su lugar diag(W_sym 1) - W_sym, que conserva masa exactamente y mantiene la normalizacion espectral.
- La jerarquia completa FusionData / Relation / Scale / Block / Trace / FitState de "estimador" y "escala". Sube el costo de entrada (27 parametros en el constructor, siete conceptos donde hoy hay dos dicts) y elimina la entrada minima desde un dict de sparse, que es el patron 1 del README y como piensan los ocho notebooks. Se conserva solo lo que resuelve un defecto medido: `Relation` como dataclass de cinco campos y el nombre de relacion como clave.
- La parametrizacion por codificador G_i = Z W, con Z las vistas concatenadas, de "minimo". Elimina por construccion la brecha de estimador, pero empeora en MovieLens (AP 0.601 contra 0.718) porque Z tiene 6548 columnas para 1986 filas. Solo regulariza cuando la suma de n_dst es mucho menor que n_i, condicion que si cumple un caso con muchas entidades y pocos atributos, asi que queda como experimento anotado y no como API.
- Las promesas de paralelismo agresivo de las tres propuestas (5.8x en M @ X, 5.0x en M^T @ G, 15x de ganancia total). Medido, el techo es 3.1x y 2.1x, los productos sparse corren a 1.7 GB/s contra 33.8 GB/s de memcpy porque estan limitados por acceso aleatorio, y el termino dominante a rango 50 no es el sparse sino el elementwise sobre arreglos de (n_t, c_t), 9.3 s de una iteracion de 11.7 s. `n_threads` queda en 1 por default y opt-in; la ganancia real viene de la identidad de traza, del bloque en sitio y de eta=1.0, cerca de 3x total.

## Preguntas abiertas

- ¿Aceptas que `dfmf_sparse` quede congelado como ruta de compatibilidad (los ocho notebooks no se tocan y reproducen report/main.md) y que la API nueva viva en `fit`, o prefieres migrar los notebooks y tener un solo camino? La primera opcion cuesta unas 20 lineas de wrapper y deja dos rutas conviviendo hasta que decidas borrar la vieja.
- El termino del grafo cambia respecto del Laplaciano crudo, que en un caso medido pesaba 130x el termino de datos. Con alpha_graph calibrado, la estructura espacial que resulte puede cambiar respecto de resultados previos, asi que conviene re-correr antes de citar cifras nuevas.
- ¿float32 por default para los factores? Baja el pico a la mitad, pero exige convertir los datos de las relaciones aguas arriba (la conversion dentro de fit duplicaria el nnz) y obliga a acumular la perdida en float64 con un piso de tol cerca de 1e-6. La alternativa es float64 por default y float32 explicito solo en las corridas grandes.
- Cuando no hay ground truth, el criterio de exito de una seleccion de entidades queda implicito. Definir la metrica que decide ahi: similitud de distribuciones marginales contra una referencia, estabilidad del conjunto seleccionado entre semillas, u otra.
- Las mascaras por entrada con peso de fondo (leer los ceros como negativos debiles, no como no observados) quedan fuera de esta iteracion. ¿Hay algun caso tuyo que las necesite ya, o el regimen semi supervisado por fila cubre lo que tienes en mano?