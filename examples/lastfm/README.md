# Caso de uso: Last.fm (HetRec 2011)

Evaluación de las familias mezcladas: una relación de conteos sobredispersos
(reproducciones artista-usuario) junto a una relación de etiquetas (tags por
artista), compartiendo el factor de artistas.

## Datos

Last.fm 2k del workshop HetRec 2011, distribuido por GroupLens. Se descarga la
primera vez (unos 2.5 MB) a `data/` o al directorio de la variable de entorno
`LASTFM_DIR`. Uso no comercial según el README del dataset, que pide citar a
Last.fm y al workshop:

- Cantador, I., Brusilovsky, P. y Kuflik, T. (2011). Second Workshop on
  Information Heterogeneity and Fusion in Recommender Systems (HetRec2011).
  En Proceedings of the 5th ACM Conference on Recommender Systems (RecSys 2011).
  (Referencia tomada del README del dataset; verificar antes de citarla en un
  paper.)

1892 usuarios, 17 632 artistas, 92 834 registros de escucha. Los conteos de
reproducción van de 1 a 352 698, con índice de dispersión 18 883: el régimen de
sobredispersión extrema donde una likelihood de conteo debería importar.

Las etiquetas son los 100 tags más usados; un artista lleva un tag si al menos 2
usuarios distintos se lo aplicaron, lo que deja 5531 artistas etiquetados. La red
de amistades del dataset queda para una fase posterior (el suavizado por grafo no
está soportado junto a relaciones Poisson todavía).

## Protocolo

Los artistas etiquetados se parten 60/20/20. La relación de etiquetas entra al
ajuste con máscara de filas sobre el entrenamiento; validación y prueba se
predicen con `predict_proba` desde el factor compartido, así que la etiqueta de
los artistas evaluados no puede filtrarse. Cada brazo elige sus hiperparámetros
en validación y se evalúa en prueba con 4 semillas apareadas.

- **Brazo A (monofamilia)**: escuchas gaussianas con la mejor transformación de
  {crudo, sqrt, log1p} y peso de etiquetas barrido.
- **Brazo B (mezclado)**: escuchas `family="poisson"` sobre los conteos crudos y
  el mismo barrido de peso.

Criterio de éxito, declarado antes de correr: B le gana a A en AP media de prueba
por 2 errores estándar de la diferencia apareada. AP es la precisión promedio por
artista sobre los 100 tags, promediada sobre los artistas evaluados.

## Resultados

| | AP en prueba |
|---|---|
| baseline de popularidad | 0.204 |
| brazo B, mezclado (Poisson + gaussiana, peso 100) | 0.265 +- 0.012 |
| brazo A, monofamilia (log1p, peso 100) | 0.314 +- 0.005 |

Diferencia apareada B - A: -0.050 +- 0.017 (-2.9 SE). El criterio declarado no se
cumplió: el brazo mezclado pierde contra la monofamilia bien transformada.

Ambos brazos superan con claridad al baseline, así que el factor compartido sí
transporta señal de las escuchas a los tags; la pregunta era qué pérdida la
transporta mejor, y la respuesta es la cuadrática sobre log1p. El mecanismo
probable: la KL es linealmente sensible a la masa, así que el ajuste se lo llevan
los conteos gigantes de los artistas y usuarios de cabeza (hasta 352 mil
reproducciones), mientras que log1p declara que la magnitud satura, y la señal que
predice tags vive en ese régimen saturado. Es consistente con el resultado de 20
Newsgroups (la likelihood compra lo que la transformación compra, como máximo) y
con la práctica estándar en retroalimentación implícita, donde los conteos de
reproducción se comprimen con log antes de factorizar.

Dos salvedades de protocolo. La grilla de pesos se extendió tras una primera
corrida de validación (la nota está en el script); esa corrida además destapó un
defecto real de la librería, corregido antes de la corrida final: sin calibrar el
gradiente KL por la desviación nula, la relación de conteos aplastaba a la
gaussiana y el peso entre familias no tenía ningún efecto. Y el brazo A eligió el
borde de la grilla (peso 100) con la curva de validación ya casi plana (0.318 a
0.323 entre 30 y 100), así que el margen restante por ese lado es chico.

## Ejecución

```bash
uv run python examples/lastfm/comparacion.py
```
