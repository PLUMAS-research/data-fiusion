# Caso de uso: 20 Newsgroups

Evaluación del anclaje de componentes (`supervision`) sobre texto, que es el régimen
para el que TS-NMF fue diseñado.

En MovieLens el anclaje mejoró la legibilidad y empeoró la predicción. Ese caso tiene
varias relaciones compitiendo, y ratings y elenco traen estructura que no se alinea con
el género, así que no dice mucho sobre el mecanismo en sí. Acá hay una sola relación
documento-término y categorías que sí tienen correspondencia con el vocabulario.

## Datos

20 Newsgroups vía `sklearn.datasets.fetch_20newsgroups`, con cabeceras, pies y citas
removidos. Se descarga la primera vez (unos 14 MB) y queda en la caché de sklearn.

18 846 documentos, 20 categorías, vocabulario TF-IDF de 20 000 términos con `min_df=5` y
`max_df=0.5`. Los documentos que quedan sin ningún término tras el filtrado se descartan.

## Diferencia con el paper

TS-NMF factoriza $V \approx (W \circ L) H$, donde cada tópico es una distribución libre
sobre **todos** los términos. `data-fiusion` hace tri-factorización,
$V \approx G_{doc}\, S\, G_{term}^T$, que además agrupa los términos en $c_{term}$
grupos. Es una restricción adicional, así que los valores absolutos no son comparables
con los del paper. Lo comparable es el efecto de subir la tasa de supervisión.

Medido, esa restricción no cuesta nada en este dataset: sin supervisar, la
tri-factorización da ARI 0.185 y el NMF de `sklearn` da 0.184 con el mismo presupuesto
de iteraciones.

## Resultados

Todas las métricas van sobre los documentos **no** supervisados. Los supervisados están
forzados a su categoría, así que evaluarlos sería circular.

| supervisión | ARI | NMI | acierto directo |
|---|---|---|---|
| 0% | 0.185 | 0.317 | 0.036 |
| 5% | 0.217 | 0.346 | 0.381 |
| 10% | 0.227 | 0.360 | 0.425 |
| 20% | 0.250 | 0.385 | 0.461 |
| 50% | 0.292 | 0.449 | 0.534 |

Contra el ajuste sin supervisar: +0.032 de ARI con 5% (2.4 SE), +0.042 con 10% (3.8 SE),
+0.066 con 20% (6.5 SE) y +0.108 con 50% (8.5 SE). El efecto es monótono y significativo
desde 10%, que coincide con lo que reporta el paper.

**Acierto directo** es la fracción de documentos cuyo argmax cae en la componente de su
categoría, sin emparejar nada. Solo tiene sentido cuando hay anclaje: con 0% de
supervisión las componentes no corresponden a categorías, y el 0.036 es el azar de que
coincidan.

El rango del tipo `termino` casi no influye: entre 20 y 400 grupos, el ARI sin supervisar
va de 0.160 a 0.206 y con 20% de supervisión de 0.246 a 0.255.

## Los tópicos que salen

Con 20% de supervisión, leyendo el backbone hacia el espacio de términos:

| componente | términos con más peso |
|---|---|
| `sci.space` | space, nasa, shuttle, launch, orbit, moon |
| `sci.crypt` | key, chip, encryption, clipper, keys, government |
| `talk.politics.mideast` | israel, armenian, jews, armenians, israeli, turkish |
| `comp.sys.ibm.pc.hardware` | drive, scsi, ide, drives, controller, disk |
| `soc.religion.christian` | god, jesus, bible, christ, believe, christian |
| `rec.motorcycles` | bike, just, ve, ride, like, bikes |

Tres salen mal y conviene decirlo: `sci.med` da "geb, dsl, n3jxp, cadre, chastity, pitt",
que es la firma de un usuario frecuente del grupo y no vocabulario médico, un artefacto
conocido de este corpus. `sci.electronics` da términos genéricos de saludo, y
`talk.religion.misc` queda mezclado con vocabulario de hardware.

## Qué concluye esto sobre el mecanismo

El anclaje funciona donde el paper dice que funciona. El resultado contrario en MovieLens
no era del mecanismo sino del caso: cuando hay varias relaciones y su estructura no se
alinea con la etiqueta, forzar las componentes a la etiqueta destruye información útil.
Con una sola relación alineada con las categorías, agrega información y mejora.

Eso da una regla práctica: `supervision` conviene cuando la relación supervisada es la
principal fuente de estructura, y no cuando compite con otras que aportan estructura
distinta.

## Ejecución

```bash
uv run python examples/newsgroups/supervision_texto.py
```

Unos 15 minutos, dominados por los 15 ajustes del barrido de tasas.
