# Plan: receta de stacking + MLP con VOS, evaluadas con protocolo TabArena-v0.1 (clasificación)

> Documento maestro. Si una sesión nueva retoma este trabajo: leer este archivo completo,
> revisar la **Bitácora** al final y seguir con el primer paso sin marcar. No re-decidir lo que
> ya está en **Decisiones tomadas**.

Fecha del plan: 2026-11-05. Autor: Javier Rizo + Claude.

---

## 0. Objetivo y alcance

- Construir y evaluar, de forma reproducible, dos soluciones:
  1. **Receta de stacking**: stacking de modelos base diversos con meta-modelo sencillo entrenado solo con predicciones OOF.
  2. **Receta de stacking + MLP con VOS**: la misma receta con un cuarto modelo base (MLP en NumPy entrenado con Virtual Outlier Synthesis), comparado contra el mismo MLP **sin** VOS.
- Evaluación: **réplica manual del protocolo de TabArena-v0.1, solo tareas de clasificación**.
- Entregable: **un repositorio comprobable con README**. No se sube nada a TabArena ni a la plataforma del curso como parte de este plan.

## 1. Decisiones tomadas (no re-discutir)

| Tema | Decisión |
|---|---|
| Benchmark | TabArena-v0.1, solo clasificación (binaria y multiclase) |
| Protocolo | Replicado a mano y documentado (sin paquetes `tabarena`/`autogluon`) |
| Datos | Por conseguir (ver §4). OpenML da 403 desde el sandbox de Claude; GitHub sí funciona |
| VOS | Sí: MLP propio en NumPy + VOS fiel al Algoritmo 1 del paper, como base aislada |
| Repo | Lo crea Javier (GitHub); Claude sube el contenido |
| AUC por dataset de entregas previas del curso | No se usará |
| Receta | Solo NumPy, pandas, scikit-learn. Python 3.11. `predict_proba`, pickle sin `lambda` |
| Límites de diseño (del curso) | ≤ 5 MB de pickle; ≤ 5 min por dataset en servidor; **meta: ≤ ~60 s por dataset en laptop** (servidor hasta ~5× más lento) |

## 2. Lo que se aprendió de la versión anterior (y se corrige)

**Bugs confirmados en `submission.py` original** (reproducidos):
1. La rama LR daba AUC = 0.500 en datasets 100 % categóricos: descartaba las categóricas y quedaba solo `__num_missing` (constante); el "blindaje anti-vacío" nunca se disparaba.
2. El peso fijo 70/30 destruía al mejor modelo cuando una rama dominaba (Hill_Valley: LR 0.926, HGB 0.567, mezcla 0.619).
3. Al reentrenar al 100 %, una columna con ~250 categorías podía pasar de baja a alta cardinalidad y llegar al HGB como hash numérico.
4. Una columna 100 % vacía en train tumba el binning de HGB.
5. `XtV` (15 % de validación) se calculaba y no se usaba. El docstring dice "soft voting en multiclase" pero el código usa media geométrica siempre.

**Lo que hice mal en la iteración anterior (no repetir):**
- No seguí la guía del curso: faltó el modelo base para datasets pequeños, las bases compartían preprocesamiento y selección de variables, el meta-modelo era excepción y no regla, no comparé contra el **promedio de probabilidades**, la LR no tenía indicadores de faltante.
- Banco de prueba sesgado: diseño final elegido mirando el test de los mismos 27 datasets; una sola partición 75/25; datasets elegidos y transformados por mí (categóricas por umbral, faltantes MCAR inyectados, 4 datasets con AUC = 1.0); media dominada por un dataset; métrica multiclase asumida; tiempos en sandbox ruidoso.
- **Ese banco (PMLB + sintéticos) queda solo como banco de desarrollo/humo, nunca como evidencia.**

## 3. Protocolo TabArena-v0.1 (según el paper, arXiv 2506.16791) y cómo se replica

| Elemento en el paper | Réplica manual |
|---|---|
| 51 datasets curados (500–250,000 filas), de OpenML | Solo el subconjunto de clasificación. Lista de task IDs tomada de la metadata pública de TabArena; número exacto **por confirmar** |
| < 2,500 filas: 10 repeticiones × 3-fold; resto: 3 repeticiones × 3-fold; estratificado | Igual. Si se consiguen los splits oficiales de OpenML se usan; si no, `RepeatedStratifiedKFold` con semillas fijas, y se documenta la desviación |
| Métrica: ROC-AUC (binario), log-loss (multiclase) | Igual. Además se reporta AUC OvR por clase en multiclase, como diagnóstico |
| Agregado: Elo (RF por defecto = 1000) con IC por bootstrap (200 rondas) | Elo por comparaciones pareadas implementado a mano + rango medio + tasa de victorias, con IC por bootstrap sobre datasets. Se documenta que no es el código oficial |
| Ajuste de hiperparámetros con validación interna 8-fold | No se ajustan hiperparámetros con los datos de evaluación. Lo que se decida, se decide en el banco de desarrollo o con validación interna dentro del fit |
| Límite 1 h por configuración, hardware recomendado 8 núcleos | Se usan los límites más estrictos del curso (§1). Se reporta el hardware real |

## 4. Datos (paso pendiente)

- Problema: `openml.org` responde 403 desde el sandbox de Claude. GitHub (`raw.githubusercontent.com`, `media.githubusercontent.com` para LFS) sí responde.
- Opciones, en orden:
  1. Encontrar un espejo en GitHub de los datasets de TabArena (o de los de OpenML que la componen).
  2. Javier descarga desde su máquina con un script `bench/fetch_data.py` (que Claude escribe) y sube los archivos (parquet/csv) al repo o al chat.
  3. Javier corre todo el benchmark en su máquina con el mismo script.
- Antes de evaluar: verificar por dataset filas, columnas, número de clases, tipos y huella (md5) contra la metadata.

## 5. Variantes a evaluar (pre-registradas)

| ID | Qué es | Para qué |
|---|---|---|
| RF | `RandomForestClassifier` por defecto | Ancla del Elo (como en el paper) |
| R0 | `submission.py` original de Javier | Línea base |
| B-LIN | Solo la base lineal | Diagnóstico por rama |
| B-HGB | Solo la base HGB | Diagnóstico por rama |
| B-SMALL | Solo la base para datasets pequeños | Diagnóstico por rama |
| R1-AVG | Las 3 bases, **promedio aritmético** de probabilidades | Comparación recomendada por la guía del curso |
| R1 | Las 3 bases + meta-modelo LR sobre predicciones OOF | **Receta de stacking** |
| MLP | MLP NumPy **sin** VOS (solo) | Control del experimento VOS |
| MLP-VOS | El mismo MLP **con** VOS (solo) | Experimento VOS |
| R2 | R1 + MLP sin VOS como 4.ª base | Control |
| R3 | R1 + MLP-VOS como 4.ª base | Experimento VOS dentro del stack |
| LR-SMOTE | Regresión logística + SMOTE (NumPy) con corrección de prior | Tercera receta, pedida por Javier el 2026-11-05 |
| R4 | R1 + logística/SMOTE-NC como 4.ª rama | Demostrativa: cómo se encapsula una rama con SMOTE en el stacking |

La comparación que responde "¿sirve VOS?" es **MLP-VOS vs MLP** y **R3 vs R2**: mismas semillas, mismos folds, mismo código salvo el término VOS.

## 6. Diseño de la Receta de stacking (R1)

Cada base tiene **su propio preprocesamiento**:

1. **Lineal** (relaciones lineales, datasets anchos o dispersos): regresión logística L2.
   - Numéricas: imputación por mediana **+ indicador de faltante**, escalado robusto.
   - Categóricas de baja cardinalidad: one-hot con categoría "rara" y manejo de desconocidas.
   - Alta cardinalidad: target encoding OOF + frecuencia.
   - Sin la selección por información mutua del HGB.
2. **No lineal** (interacciones, datos mixtos, faltantes): HistGradientBoosting con categóricas y NaN nativos, target encoding OOF para alta cardinalidad, early stopping interno. Incluye los arreglos de los bugs 3 y 4.
3. **Datasets pequeños** (donde lo complejo sobreajusta): candidato a fijar en el banco de desarrollo **antes** de evaluar: ExtraTrees de tamaño acotado (hojas mínimas grandes) o HGB poco profundo y muy regularizado. Imputación con centinela (no depender del soporte de NaN de versiones recientes).
4. **Meta-modelo**: regresión logística con L2 fuerte sobre los logits OOF de cada base.
   - OOF con K-fold estratificado. K según tamaño, regla fija: K=5 si n < 5,000; K=3 si n < 50,000; K=2 en adelante (sujeto a la medición de tiempo).
   - Dentro de cada fold se ajustan: preprocesamiento, target encoder, selección de variables, early stopping. Nada ve las filas que predice.
   - Clases raras: si una clase tiene menos ejemplos que K, se baja K o se usa KFold, y siempre se alinea con `classes_`.
   - Objetivo del meta: log-loss (sirve para AUC y para log-loss). **Sin** `class_weight="balanced"` en el meta, para no descalibrar el log-loss multiclase.
5. **Presupuesto de 5 MB** repartido desde el inicio entre las bases (p. ej. HGB ≤ 2.5 MB, pequeña ≤ 1 MB, LR + prep + meta ≤ 0.5 MB, MLP ≤ 0.5 MB), con margen.

## 7. Diseño de MLP con VOS (fiel al paper: Du et al., ICLR 2022, Algoritmo 1)

- Qué es VOS: método de **detección fuera de distribución (OOD)**. Ajusta una gaussiana por clase en el espacio de la **penúltima capa** de una red, muestrea **outliers virtuales** en las zonas de baja verosimilitud y añade una **pérdida de regularización de incertidumbre** (energía) al entrenar.
- Por qué no se aplica a HGB/LR: no tienen espacio latente ni pérdida diferenciable a la que añadir el término. Por eso VOS solo entra como MLP propio.
- `MLPClassifier` de sklearn **no sirve**: no permite añadir una pérdida propia. Se implementa en NumPy (ya hubo un prototipo así en el notebook de segmentación de clientes, act10).
- Algoritmo (por iteración): actualizar la cola por clase con las representaciones de la penúltima capa → estimar media por clase y covarianza compartida → muestrear candidatos y quedarse con los de menor verosimilitud → pérdida = entropía cruzada + β · pérdida de incertidumbre (logística sobre la energía, ID vs virtual) → retropropagación.
- Hiperparámetros de partida (los del prototipo anterior; se fijan **antes** de evaluar y no se ajustan con los datos de evaluación): `n_hidden=32, beta=0.1, start_epoch=20, queue=300, pool=1000, t=5, lr=1e-2, epochs=60, batch=256`.
- Para respetar ≤ 60 s en laptop: submuestreo de filas para entrenar el MLP en datasets grandes y tope de épocas; regla fija antes de medir.
- Advertencia: TabArena no mide OOD, y el paper de VOS no afirma mejorar el desempeño dentro de la distribución. **Hipótesis: VOS no cambia el score o lo empeora levemente.** Si sale así, se reporta como resultado negativo. No se presentará ninguna adaptación distinta como "VOS".

## 8. Reglas de decisión (escritas antes de correr)

- **R1 reemplaza a R0** si gana el Elo con IC 95 % que no se traslapa, o si la diferencia pareada media (por dataset) tiene IC 95 % por bootstrap que excluye 0, **y** no falla en ningún dataset ni rompe límites.
- **R1 vs R1-AVG**: si el meta no supera al promedio con la misma regla, se reporta y se recomienda el promedio (más simple).
- **VOS se recomienda** solo si R3 > R2 (y MLP-VOS > MLP) con IC que excluya 0. Si no, resultado negativo documentado.
- **Receta "lista"** solo si: 0 errores en todos los datasets y en las pruebas de borde; pickle ≤ 5 MB en todos; fit ≤ ~60 s en laptop en el peor caso medido (100k filas y ~1,800 columnas).
- Prohibido: elegir arquitectura, variables, pesos o hiperparámetros mirando los resultados de evaluación. Si se cambia algo después de ver resultados, se marca como "post-hoc" en el README.

## 9. Pruebas de robustez (antes de cualquier conclusión)

- Peor caso: ~100k filas; ~1,800 columnas; categóricas de alta cardinalidad; multiclase hasta 8 clases; clase con 1–3 ejemplos; categorías desconocidas en test; test con filas todo NaN; columna vacía y constante; booleanas.
- Compatibilidad: Python 3.11 con scikit-learn 1.4 y 1.6 (ya se probó con uv en la iteración anterior).
- Comprobar en cada caso: forma de `predict_proba` = (n, k), suma 1, finitos, alineación con `classes_`, pickle ida y vuelta idéntico.

## 10. Estructura del repo

```
README.md            # qué es, cómo reproducir, resultados: medido / pendiente / hipótesis
PLAN.md              # este documento
requirements.txt     # versiones fijadas (numpy, pandas, scikit-learn; runner aparte)
recipe/
  stacking.py        # receta de stacking (R1, R1-AVG, bases solas) — autocontenido
  mlp_vos.py         # MLP NumPy con/sin VOS — autocontenido
  logistic_smote.py  # logística + SMOTE — autocontenido
bench/
  fetch_data.py      # descarga + verificación (md5, forma, clases)
  splits.py          # réplica de las particiones de TabArena-v0.1
  run.py             # corre una variante en todos los datasets/folds; guarda JSON crudo
  aggregate.py       # métricas por dataset/clase, Elo, rangos, IC bootstrap
  variants.py        # todas las variantes (R2/R3 = stacking + MLPBase externa)
  stress.py          # 100k filas, 1,800 columnas, tiempos y tamaño de pickle
  edge.py            # casos de borde
results/
  raw/               # JSON por variante, dataset y fold
  summary.md         # tablas finales
legacy/
  submission_original.py
  submission_iter1.py        # la versión de la iteración anterior (mezcla aprendida)
  banco_desarrollo/          # banco PMLB + sintéticos: SOLO desarrollo, no evidencia
```

## 11. Pasos (checklist)

- [x] 1. Javier crea el repo en GitHub y pasa owner/nombre. Claude lo conecta y sube el esqueleto + este PLAN.md + `legacy/`.
- [~] 2. (Lista: hecha → `bench/tabarena_v0.1_clasificacion.csv`. Datos: pendiente.) Conseguir la lista de tareas de clasificación de TabArena-v0.1 (metadata pública) y los datos (§4). Verificar huellas.
- [ ] 3. `splits.py` + `run.py` + `aggregate.py`; probar con RF y R0 en 2–3 datasets.
- [x] 4. Implementar R1 (3 bases + meta OOF) y R1-AVG. Elegir la base "pequeña" **solo** con el banco de desarrollo.
- [x] 5. Implementar `mlp_vos.py` (con y sin VOS). Prueba unitaria: pérdida baja, muestreo de outliers en baja verosimilitud, `predict_proba` válido.
- [x] 6. Pruebas de borde y peor caso (§9). Medir tiempos y pickle. Ajustar presupuestos si algo rompe límites (decisión documentada).
- [ ] 7. Correr todas las variantes de §5 con el protocolo de §3.
- [ ] 8. `aggregate.py` → `results/summary.md`; aplicar reglas de §8.
- [ ] 9. README final: medido / pendiente / hipótesis; hardware; desviaciones del protocolo oficial.

## 12. Bitácora

- 2026-11-05: Iteración anterior hecha (mezcla aprendida con holdout/OOF; banco PMLB sesgado). Resultados en `banco_de_prueba.zip`, enviado a Javier: va a `legacy/banco_desarrollo/`. La receta de esa iteración (`submission.py` entregado ese día) va a `legacy/submission_iter1.py`.
- 2026-11-05: Plan aprobado por Javier: MLP-VOS sí; TabArena-v0.1 clasificación; protocolo replicado a mano; datos por buscar; repo lo crea Javier.
- 2026-11-05: Repo `Revaj/testbench` creado; esqueleto subido (paso 1). Siguiente: paso 2 (lista de tareas de clasificación de TabArena-v0.1 y datos).
- 2026-11-05: Lista oficial obtenida de github.com/autogluon/tabarena (Apache-2.0, commit 1fa0a62), archivo `packages/tabarena/src/tabarena/benchmark/task/metadata/sources/data/TabArena-v0.1_tasks_metadata.csv`. Clasificación = **38 datasets: 30 binarios (métrica roc_auc) y 8 multiclase (log_loss)**, task IDs de OpenML 363613–363712; <2,500 filas → 30 splits (10×3), resto → 9 (3×3). Copia resumida en `bench/tabarena_v0.1_clasificacion.csv`. Coincide exactamente con los 38 del reto del curso (30 bin + 8 multi con 3/5/8 clases, 4–1,776 columnas): **hipótesis no verificada** de que son los mismos datasets; la candidata para dataset_37 (3 clases, AUC 0.49) sería hiva_agnostic (3 clases, 1,617 columnas, clase minoritaria 1.1 %). Los splits oficiales son los de cada task de OpenML: se usarán esos en vez de replicarlos. Siguiente: descargar datos+splits desde la máquina de Javier (OpenML da 403 desde el sandbox).
- 2026-11-05: `bench/fetch_data.py` escrito (solo urllib + pandas; parser ARFF propio probado con datos simulados, no con OpenML real porque el sandbox no llega). Siguiente: Javier lo corre, revisa que salga 38/38 verificados y sube `data/manifest.json` como `bench/data_manifest.json` (los datos no van al repo).

- 2026-11-05: Recetas implementadas en `recipe/tabular.py` sin mirar ningún dataset (Javier pidió hacerlas "sin datos",
  ignorando lo anterior): R1, R1-AVG, R2, R3, MLP, MLP-VOS, LR-SMOTE y bases solas. Cambios respecto a §6/§7, decididos por
  regla y no con datos: base pequeña = ExtraTrees (no se usó el banco de desarrollo para elegirla); target encoding con
  `sklearn.preprocessing.TargetEncoder` (cross-fitting); OOF con >30k filas sobre submuestra de 30k; HGB limita a 256
  columnas numéricas (F de ANOVA en el fold) porque con 1,776 columnas el binning pasaba de 5 MB y el fit de 60 s; MLP en
  dos pasadas (la primera fija épocas, la segunda reentrena con VOS desde el 40 % — sin esto el early stopping guardaba
  pesos de antes de VOS y MLP-VOS salía idéntica a MLP). Pruebas sintéticas: bordes 40/40 en sklearn 1.4.2/1.6.1/1.9.1,
  gradcheck VOS OK, peor caso ≤ 43 s y ≤ 2.6 MB (tabla en README). Entregables de un archivo en `submissions/`.
  Siguiente: Javier corre `bench/fetch_data.py`; luego pasos 3, 7, 8, 9.

- 2026-11-06: Encontrado y corregido: OpenML guarda 10 repeticiones para todas las tareas, pero TabArena-v0.1 usa
  solo 3 en 26 de los 38 datasets (los de ≥ 2,500 filas). `fetch_data.py` ahora filtra a las repeticiones de TabArena
  (antes la verificación habría marcado DIF y la evaluación habría corrido 3× más particiones). También descarga en
  paralelo (`--jobs`, 4 por defecto). Lista de enlaces directos en `bench/DESCARGAS.md` / `bench/descargas.csv`, con
  `dataset_id` tomado de `curated_tabarena_dataset_metadata.csv` del repo oficial.

- 2026-11-06: A pedido de Javier, `recipe/tabular.py` se dividió en tres recetas autocontenidas (`recipe/stacking.py`,
  `recipe/mlp_vos.py`, `recipe/logistic_smote.py`) y se borraron `tabular.py`, `submissions/` y `tools/`. El código se
  copió tal cual: las 10 variantes predicen lo mismo que antes (diferencia máxima 1.3e-13, redondeo). R2/R3 = stacking con
  `extra_bases={"mlp": MLPBase(...)}`, armadas en `bench/variants.py`. Bordes 40/40 en sklearn 1.4.2/1.6.1/1.9.1; gradcheck OK;
  cada archivo corre solo en una carpeta vacía y su pickle se recarga en un proceso nuevo.

- 2026-11-06: A pedido de Javier ("no mezcles el preprocesamiento de cada parte"): se eliminó el `Featurizer`
  compartido y `_linear_ct`. Ahora cada modelo (logística, HGB, ExtraTrees, MLP, logística+SMOTE) recibe la tabla
  cruda y ajusta su propio preprocesamiento, con sus propios umbrales (tabla en README). La MLP como 4.ª base del
  stacking también usa el suyo. SMOTE pasó a SMOTE-NC (categóricas por moda de vecinos, no interpolando one-hot).
  `tests/check_independence.py` verifica 0 objetos de preprocesamiento compartidos entre bases.
  Bug encontrado y corregido: en scikit-learn 1.9 `TargetEncoder(shuffle=..., random_state=...)` está obsoleto y
  baraja al azar → resultados no reproducibles (la misma receta daba 18 o 22 épocas). Ahora se pasa un
  `StratifiedKFold` con semilla; `tests/check_recipes.py repro` verifica que dos ajustes dan lo mismo.

- 2026-11-06: Variante demostrativa R4 = stacking + logística/SMOTE-NC como 4.ª rama (`logistic_smote.LRSmoteBase`,
  pedida por Javier). Su preprocesamiento y su SMOTE quedan dentro de su fit (en el stacking: solo filas de
  entrenamiento de cada fold). Bordes 4/4, reproducible, 0 objetos compartidos, peor caso 31.6 s y 3.07 MB.

- 2026-11-06: Pasada rápida (1.ª partición × 38 datasets) antes del protocolo completo. Hallazgos y ajustes (todos por
  límites o errores, ninguno por mirar el AUC):
  1. Pickle de R3 en kddcup09_appetency = 5.40 MB (> 5 MB). Causa: cada base guardaba hasta 5,000 conteos por columna
     como Series de texto (~0.84 MB por base). `CountEncoder` ahora guarda hash de 64 bits + conteo, solo categorías con
     ≥ 2 filas, ≤ 20,000 en total. Además el stacking mide el pickle real al final: si pasa de 4.6 MB reajusta el HGB con
     menos iteraciones y, si no basta, recorta árboles del ExtraTrees. kddcup R3 quedó en 3.08 MB.
  2. Con el nuevo conteo, el estandarizado de target encoding / frecuencias producía valores de hasta 155 → la MLP se
     volvía 15× más lenta (pesos subnormales). Ahora esas columnas se recortan a ±5 en logística, MLP y SMOTE, y la MLP
     pone a 0 pesos < 1e-30 al final de cada época.
  3. Tiempos: correr otro proceso pesado a la vez vuelve al HGB (OpenMP) hasta 100× más lento. El benchmark corre solo,
     sin nada en paralelo. Los resultados de la pasada rápida se descartaron.
  4. R0 (receta original) falla en 5 datasets con "window shape cannot be larger than input array shape" (bug conocido,
     §2 punto 4). Se deja tal cual: es la referencia.
  5. hiva_agnostic: todas las recetas ≈ 0.5 de AUC e incluso un Bernoulli NB / logística simples con CV dan 0.44–0.46
     (activo vs. inactivo). Sin señal aprendible en estos datos tal como se descargaron; posible problema del dataset o
     de la descarga, sin verificar. Se reporta aparte.

- 2026-11-06: Con la tanda 1 completa (R1, MLP-VOS, LR-SMOTE × 594 particiones, 0 errores), hiva_agnostic quedó como
  peor caso de tiempo de R1: 89 s de fit medio (103 s máx.), por encima de la guía de ~60 s. Causa: 1,617 columnas
  categóricas. Ajuste por límite de tiempo (no por AUC): el HGB ahora usa a lo más 256 columnas EN TOTAL (antes el tope
  era solo para numéricas), repartiendo el cupo y eligiendo categóricas por información mutua (en el fold). Solo cambia
  hiva_agnostic (único dataset con > 256 columnas que incluyen categóricas); Bioresponse da exactamente el mismo AUC.
  hiva baja a ~64 s, todavía al límite: se reporta como riesgo. Se probó vectorizar los codificadores (idénticos en los
  38 datasets) pero era más lento en datasets largos; se descartó. Se re-corre R1 solo en hiva_agnostic.

## Fuentes

- TabArena: A Living Benchmark for Machine Learning on Tabular Data — https://arxiv.org/abs/2506.16791
- VOS: Learning What You Don't Know by Virtual Outlier Synthesis (Du et al., ICLR 2022) — https://arxiv.org/abs/2202.01197
