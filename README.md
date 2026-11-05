# testbench

Tres recetas de clasificación tabular (solo NumPy, pandas y scikit-learn; Python 3.11, scikit-learn ≥ 1.4)
para evaluarse con una réplica manual del protocolo de TabArena-v0.1 (38 tareas de clasificación).

**Estado: recetas implementadas y probadas solo con datos sintéticos. Todavía no hay resultados con
los datasets reales.** Plan completo, decisiones y bitácora: [`PLAN.md`](PLAN.md).

## Recetas

Tres archivos autocontenidos en [`recipe/`](recipe/), cada uno listo para entregar tal cual
(`build_model()` → `fit` → `predict_proba`, columnas en el orden de `classes_`):

| Archivo | Qué es |
|---|---|
| [`recipe/stacking.py`](recipe/stacking.py) | Stacking: regresión logística L2 + HistGradientBoosting + ExtraTrees, cada una con su preprocesamiento; meta-modelo logístico L2 entrenado solo con predicciones OOF |
| [`recipe/mlp_vos.py`](recipe/mlp_vos.py) | MLP en NumPy con VOS (Virtual Outlier Synthesis) |
| [`recipe/logistic_smote.py`](recipe/logistic_smote.py) | Regresión logística + SMOTE (implementado en NumPy) con corrección de prior |

Las variantes de comparación se arman en [`bench/variants.py`](bench/variants.py) sin duplicar código:

| Variante | Cómo se construye | Para qué |
|---|---|---|
| `R1` | `stacking.build_model()` | Receta de stacking |
| `R1-AVG` | `stacking.build_model("R1-AVG")` | Mismas 3 bases, promedio simple |
| `R2` | stacking + `mlp_vos.MLPBase(use_vos=False)` como 4.ª base | Control de R3 |
| `R3` | stacking + `mlp_vos.MLPBase(use_vos=True)` como 4.ª base | VOS dentro del stack |
| `MLP` | `mlp_vos.build_model(use_vos=False)` | Control de MLP-VOS (misma red, sin VOS) |
| `MLP-VOS` | `mlp_vos.build_model()` | MLP con VOS sola |
| `LR-SMOTE` | `logistic_smote.build_model()` | Logística + SMOTE |
| `B-LIN`, `B-HGB`, `B-SMALL` | `stacking.build_model("B-...")` | Cada base sola |

### Principio: cada modelo tiene su propio preprocesamiento

Ningún modelo comparte codificación, imputación ni escalado con otro. Cada uno recibe la tabla **cruda** y
ajusta su propio preprocesamiento dentro de su `fit` (y por tanto dentro de cada fold del stacking), con los
umbrales que le convienen. Así lo que necesita uno (por ejemplo, escalar para la logística o la red) no le
estorba a otro (los árboles no lo necesitan), y cada modelo ataca los datos donde es mejor. Lo único común
es código de utilería (alinear probabilidades, detectar qué columnas son numéricas) y las clases de
herramientas (`CategoryCapper`, `CountEncoder`, `FreqOrdinal`), de las que cada modelo crea sus propias
instancias. `tests/check_independence.py` comprueba que en el stacking con MLP ninguna base comparte un
solo objeto de preprocesamiento con otra y que todas reciben columnas crudas.

| | Numéricas | Faltantes | Categóricas pocas | Categóricas muchas | Escalado |
|---|---|---|---|---|---|
| Logística (stacking) | mediana | indicador | one-hot (≤ 30) | target encoding CV + frecuencia (> 30) | robusto + recorte |
| HistGradientBoosting | tal cual | nativos | nativas (254 más frecuentes) | nativas + target encoding CV + frecuencia (> 254) | ninguno |
| ExtraTrees | mediana | indicador | código ordinal por frecuencia | ordinal + frecuencia (> 30), sin target encoding | ninguno |
| MLP (con/sin VOS) | mediana | indicador aparte | one-hot (≤ 20) | target encoding CV + frecuencia (> 20) | cuantiles → normal (robusto si hay demasiadas columnas) |
| Logística + SMOTE-NC | mediana | indicador (categórico para SMOTE) | código → one-hot después de SMOTE (≤ 30) | target encoding CV + frecuencia (> 30) | estandarizado + recorte (SMOTE usa distancias) |

### Receta de stacking (R1)
- **Bases:** las tres primeras filas de la tabla. HGB con early stopping interno, iteraciones acotadas por el
  presupuesto de pickle y, con > 256 columnas numéricas, las 256 de mayor F de ANOVA (calculada con sus filas).
  ExtraTrees de tamaño acotado (64 hojas por árbol, número de árboles según presupuesto).
- **Meta-modelo:** regresión logística L2 sobre los logits OOF de cada base (5 folds si n < 5,000 y no es ancho;
  3 si no; con más de 30,000 filas, OOF sobre una submuestra estratificada de 30,000). En cada fold cada base
  ajusta desde cero su preprocesamiento y su modelo. Luego cada base se reentrena con todas las filas.
- **Límites:** presupuesto de 4.4 MB de pickle repartido entre bases (el HGB usa lo que sobra, medido).

### MLP + VOS
MLP de dos capas ocultas (ReLU, Adam, L2) escrita en NumPy, con VOS fiel al Algoritmo 1 de Du et al. (ICLR 2022):
cola por clase en la penúltima capa, gaussianas por clase con covarianza compartida, outlier virtual = la muestra
de menor verosimilitud entre 10,000 (muestreada de forma exacta vía el máximo de χ²), energía libre con pesos
aprendidos y pérdida logística ID vs virtual con β = 0.1. Entrenamiento en dos pasadas idénticas con y sin VOS:
la primera (sin VOS, con validación) fija el número de épocas; la segunda reentrena con todas las filas y activa VOS
desde el 40 % de las épocas, como el paper. Desviación documentada: la cola de cada clase mide min(1000, n_clase).
Los gradientes (incluido el término VOS) se verifican contra diferencias finitas en `tests/gradcheck_vos.py`.
Como 4.ª base del stacking usa su propio preprocesamiento (no el de la logística).

### Logística + SMOTE-NC
SMOTE-NC (Chawla et al., 2002, §6.1) implementado en NumPy, solo con filas de entrenamiento: las numéricas se
interpolan entre un punto y un vecino de su clase; las categóricas (y los indicadores de faltante) toman la
categoría más frecuente entre sus k = 5 vecinos, así nunca salen categorías "a medias" como pasaría
interpolando un one-hot. Cada clase se lleva al tamaño de la mayoritaria (tope de 300,000 filas). Como SMOTE
cambia las proporciones de clase, los logits se corrigen al prior original (+ log π/π'), necesario para log-loss;
en AUC binario no cambia el orden.

## Datos

Los datos no se guardan en el repo (licencias de origen distintas). Se descargan de OpenML con:

```bash
python bench/fetch_data.py            # los 38 datasets de clasificación de TabArena-v0.1
python bench/fetch_data.py credit-g   # o solo algunos
```

Requiere Python 3.11, numpy y pandas (usa la API REST de OpenML, sin paquetes extra). Descarga 4 datasets
en paralelo (`--jobs=N` para cambiarlo) y, si se corta, al volver a correrlo retoma donde se quedó.
Enlaces directos de cada dataset y de sus particiones, para bajarlos a mano: [`bench/DESCARGAS.md`](bench/DESCARGAS.md). Escribe en
`data/` los datos, el esquema y las particiones oficiales de cada task, y verifica filas, columnas,
clases y tamaños de cada partición contra `bench/TabArena-v0.1_tasks_metadata.csv` (copia sin
modificar de [autogluon/tabarena](https://github.com/autogluon/tabarena), Apache-2.0).
El resultado queda en `data/manifest.json` (con md5 de cada archivo); ese manifest sí se sube al
repo como `bench/data_manifest.json` para que cualquiera compruebe que bajó lo mismo.

## Pruebas hechas (solo datos sintéticos; no son evidencia de desempeño)

```bash
python tests/check_recipes.py edge     # casos raros, todas las variantes
python tests/check_recipes.py smoke    # 80/20 sintético
python tests/check_recipes.py stress   # peor caso: tiempo y tamaño de pickle
python tests/check_recipes.py repro    # dos ajustes iguales -> mismas probabilidades
python tests/check_independence.py     # ninguna base comparte preprocesamiento con otra
python tests/gradcheck_vos.py          # gradientes de la MLP y de VOS vs diferencias finitas
```

- **Casos raros:** 40/40 correctos (10 variantes × 4 casos: 8 clases con la menor de 2 ejemplos, binario con
  3 positivos, etiquetas de texto, clase de ~0.8 %; en test: categorías nunca vistas, filas todo NaN, columna
  vacía y constante). Probado en Python 3.13 + scikit-learn 1.9.1 y en Python 3.11 + scikit-learn 1.4.2 y 1.6.1.
- **Reproducibilidad:** 10/10 variantes dan las mismas probabilidades en dos ajustes, en las tres versiones.
- **Independencia del preprocesamiento:** 0 objetos compartidos entre las 4 bases del stacking con MLP.
- **Gradientes:** error relativo máximo 1.4e-7.
- **Peor caso** (sandbox de 2 CPU; el servidor del curso puede ser hasta 5× más lento):

| Caso sintético | R1 fit | R3 fit | MLP-VOS fit | LR-SMOTE fit | Pickle máx. |
|---|---|---|---|---|---|
| 100k × 39, 5 clases, alta cardinalidad | 14.5 s | 24.0 s | 3.7 s | 8.4 s | 3.18 MB |
| 3,750 × 1,776, binario | 15.1 s | 19.3 s | 1.5 s | 0.5 s | 2.64 MB |
| 3,850 × 1,617, 3 clases (una de 1 %) | 8.5 s | 13.3 s | 1.4 s | 0.8 s | 2.28 MB |
| 150k × 10, binario desbalanceado | 2.6 s | 10.9 s | 3.6 s | 0.3 s | 1.07 MB |

## Medido / pendiente / hipótesis

| | Contenido |
|---|---|
| Medido | Que las recetas corren, respetan límites de tamaño y devuelven probabilidades válidas en casos sintéticos extremos; gradientes correctos |
| Pendiente | Datos reales (`bench/fetch_data.py`), réplica de particiones, evaluación de todas las variantes con el protocolo de TabArena-v0.1 |
| Hipótesis | VOS no mejora métricas dentro de la distribución (el paper es de detección OOD); SMOTE no mejora AUC y empeora log-loss si no se corrige el prior (aquí sí se corrige) |

## Otros

- `legacy/submission_original.py`: receta original (HGB 70 % + LR 30 %).
- `legacy/submission_iter1.py`: primera iteración (mezcla aprendida fuera de muestra).
- `legacy/banco_desarrollo/`: banco PMLB + sintéticos de la primera iteración. **Sesgado; no es evidencia** (ver PLAN.md §2).
