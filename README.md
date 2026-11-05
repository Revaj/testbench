# testbench

Tres recetas de clasificación tabular (solo NumPy, pandas y scikit-learn; Python 3.11, scikit-learn ≥ 1.4)
para evaluarse con una réplica manual del protocolo de TabArena-v0.1 (38 tareas de clasificación).

**Estado: recetas implementadas y probadas solo con datos sintéticos. Todavía no hay resultados con
los datasets reales.** Plan completo, decisiones y bitácora: [`PLAN.md`](PLAN.md).

## Recetas

Todo vive en [`recipe/tabular.py`](recipe/tabular.py); `VARIANT` elige la receta. En
[`submissions/`](submissions/) hay copias de un solo archivo listas para entregar
(`build_model()` → `fit` → `predict_proba`), generadas con `python tools/make_submissions.py`.

| Archivo | VARIANT | Qué es |
|---|---|---|
| `submissions/receta_stacking.py` | `R1` | Stacking: regresión logística L2 + HistGradientBoosting + ExtraTrees, cada una con su preprocesamiento; meta-modelo logístico L2 entrenado solo con predicciones OOF |
| `submissions/receta_stacking_vos.py` | `R3` | R1 + MLP en NumPy con VOS como 4.ª base |
| `submissions/mlp_vos.py` | `MLP-VOS` | MLP en NumPy con VOS (Virtual Outlier Synthesis), sola |
| `submissions/logistic_smote.py` | `LR-SMOTE` | Regresión logística + SMOTE (implementado en NumPy) con corrección de prior |

Variantes de control (mismo archivo, otro `VARIANT`): `R1-AVG` (promedio simple de las 3 bases),
`R2` (R1 + MLP **sin** VOS, control de R3), `MLP` (control de MLP-VOS), `B-LIN`, `B-HGB`, `B-SMALL`.

### Receta de stacking (R1)
- **Codificación compartida, ajustada dentro de cada fold:** columnas categóricas con ≤ 30 valores van
  como categóricas; con > 30, target encoding con validación cruzada (`TargetEncoder`, las filas de
  entrenamiento nunca ven su propia etiqueta) + log de frecuencia, guardando a lo más 2,000 categorías por columna.
- **Base lineal:** imputación por mediana + indicador de faltante, escalado robusto con recorte, one-hot
  (categorías raras agrupadas, desconocidas ignoradas), regresión logística L2 (C=1, o 0.1 si hay pocas filas por columna).
- **Base no lineal:** HistGradientBoosting con categóricas y faltantes nativos, early stopping interno;
  iteraciones acotadas por el presupuesto de pickle; con > 256 columnas numéricas usa las 256 de mayor F de ANOVA (calculada en el fold).
- **Base para datos pequeños:** ExtraTrees de tamaño acotado (64 hojas por árbol, número de árboles según presupuesto).
- **Meta-modelo:** regresión logística L2 sobre los logits OOF de cada base (5 folds si n < 5,000 y no es ancho; 3 si no;
  con más de 30,000 filas, OOF sobre una submuestra estratificada de 30,000). Luego cada base se reentrena con todas las filas.
- **Límites:** presupuesto de 4.4 MB de pickle repartido entre bases (el HGB usa lo que sobra, medido).

### MLP + VOS
MLP de dos capas ocultas (ReLU, Adam, L2) escrita en NumPy, con VOS fiel al Algoritmo 1 de Du et al. (ICLR 2022):
cola por clase en la penúltima capa, gaussianas por clase con covarianza compartida, outlier virtual = la muestra
de menor verosimilitud entre 10,000 (muestreada de forma exacta vía el máximo de χ²), energía libre con pesos
aprendidos y pérdida logística ID vs virtual con β = 0.1. Entrenamiento en dos pasadas idénticas con y sin VOS:
la primera (sin VOS, con validación) fija el número de épocas; la segunda reentrena con todas las filas y activa VOS
desde el 40 % de las épocas, como el paper. Desviación documentada: la cola de cada clase mide min(1000, n_clase).
Los gradientes (incluido el término VOS) se verifican contra diferencias finitas en `tests/gradcheck_vos.py`.

### Logística + SMOTE
Mismo preprocesamiento que la base lineal; SMOTE (k = 5) en el espacio codificado, solo con filas de entrenamiento,
llevando cada clase al tamaño de la mayoritaria (tope de 300,000 filas). Como SMOTE cambia las proporciones de clase,
los logits se corrigen al prior original (+ log π/π'), necesario para log-loss; en AUC binario no cambia el orden.

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
python tests/gradcheck_vos.py          # gradientes de la MLP y de VOS vs diferencias finitas
```

- **Casos raros:** 40/40 correctos (10 variantes × 4 casos: 8 clases con la menor de 2 ejemplos, binario con
  3 positivos, etiquetas de texto, clase de ~0.8 %; en test: categorías nunca vistas, filas todo NaN, columna
  vacía y constante). Probado en Python 3.13 + scikit-learn 1.9.1 y en Python 3.11 + scikit-learn 1.4.2 y 1.6.1.
- **Gradientes:** error relativo máximo 1.4e-7.
- **Peor caso** (sandbox de 2 CPU; el servidor del curso puede ser hasta 5× más lento):

| Caso sintético | R1 fit | R3 fit | MLP-VOS fit | LR-SMOTE fit | Pickle máx. |
|---|---|---|---|---|---|
| 100k × 39, 5 clases, alta cardinalidad | 23.6 s | 43.0 s | 6.5 s | 4.1 s | 2.60 MB |
| 3,750 × 1,776, binario | 27.3 s | 34.8 s | 3.4 s | 1.4 s | 2.62 MB |
| 3,850 × 1,617, 3 clases (una de 1 %) | 26.9 s | 39.3 s | 2.6 s | 2.2 s | 2.28 MB |
| 150k × 10, binario desbalanceado | 4.3 s | 17.8 s | 5.4 s | 0.6 s | 1.09 MB |

(Los tiempos de 3,850 × 1,617 son de antes de pasar los datos anchos a 3 folds; deberían bajar.)

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
