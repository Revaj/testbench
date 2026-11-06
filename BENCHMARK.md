# Cómo correr el benchmark

Replica a mano el protocolo de clasificación de TabArena-v0.1: los 38 datasets, sus particiones oficiales de
OpenML, ROC-AUC (binario), log-loss (multiclase) y un Elo tipo Bradley-Terry anclado en el random forest (1000).
Todo se corre desde la raíz del repo.

## 1. Requisitos

- Python 3.11 (también probado en 3.13).
- `numpy`, `pandas`, `scikit-learn` (≥ 1.4) y `matplotlib` (solo para las gráficas de `aggregate.py`).
- Una máquina sin otros procesos pesados. Los tiempos de fit se miden de verdad, y correr dos cosas a la vez
  los infla mucho (vimos HGB hasta 100× más lento por competir por los hilos de OpenMP).

```bash
pip install numpy pandas scikit-learn matplotlib
```

## 2. Bajar y validar los datos

```bash
python bench/fetch_data.py              # los 38 datasets → data/<dataset>/{data.csv.gz, splits.csv.gz, schema.json}
python bench/fetch_data.py credit-g     # o solo algunos
python bench/validate_data.py           # debe terminar en "38/38 datasets válidos"
```

`fetch_data.py` se puede interrumpir: al volver a correrlo retoma donde se quedó. Si OpenML va lento, en
[`bench/DESCARGAS.md`](bench/DESCARGAS.md) están los enlaces directos de cada dataset y sus particiones.
`validate_data.py` revisa filas, columnas, clases, particiones (sin traslape, test cubre cada fila una vez
por repetición) y md5 contra el manifest. No corras el benchmark si no da 38/38.

## 3. Correr variantes

Cada variante es una receta o un control, definido en [`bench/variants.py`](bench/variants.py):

| Variante | Qué es |
|---|---|
| `R1` | Stacking (logística + HGB + ExtraTrees, meta-modelo logístico con OOF) |
| `R1-AVG` | Mismas 3 bases, promedio simple (control del meta-modelo) |
| `R2` / `R3` | Stacking + MLP sin / con VOS como 4.ª base |
| `R4` | Stacking + logística con SMOTE como 4.ª base (demostrativa) |
| `MLP` / `MLP-VOS` | MLP sola sin / con VOS |
| `LR-SMOTE` | Logística + SMOTE-NC |
| `KNN` / `KNN-SMOTE` | kNN sin / con SMOTE |
| `ET` / `ET-BAL` | ExtraTrees sin pesos / con `class_weight="balanced"` |
| `B-LIN`, `B-HGB`, `B-SMALL` | Cada base del stacking sola |
| `R0` | Receta original (HGB 70 % + LR 30 %), referencia |
| `RF` | Random forest por defecto, ancla del Elo |

```bash
# todas las particiones oficiales de los 38 datasets, para las variantes indicadas
python bench/run.py --data data --variants KNN-SMOTE,KNN,ET-BAL,ET --out results/raw

# prueba rápida: solo la primera partición de dos datasets
python bench/run.py --data data --variants KNN-SMOTE --datasets credit-g,splice --max-splits 1 --out /tmp/prueba
```

- Corre los datasets de chico a grande y escribe un JSON por (variante, dataset) en
  `results/raw/<variante>/<dataset>.json`, con ROC-AUC (y por clase en multiclase), log-loss, tiempo de fit,
  tiempo de `predict_proba`, tamaño del pickle y el error, si lo hubo.
- **Es reanudable:** si lo cortas o se cae, vuelve a correr el mismo comando y salta las particiones ya hechas.
- Cada partición tiene un tope de 900 s. Si lo pasa, o si truena, se guarda el error y el dataset cuenta como
  derrota en el Elo.
- Para comparar justo, corre una variante y su control en la misma máquina (por ejemplo `KNN-SMOTE,KNN`).
- Para rehacer una variante desde cero, borra su carpeta `results/raw/<variante>/`.

Todo de una vez (por tandas, reanudable, y al final agrega):

```bash
bash bench/run_all.sh                   # DATA=otra/carpeta bash bench/run_all.sh para cambiar la ruta de datos
```

### Cuánto tarda

Suma de fit + predict en las 594 particiones, en un sandbox de 2 CPU (sin contar carga de datos ni pickle):
R1 ≈ 1.0 h, R2 ≈ 1.7 h, R4 ≈ 1.9 h, R3 ≈ 2.0 h, R0 ≈ 0.6 h; LR-SMOTE, MLP, MLP-VOS, R1-AVG y RF ≈ 0.2–0.3 h
cada una. kNN y ExtraTrees aún no se han corrido; por las pruebas sintéticas deberían quedar en el grupo rápido.
Si tu sesión corta procesos largos, lanza tandas de pocas variantes; como es reanudable, no se pierde nada.

## 4. Agregar resultados

```bash
python bench/aggregate.py --raw results/raw --out results --title "TabArena-v0.1 (clasificación): resultados"
```

Genera en `results/`:

| Archivo | Contenido |
|---|---|
| `summary.md` | Tabla principal (AUC media, fit medio y peor, pickle medio y peor, errores), Elo con IC bootstrap y comparaciones pareadas con IC 95 % |
| `por_dataset.csv` | Por receta y dataset: ROC-AUC, tamaño de pickle y tiempo de fit (promedio de sus particiones) |
| `por_receta.csv` | El promedio de esas tres métricas por receta |
| `resumen_recetas.png` | Gráfica de AUC, fit y pickle por receta |
| `auc_por_dataset.png` | Mapa de calor de AUC por dataset y receta |

Solo entran las variantes que tienen resultados en `results/raw/`, así que se puede agregar con corridas
parciales. Las comparaciones pareadas (por ejemplo `KNN-SMOTE` vs `KNN`, `ET-BAL` vs `ET`) se calculan solo
sobre los datasets que ambas variantes completaron.

## 5. Antes de correr: pruebas rápidas (datos sintéticos)

Tardan unos minutos y detectan errores sin tocar los datos reales:

```bash
python tests/check_recipes.py edge                          # casos raros, todas las variantes
python tests/check_recipes.py repro                         # dos ajustes iguales → mismas probabilidades
python tests/check_recipes.py stress KNN-SMOTE ET-BAL       # peor caso: tiempo de fit y tamaño de pickle
STRESS_ONLY="150k" python tests/check_recipes.py stress R1  # solo un caso de peor caso
```

## 6. Cómo leer los resultados sin sesgo

- Ningún hiperparámetro de las recetas se eligió con estos datos. Si cambias una receta después de ver
  resultados, ya no es una evaluación limpia: dilo en la bitácora ([`PLAN.md`](PLAN.md)).
- Una diferencia cuenta solo si su IC 95 % pareado excluye 0. Mira también el gana–pierde por dataset.
- El Elo es una réplica propia y solo entre las variantes que corriste; no es comparable con el leaderboard
  oficial de TabArena.
- Los tiempos dependen de la máquina. El servidor del curso puede ser ~5× más lento, así que se busca un fit
  ≤ ~60 s por dataset aquí, con el límite real de 5 min.
