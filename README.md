# testbench

Receta de clasificación tabular (stacking según la guía del curso) y experimento con
MLP + VOS (Virtual Outlier Synthesis), evaluados con una réplica manual del protocolo de
TabArena-v0.1 (solo clasificación).

**Estado: en construcción. Todavía no hay resultados medidos con este protocolo.**

- Plan completo, decisiones y checklist: [`PLAN.md`](PLAN.md)
- `legacy/submission_original.py`: receta original (HGB 70 % + LR 30 %).
- `legacy/submission_iter1.py`: primera iteración (mezcla aprendida fuera de muestra).
- `legacy/banco_desarrollo/`: banco PMLB + sintéticos de la primera iteración.
  **Sesgado; sirve solo para desarrollo, no como evidencia** (ver PLAN.md §2).

## Datos

Los datos no se guardan en el repo (licencias de origen distintas). Se descargan de OpenML con:

```bash
python bench/fetch_data.py            # los 38 datasets de clasificación de TabArena-v0.1
python bench/fetch_data.py credit-g   # o solo algunos
```

Requiere Python 3.11, numpy y pandas (usa la API REST de OpenML, sin paquetes extra). Escribe en
`data/` los datos, el esquema y las particiones oficiales de cada task, y verifica filas, columnas,
clases y tamaños de cada partición contra `bench/TabArena-v0.1_tasks_metadata.csv` (copia sin
modificar de [autogluon/tabarena](https://github.com/autogluon/tabarena), Apache-2.0).
El resultado queda en `data/manifest.json` (con md5 de cada archivo); ese manifest sí se sube al
repo como `bench/data_manifest.json` para que cualquiera compruebe que bajó lo mismo.

## Medido / pendiente / hipótesis

| | Contenido |
|---|---|
| Medido | Nada todavía con el protocolo TabArena-v0.1 |
| Pendiente | Datos, réplica de particiones, receta la guía del curso, MLP con/sin VOS, evaluación |
| Hipótesis | VOS no mejora métricas dentro de la distribución (el paper es de detección OOD) |
