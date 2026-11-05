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

## Medido / pendiente / hipótesis

| | Contenido |
|---|---|
| Medido | Nada todavía con el protocolo TabArena-v0.1 |
| Pendiente | Datos, réplica de particiones, receta la guía del curso, MLP con/sin VOS, evaluación |
| Hipótesis | VOS no mejora métricas dentro de la distribución (el paper es de detección OOD) |
