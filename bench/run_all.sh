#!/usr/bin/env bash
# Corre el benchmark completo por tandas, en orden. Es reanudable: si se interrumpe, vuelve a correrlo
# y salta las particiones ya hechas. No corras nada pesado en paralelo: altera los tiempos.
set -e
cd "$(dirname "$0")/.."
DATA=${DATA:-data}
for V in "R1,MLP-VOS,LR-SMOTE" "R3" "R0,RF,MLP,R1-AVG" "R2" "R4"; do
  python3 bench/run.py --data "$DATA" --variants "$V" --out results/raw
done
python3 bench/aggregate.py --raw results/raw --out results --title "TabArena-v0.1 (clasificación): resultados"
