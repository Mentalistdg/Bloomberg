"""Orquestador end-to-end: ejecuta los 5 scripts del pipeline en orden.

Este script importa y ejecuta secuencialmente cada paso del pipeline.
Si algún paso falla (retorna código != 0), el pipeline se detiene
inmediatamente.

Orden de ejecución:
    01_build_features       sqlite → panel mensual base
    02_eda_report           gráficos diagnósticos de datos
    03_build_features_full  panel → 17 features + target
    04_train_and_evaluate   walk-forward CV + métricas + validación
    05_build_app_data       artefactos → JSONs para dashboard

Uso:
    uv run python -m scripts.run_all
"""

from __future__ import annotations

import importlib
import sys
import time

STEPS = [
    ("scripts.01_build_features", "panel mensual base"),
    ("scripts.02_eda_report",     "plots de validación"),
    ("scripts.03_build_features_full", "feature engineering + target"),
    ("scripts.04_train_and_evaluate", "walk-forward + métricas + bootstrap + DM"),
    ("scripts.05_build_app_data", "JSONs para el dashboard"),
]


def main() -> int:
    t0 = time.time()
    for module_name, label in STEPS:
        print("\n" + "=" * 72)
        print(f"  EJECUTANDO: {module_name}  ({label})")
        print("=" * 72)
        m = importlib.import_module(module_name)
        rc = m.main() if hasattr(m, "main") else 0
        if rc not in (None, 0):
            print(f"\n!!! {module_name} retornó {rc}, abortando.")
            return rc
    print("\n" + "=" * 72)
    print(f"  PIPELINE COMPLETO — {time.time()-t0:.1f}s")
    print("=" * 72)
    return 0


if __name__ == "__main__":
    sys.exit(main() or 0)
