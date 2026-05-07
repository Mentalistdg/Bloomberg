"""Script 03/05 — Ingeniería de features y creación de targets multi-horizonte.

Toma el panel mensual base (output del script 01) y le aplica toda la
ingeniería de features definida en src/features.py, produciendo un
dataset con 23 features predictoras y targets forward a 3, 6 y 12 meses.

Las features se agrupan en:
  - CORE (10):    retornos trailing + riesgo + 3 intra-mes
  - EXTENDED (4): fee + concentración + flags de disponibilidad
  - RANK (9):     percentiles cross-seccionales por mes

Los targets son retornos compuestos forward a 3, 6 y 12 meses, cada uno
convertido a percentil cross-seccional. Son las ÚNICAS variables forward.

Input:  artifacts/panel_raw.parquet
Output: artifacts/panel_features.parquet

Uso:
    python -m scripts.03_build_features_full

Funciones utilizadas de src/:
    src.features  →  build_features, FEATURE_COLS
"""

from __future__ import annotations

import sys
import time

import pandas as pd

from src.features import FEATURE_COLS, build_features
from src.paths import ARTIFACTS_DIR

HORIZONS = [3, 6, 12]


def main() -> None:
    t0 = time.time()

    # Paso 1: cargar panel mensual base (output del script 01)
    panel = pd.read_parquet(ARTIFACTS_DIR / "panel_raw.parquet")
    print(f">>> panel_raw: {len(panel):,} filas, {panel['fondo'].nunique()} fondos")

    # Paso 2: aplicar toda la ingeniería de features + targets multi-horizonte
    df = build_features(panel, horizons=HORIZONS)
    print(f">>> panel con features: {len(df):,} filas, {len(FEATURE_COLS)} features")

    # Paso 3: reportar cobertura por horizonte
    feat_obs = df[FEATURE_COLS].notna().all(axis=1).sum()
    print(f">>> features completas (sin NaN):    {feat_obs:,}")
    for h in HORIZONS:
        target_col = f"target_rank_{h}m"
        target_obs = df[target_col].notna().sum()
        full_obs = (df[target_col].notna() & df[FEATURE_COLS].notna().all(axis=1)).sum()
        print(f">>> horizonte {h:>2d}m — target observable: {target_obs:>7,}  "
              f"modelable: {full_obs:>7,}")

    # Paso 4: guardar panel completo (el filtrado a obs modelables se hace en script 04)
    out = ARTIFACTS_DIR / "panel_features.parquet"
    df.to_parquet(out, index=False)
    print(f"\n>>> guardado en {out.relative_to(ARTIFACTS_DIR.parent)} en {time.time()-t0:.1f}s")


if __name__ == "__main__":
    sys.exit(main())
