"""03 - Construye features completas + target sobre el panel mensual base.

Lee:    artifacts/panel_raw.parquet
Output: artifacts/panel_features.parquet
"""

from __future__ import annotations

import sys
import time

import pandas as pd

from src.features import FEATURE_COLS, build_features
from src.paths import ARTIFACTS_DIR


def main() -> None:
    t0 = time.time()
    panel = pd.read_parquet(ARTIFACTS_DIR / "panel_raw.parquet")
    print(f">>> panel_raw: {len(panel):,} filas, {panel['fondo'].nunique()} fondos")

    df = build_features(panel)
    print(f">>> panel con features: {len(df):,} filas")

    target_obs = df["target_rank"].notna().sum()
    feat_obs = df[FEATURE_COLS].notna().all(axis=1).sum()
    full_obs = (df["target_rank"].notna() & df[FEATURE_COLS].notna().all(axis=1)).sum()
    print(f">>> target observable:               {target_obs:,}")
    print(f">>> features completas (sin NaN):    {feat_obs:,}")
    print(f">>> ambos (modelable end-to-end):    {full_obs:,}")

    out = ARTIFACTS_DIR / "panel_features.parquet"
    df.to_parquet(out, index=False)
    print(f"\n>>> guardado en {out.relative_to(ARTIFACTS_DIR.parent)} en {time.time()-t0:.1f}s")


if __name__ == "__main__":
    sys.exit(main())
