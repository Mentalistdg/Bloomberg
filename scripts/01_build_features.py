"""Script 01/05 — Construye el panel mensual base desde la base sqlite.

Este es el primer paso del pipeline. Carga las tres tablas crudas de la
base de datos (historico, fees, subyacentes), calcula el retorno total
diario incluyendo eventos de capital, y produce un panel mensual
fondo x fin-de-mes que sirve como input para los siguientes scripts.

Input:  usa_fondos_pp.sqlite (tablas: historico, fees, subyacentes)
Output: artifacts/panel_raw.parquet
        Columnas: fondo, mes, tri_eom, ret_mensual, n_dias_obs,
                  fee, pct_acum, n_instrumentos

Siguiente paso: script 02 (EDA) o script 03 (feature engineering).

Uso:
    python -m scripts.01_build_features

Funciones utilizadas de src/:
    src.data  →  load_historico, load_fees, load_subyacentes,
                 compute_daily_total_return, build_monthly_panel,
                 attach_fees_monthly, attach_subyacentes
"""

from __future__ import annotations

import sys
import time

from src.data import (
    attach_fees_monthly,
    attach_subyacentes,
    build_monthly_panel,
    compute_daily_total_return,
    load_fees,
    load_historico,
    load_subyacentes,
)
from src.paths import ARTIFACTS_DIR


def main() -> None:
    t0 = time.time()
    print(">>> 1. cargando tablas crudas")
    historico = load_historico()
    fees = load_fees()
    sub = load_subyacentes()
    print(f"    historico:    {len(historico):>10,} filas, "
          f"{historico['fondo'].nunique()} fondos")
    print(f"    fees:         {len(fees):>10,} filas (post-dropna), "
          f"{fees['fondo'].nunique()} fondos")
    print(f"    subyacentes:  {len(sub):>10,} filas, "
          f"{sub['fondo'].nunique()} fondos")

    print("\n>>> 2. retorno total diario (precio + evento_pct, winsorizado p99.5)")
    daily = compute_daily_total_return(historico)
    cubrió = daily["ret_total"].notna().sum()
    print(f"    días totales con retorno calculable: {cubrió:,}")

    print("\n>>> 3. panel mensual fondo x mes")
    panel = build_monthly_panel(daily)
    print(f"    panel mensual: {len(panel):,} filas, "
          f"{panel['fondo'].nunique()} fondos, "
          f"{panel['mes'].min().date()} -> {panel['mes'].max().date()}")

    print("\n>>> 4. adjuntar fees (forward-fill por fondo)")
    panel = attach_fees_monthly(panel, fees)
    cob_fee = panel["fee"].notna().mean()
    print(f"    cobertura de fee post-ffill: {cob_fee:.1%}")

    print("\n>>> 5. adjuntar concentración (subyacentes, ffill por fondo)")
    panel = attach_subyacentes(panel, sub)
    cob_n = panel["n_instrumentos"].notna().mean()
    print(f"    cobertura de n_instrumentos post-ffill: {cob_n:.1%}")

    out = ARTIFACTS_DIR / "panel_raw.parquet"
    panel.to_parquet(out, index=False)
    print(f"\n>>> guardado en {out.relative_to(ARTIFACTS_DIR.parent)}")
    print(f">>> tiempo total: {time.time() - t0:.1f}s")


if __name__ == "__main__":
    sys.exit(main())
