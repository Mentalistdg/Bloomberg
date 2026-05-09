"""Script 05/05 — Transforma artefactos del pipeline en JSONs para el dashboard.

Este script es el puente entre el pipeline de modelado (scripts 01-04)
y la aplicación web (FastAPI + React). Toma los artefactos producidos
por los scripts anteriores y los convierte en JSONs optimizados para
consumo del frontend.

No realiza ningún cálculo de modelado — solo reestructura datos.

Input:  artifacts/scores.parquet, metrics.json, drivers_elastic.csv,
        drivers_lgbm.csv, panel_features.parquet,
        assets/usa_fondos_pp.sqlite (para curvas de equity sin winsorización)
Output: app/backend/data/
          funds_summary.json  — ranking de fondos con score y métricas
          fund_detail.json    — serie temporal mensual por fondo
          drivers.json        — coeficientes ElasticNet e importancias LightGBM
          backtest.json       — retorno por quintil por mes (backtest visual)
          meta.json           — metadata global (fechas, n_fondos, métricas resumen)

Uso:
    python -m scripts.05_build_app_data
"""

from __future__ import annotations

import json
import sys

import numpy as np
import pandas as pd

from src.data import load_historico
from src.paths import APP_DATA_DIR, ARTIFACTS_DIR

SCORE_COL = "score_elastic"  # modelo principal


def _build_display_equity(fondos: list[str]) -> dict[str, pd.DataFrame]:
    """Construye curvas de equity desde precios crudos, SIN winsorización.

    La winsorización (p99.5) se usa en el pipeline de modelado para
    controlar outliers en features, pero distorsiona las curvas de
    display: en fondos con movimientos extremos legítimos (crisis 2008,
    COVID 2020) el clipping acumula errores de hasta 30%+ en la
    trayectoria. Para visualización se usa el retorno total real.

    Returns: dict fondo → DataFrame con columnas [mes, equity].
    """
    hist = load_historico()
    hist = hist[hist["fondo"].isin(fondos)].sort_values(["fondo", "fecha"])

    # Retorno total diario = variación NAV + eventos de capital (sin clip)
    hist["ret_precio"] = hist.groupby("fondo")["precio"].pct_change()
    hist["ret_total"] = hist["ret_precio"].fillna(0) + hist["evento_pct"].fillna(0)
    hist.loc[hist.groupby("fondo").head(1).index, "ret_total"] = 0

    # TRI diario
    hist["tri"] = hist.groupby("fondo")["ret_total"].transform(
        lambda s: (1 + s).cumprod()
    )

    # EOM: último día hábil de cada mes
    hist["mes"] = hist["fecha"].dt.to_period("M").dt.to_timestamp("M")
    eom = hist.sort_values("fecha").groupby(["fondo", "mes"]).tail(1)
    eom = eom[["fondo", "mes", "tri"]].copy()

    # Normalizar cada fondo a base 100
    eom["equity"] = eom.groupby("fondo")["tri"].transform(
        lambda s: 100.0 * s / s.iloc[0]
    )

    result = {}
    for fid, g in eom.groupby("fondo"):
        result[fid] = g[["mes", "equity"]].reset_index(drop=True)
    return result


def main() -> None:
    APP_DATA_DIR.mkdir(parents=True, exist_ok=True)

    print(">>> cargando artefactos")
    scores = pd.read_parquet(ARTIFACTS_DIR / "scores.parquet")
    panel = pd.read_parquet(ARTIFACTS_DIR / "panel_features.parquet")
    coefs = pd.read_csv(ARTIFACTS_DIR / "drivers_elastic.csv")
    importances = pd.read_csv(ARTIFACTS_DIR / "drivers_lgbm.csv")
    with open(ARTIFACTS_DIR / "metrics.json") as f:
        metrics = json.load(f)
    fold_diag = pd.read_csv(ARTIFACTS_DIR / "fold_diagnostics.csv")

    # ---- 1. funds_summary.json
    # Solo fondos del último mes de scoring para tener un corte
    # cross-seccional consistente (mismo período para todos).
    # Fondos scorados en meses anteriores (por salir del walk-forward)
    # no son comparables con el ranking actual.
    latest_month = scores["mes"].max()
    # Para el overview, usar el último mes con target realizado (OOS validado)
    # en vez del mes de producción donde target_ret_12m es NaN (futuro).
    has_target = scores.dropna(subset=["target_ret_12m"])
    if len(has_target) > 0:
        overview_month = has_target["mes"].max()
    else:
        overview_month = latest_month
    last_score = scores[scores["mes"] == overview_month].copy()
    print(f"    filtro: overview_month={overview_month.date()} "
          f"(latest={latest_month.date()}, "
          f"{len(last_score)} de {scores['fondo'].nunique()} totales)")
    last_score["decil"] = pd.qcut(
        last_score[SCORE_COL].rank(method="first"),
        10, labels=False, duplicates="drop",
    ) + 1

    last_panel = panel.sort_values("mes").groupby("fondo").tail(1).set_index("fondo")

    funds = []
    for _, row in last_score.iterrows():
        fid = row["fondo"]
        p = last_panel.loc[fid] if fid in last_panel.index else None
        funds.append({
            "fondo": fid,
            "fecha_score": str(row["mes"].date()),
            "score": float(row[SCORE_COL]),
            "decil": int(row["decil"]) if pd.notna(row["decil"]) else None,
            "target_realizado_12m": (None if pd.isna(row["target_ret_12m"])
                                     else float(row["target_ret_12m"])),
            "ret_12m_trailing": (None if p is None or pd.isna(p.get("ret_12m"))
                                 else float(p["ret_12m"])),
            "vol_12m": (None if p is None or pd.isna(p.get("vol_12m"))
                        else float(p["vol_12m"])),
            "sharpe_12m": (None if p is None or pd.isna(p.get("sharpe_12m"))
                           else float(p["sharpe_12m"])),
            "max_dd_12m": (None if p is None or pd.isna(p.get("max_dd_12m"))
                           else float(p["max_dd_12m"])),
            "fee": (None if p is None or pd.isna(p.get("fee")) else float(p["fee"])),
            "fee_observado": (None if p is None else int(p.get("fee_observado", 0))),
            "n_instrumentos": (None if p is None or pd.isna(p.get("n_instrumentos"))
                               else int(p["n_instrumentos"])),
        })
    funds.sort(key=lambda x: x["score"], reverse=True)
    with open(APP_DATA_DIR / "funds_summary.json", "w") as f:
        json.dump(funds, f, indent=2)
    print(f"    funds_summary.json: {len(funds)} fondos")

    # ---- 2. fund_detail.json — historia mensual por fondo
    # Curvas de equity desde precios crudos (sin winsorización del pipeline)
    fund_ids = [f["fondo"] for f in funds]
    print("    construyendo curvas de equity desde precios crudos (sin winsor)...")
    display_equity = _build_display_equity(fund_ids)

    detail = {}
    panel_by_fondo = panel.set_index("fondo")
    scores_by_fondo = scores.set_index("fondo")
    for fid in fund_ids:
        if fid not in panel_by_fondo.index:
            continue
        p = panel_by_fondo.loc[[fid]].sort_values("mes")
        s = (scores_by_fondo.loc[[fid]].sort_values("mes")
             if fid in scores_by_fondo.index else None)
        merged = p.merge(s[["mes", SCORE_COL]] if s is not None else
                          pd.DataFrame(columns=["mes", SCORE_COL]),
                          on="mes", how="left")
        # Equity desde precios crudos, alineada a las fechas del panel
        if fid in display_equity:
            eq_df = display_equity[fid]
            merged = merged.merge(eq_df, on="mes", how="left")
            equity = merged["equity"].values
        else:
            # Fallback: reconstruir desde ret_mensual del panel
            ret = merged["ret_mensual"].fillna(0).values
            equity = 100 * np.cumprod(1 + ret)
        detail[fid] = {
            "dates": [str(d.date()) for d in merged["mes"]],
            "ret_mensual": [None if pd.isna(r) else float(r) for r in merged["ret_mensual"]],
            "equity": equity.tolist(),
            "score": [None if pd.isna(v) else float(v) for v in merged[SCORE_COL]],
            "fee": [None if pd.isna(v) else float(v) for v in merged["fee"]],
            "n_instrumentos": [None if pd.isna(v) else int(v) for v in merged["n_instrumentos"]],
        }
    with open(APP_DATA_DIR / "fund_detail.json", "w") as f:
        json.dump(detail, f)
    print(f"    fund_detail.json:   {len(detail)} fondos")

    # ---- 3. drivers.json
    feat_cols = [c for c in coefs.columns if c not in ("fold", "alpha", "l1_ratio")]
    drivers = {
        "elastic_net_coefs": {
            "features": feat_cols,
            "mean": coefs[feat_cols].mean().tolist(),
            "std": coefs[feat_cols].std().tolist(),
        },
        "elastic_net_hyperparams": {
            "alpha_mean": float(coefs["alpha"].mean()),
            "l1_ratio_mean": float(coefs["l1_ratio"].mean()),
        },
        "lgbm_importance": {
            "features": [c for c in importances.columns if c != "fold"],
            "mean": importances.drop(columns="fold").mean().tolist(),
        },
    }
    with open(APP_DATA_DIR / "drivers.json", "w") as f:
        json.dump(drivers, f, indent=2)
    print(f"    drivers.json:       {len(feat_cols)} features")

    # ---- 4. backtest.json — top-decil vs bottom-decil vs universo + naive

    # Precomputar retornos forward parciales (para meses sin target_ret_12m)
    panel_sorted = panel[["fondo", "mes", "ret_mensual"]].sort_values(["fondo", "mes"])
    last_month = panel_sorted["mes"].max()

    partial_list = []
    for fondo, group in panel_sorted.groupby("fondo"):
        rets = group["ret_mensual"].values
        meses = group["mes"].values
        n = len(rets)
        partial = np.full(n, np.nan)
        months_fwd = np.zeros(n, dtype=int)
        for i in range(n):
            end = min(i + 13, n)  # hasta 12 meses forward
            k = end - i - 1
            if k > 0:
                fwd = rets[i + 1 : end]
                if not np.any(np.isnan(fwd)):
                    partial[i] = np.prod(1 + fwd) - 1
                    months_fwd[i] = k
        partial_list.append(pd.DataFrame({
            "fondo": fondo, "mes": meses,
            "partial_fwd_ret": partial, "months_forward": months_fwd,
        }))
    partial_rets = pd.concat(partial_list, ignore_index=True).set_index(["fondo", "mes"])

    bt_rows = []
    for date, g in scores.groupby("mes"):
        if len(g) < 20:  # mínimo 2 fondos por decil
            continue
        gg = g.copy()

        is_prod = bool(g["fold"].iloc[0] == -1) if "fold" in g.columns else False
        is_ext = bool(g["fold"].iloc[0] == -2) if "fold" in g.columns else False

        if is_ext:
            # Merge partial returns para meses extendidos (sin target completo)
            gg = gg.merge(partial_rets, left_on=["fondo", "mes"],
                          right_index=True, how="left")
            ret_col = "partial_fwd_ret"
            avg_months = int(gg["months_forward"].median())
        else:
            ret_col = "target_ret_12m"
            avg_months = 12

        # ElasticNet (deciles)
        gg["d"] = pd.qcut(gg[SCORE_COL].rank(method="first"), 10,
                          labels=False, duplicates="drop")
        top = gg[gg["d"] == 9][ret_col].mean()
        bot = gg[gg["d"] == 0][ret_col].mean()
        eq = gg[ret_col].mean()

        # Naive benchmark (deciles)
        gg["d_naive"] = pd.qcut(gg["score_benchmark"].rank(method="first"), 10,
                                labels=False, duplicates="drop")
        naive_top = gg[gg["d_naive"] == 9][ret_col].mean()
        naive_bot = gg[gg["d_naive"] == 0][ret_col].mean()

        bt_rows.append({
            "date": str(date.date()),
            "top_d10": float(top) if pd.notna(top) else None,
            "bot_d1": float(bot) if pd.notna(bot) else None,
            "universe": float(eq) if pd.notna(eq) else None,
            "spread": float(top - bot) if pd.notna(top) and pd.notna(bot) else None,
            "naive_top_d10": float(naive_top) if pd.notna(naive_top) else None,
            "naive_bot_d1": float(naive_bot) if pd.notna(naive_bot) else None,
            "naive_spread": (float(naive_top - naive_bot)
                             if pd.notna(naive_top) and pd.notna(naive_bot) else None),
            "n_funds": int(len(gg)),
            "is_production": is_prod or is_ext,
            "is_partial": is_ext,
            "months_forward": avg_months,
        })
    with open(APP_DATA_DIR / "backtest.json", "w") as f:
        json.dump(bt_rows, f, indent=2)
    print(f"    backtest.json:      {len(bt_rows)} fechas")

    # ---- 5. meta.json — info global
    meta = {
        "as_of": str(overview_month.date()),
        "n_funds": int(scores["fondo"].nunique()),
        "n_months": int(scores["mes"].nunique()),
        "n_folds": int(scores["fold"].nunique()),
        "metrics_summary": {
            label: {
                "ic_mean": metrics[label]["ic_summary"]["mean"],
                "ic_ir": metrics[label]["ic_summary"]["ic_ir"],
                "ic_hit_meses": metrics[label]["ic_summary"]["hit"],
                "ic_ci95_low": metrics[label]["ic_bootstrap"]["ci_low"],
                "ic_ci95_high": metrics[label]["ic_bootstrap"]["ci_high"],
                "spread_q5_q1": metrics[label]["spread_q5_q1_mean"],
                "hit_top25": metrics[label]["hit_rate_top25_mean"],
            }
            for label in ["elastic", "lgbm", "benchmark", "axiomatic"]
            if label in metrics
        },
        "diebold_mariano_elastic_vs_benchmark": metrics.get(
            "diebold_mariano_elastic_vs_benchmark"),
        "diebold_mariano_elastic_vs_axiomatic": metrics.get(
            "diebold_mariano_elastic_vs_axiomatic"),
        "fold_boundaries": [
            {"fold": int(row["fold"]),
             "val_start": row["val_start"],
             "val_end": row["val_end"]}
            for _, row in fold_diag.iterrows()
        ],
        "primary_model": "elastic",
        "training_target": "target_sharpe_rank_12m",
        "scorers_compared": ["elastic", "lgbm", "benchmark", "axiomatic"],
    }
    with open(APP_DATA_DIR / "meta.json", "w") as f:
        json.dump(meta, f, indent=2, default=str)
    print(f"    meta.json: as_of={meta['as_of']}, primary_model={meta['primary_model']}")

    print(f"\n>>> JSONs en {APP_DATA_DIR}")


if __name__ == "__main__":
    sys.exit(main())
