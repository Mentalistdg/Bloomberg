"""Script 05/05 — Transforma artefactos del pipeline en JSONs para el dashboard.

Este script es el puente entre el pipeline de modelado (scripts 01-04)
y la aplicación web (FastAPI + React). Toma los artefactos producidos
por los scripts anteriores y los convierte en JSONs optimizados para
consumo del frontend.

No realiza ningún cálculo de modelado — solo reestructura datos.

Input:  artifacts/scores.parquet, metrics.json, drivers_elastic.csv,
        drivers_lgbm.csv, panel_features.parquet
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

from src.paths import APP_DATA_DIR, ARTIFACTS_DIR

SCORE_COL = "score_elastic"  # modelo principal


def main() -> None:
    APP_DATA_DIR.mkdir(parents=True, exist_ok=True)

    print(">>> cargando artefactos")
    scores = pd.read_parquet(ARTIFACTS_DIR / "scores.parquet")
    panel = pd.read_parquet(ARTIFACTS_DIR / "panel_features.parquet")
    coefs = pd.read_csv(ARTIFACTS_DIR / "drivers_elastic.csv")
    importances = pd.read_csv(ARTIFACTS_DIR / "drivers_lgbm.csv")
    with open(ARTIFACTS_DIR / "metrics.json") as f:
        metrics = json.load(f)

    # ---- 1. funds_summary.json
    # Para cada fondo, su última observación con score, decil y
    # estadísticas históricas.
    last_score = scores.sort_values("mes").groupby("fondo").tail(1)
    last_score["decil"] = last_score.groupby("mes")[SCORE_COL].transform(
        lambda s: pd.qcut(s.rank(method="first"), 10, labels=False, duplicates="drop") + 1
    )

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
    detail = {}
    panel_by_fondo = panel.set_index("fondo")
    scores_by_fondo = scores.set_index("fondo")
    for fid in [f["fondo"] for f in funds]:
        if fid not in panel_by_fondo.index:
            continue
        p = panel_by_fondo.loc[[fid]].sort_values("mes")
        s = (scores_by_fondo.loc[[fid]].sort_values("mes")
             if fid in scores_by_fondo.index else None)
        merged = p.merge(s[["mes", SCORE_COL]] if s is not None else
                          pd.DataFrame(columns=["mes", SCORE_COL]),
                          on="mes", how="left")
        # equity curve normalizada a 100
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

    # ---- 4. backtest.json — top-quintile vs bottom-quintile vs universo
    bt_rows = []
    for date, g in scores.groupby("mes"):
        if len(g) < 10:
            continue
        gg = g.copy()
        gg["q"] = pd.qcut(gg[SCORE_COL].rank(method="first"), 5,
                          labels=False, duplicates="drop")
        top = gg[gg["q"] == 4]["target_ret_12m"].mean()
        bot = gg[gg["q"] == 0]["target_ret_12m"].mean()
        eq = gg["target_ret_12m"].mean()
        bt_rows.append({
            "date": str(date.date()),
            "top_q5": float(top) if pd.notna(top) else None,
            "bot_q1": float(bot) if pd.notna(bot) else None,
            "universe": float(eq) if pd.notna(eq) else None,
            "spread": float(top - bot) if pd.notna(top) and pd.notna(bot) else None,
            "n_funds": int(len(gg)),
        })
    with open(APP_DATA_DIR / "backtest.json", "w") as f:
        json.dump(bt_rows, f, indent=2)
    print(f"    backtest.json:      {len(bt_rows)} fechas")

    # ---- 5. meta.json — info global
    meta = {
        "as_of": str(scores["mes"].max().date()),
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
