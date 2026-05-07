"""Script 04/05 — Entrenamiento walk-forward, prediccion y evaluacion multi-horizonte.

Entrena los modelos (ElasticNet, LightGBM, benchmark naive) usando validacion
walk-forward con embargo temporal, tanto con el set completo de features (23)
como con el set reducido (5 features curadas). Compara horizontes y feature sets.

Input:  artifacts/panel_features.parquet
Output: artifacts/scores.parquet, metrics.json, drivers (backward-compat, 12m full)
        artifacts/scores_{h}m.parquet, *_{h}m_reduced.* (por horizonte y feature set)
        artifacts/horizon_comparison.csv (tabla resumen completa)
        artifacts/plots/signal_*_{h}m.png, *_{h}m_reduced.png

Uso:
    python -m scripts.04_train_and_evaluate
"""

from __future__ import annotations

import json
import sys
import time

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src.features import (
    CORE_FEATURES,
    FEATURE_COLS,
    REDUCED_FEATURES,
    get_modeling_frame,
)
from src.metrics import hit_rate_top_quartile, ic_per_date, ic_summary, quintile_spread_per_date
from src.model import ElasticNetModel, LightGBMModel, benchmark_naive_score
from src.paths import ARTIFACTS_DIR, PLOTS_DIR
from src.splits import walk_forward_folds
from src.validation import bootstrap_mean_ci, diebold_mariano

# --- Configuracion del walk-forward ---
MIN_TRAIN_MONTHS = 60   # minimo 5 anios de historia antes del primer fold
VAL_MONTHS = 12          # cada fold evalua sobre 12 meses out-of-sample
HORIZONS = [3, 6, 12]    # horizontes de prediccion a comparar

# Feature sets a evaluar: nombre -> lista de columnas
FEATURE_SETS = {
    "full": FEATURE_COLS,          # 23 features
    "reduced": REDUCED_FEATURES,   # 5 features curadas
}

plt.rcParams.update({"figure.dpi": 110, "savefig.dpi": 130, "savefig.bbox": "tight",
                     "axes.spines.top": False, "axes.spines.right": False,
                     "axes.grid": True, "grid.alpha": 0.25, "font.size": 10})


def run_walk_forward(df: pd.DataFrame, horizon: int,
                     feature_cols: list[str]) -> tuple:
    """Ejecuta el loop de walk-forward para un horizonte y feature set dados.

    El embargo se ajusta al horizonte del target. El benchmark siempre usa
    las columnas completas del DataFrame (no depende de feature_cols).
    """
    embargo = horizon
    folds = list(walk_forward_folds(df["mes"], MIN_TRAIN_MONTHS, VAL_MONTHS, embargo))
    print(f"    {len(folds)} folds generados (embargo={embargo}m, {len(feature_cols)} features)")

    elastic = ElasticNetModel()
    lgbm = LightGBMModel()

    preds_rows = []
    fold_diag = []
    coefs_per_fold: list[dict] = []
    importance_per_fold: list[dict] = []

    for fold in folds:
        train_mask = df["mes"].isin(fold.train_dates)
        val_mask = df["mes"].isin(fold.val_dates)
        train = df.loc[train_mask]
        val = df.loc[val_mask]
        if len(train) < 200 or len(val) < 30:
            continue

        X_tr, y_tr = train[feature_cols], train["target_rank"].values
        X_vl = val[feature_cols]

        en_pred, en_info = elastic.fit_predict(X_tr, y_tr, X_vl)
        lgbm_pred, lgbm_info = lgbm.fit_predict(X_tr, y_tr, X_vl)
        # Benchmark siempre usa las mismas 2 features (ret_12m_rank, fee_rank)
        bench_pred = benchmark_naive_score(val)

        rows = val[["mes", "fondo", "target_ret", "target_rank"]].copy()
        rows["score_elastic"] = en_pred
        rows["score_lgbm"] = lgbm_pred
        rows["score_benchmark"] = bench_pred
        rows["fold"] = fold.fold_id
        preds_rows.append(rows)

        coefs_per_fold.append({"fold": fold.fold_id, "alpha": en_info["alpha"],
                               "l1_ratio": en_info["l1_ratio"], **en_info["coefs"]})
        importance_per_fold.append({"fold": fold.fold_id, **lgbm_info["feature_importance"]})
        fold_diag.append(dict(
            fold=fold.fold_id,
            train_start=str(fold.train_dates[0].date()),
            train_end=str(fold.train_end.date()),
            val_start=str(fold.val_start.date()),
            val_end=str(fold.val_end.date()),
            n_train=int(train_mask.sum()),
            n_val=int(val_mask.sum()),
            elastic_alpha=en_info["alpha"],
            elastic_l1_ratio=en_info["l1_ratio"],
        ))
        print(f"    fold {fold.fold_id:>2d}  "
              f"train {fold.train_dates[0].date()}->{fold.train_end.date()}  "
              f"val {fold.val_start.date()}->{fold.val_end.date()}  "
              f"n_tr={train_mask.sum():>5d} n_vl={val_mask.sum():>4d}")

    scores = pd.concat(preds_rows, ignore_index=True)
    coefs = pd.DataFrame(coefs_per_fold)
    importances = pd.DataFrame(importance_per_fold)
    diag = pd.DataFrame(fold_diag)
    return scores, coefs, importances, diag


def evaluate_signal(scores: pd.DataFrame, score_col: str) -> dict:
    """Evalua la calidad predictiva de un score: IC, bootstrap, quintiles, hit rate."""
    ic = ic_per_date(scores, score_col, "target_rank")
    summary = ic_summary(ic)
    boot = bootstrap_mean_ci(ic)
    quint = quintile_spread_per_date(scores, score_col, "target_ret")
    hit = hit_rate_top_quartile(scores, score_col, "target_ret")
    return {
        "ic_summary": summary,
        "ic_bootstrap": boot,
        "ic_series_index": [str(d.date()) for d in ic.index],
        "ic_series_values": ic.tolist(),
        "spread_q5_q1_mean": float(quint["spread"].mean()) if len(quint) else np.nan,
        "spread_q5_q1_pos_pct": float((quint["spread"] > 0).mean()) if len(quint) else np.nan,
        "hit_rate_top25_mean": float(hit.mean()) if len(hit) else np.nan,
    }


def plot_diagnostics(scores: pd.DataFrame, ic_series: pd.Series,
                     label: str, suffix: str = "") -> None:
    """Genera graficos de IC y quintiles. suffix diferencia horizonte/feature set."""
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))

    axes[0].plot(ic_series.index, ic_series.values, color="#1f4e79", lw=1.2)
    axes[0].axhline(0, color="black", lw=0.5)
    axes[0].axhline(ic_series.mean(), color="red", ls="--", lw=1, label=f"media={ic_series.mean():.3f}")
    axes[0].set_title(f"IC mensual ({label}{suffix})")
    axes[0].set_ylabel("Spearman corr")
    axes[0].legend()

    quint = quintile_spread_per_date(scores, f"score_{label}", "target_ret")
    if len(quint):
        cols = [c for c in quint.columns if c.startswith("q")]
        means = quint[cols].mean()
        axes[1].bar(range(len(means)), means.values * 100, color="#2e7d32")
        axes[1].set_xticks(range(len(means)))
        axes[1].set_xticklabels(cols)
        axes[1].set_ylabel("retorno realizado promedio (%)")
        axes[1].set_title(f"Retorno por quintil ({label}{suffix})")
        axes[1].axhline(0, color="black", lw=0.5)

    fig.tight_layout()
    fig.savefig(PLOTS_DIR / f"signal_{label}{suffix}.png")
    plt.close(fig)


def plot_drivers(coefs: pd.DataFrame, suffix: str = "") -> None:
    feature_cols_in_coefs = [c for c in coefs.columns if c not in ("fold", "alpha", "l1_ratio")]
    avg = coefs[feature_cols_in_coefs].mean().sort_values()
    fig, ax = plt.subplots(figsize=(7, max(4, len(avg) * 0.35)))
    colors = ["#c62828" if v < 0 else "#2e7d32" for v in avg.values]
    ax.barh(avg.index, avg.values, color=colors, alpha=0.8)
    ax.axvline(0, color="black", lw=0.5)
    ax.set_title(f"Coeficientes ElasticNet{suffix}")
    ax.set_xlabel("coeficiente (variables estandarizadas)")
    fig.tight_layout()
    fig.savefig(PLOTS_DIR / f"drivers_elastic{suffix}.png")
    plt.close(fig)


def _prepare_df_for_horizon(df_full: pd.DataFrame, horizon: int) -> pd.DataFrame:
    """Filtra a observaciones modelables y renombra columnas de target
    a nombres genericos (target_ret, target_rank)."""
    df = get_modeling_frame(df_full, horizon=horizon)
    df = df[df["mes"] >= "2010-01-01"].copy()
    df = df.rename(columns={
        f"target_ret_{horizon}m": "target_ret",
        f"target_rank_{horizon}m": "target_rank",
    })
    return df


def main() -> None:
    t0 = time.time()
    PLOTS_DIR.mkdir(parents=True, exist_ok=True)

    print(">>> 1. cargar features")
    df_full = pd.read_parquet(ARTIFACTS_DIR / "panel_features.parquet")

    comparison_rows = []

    for horizon in HORIZONS:
        print(f"\n{'='*60}")
        print(f">>> HORIZONTE {horizon}m")
        print(f"{'='*60}")

        df = _prepare_df_for_horizon(df_full, horizon)
        print(f"    panel modelable post-2010: {len(df):,} filas, "
              f"{df['fondo'].nunique()} fondos, "
              f"{df['mes'].min().date()} -> {df['mes'].max().date()}")

        for feat_name, feat_cols in FEATURE_SETS.items():
            is_full = feat_name == "full"
            tag = f"_{horizon}m" if is_full else f"_{horizon}m_reduced"
            label_prefix = "" if is_full else " [reduced]"

            print(f"\n--- {feat_name.upper()} ({len(feat_cols)} features) ---")

            scores, coefs, importances, diag = run_walk_forward(df, horizon, feat_cols)
            print(f"    predicciones OOS: {len(scores):,} filas, "
                  f"{scores['fold'].nunique()} folds")

            # Evaluar los 3 modelos
            results = {}
            for label in ["elastic", "lgbm", "benchmark"]:
                col = f"score_{label}"
                ev = evaluate_signal(scores, col)
                results[label] = ev
                ic = pd.Series(ev["ic_series_values"],
                               index=pd.to_datetime(ev["ic_series_index"]))
                plot_diagnostics(scores, ic, label, suffix=tag)
                s = ev["ic_summary"]
                b = ev["ic_bootstrap"]
                print(f"    {label:>10s}  IC={s['mean']:+.4f}  IR={s['ic_ir']:+.2f}  "
                      f"hit={s['hit']:.1%}  Q5-Q1={ev['spread_q5_q1_mean']*100:+.2f}%  "
                      f"CI95=[{b['ci_low']:+.4f}, {b['ci_high']:+.4f}]")

                comparison_rows.append({
                    "horizon": horizon,
                    "features": feat_name,
                    "n_features": len(feat_cols),
                    "model": label,
                    "ic_mean": s["mean"],
                    "ic_ir": s["ic_ir"],
                    "ic_ci95_low": b["ci_low"],
                    "ic_ci95_high": b["ci_high"],
                    "spread_q5_q1": ev["spread_q5_q1_mean"],
                    "hit_rate_top25": ev["hit_rate_top25_mean"],
                    "n_folds": scores["fold"].nunique(),
                })

            # Diebold-Mariano: ElasticNet vs benchmark
            losses_en = -ic_per_date(scores, "score_elastic", "target_rank")
            losses_bm = -ic_per_date(scores, "score_benchmark", "target_rank")
            dm = diebold_mariano(losses_en, losses_bm, h=horizon)
            print(f"    DM(elastic vs bench): stat={dm['stat']:+.3f}  p={dm['p_value']:.4f}")
            results["diebold_mariano_elastic_vs_benchmark"] = dm

            # Drivers
            plot_drivers(coefs, suffix=tag)

            # Guardar artefactos por horizonte+feature_set
            scores_out = scores.rename(columns={
                "target_ret": f"target_ret_{horizon}m",
                "target_rank": f"target_rank_{horizon}m",
            })
            scores_out.to_parquet(ARTIFACTS_DIR / f"scores{tag}.parquet", index=False)
            coefs.to_csv(ARTIFACTS_DIR / f"drivers_elastic{tag}.csv", index=False)
            importances.to_csv(ARTIFACTS_DIR / f"drivers_lgbm{tag}.csv", index=False)
            diag.to_csv(ARTIFACTS_DIR / f"fold_diagnostics{tag}.csv", index=False)

            # Backward-compat: archivos sin sufijo = 12m full
            if horizon == 12 and is_full:
                scores_compat = scores.rename(columns={
                    "target_ret": "target_ret_12m",
                    "target_rank": "target_rank",
                })
                scores_compat.to_parquet(ARTIFACTS_DIR / "scores.parquet", index=False)
                coefs.to_csv(ARTIFACTS_DIR / "drivers_elastic.csv", index=False)
                importances.to_csv(ARTIFACTS_DIR / "drivers_lgbm.csv", index=False)
                diag.to_csv(ARTIFACTS_DIR / "fold_diagnostics.csv", index=False)
                with open(ARTIFACTS_DIR / "metrics.json", "w") as f:
                    json.dump(results, f, indent=2, default=str)
                plot_drivers(coefs)  # sin sufijo

    # Tabla comparativa final
    print(f"\n{'='*60}")
    print(">>> TABLA COMPARATIVA")
    print(f"{'='*60}")
    comparison = pd.DataFrame(comparison_rows)
    # Mostrar ordenado por horizonte, luego features, luego modelo
    comparison = comparison.sort_values(["horizon", "features", "model"]).reset_index(drop=True)
    print(comparison.to_string(index=False))
    comparison.to_csv(ARTIFACTS_DIR / "horizon_comparison.csv", index=False)

    print(f"\n>>> tiempo total: {time.time() - t0:.1f}s")


if __name__ == "__main__":
    sys.exit(main())
