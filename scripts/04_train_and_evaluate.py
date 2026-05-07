"""04 - Walk-forward training, prediction, evaluation.

Pipeline:
  1. lee panel_features.parquet
  2. genera folds walk-forward
  3. en cada fold entrena ElasticNet y LightGBM, predice sobre val
  4. acumula predicciones de todos los folds
  5. calcula IC, spread Q5-Q1, hit rate top-quartile globalmente y por fold
  6. corre validación estadística (bootstrap del IC, Diebold-Mariano vs
     benchmark naive)
  7. guarda artifacts: scores.parquet, metrics.json, fold_diagnostics.json,
     drivers.csv, plots
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src.features import CORE_FEATURES, EXTENDED_FEATURES, FEATURE_COLS, RANK_FEATURES, get_modeling_frame
from src.metrics import hit_rate_top_quartile, ic_per_date, ic_summary, quintile_spread_per_date
from src.model import ElasticNetModel, LightGBMModel, benchmark_naive_score
from src.paths import ARTIFACTS_DIR, PLOTS_DIR
from src.splits import walk_forward_folds
from src.validation import bootstrap_mean_ci, diebold_mariano

MIN_TRAIN_MONTHS = 60
VAL_MONTHS = 12
EMBARGO_MONTHS = 12

plt.rcParams.update({"figure.dpi": 110, "savefig.dpi": 130, "savefig.bbox": "tight",
                     "axes.spines.top": False, "axes.spines.right": False,
                     "axes.grid": True, "grid.alpha": 0.25, "font.size": 10})


def run_walk_forward(df: pd.DataFrame) -> pd.DataFrame:
    folds = list(walk_forward_folds(df["mes"], MIN_TRAIN_MONTHS, VAL_MONTHS, EMBARGO_MONTHS))
    print(f">>> {len(folds)} folds generados")

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

        X_tr, y_tr = train[FEATURE_COLS], train["target_rank"].values
        X_vl, y_vl = val[FEATURE_COLS], val["target_rank"].values

        en_pred, en_info = elastic.fit_predict(X_tr, y_tr, X_vl)
        lgbm_pred, lgbm_info = lgbm.fit_predict(X_tr, y_tr, X_vl)
        bench_pred = benchmark_naive_score(X_vl)

        rows = val[["mes", "fondo", "target_ret_12m", "target_rank"]].copy()
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
    ic = ic_per_date(scores, score_col, "target_rank")
    summary = ic_summary(ic)
    boot = bootstrap_mean_ci(ic)
    quint = quintile_spread_per_date(scores, score_col, "target_ret_12m")
    hit = hit_rate_top_quartile(scores, score_col, "target_ret_12m")
    return {
        "ic_summary": summary,
        "ic_bootstrap": boot,
        "ic_series_index": [str(d.date()) for d in ic.index],
        "ic_series_values": ic.tolist(),
        "spread_q5_q1_mean": float(quint["spread"].mean()) if len(quint) else np.nan,
        "spread_q5_q1_pos_pct": float((quint["spread"] > 0).mean()) if len(quint) else np.nan,
        "hit_rate_top25_mean": float(hit.mean()) if len(hit) else np.nan,
    }


def plot_diagnostics(scores: pd.DataFrame, ic_series: pd.Series, label: str) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))

    axes[0].plot(ic_series.index, ic_series.values, color="#1f4e79", lw=1.2)
    axes[0].axhline(0, color="black", lw=0.5)
    axes[0].axhline(ic_series.mean(), color="red", ls="--", lw=1, label=f"media={ic_series.mean():.3f}")
    axes[0].set_title(f"IC mensual ({label})")
    axes[0].set_ylabel("Spearman corr")
    axes[0].legend()

    quint = quintile_spread_per_date(scores, f"score_{label}", "target_ret_12m")
    if len(quint):
        cols = [c for c in quint.columns if c.startswith("q")]
        means = quint[cols].mean()
        axes[1].bar(range(len(means)), means.values * 100, color="#2e7d32")
        axes[1].set_xticks(range(len(means)))
        axes[1].set_xticklabels(cols)
        axes[1].set_ylabel("retorno realizado 12m promedio (%)")
        axes[1].set_title(f"Retorno por quintil del score ({label})")
        axes[1].axhline(0, color="black", lw=0.5)

    fig.tight_layout()
    fig.savefig(PLOTS_DIR / f"signal_{label}.png")
    plt.close(fig)


def plot_drivers(coefs: pd.DataFrame) -> None:
    feature_cols_in_coefs = [c for c in coefs.columns if c not in ("fold", "alpha", "l1_ratio")]
    avg = coefs[feature_cols_in_coefs].mean().sort_values()
    fig, ax = plt.subplots(figsize=(7, 6))
    colors = ["#c62828" if v < 0 else "#2e7d32" for v in avg.values]
    ax.barh(avg.index, avg.values, color=colors, alpha=0.8)
    ax.axvline(0, color="black", lw=0.5)
    ax.set_title("Coeficientes ElasticNet — promedio entre folds")
    ax.set_xlabel("coeficiente (variables estandarizadas)")
    fig.tight_layout()
    fig.savefig(PLOTS_DIR / "drivers_elastic.png")
    plt.close(fig)


def main() -> None:
    t0 = time.time()
    PLOTS_DIR.mkdir(parents=True, exist_ok=True)

    print(">>> 1. cargar features y filtrar a observaciones modelables")
    df_full = pd.read_parquet(ARTIFACTS_DIR / "panel_features.parquet")
    df = get_modeling_frame(df_full)
    df = df[df["mes"] >= "2010-01-01"].copy()  # comienza cuando hay masa crítica de fondos
    print(f"    panel modelable post-2010: {len(df):,} filas, "
          f"{df['fondo'].nunique()} fondos, "
          f"{df['mes'].min().date()} -> {df['mes'].max().date()}")

    print("\n>>> 2. walk-forward CV")
    scores, coefs, importances, diag = run_walk_forward(df)
    print(f"\n    predicciones acumuladas: {len(scores):,} filas a través de {scores['fold'].nunique()} folds")

    print("\n>>> 3. evaluación de señales")
    results = {}
    for label in ["elastic", "lgbm", "benchmark"]:
        col = f"score_{label}"
        ev = evaluate_signal(scores, col)
        results[label] = ev
        ic = pd.Series(ev["ic_series_values"],
                        index=pd.to_datetime(ev["ic_series_index"]))
        plot_diagnostics(scores, ic, label)
        s = ev["ic_summary"]
        b = ev["ic_bootstrap"]
        print(f"    {label:>10s}  IC mean={s['mean']:+.4f}  IR={s['ic_ir']:+.2f}  "
              f"hit_meses={s['hit']:.1%}  Q5-Q1={ev['spread_q5_q1_mean']*100:+.2f}%  "
              f"CI95=[{b['ci_low']:+.4f}, {b['ci_high']:+.4f}]")

    print("\n>>> 4. Diebold-Mariano (ElasticNet vs benchmark)")
    losses_en = -ic_per_date(scores, "score_elastic", "target_rank")
    losses_bm = -ic_per_date(scores, "score_benchmark", "target_rank")
    dm = diebold_mariano(losses_en, losses_bm, h=12)
    print(f"    DM stat={dm['stat']:+.3f}  p={dm['p_value']:.4f}  N={dm['n']}")
    results["diebold_mariano_elastic_vs_benchmark"] = dm

    print("\n>>> 5. drivers (coeficientes ElasticNet promedio entre folds)")
    plot_drivers(coefs)

    print("\n>>> 6. guardando artifacts")
    scores.to_parquet(ARTIFACTS_DIR / "scores.parquet", index=False)
    coefs.to_csv(ARTIFACTS_DIR / "drivers_elastic.csv", index=False)
    importances.to_csv(ARTIFACTS_DIR / "drivers_lgbm.csv", index=False)
    diag.to_csv(ARTIFACTS_DIR / "fold_diagnostics.csv", index=False)
    with open(ARTIFACTS_DIR / "metrics.json", "w") as f:
        json.dump(results, f, indent=2, default=str)

    print(f"\n>>> tiempo total: {time.time() - t0:.1f}s")


if __name__ == "__main__":
    sys.exit(main())
