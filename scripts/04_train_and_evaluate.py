"""Script 04 — Entrenamiento walk-forward y evaluacion (horizonte 6m, target Sortino).

Entrena ElasticNet (primario), LightGBM (sanity check) y benchmark naive
usando validacion walk-forward rolling window (10 años) con embargo de
6 meses. El target de entrenamiento es target_sortino_rank_6m
(Sortino & Price 1994).

Input:  artifacts/panel_features.parquet
Output: artifacts/scores.parquet, metrics.json, drivers, fold_diagnostics
        artifacts/plots/signal_*.png, drivers_elastic.png

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
    FEATURE_COLS,
    get_modeling_frame,
)
from src.metrics import (
    hit_rate_top_quartile,
    ic_per_date,
    ic_summary,
    multi_lens_evaluation,
    quintile_spread_per_date,
    rank_persistence,
)
from src.model import ElasticNetModel, LightGBMModel, benchmark_naive_score
from src.paths import ARTIFACTS_DIR, PLOTS_DIR
from src.splits import walk_forward_folds
from src.validation import bootstrap_mean_ci, diebold_mariano

# --- Configuracion ---
HORIZON = 6              # horizonte de prediccion
MIN_TRAIN_MONTHS = 60    # minimo 5 anios de historia antes del primer fold
VAL_MONTHS = 12          # cada fold evalua sobre 12 meses out-of-sample
MAX_TRAIN_MONTHS = 120   # ventana rolling: 10 anios de historia

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
    folds = list(walk_forward_folds(df["mes"], MIN_TRAIN_MONTHS, VAL_MONTHS, embargo,
                                     max_train_months=MAX_TRAIN_MONTHS))
    print(f"    {len(folds)} folds generados (embargo={embargo}m, rolling={MAX_TRAIN_MONTHS}m, "
          f"{len(feature_cols)} features)")

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
        # Benchmark naive: ret_12m_rank - fee_rank
        bench_pred = benchmark_naive_score(val)

        # Capturar TODOS los targets disponibles para validacion multi-lente
        target_capture = ["mes", "fondo", "target_ret", "target_rank"]
        for opt in ("target_sharpe", "target_sortino", "target_max_dd"):
            if opt in val.columns:
                target_capture.append(opt)
        rows = val[target_capture].copy()
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
    """Evalua la calidad predictiva de un score: IC, bootstrap, deciles, hit rate."""
    ic = ic_per_date(scores, score_col, "target_rank")
    summary = ic_summary(ic)
    boot = bootstrap_mean_ci(ic)
    decile_spread = quintile_spread_per_date(scores, score_col, "target_sortino", n_q=10)
    hit = hit_rate_top_quartile(scores, score_col, "target_sortino")
    return {
        "ic_summary": summary,
        "ic_bootstrap": boot,
        "ic_series_index": [str(d.date()) for d in ic.index],
        "ic_series_values": ic.tolist(),
        "spread_d10_d1_mean": float(decile_spread["spread"].mean()) if len(decile_spread) else np.nan,
        "spread_d10_d1_pos_pct": float((decile_spread["spread"] > 0).mean()) if len(decile_spread) else np.nan,
        "hit_rate_top25_mean": float(hit.mean()) if len(hit) else np.nan,
    }


def plot_diagnostics(scores: pd.DataFrame, ic_series: pd.Series,
                     label: str, suffix: str = "") -> None:
    """Genera graficos de IC y quintiles."""
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))

    axes[0].plot(ic_series.index, ic_series.values, color="#1f4e79", lw=1.2)
    axes[0].axhline(0, color="black", lw=0.5)
    axes[0].axhline(ic_series.mean(), color="red", ls="--", lw=1, label=f"media={ic_series.mean():.3f}")
    axes[0].set_title(f"IC mensual ({label}{suffix})")
    axes[0].set_ylabel("Spearman corr")
    axes[0].legend()

    decile = quintile_spread_per_date(scores, f"score_{label}", "target_sortino", n_q=10)
    if len(decile):
        cols = [c for c in decile.columns if c.startswith("q")]
        means = decile[cols].mean()
        axes[1].bar(range(len(means)), means.values, color="#2e7d32")
        axes[1].set_xticks(range(len(means)))
        axes[1].set_xticklabels(cols)
        axes[1].set_ylabel("Sortino realizado promedio")
        axes[1].set_title(f"Sortino por decil ({label}{suffix})")
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


def _prepare_df_for_horizon(df_full: pd.DataFrame, horizon: int,
                            require_target: bool = True) -> pd.DataFrame:
    """Filtra a observaciones modelables y renombra columnas de target
    a nombres genericos.

    Convencion:
      target_rank    = target_sortino_rank_{h}m  (target de ENTRENAMIENTO)
      target_ret     = target_ret_{h}m           (retorno realizado)
      target_sharpe  = target_sharpe_{h}m        (Sharpe realizado, lente)
      target_sortino = target_sortino_{h}m       (Sortino realizado, lente)
      target_max_dd  = target_max_dd_{h}m        (Max DD realizado, lente)
    """
    df = get_modeling_frame(df_full, horizon=horizon, target="sortino",
                            require_target=require_target)
    rename_map = {
        f"target_sortino_rank_{horizon}m": "target_rank",
        f"target_ret_{horizon}m": "target_ret",
        f"target_sharpe_{horizon}m": "target_sharpe",
        f"target_sortino_{horizon}m": "target_sortino",
        f"target_max_dd_{horizon}m": "target_max_dd",
    }
    rename_map = {k: v for k, v in rename_map.items() if k in df.columns}
    df = df.rename(columns=rename_map)
    return df


def main() -> None:
    t0 = time.time()
    PLOTS_DIR.mkdir(parents=True, exist_ok=True)

    print(">>> 1. cargar features")
    df_full = pd.read_parquet(ARTIFACTS_DIR / "panel_features.parquet")

    print(f"\n{'='*60}")
    print(f">>> HORIZONTE {HORIZON}m")
    print(f"{'='*60}")

    df = _prepare_df_for_horizon(df_full, HORIZON)
    print(f"    panel modelable: {len(df):,} filas, "
          f"{df['fondo'].nunique()} fondos, "
          f"{df['mes'].min().date()} -> {df['mes'].max().date()}")

    # --- Walk-forward ---
    scores, coefs, importances, diag = run_walk_forward(df, HORIZON, FEATURE_COLS)
    print(f"    predicciones OOS: {len(scores):,} filas, "
          f"{scores['fold'].nunique()} folds")

    # --- Evaluar 3 modelos (ElasticNet, LightGBM, naive) ---
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
        print(f"    {label:>10s}  IC={s['mean']:+.4f}  IR={s['ic_ir']:+.2f}  "
              f"hit={s['hit']:.1%}  D10-D1={ev['spread_d10_d1_mean']:+.3f}  "
              f"CI95=[{b['ci_low']:+.4f}, {b['ci_high']:+.4f}]")

    # --- Diebold-Mariano: ElasticNet vs benchmark ---
    losses_en = -ic_per_date(scores, "score_elastic", "target_rank")
    losses_bm = -ic_per_date(scores, "score_benchmark", "target_rank")
    dm = diebold_mariano(losses_en, losses_bm, h=HORIZON)
    print(f"    DM(elastic vs bench): stat={dm['stat']:+.3f}  p={dm['p_value']:.4f}")
    results["diebold_mariano_elastic_vs_benchmark"] = dm

    # --- Validacion multi-lente ---
    multi_lens_targets = [c for c in
                          ["target_ret", "target_sharpe", "target_sortino", "target_max_dd"]
                          if c in scores.columns]
    multi_lens = {}
    for label in ["elastic", "lgbm", "benchmark"]:
        multi_lens[label] = multi_lens_evaluation(
            scores, f"score_{label}", multi_lens_targets,
        )
        pers = rank_persistence(scores, f"score_{label}", lag=HORIZON)
        multi_lens[label]["_rank_persistence_lag_h"] = pers
    results["multi_lens"] = multi_lens

    print("    --- Multi-lens (Q5-Q1 spread medio sobre target realizado):")
    for label in ["elastic", "lgbm", "benchmark"]:
        parts = []
        for tcol in multi_lens_targets:
            ml = multi_lens[label].get(tcol, {})
            if "q5_minus_q1_mean" in ml:
                v = ml["q5_minus_q1_mean"]
                parts.append(f"{tcol[7:]}={v:+.3f}")
        print(f"      {label:>10s}  " + "  ".join(parts))

    # --- Drivers plot ---
    plot_drivers(coefs)

    # --- Scoring de produccion (post walk-forward) ---
    last_val_end = diag["val_end"].max()
    prod_mask = df["mes"] > pd.Timestamp(last_val_end)
    prod_data = df.loc[prod_mask]

    if len(prod_data) > 0:
        print(f"\n    --- PRODUCCION: {prod_data['mes'].nunique()} meses "
              f"post walk-forward ({prod_data['mes'].min().date()} -> "
              f"{prod_data['mes'].max().date()})")

        train_all = df.loc[df["mes"] <= pd.Timestamp(last_val_end)]
        # Aplicar rolling window al train de produccion
        unique_train_months = sorted(train_all["mes"].unique())
        if len(unique_train_months) > MAX_TRAIN_MONTHS:
            cutoff = unique_train_months[-MAX_TRAIN_MONTHS]
            train_all = train_all.loc[train_all["mes"] >= cutoff]
        X_tr = train_all[FEATURE_COLS]
        y_tr = train_all["target_rank"].values
        X_prod = prod_data[FEATURE_COLS]

        elastic_prod = ElasticNetModel()
        lgbm_prod = LightGBMModel()
        en_pred, _ = elastic_prod.fit_predict(X_tr, y_tr, X_prod)
        lgbm_pred, _ = lgbm_prod.fit_predict(X_tr, y_tr, X_prod)
        bench_pred = benchmark_naive_score(prod_data)

        target_cols = ["mes", "fondo", "target_ret", "target_rank"]
        for opt in ("target_sharpe", "target_sortino", "target_max_dd"):
            if opt in prod_data.columns:
                target_cols.append(opt)
        prod_rows = prod_data[target_cols].copy()
        prod_rows["score_elastic"] = en_pred
        prod_rows["score_lgbm"] = lgbm_pred
        prod_rows["score_benchmark"] = bench_pred
        prod_rows["fold"] = -1  # marcador: produccion, no walk-forward

        scores = pd.concat([scores, prod_rows], ignore_index=True)
        print(f"    produccion: {len(prod_rows)} obs agregadas (fold=-1)")
    else:
        train_all = df.loc[df["mes"] <= df["mes"].max()]
        # Aplicar rolling window al train de produccion
        unique_train_months = sorted(train_all["mes"].unique())
        if len(unique_train_months) > MAX_TRAIN_MONTHS:
            cutoff = unique_train_months[-MAX_TRAIN_MONTHS]
            train_all = train_all.loc[train_all["mes"] >= cutoff]
        X_tr = train_all[FEATURE_COLS]
        y_tr = train_all["target_rank"].values
        elastic_prod = ElasticNetModel()
        lgbm_prod = LightGBMModel()
        elastic_prod.fit_predict(X_tr, y_tr, train_all[FEATURE_COLS])
        lgbm_prod.fit_predict(X_tr, y_tr, train_all[FEATURE_COLS])

    # --- Scoring extendido (features sin target) ---
    prod_months = set(scores.loc[scores["fold"] == -1, "mes"].unique()) if len(scores) else set()
    last_prod = max(prod_months) if prod_months else pd.Timestamp(last_val_end)
    df_ext = _prepare_df_for_horizon(df_full, HORIZON, require_target=False)
    ext_mask = (df_ext["mes"] > last_prod)
    ext_data = df_ext.loc[ext_mask]

    if len(ext_data) > 0:
        X_ext = ext_data[FEATURE_COLS]
        en_ext, _ = elastic_prod.fit_predict(X_tr, y_tr, X_ext)
        lgbm_ext, _ = lgbm_prod.fit_predict(X_tr, y_tr, X_ext)
        bench_ext = benchmark_naive_score(ext_data)

        ext_rows = ext_data[["mes", "fondo"]].copy()
        ext_rows["target_ret"] = np.nan
        ext_rows["target_rank"] = np.nan
        for opt in ("target_sharpe", "target_sortino", "target_max_dd"):
            if opt in scores.columns:
                ext_rows[opt] = np.nan
        ext_rows["score_elastic"] = en_ext
        ext_rows["score_lgbm"] = lgbm_ext
        ext_rows["score_benchmark"] = bench_ext
        ext_rows["fold"] = -2  # marcador: produccion extendida

        scores = pd.concat([scores, ext_rows], ignore_index=True)
        print(f"    extendido: {len(ext_rows)} obs "
              f"({ext_data['mes'].nunique()} meses, fold=-2)")

    # --- Guardar artefactos ---
    scores_out = scores.rename(columns={
        "target_ret": "target_ret_6m",
        "target_rank": "target_rank",
    })
    scores_out.to_parquet(ARTIFACTS_DIR / "scores.parquet", index=False)
    coefs.to_csv(ARTIFACTS_DIR / "drivers_elastic.csv", index=False)
    importances.to_csv(ARTIFACTS_DIR / "drivers_lgbm.csv", index=False)
    diag.to_csv(ARTIFACTS_DIR / "fold_diagnostics.csv", index=False)
    with open(ARTIFACTS_DIR / "metrics.json", "w") as f:
        json.dump(results, f, indent=2, default=str)

    print(f"\n>>> tiempo total: {time.time() - t0:.1f}s")


if __name__ == "__main__":
    sys.exit(main())
