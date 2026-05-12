"""Métricas de evaluación del poder predictivo del scoring.

Este módulo calcula las métricas que responden a la pregunta central:
"¿el score del modelo permite distinguir fondos atractivos de poco
atractivos?". No se mide precisión absoluta (MAE, RMSE) sino calidad
del ranking — lo que importa para la decisión de inversión.

Tres métricas complementarias:

  Information Coefficient (IC):
    Correlación de Spearman por mes entre el score predicho y el target
    realizado. Métrica estándar en quant equity. Un IC de +0.05
    ya se considera útil en la industria; IC ~ 0 indica que el modelo
    no tiene poder predictivo cross-seccional.

  Spread D10-D1:
    Diferencia de retorno realizado promedio entre el decil superior
    (D10, fondos con score más alto) y el inferior (D1, score más bajo).
    Responde directamente: "¿los fondos que el modelo recomienda rinden
    más que los que no?". Un spread positivo indica que sí.

  Hit rate top-quartile:
    Porcentaje de fondos en el top-25% del score que superan la mediana
    del universo en retorno realizado. Más interpretable para audiencias
    no-cuantitativas. Un 50% = azar; >50% = valor agregado.

Usado por: scripts/04_train_and_evaluate.py
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import stats


def ic_per_date(df: pd.DataFrame, score_col: str, target_col: str,
                date_col: str = "mes") -> pd.Series:
    """Spearman rank correlation por fecha. Devuelve serie indexada por fecha."""
    out = {}
    for d, g in df.groupby(date_col):
        if g[score_col].nunique() < 3 or g[target_col].nunique() < 3:
            continue
        rho, _ = stats.spearmanr(g[score_col], g[target_col])
        out[d] = rho
    return pd.Series(out, name="ic").sort_index()


def ic_summary(ic_series: pd.Series) -> dict:
    """Estadísticas resumen del IC mensual."""
    s = ic_series.dropna()
    if s.empty:
        return dict(mean=np.nan, std=np.nan, ic_ir=np.nan, hit=np.nan, n=0)
    return dict(
        mean=float(s.mean()),
        std=float(s.std()),
        ic_ir=float(s.mean() / s.std()) if s.std() > 0 else np.nan,
        hit=float((s > 0).mean()),
        n=int(len(s)),
    )


def quintile_spread_per_date(df: pd.DataFrame, score_col: str, target_col: str,
                              date_col: str = "mes", n_q: int = 5) -> pd.DataFrame:
    """Retorno realizado promedio por quintil del score, por fecha.
    Columnas: date, q1..q5, spread (q5-q1)."""
    rows = []
    for d, g in df.groupby(date_col):
        if len(g) < n_q * 2:
            continue
        try:
            q = pd.qcut(g[score_col].rank(method="first"), n_q, labels=False, duplicates="drop")
        except ValueError:
            continue
        # Limpiar target: (1) inf → NaN, (2) winsorizar p1/p99 dentro de cada
        # fecha para que valores extremos finitos (Sortino ~1e3 con downside_dev
        # cercano a cero) no dominen el mean() por quintil.
        target_clean = g[target_col].replace([np.inf, -np.inf], np.nan)
        p01, p99 = target_clean.quantile(0.01), target_clean.quantile(0.99)
        target_clean = target_clean.clip(p01, p99)
        means = target_clean.groupby(q).mean()
        if len(means) < n_q:
            continue
        row = {date_col: d}
        for i in range(n_q):
            row[f"q{i+1}"] = float(means.iloc[i])
        row["spread"] = float(means.iloc[-1] - means.iloc[0])
        rows.append(row)
    return pd.DataFrame(rows)


def hit_rate_top_quartile(df: pd.DataFrame, score_col: str, target_col: str,
                           date_col: str = "mes") -> pd.Series:
    """% de fondos top-25% por score que superan la mediana del universo."""
    out = {}
    for d, g in df.groupby(date_col):
        if len(g) < 8:
            continue
        target_clean = g[target_col].replace([np.inf, -np.inf], np.nan)
        med = target_clean.median()
        thr = g[score_col].quantile(0.75)
        top = g[g[score_col] >= thr]
        if len(top) == 0:
            continue
        out[d] = float((top[target_col].replace([np.inf, -np.inf], np.nan) > med).mean())
    return pd.Series(out, name="hit_rate_top25").sort_index()


# --- validación multi-lente -------------------------------------------------

def long_short_information_ratio(spread_series: pd.Series,
                                  periods_per_year: int = 12) -> dict:
    """Information Ratio del long-short top-bottom.

    Es el Sharpe de la estrategia "comprar el top del score y vender
    el bottom". Captura si la señal del modelo se traduce en una
    estrategia rentable de manera estable, no solo episódica.

    IR = mean(spread) / std(spread) * sqrt(periods_per_year)

    Métrica estándar en gestión activa de portafolios. IR > 0.5 anualizado
    se considera bueno; > 1 excelente (raro fuera de muestra).
    """
    s = pd.to_numeric(spread_series, errors="coerce").dropna()
    if len(s) < 2 or s.std() == 0:
        return dict(ir=np.nan, mean=np.nan, std=np.nan, n=int(len(s)))
    mean = float(s.mean())
    std = float(s.std())
    ir = (mean / std) * np.sqrt(periods_per_year)
    return dict(ir=float(ir), mean=mean, std=std, n=int(len(s)))


def rank_persistence(scores_df: pd.DataFrame, score_col: str, lag: int = 12,
                     date_col: str = "mes", fund_col: str = "fondo") -> dict:
    """Persistencia del rank predicho por el modelo: correlación entre el
    rank que el modelo asignó a un fondo en T y el rank que le asignó en
    T+lag (default 12 meses).

    Si la persistencia es alta, el modelo recomienda los mismos fondos
    consistentemente — lo que es deseable para una decisión de selección
    a largo plazo (turnover bajo, costos de transacción bajos). Si es
    baja, el modelo cambia mucho de opinión y la estrategia implicaría
    rotación frecuente.

    Devuelve correlación de Spearman pooled sobre todos los pares
    (fondo, T) — (mismo fondo, T+lag) disponibles.
    """
    df = scores_df[[date_col, fund_col, score_col]].copy()
    df["rank_t"] = df.groupby(date_col)[score_col].rank(pct=True, method="average")
    df = df.sort_values([fund_col, date_col])
    df["rank_t_plus_lag"] = df.groupby(fund_col)["rank_t"].shift(-lag)
    paired = df.dropna(subset=["rank_t", "rank_t_plus_lag"])
    if len(paired) < 50:
        return dict(persistence=np.nan, n_pairs=int(len(paired)))
    rho, _ = stats.spearmanr(paired["rank_t"], paired["rank_t_plus_lag"])
    return dict(persistence=float(rho), n_pairs=int(len(paired)))


def multi_lens_evaluation(scores_df: pd.DataFrame, score_col: str,
                           target_cols: list[str],
                           date_col: str = "mes",
                           n_q: int = 10) -> dict:
    """Evalúa el mismo score contra MÚLTIPLES targets forward. Cada target
    es una "lente": una pregunta financiera distinta.

    Por cada target reporta:
      - IC mensual promedio (correlación rank-rank)
      - IC IR (mean/std del IC mensual)
      - Spread top-bottom promedio del target realizado (D10-D1 con n_q=10)
      - Long-Short IR (Sharpe del spread top-bottom)
      - Hit rate top-25%

    La idea: un score robusto debería discriminar bien en VARIAS lentes,
    no solo en la métrica con la que fue entrenado. Un score que solo
    funciona en su métrica de entrenamiento es sospechoso de overfitting
    a la definición de target.

    Parameters
    ----------
    target_cols : list of str
        Nombres de columnas de target en scores_df. Pueden ser nivel
        ("target_ret_12m") o rank ("target_rank_12m") — la función no
        distingue, solo usa Spearman para IC y agrupa para spread.
    n_q : int
        Número de grupos para el spread (default 10 = deciles).
    """
    out = {}
    for tcol in target_cols:
        if tcol not in scores_df.columns:
            out[tcol] = dict(error=f"columna {tcol!r} no presente")
            continue
        ic = ic_per_date(scores_df, score_col, tcol, date_col=date_col)
        ic_stats = ic_summary(ic)
        spread = quintile_spread_per_date(scores_df, score_col, tcol, date_col=date_col, n_q=n_q)
        ls_ir = (long_short_information_ratio(spread["spread"])
                 if "spread" in spread.columns and len(spread) > 0
                 else dict(ir=np.nan, n=0))
        hit = hit_rate_top_quartile(scores_df, score_col, tcol, date_col=date_col)
        out[tcol] = {
            "ic_mean": ic_stats["mean"],
            "ic_ir": ic_stats["ic_ir"],
            "ic_hit": ic_stats["hit"],
            "q5_minus_q1_mean": float(spread["spread"].mean()) if len(spread) > 0 else np.nan,
            "long_short_ir": ls_ir["ir"],
            "hit_rate_top25": float(hit.mean()) if len(hit) > 0 else np.nan,
            "n_months": int(len(ic.dropna())),
        }
    return out
