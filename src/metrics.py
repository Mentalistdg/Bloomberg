"""Métricas relevantes para evaluar un modelo de scoring de fondos.

Las métricas elegidas responden a la decisión de inversión —
identificar fondos atractivos vs. poco atractivos — y no a la
precisión absoluta de la predicción:

  - **Information Coefficient (IC)**: correlación de Spearman por mes
    entre el score predicho y el retorno realizado a 12m. Métrica
    estándar en quant equity para medir poder predictivo cross-seccional.
  - **Spread Q5-Q1**: diferencia de retorno realizado entre el quintil
    superior e inferior por score. Métrica directa de "valor de la señal":
    si el modelo dice "este es top-20%", ¿efectivamente performaron
    mejor que el bottom-20%?
  - **Hit rate top-quartile**: % de fondos en el top-25% del score que
    superan la mediana del universo en retorno realizado. Más
    interpretable para audiencias no-quant.
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
        means = g.groupby(q)[target_col].mean()
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
        med = g[target_col].median()
        thr = g[score_col].quantile(0.75)
        top = g[g[score_col] >= thr]
        if len(top) == 0:
            continue
        out[d] = float((top[target_col] > med).mean())
    return pd.Series(out, name="hit_rate_top25").sort_index()
