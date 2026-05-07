"""Validación estadística de la señal del modelo.

Este módulo responde la pregunta: "¿la señal predictiva es
estadísticamente significativa o podría ser producto del azar?".
Sin estas pruebas, un IC positivo podría ser ruido muestral.

Dos herramientas estadísticas:

  Bootstrap del IC (bootstrap_mean_ci):
    Remuestrea con reposición la serie mensual de IC (5,000 iteraciones)
    para construir un intervalo de confianza del IC promedio. Si el
    intervalo incluye al cero, NO se puede rechazar la hipótesis nula
    de "cero poder predictivo" al nivel de confianza elegido (95%).
    Más robusto que un t-test cuando la serie es corta o tiene colas
    pesadas.

  Diebold-Mariano (diebold_mariano):
    Test formal de superioridad predictiva: ¿el modelo A es
    significativamente mejor que el modelo B? Compara las pérdidas
    (= -IC) de ambos modelos mes a mes, con corrección Newey-West
    por autocorrelación (necesaria porque el target es a 12 meses,
    generando dependencia temporal en las pérdidas). Un p-valor alto
    (ej. 0.66) indica que NO hay diferencia significativa.

Usado por: scripts/04_train_and_evaluate.py
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import stats


def bootstrap_mean_ci(
    series: pd.Series,
    n_iter: int = 5_000,
    alpha: float = 0.05,
    seed: int = 42,
) -> dict:
    """IC promedio con CI bootstrap percentil."""
    arr = series.dropna().values
    if len(arr) < 5:
        return dict(mean=float(np.mean(arr)) if len(arr) else np.nan,
                    ci_low=np.nan, ci_high=np.nan, n=len(arr))
    rng = np.random.default_rng(seed)
    means = np.empty(n_iter)
    n = len(arr)
    # Remuestrear n_iter veces con reposición y calcular la media de cada muestra
    for i in range(n_iter):
        idx = rng.integers(0, n, n)  # índices aleatorios con reposición
        means[i] = arr[idx].mean()
    # El intervalo se obtiene de los percentiles alpha/2 y 1-alpha/2
    # de la distribución bootstrap de medias
    return dict(
        mean=float(arr.mean()),
        ci_low=float(np.quantile(means, alpha / 2)),
        ci_high=float(np.quantile(means, 1 - alpha / 2)),
        n=int(n),
    )


def diebold_mariano(
    losses_a: pd.Series,
    losses_b: pd.Series,
    h: int = 1,
) -> dict:
    """Test de Diebold-Mariano (1995) con corrección Newey-West.

    H0: las pérdidas tienen igual media (no hay diferencia predictiva).
    Pérdidas más altas = peor modelo. El estadístico DM negativo => A
    es mejor que B.
    """
    a = losses_a.dropna()
    b = losses_b.dropna()
    # Alinear ambas series a las mismas fechas
    common = a.index.intersection(b.index)
    # d_t = loss_A_t - loss_B_t. Si d < 0 en promedio, A es mejor
    d = (a.loc[common] - b.loc[common]).values
    n = len(d)
    if n < 10:
        return dict(stat=np.nan, p_value=np.nan, n=n)

    d_bar = d.mean()
    # Varianza con corrección Newey-West por autocorrelación de orden h-1
    # (necesaria porque el target a 12m genera dependencia serial en pérdidas)
    gamma0 = np.var(d, ddof=1)
    var = gamma0
    for k in range(1, h):
        gk = np.cov(d[:-k], d[k:], ddof=1)[0, 1]
        var += 2 * (1 - k / h) * gk  # kernel de Bartlett
    if var <= 0:
        return dict(stat=np.nan, p_value=np.nan, n=n)
    # Estadístico DM ~ N(0,1) bajo H0
    stat = d_bar / np.sqrt(var / n)
    p = 2 * (1 - stats.norm.cdf(abs(stat)))  # test bilateral
    return dict(stat=float(stat), p_value=float(p), n=int(n))
