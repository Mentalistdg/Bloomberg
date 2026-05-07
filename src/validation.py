"""Validación estadística de la señal del modelo.

  - **Bootstrap del IC**: se remuestrea con reposición la serie mensual
    de IC para construir intervalos de confianza del IC promedio.
    Más robusto que asumir normalidad cuando la serie es corta o tiene
    colas pesadas.
  - **Diebold-Mariano**: test de superioridad predictiva entre dos
    modelos sobre la misma serie de pérdidas. Versión clásica con
    ajuste por autocorrelación de orden 1.
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
    for i in range(n_iter):
        idx = rng.integers(0, n, n)
        means[i] = arr[idx].mean()
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
    common = a.index.intersection(b.index)
    d = (a.loc[common] - b.loc[common]).values
    n = len(d)
    if n < 10:
        return dict(stat=np.nan, p_value=np.nan, n=n)

    d_bar = d.mean()
    gamma0 = np.var(d, ddof=1)
    var = gamma0
    for k in range(1, h):
        gk = np.cov(d[:-k], d[k:], ddof=1)[0, 1]
        var += 2 * (1 - k / h) * gk
    if var <= 0:
        return dict(stat=np.nan, p_value=np.nan, n=n)
    stat = d_bar / np.sqrt(var / n)
    p = 2 * (1 - stats.norm.cdf(abs(stat)))
    return dict(stat=float(stat), p_value=float(p), n=int(n))
