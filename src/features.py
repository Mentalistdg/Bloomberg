"""Ingeniería de features sobre el panel mensual.

Este módulo transforma el panel mensual crudo (output de data.py) en un
dataset listo para modelar, con 23 features y targets forward multi-horizonte.
Es el núcleo de la preparación de datos del pipeline.

Convención anti-leakage (CRÍTICA):
    Todas las features al final del mes t usan EXCLUSIVAMENTE información
    disponible hasta el cierre del mes t (rolling windows hacia atrás).
    Los targets forward son las ÚNICAS variables que miran al futuro.
    Esto garantiza que no hay contaminación de información futura en las
    features predictoras.

Convención sobre fees y NAV:
    El precio es NAV de mutual fund USA, ya neto de expense ratio
    (descontado diariamente del NAV). Por eso el target derivado de NAV
    no requiere ajuste adicional por fees. La columna `fee` se usa como
    feature explicativa (un fondo con fee alto necesita generar más
    retorno bruto para compensar).

Las 23 features se dividen en tres grupos:

    CORE (10):     Derivadas del retorno + intra-mes (densas tras 12m warmup).
                   ret_1m, ret_3m, ret_6m, ret_12m, vol_12m, max_dd_12m,
                   sharpe_12m, vol_intrames, autocorr_diaria, ratio_dias_cero.
                   Se exige no-NaN para que una observación sea modelable.

    EXTENDED (4):  fee, log_n_instrumentos + 2 flags binarias.
                   - fee: cobertura ~98% post ffill+bfill intra-fondo
                     (el dataset solo reporta fees desde 2024-01-31, pero
                     el fee es estructuralmente estable en mutual funds USA,
                     por lo que se imputa hacia atrás dentro de cada fondo).
                     Flag `fee_observado` (set en data.py) marca si el valor
                     del mes específico era original (1) o imputado (0).
                   - log_n_instrumentos: cobertura limitada, imputado con
                     mediana cross-seccional + flag `concentracion_disponible`.

    RANK (9):      Percentil cross-seccional dentro del mes para: ret_3m,
                   ret_12m, vol_12m, sharpe_12m, fee, log_n_instrumentos,
                   vol_intrames, autocorr_diaria, ratio_dias_cero.
                   Aportan robustez ante cambios de escala entre períodos.

Usado por: scripts/03_build_features_full.py
"""

from __future__ import annotations

import numpy as np
import pandas as pd

# Tasa libre de riesgo constante (proxy T-bill 3m promedio histórico)
RF_ANNUAL = 0.02                              # 2% anual
RF_MONTHLY = (1 + RF_ANNUAL) ** (1 / 12) - 1  # ~0.001652 mensual


# --- helpers --------------------------------------------------------------

def _rolling_compound(s: pd.Series, window: int) -> pd.Series:
    """prod(1+r) - 1 sobre ventana móvil. NaN si no hay window obs."""
    return (1 + s).rolling(window, min_periods=window).apply(np.prod, raw=True) - 1


def _rolling_max_drawdown(s: pd.Series, window: int) -> pd.Series:
    """Min drawdown observado sobre últimos `window` meses (valor negativo)."""
    cum = (1 + s.fillna(0)).cumprod()
    rolling_max = cum.rolling(window, min_periods=window).max()
    return (cum / rolling_max - 1).rolling(window, min_periods=window).min()


# --- features core --------------------------------------------------------

def add_return_features(df: pd.DataFrame) -> pd.DataFrame:
    """Calcula retornos compuestos trailing a 1, 3, 6 y 12 meses por fondo.
    Cada ventana usa solo datos pasados (rolling hacia atrás)."""
    g = df.groupby("fondo", group_keys=False)
    for h in [1, 3, 6, 12]:
        # Retorno compuesto = prod(1 + r_i) - 1 sobre los últimos h meses
        df[f"ret_{h}m"] = g["ret_mensual"].apply(lambda s: _rolling_compound(s, h))
    return df


def add_risk_features(df: pd.DataFrame) -> pd.DataFrame:
    """Calcula métricas de riesgo rolling a 12 meses por fondo."""
    g = df.groupby("fondo", group_keys=False)["ret_mensual"]
    # Volatilidad anualizada: desviación estándar mensual * sqrt(12)
    df["vol_12m"] = g.apply(lambda s: s.rolling(12, min_periods=12).std() * np.sqrt(12))
    # Máximo drawdown: peor caída desde un peak en los últimos 12 meses
    df["max_dd_12m"] = df.groupby("fondo", group_keys=False)["ret_mensual"].apply(
        lambda s: _rolling_max_drawdown(s, 12)
    )
    # Sharpe 12m: (retorno 12m − Rf) / volatilidad 12m, con Rf = 2% anual
    df["sharpe_12m"] = (df["ret_12m"] - RF_ANNUAL) / df["vol_12m"]
    # Sortino 12m: (ret_12m − Rf) / downside_deviation_12m anualizada
    # Fórmula Sortino & Price (1994): downside_dev = sqrt(mean(min(r,0)²)) * sqrt(12)
    downside_vol_12m = g.apply(
        lambda s: s.clip(upper=0).pow(2).rolling(12, min_periods=12).mean().pipe(np.sqrt) * np.sqrt(12)
    )
    df["sortino_12m"] = (df["ret_12m"] - RF_ANNUAL) / downside_vol_12m
    df["sortino_12m"] = df["sortino_12m"].replace([np.inf, -np.inf], np.nan)
    return df


def add_stylistic_features(df: pd.DataFrame) -> pd.DataFrame:
    """Features que capturan propiedades estilísticas de la distribución
    de retornos y eventos del fondo, complementarias a nivel y volatilidad.

    skewness_12m: asimetría de la distribución de retornos mensuales en
        los últimos 12 meses. Positiva = cola más larga al alza. Negativa
        = cola más larga a la baja (típico de fondos con retornos
        consistentes pequeños interrumpidos por pérdidas grandes).
    hit_rate_12m: fracción de meses con retorno positivo en los últimos
        12 meses. Mide consistencia (no nivel) — un fondo puede tener
        hit rate alto con retornos chicos o hit rate bajo con retornos
        grandes ocasionales.
    distribution_yield_12m: suma de eventos de capital
        (evento_pct_mes, ya winsorizada en data.py) en los últimos
        12 meses. Captura el "yield" anual de distribuciones del fondo
        (dividendos, capital gains distributions). Es señal de estilo
        income/value (alta) vs growth/total-return (baja, ~0).
    """
    g = df.groupby("fondo", group_keys=False)["ret_mensual"]
    df["skewness_12m"] = g.apply(lambda s: s.rolling(12, min_periods=12).skew())
    df["hit_rate_12m"] = g.apply(
        lambda s: s.rolling(12, min_periods=12).apply(lambda x: (x > 0).mean(), raw=True)
    )
    df["distribution_yield_12m"] = (
        df.groupby("fondo", group_keys=False)["evento_pct_mes"]
        .apply(lambda s: s.fillna(0).rolling(12, min_periods=12).sum())
    )
    return df


def add_persistence_features(df: pd.DataFrame) -> pd.DataFrame:
    """Estabilidad del rank del fondo: cuán similar es su posición
    relativa al universo hoy comparado con 12 meses atrás.

    persistencia_rank_12m = 1 - |sharpe_12m_rank_t - sharpe_12m_rank_{t-12}|

    Vale ~1 si el fondo mantiene su posición (top sigue top, o bottom
    sigue bottom). Vale ~0 si saltó entre extremos. Información
    incremental sobre el nivel del rank: un fondo puede estar siempre
    arriba o saltar entre rankings altos y bajos — son cualitativamente
    distintos. Esta función debe ejecutarse DESPUÉS de
    add_cross_sectional_ranks para que sharpe_12m_rank exista.
    """
    rank_lagged = df.groupby("fondo")["sharpe_12m_rank"].shift(12)
    df["persistencia_rank_12m"] = 1 - (df["sharpe_12m_rank"] - rank_lagged).abs()
    return df


# --- features estructurales ----------------------------------------------

def add_extended_features(df: pd.DataFrame) -> pd.DataFrame:
    """Prepara features de concentración y log-transform de n_instrumentos.

    La flag `fee_observado` ya viene seteada desde data.attach_fees_monthly
    (1 = valor original del mes, 0 = imputado por ffill/bfill intra-fondo),
    por lo que no se recalcula acá.

    Para concentración se crea `concentracion_disponible` que distingue
    fondo con dato de concentración disponible (vía ffill) de fondos sin
    ningún reporte. La concentración SÍ varía en el tiempo (es snapshot del
    portafolio en una fecha), por lo que NO se aplica bfill — solo ffill
    desde el último reporte conocido.
    """
    df["concentracion_disponible"] = df["n_instrumentos"].notna().astype(int)
    # Log-transform para reducir asimetría de n_instrumentos
    df["log_n_instrumentos"] = np.log1p(df["n_instrumentos"])
    return df


def impute_extended(df: pd.DataFrame) -> pd.DataFrame:
    """Imputa fee, log_n_instrumentos y pct_acum con la mediana
    cross-seccional del universo activo en cada mes. Las flags
    `fee_observado` y `concentracion_disponible` ya capturan el efecto
    de información faltante para que el modelo distinga dato real de
    imputado.

    Para `fee` la imputación cross-seccional solo aplica a los ~2 fondos
    sin ningún reporte (la imputación intra-fondo ffill+bfill ya cubrió
    el resto en data.attach_fees_monthly).
    """
    for col in ["fee", "log_n_instrumentos", "pct_acum"]:
        med = df.groupby("mes")[col].transform("median")
        df[col] = df[col].fillna(med)
        # fallback final: mediana global
        df[col] = df[col].fillna(df[col].median())
    return df


# --- ranks cross-seccionales ---------------------------------------------

RANK_COLS = [
    "ret_3m", "ret_12m", "vol_12m", "sharpe_12m", "max_dd_12m",
    "fee", "log_n_instrumentos", "pct_acum",
    "vol_intrames", "autocorr_diaria", "ratio_dias_cero",
    "skewness_12m", "hit_rate_12m",
]


def add_cross_sectional_ranks(df: pd.DataFrame) -> pd.DataFrame:
    """Calcula el percentil rank de cada fondo dentro de su mes para las
    features seleccionadas. Esto normaliza las features al rango [0, 1]
    y las hace robustas a cambios de régimen entre períodos."""
    for c in RANK_COLS:
        df[f"{c}_rank"] = df.groupby("mes")[c].rank(pct=True, method="average")
    return df


# --- target ---------------------------------------------------------------

def add_target(df: pd.DataFrame, horizon: int = 12) -> pd.DataFrame:
    """Targets forward para entrenar y validar el scoreador.

    Construye tres métricas forward (mira `horizon` meses hacia adelante) y
    los rankings cross-seccionales correspondientes. Todas se calculan
    sobre la misma ventana de retornos forward (T+1 a T+horizon).

    Targets generados:
      - target_ret_{h}m / target_rank_{h}m:
            Retorno total compuesto a `horizon` meses y su percentil
            dentro del mes T. Métrica naive — captura ganancia bruta sin
            ajustar por riesgo. Se mantiene para reportar resultados en
            términos interpretables (Q5-Q1 spread en %).
      - target_sharpe_{h}m / target_sharpe_rank_{h}m:
            Sharpe ratio anualizado a `horizon` meses
            ((mean / std) * sqrt(12)) y su percentil dentro del mes T.
            Captura retorno y riesgo total. Lente de validación.
      - target_sortino_{h}m / target_sortino_rank_{h}m:
            Sortino ratio anualizado (Sortino & Price 1994):
            (mean − Rf) / sqrt(mean(min(r,0)²)) * sqrt(12).
            Solo penaliza volatilidad a la baja — es el TARGET PRINCIPAL
            DE ENTRENAMIENTO porque para renta variable la volatilidad
            al alza es deseable, no penalizable.
    """
    fwd = pd.concat(
        [df.groupby("fondo")["ret_mensual"].shift(-i) for i in range(1, horizon + 1)],
        axis=1,
    )

    # --- Target 1: Retorno forward compuesto (interpretable, métrica reportada)
    # min_count=horizon asegura NaN si faltan meses (fondo no sobrevive h meses)
    df[f"target_ret_{horizon}m"] = (1 + fwd).prod(axis=1, min_count=horizon) - 1
    df[f"target_rank_{horizon}m"] = df.groupby("mes")[f"target_ret_{horizon}m"].rank(
        pct=True, method="average"
    )

    # --- Target 2: Sharpe forward anualizado (target de entrenamiento)
    # Solo se calcula cuando los `horizon` retornos forward están todos
    # presentes — un fondo que no sobrevive el horizonte completo NO tiene
    # Sharpe forward observable.
    n_valid = fwd.notna().sum(axis=1)
    all_present = n_valid == horizon
    mean_fwd = fwd.mean(axis=1)
    excess_mean_fwd = mean_fwd - RF_MONTHLY
    std_fwd = fwd.std(axis=1)
    sharpe = (excess_mean_fwd / std_fwd) * np.sqrt(12)
    sharpe = sharpe.replace([np.inf, -np.inf], np.nan).where(all_present, np.nan)
    df[f"target_sharpe_{horizon}m"] = sharpe
    df[f"target_sharpe_rank_{horizon}m"] = df.groupby("mes")[f"target_sharpe_{horizon}m"].rank(
        pct=True, method="average"
    )

    # --- Target 3: Sortino forward (TARGET DE ENTRENAMIENTO)
    # Como Sharpe pero penaliza solo volatilidad a la baja. Más robusto
    # ante asimetría — premia fondos que ganan asimétricamente arriba.
    # Fórmula Sortino & Price (1994): downside_dev = sqrt(mean(min(r,0)²))
    # Usa ddof=0 e incluye ceros para retornos positivos → no produce NaN
    # cuando hay 0-1 retornos negativos en la ventana.
    downside_fwd = fwd.clip(upper=0)
    downside_dev_fwd = np.sqrt((downside_fwd ** 2).mean(axis=1))
    sortino = (excess_mean_fwd / downside_dev_fwd) * np.sqrt(12)
    # downside_dev == 0 con excess > 0 ⇒ +inf (fondo sin riesgo downside)
    # rank() lo ubica en el tope (~1.0), se mantiene intencionalmente.
    sortino = sortino.replace([-np.inf], np.nan)
    sortino = sortino.where(all_present, np.nan)
    df[f"target_sortino_{horizon}m"] = sortino
    df[f"target_sortino_rank_{horizon}m"] = df.groupby("mes")[
        f"target_sortino_{horizon}m"
    ].rank(pct=True, method="average")

    # --- Target 4: Max drawdown forward (lente de validación)
    # Peor caída pico-valle dentro del horizonte. Mide riesgo de cola
    # realizado. Captura "qué tan mal momento tuvo el fondo en el período"
    # más allá de la volatilidad promedio.
    def _fwd_max_dd(row):
        if row.isna().any():
            return np.nan
        cum = (1 + row).cumprod()
        return float((cum / cum.cummax() - 1).min())

    df[f"target_max_dd_{horizon}m"] = fwd.apply(_fwd_max_dd, axis=1)

    return df


# --- pipeline -------------------------------------------------------------

CORE_FEATURES = [
    "ret_1m", "ret_3m", "ret_6m", "ret_12m",
    "vol_12m", "max_dd_12m", "sharpe_12m",
    "vol_intrames", "autocorr_diaria", "ratio_dias_cero",
    "skewness_12m", "hit_rate_12m", "distribution_yield_12m",
    "persistencia_rank_12m",
]
EXTENDED_FEATURES = [
    "fee", "log_n_instrumentos", "pct_acum",
    "fee_observado", "concentracion_disponible",
]
RANK_FEATURES = [f"{c}_rank" for c in RANK_COLS]
FEATURE_COLS = CORE_FEATURES + EXTENDED_FEATURES + RANK_FEATURES

# Subconjunto curado: 5 features con señal real e independiente entre sí.
# Elimina redundancia raw/rank, features con cobertura <12% (fee, concentración),
# y features sin señal cross-seccional (ratio_dias_cero, vol_12m).
REDUCED_FEATURES = [
    "sharpe_12m_rank",    # momentum risk-adjusted (subsume ret_12m, IR=+0.10)
    "ret_3m_rank",        # momentum corto, complementario al de 12m (IR=+0.08)
    "max_dd_12m",         # riesgo de cola, no colineal con sharpe (IR=-0.10)
    "vol_intrames",       # microestructura intra-mes (IR=+0.05)
    "autocorr_diaria",    # persistencia intra-mes (IR=+0.07)
]


def build_features(panel: pd.DataFrame, horizons: list[int] | None = None) -> pd.DataFrame:
    """Pipeline completo. Devuelve el panel con todas las columnas — la
    selección/filtrado de filas modelables se hace en el script de
    entrenamiento.

    Parameters
    ----------
    horizons : list of int, optional
        Horizontes de target en meses. Default [12] para backward-compat.
    """
    if horizons is None:
        horizons = [12]
    df = panel.copy().sort_values(["fondo", "mes"]).reset_index(drop=True)
    df = add_return_features(df)          # ret_1m, ret_3m, ret_6m, ret_12m
    df = add_risk_features(df)            # vol_12m, max_dd_12m, sharpe_12m
    df = add_stylistic_features(df)       # skewness_12m, hit_rate_12m
    df = add_extended_features(df)        # concentracion_disponible, log_n_instrumentos (fee_observado viene de data.py)
    df = impute_extended(df)              # rellenar NaN con mediana cross-seccional
    df = add_cross_sectional_ranks(df)    # *_rank (percentil dentro del mes) sobre RANK_COLS
    df = add_persistence_features(df)     # persistencia_rank_12m (depende de sharpe_12m_rank)
    for h in horizons:
        df = add_target(df, horizon=h)    # target_ret_{h}m, target_rank_{h}m, target_sharpe_{h}m, target_sharpe_rank_{h}m
    return df


MIN_FUND_MONTHS = 36


def get_modeling_frame(
    df: pd.DataFrame,
    horizon: int = 6,
    target: str = "sortino",
    min_fund_months: int = MIN_FUND_MONTHS,
    require_target: bool = True,
) -> pd.DataFrame:
    """Filtra el panel a observaciones modelables.

    Aplica tres filtros en orden:
      1. Cobertura mínima del fondo: solo se modelan fondos con al menos
         `min_fund_months` meses de historia en el panel (default 36).
         Justificación: con menos de 3 años no hay evidencia suficiente
         para evaluar al fondo (después del warmup de 12m + horizonte
         forward de 12m, quedarían <12 obs entrenables del fondo).
      2. Target del horizonte indicado no-NaN (si require_target=True).
      3. Todas las features CORE no-NaN.

    Las features EXTENDED ya están imputadas en build_features.

    Parameters
    ----------
    horizon : int
        Horizonte forward del target en meses (default 12).
    target : {"sortino", "sharpe", "ret"}
        Cuál columna usar como target principal del entrenamiento:
        - "sortino" (recomendado): target_sortino_rank_{h}m.
          Solo penaliza volatilidad a la baja (Sortino & Price 1994).
          Más apropiado para renta variable donde upside vol es deseable.
        - "sharpe": target_sharpe_rank_{h}m.
          Premia retorno ajustado por riesgo total.
        - "ret": target_rank_{h}m (retorno forward simple). Mantiene
          compatibilidad con la versión anterior del pipeline.
    min_fund_months : int
        Mínimo de meses de historia requeridos por fondo para incluirlo
        (default 36).
    require_target : bool
        Si True (default), filtra filas donde el target sea NaN. Si False,
        permite filas sin target (útil para scoring extendido donde las
        features backward-looking existen pero el target forward no).
    """
    if target == "sortino":
        target_col = f"target_sortino_rank_{horizon}m"
    elif target == "sharpe":
        target_col = f"target_sharpe_rank_{horizon}m"
    elif target == "ret":
        target_col = f"target_rank_{horizon}m"
    else:
        raise ValueError(f"target debe ser 'sortino', 'sharpe' o 'ret', no {target!r}")

    # Filtro de cobertura mínima por fondo
    n_meses_por_fondo = df.groupby("fondo")["mes"].count()
    fondos_validos = n_meses_por_fondo[n_meses_por_fondo >= min_fund_months].index
    df = df[df["fondo"].isin(fondos_validos)]

    if require_target:
        mask = df[target_col].notna() & df[CORE_FEATURES].notna().all(axis=1)
    else:
        mask = df[CORE_FEATURES].notna().all(axis=1)
    return df.loc[mask].copy()
