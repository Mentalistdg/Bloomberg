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

    EXTENDED (4):  fee, log_n_instrumentos + 2 flags binarias (*_disponible).
                   Cobertura limitada (12% fee, 5% concentración post-ffill).
                   Se imputan con mediana cross-seccional y se marca con
                   flag para que el modelo distinga dato real vs imputado.

    RANK (9):      Percentil cross-seccional dentro del mes para: ret_3m,
                   ret_12m, vol_12m, sharpe_12m, fee, log_n_instrumentos,
                   vol_intrames, autocorr_diaria, ratio_dias_cero.
                   Aportan robustez ante cambios de escala entre períodos.

Usado por: scripts/03_build_features_full.py
"""

from __future__ import annotations

import numpy as np
import pandas as pd


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
    # Sharpe simplificado (sin risk-free rate): retorno 12m / volatilidad 12m
    df["sharpe_12m"] = df["ret_12m"] / df["vol_12m"]
    return df


# --- features estructurales ----------------------------------------------

def add_extended_features(df: pd.DataFrame) -> pd.DataFrame:
    """Prepara features de fee y concentración. Antes de imputar, se crean
    flags binarias (0/1) que marcan si el dato original estaba disponible,
    para que el modelo pueda distinguir dato real de imputado."""
    # Flag: 1 si el fondo tiene fee reportado, 0 si será imputado
    df["fee_disponible"] = df["fee"].notna().astype(int)
    # Flag: 1 si el fondo tiene dato de concentración, 0 si será imputado
    df["concentracion_disponible"] = df["n_instrumentos"].notna().astype(int)
    # Log-transform para reducir asimetría de n_instrumentos
    df["log_n_instrumentos"] = np.log1p(df["n_instrumentos"])
    return df


def impute_extended(df: pd.DataFrame) -> pd.DataFrame:
    """Imputa fee y log_n_instrumentos con la mediana cross-seccional
    del universo activo en cada mes. La flag `*_disponible` ya captura
    el efecto de información faltante."""
    for col in ["fee", "log_n_instrumentos"]:
        med = df.groupby("mes")[col].transform("median")
        df[col] = df[col].fillna(med)
        # fallback final: mediana global
        df[col] = df[col].fillna(df[col].median())
    return df


# --- ranks cross-seccionales ---------------------------------------------

RANK_COLS = [
    "ret_3m", "ret_12m", "vol_12m", "sharpe_12m", "fee", "log_n_instrumentos",
    "vol_intrames", "autocorr_diaria", "ratio_dias_cero",
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
    """Retorno total compuesto a `horizon` meses forward y su percentil
    cross-seccional dentro de los fondos con target observable en ese mes.

    Genera columnas: target_ret_{horizon}m y target_rank_{horizon}m.
    """
    fwd = pd.concat(
        [df.groupby("fondo")["ret_mensual"].shift(-i) for i in range(1, horizon + 1)],
        axis=1,
    )
    # Retorno compuesto forward: prod(1 + r_{t+1..t+h}) - 1
    # min_count=horizon asegura NaN si faltan meses (fondo no sobrevive h meses)
    df[f"target_ret_{horizon}m"] = (1 + fwd).prod(axis=1, min_count=horizon) - 1
    # Percentil cross-seccional del target dentro de cada mes
    df[f"target_rank_{horizon}m"] = df.groupby("mes")[f"target_ret_{horizon}m"].rank(
        pct=True, method="average"
    )
    return df


# --- pipeline -------------------------------------------------------------

CORE_FEATURES = [
    "ret_1m", "ret_3m", "ret_6m", "ret_12m",
    "vol_12m", "max_dd_12m", "sharpe_12m",
    "vol_intrames", "autocorr_diaria", "ratio_dias_cero",
]
EXTENDED_FEATURES = [
    "fee", "log_n_instrumentos",
    "fee_disponible", "concentracion_disponible",
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
    df = add_extended_features(df)        # fee_disponible, concentracion_disponible, log_n_instrumentos
    df = impute_extended(df)              # rellenar NaN con mediana cross-seccional
    df = add_cross_sectional_ranks(df)    # 9 features *_rank (percentil dentro del mes)
    for h in horizons:
        df = add_target(df, horizon=h)    # target_ret_{h}m y target_rank_{h}m
    return df


def get_modeling_frame(df: pd.DataFrame, horizon: int = 12) -> pd.DataFrame:
    """Filtra el panel a observaciones modelables: target observado para
    el horizonte indicado y todas las features CORE no-NaN. Las features
    EXTENDED ya están imputadas en build_features."""
    target_col = f"target_rank_{horizon}m"
    mask = df[target_col].notna() & df[CORE_FEATURES].notna().all(axis=1)
    return df.loc[mask].copy()
