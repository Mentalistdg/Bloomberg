"""Ingeniería de features sobre el panel mensual.

Convención de tiempo:
    Todas las features al final del mes t usan exclusivamente información
    disponible hasta el cierre del mes t (rolling windows hacia atrás).
    El target a 12 meses forward es prod(1+ret_{t+1..t+12})-1, percentil
    cross-seccional dentro del universo activo en cada mes.

Convención sobre fees y NAV:
    El precio es NAV de mutual fund USA, ya neto de expense ratio
    (descontado diariamente del NAV). Por eso el target derivado de NAV
    no requiere ajuste adicional por fees. La columna `fee` se usa como
    feature explicativa.

División en dos grupos de features:
    CORE:      derivadas del retorno (densas tras warmup de 12 meses).
               Se exige no-NaN — observaciones sin historia suficiente
               se descartan del conjunto modelable.
    EXTENDED:  fee y concentración. Cobertura limitada en el dataset
               (12% y 5% post-ffill). Se imputan con la mediana
               cross-seccional del mes y se agrega flag binaria
               `*_disponible` para que el modelo capture el efecto
               de información faltante.
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
    g = df.groupby("fondo", group_keys=False)
    for h in [1, 3, 6, 12]:
        df[f"ret_{h}m"] = g["ret_mensual"].apply(lambda s: _rolling_compound(s, h))
    return df


def add_risk_features(df: pd.DataFrame) -> pd.DataFrame:
    g = df.groupby("fondo", group_keys=False)["ret_mensual"]
    df["vol_12m"] = g.apply(lambda s: s.rolling(12, min_periods=12).std() * np.sqrt(12))
    df["max_dd_12m"] = df.groupby("fondo", group_keys=False)["ret_mensual"].apply(
        lambda s: _rolling_max_drawdown(s, 12)
    )
    df["sharpe_12m"] = df["ret_12m"] / df["vol_12m"]
    return df


# --- features estructurales ----------------------------------------------

def add_extended_features(df: pd.DataFrame) -> pd.DataFrame:
    df["fee_disponible"] = df["fee"].notna().astype(int)
    df["concentracion_disponible"] = df["n_instrumentos"].notna().astype(int)
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

RANK_COLS = ["ret_3m", "ret_12m", "vol_12m", "sharpe_12m", "fee", "log_n_instrumentos"]


def add_cross_sectional_ranks(df: pd.DataFrame) -> pd.DataFrame:
    for c in RANK_COLS:
        df[f"{c}_rank"] = df.groupby("mes")[c].rank(pct=True, method="average")
    return df


# --- target ---------------------------------------------------------------

def add_target(df: pd.DataFrame, horizon: int = 12) -> pd.DataFrame:
    """Retorno total compuesto a `horizon` meses forward y su percentil
    cross-seccional dentro de los fondos con target observable en ese mes.
    """
    fwd = pd.concat(
        [df.groupby("fondo")["ret_mensual"].shift(-i) for i in range(1, horizon + 1)],
        axis=1,
    )
    df[f"target_ret_{horizon}m"] = (1 + fwd).prod(axis=1, min_count=horizon) - 1
    df["target_rank"] = df.groupby("mes")[f"target_ret_{horizon}m"].rank(
        pct=True, method="average"
    )
    return df


# --- pipeline -------------------------------------------------------------

CORE_FEATURES = [
    "ret_1m", "ret_3m", "ret_6m", "ret_12m",
    "vol_12m", "max_dd_12m", "sharpe_12m",
]
EXTENDED_FEATURES = [
    "fee", "log_n_instrumentos",
    "fee_disponible", "concentracion_disponible",
]
RANK_FEATURES = [f"{c}_rank" for c in RANK_COLS]
FEATURE_COLS = CORE_FEATURES + EXTENDED_FEATURES + RANK_FEATURES


def build_features(panel: pd.DataFrame) -> pd.DataFrame:
    """Pipeline completo. Devuelve el panel con todas las columnas — la
    selección/filtrado de filas modelables se hace en el script de
    entrenamiento."""
    df = panel.copy().sort_values(["fondo", "mes"]).reset_index(drop=True)
    df = add_return_features(df)
    df = add_risk_features(df)
    df = add_extended_features(df)
    df = impute_extended(df)
    df = add_cross_sectional_ranks(df)
    df = add_target(df, horizon=12)
    return df


def get_modeling_frame(df: pd.DataFrame) -> pd.DataFrame:
    """Filtra el panel a observaciones modelables: target observado y
    todas las features CORE no-NaN. Las features EXTENDED ya están
    imputadas en build_features."""
    mask = df["target_rank"].notna() & df[CORE_FEATURES].notna().all(axis=1)
    return df.loc[mask].copy()
