"""Carga del dataset y construcción del retorno total diario y panel mensual.

El retorno total que recibe el inversionista incluye dos componentes:
- el retorno por variación de NAV: precio_t / precio_{t-1} - 1
- el retorno por distribución / evento de capital: evento_pct_t

La fórmula correcta usada en todo el pipeline es:
    ret_total_t = (precio_t / precio_{t-1} - 1) + evento_pct_t
"""

from __future__ import annotations

import sqlite3
from contextlib import contextmanager

import numpy as np
import pandas as pd

from src.paths import DB_PATH


@contextmanager
def open_db():
    """Abre la base sqlite y la cierra al salir."""
    con = sqlite3.connect(DB_PATH)
    try:
        yield con
    finally:
        con.close()


def load_historico() -> pd.DataFrame:
    """Carga la tabla historico con tipos correctos.

    Returns
    -------
    DataFrame con columnas: fecha (datetime), fondo (str), precio (float),
    evento_pct (float).
    """
    with open_db() as con:
        df = pd.read_sql(
            "SELECT fecha, securities AS fondo, precio, evento_pct FROM historico",
            con,
            parse_dates=["fecha"],
        )
    df = df.dropna(subset=["precio"]).sort_values(["fondo", "fecha"]).reset_index(drop=True)
    return df


def load_fees() -> pd.DataFrame:
    """Carga la tabla fees y agrupa duplicados (mismo fondo+fecha)."""
    with open_db() as con:
        df = pd.read_sql(
            "SELECT fecha, fondo, fee FROM fees",
            con,
            parse_dates=["fecha"],
        )
    df = (
        df.dropna(subset=["fee"])
        .groupby(["fondo", "fecha"], as_index=False)["fee"]
        .mean()
        .sort_values(["fondo", "fecha"])
        .reset_index(drop=True)
    )
    return df


def load_subyacentes() -> pd.DataFrame:
    """Carga la tabla subyacentes (snapshot de concentración por fondo)."""
    with open_db() as con:
        df = pd.read_sql(
            "SELECT fecha, nemo_fondo AS fondo, pct_acum, n_instrumentos FROM subyacentes",
            con,
            parse_dates=["fecha"],
        )
    df = df.sort_values(["fondo", "fecha"]).reset_index(drop=True)
    return df


def compute_daily_total_return(historico: pd.DataFrame) -> pd.DataFrame:
    """Calcula el retorno total diario por fondo.

    ret_precio_t = precio_t / precio_{t-1} - 1
    ret_total_t  = ret_precio_t + evento_pct_t

    El evento_pct se winsoriza al p99.5 para neutralizar outliers de
    eventos de capital extremos (devolución de capital, reorganizaciones)
    que distorsionan la serie.
    """
    df = historico.copy()
    df = df.sort_values(["fondo", "fecha"]).reset_index(drop=True)

    cap = df["evento_pct"].replace(0, np.nan).abs().quantile(0.995)
    df["evento_pct_w"] = df["evento_pct"].clip(lower=-cap, upper=cap)

    df["ret_precio"] = df.groupby("fondo")["precio"].pct_change()
    df["ret_total"] = df["ret_precio"].fillna(0) + df["evento_pct_w"].fillna(0)

    df.loc[df.groupby("fondo").head(1).index, "ret_total"] = np.nan
    return df[["fecha", "fondo", "precio", "evento_pct", "evento_pct_w", "ret_precio", "ret_total"]]


def build_total_return_index(daily: pd.DataFrame) -> pd.DataFrame:
    """Construye un total return index (TRI) por fondo, normalizado a 1.0
    en la primera observación.

    TRI_t = TRI_{t-1} * (1 + ret_total_t)
    """
    df = daily.copy()
    df["one_plus_r"] = (1 + df["ret_total"].fillna(0))
    df["tri"] = df.groupby("fondo")["one_plus_r"].cumprod()
    return df[["fecha", "fondo", "tri"]]


def build_monthly_panel(daily: pd.DataFrame) -> pd.DataFrame:
    """Construye panel mensual fondo × fin-de-mes.

    Para cada fondo y fin de mes:
      - tri_eom: total return index al cierre del mes
      - ret_mensual: retorno total del mes (compuesto desde ret_total diario)
      - n_dias_obs: cantidad de días con observación en el mes
    """
    tri = build_total_return_index(daily)
    tri["mes"] = tri["fecha"].dt.to_period("M").dt.to_timestamp("M")

    eom = tri.sort_values("fecha").groupby(["fondo", "mes"]).tail(1)
    eom = eom.rename(columns={"tri": "tri_eom"})[["fondo", "mes", "tri_eom"]]

    eom["ret_mensual"] = eom.groupby("fondo")["tri_eom"].pct_change()

    n_obs = (
        daily.dropna(subset=["ret_total"])
        .assign(mes=lambda x: x["fecha"].dt.to_period("M").dt.to_timestamp("M"))
        .groupby(["fondo", "mes"])
        .size()
        .rename("n_dias_obs")
        .reset_index()
    )
    panel = eom.merge(n_obs, on=["fondo", "mes"], how="left")
    panel = panel.sort_values(["fondo", "mes"]).reset_index(drop=True)
    return panel


def attach_fees_monthly(panel: pd.DataFrame, fees: pd.DataFrame) -> pd.DataFrame:
    """Adjunta el fee vigente al cierre de cada mes por fondo (forward-fill).

    Para fondos sin ningún fee reportado se deja NaN — la imputación a
    nivel cross-seccional se hace en features.py con flag explícita.
    """
    fees_sorted = fees.sort_values(["fondo", "fecha"])
    fees_eom = (
        fees_sorted.assign(mes=lambda x: x["fecha"].dt.to_period("M").dt.to_timestamp("M"))
        .groupby(["fondo", "mes"])
        .tail(1)[["fondo", "mes", "fee"]]
    )
    p = panel.merge(fees_eom, on=["fondo", "mes"], how="left")
    p["fee"] = p.groupby("fondo")["fee"].ffill()
    return p


def attach_subyacentes(panel: pd.DataFrame, sub: pd.DataFrame) -> pd.DataFrame:
    """Adjunta la concentración por fondo. La regla del dataset es:
    n_instrumentos = mínima cantidad de holdings necesarios para que la
    suma de pesos cruce el 30% del AUM. pct_acum es el porcentaje exacto
    acumulado en ese punto (siempre >= 30% por construcción).

    La tabla solo tiene snapshots — para cada fondo se usa la última
    observación disponible al cierre del mes (forward-fill).
    """
    s = sub.sort_values(["fondo", "fecha"]).copy()
    s["mes"] = s["fecha"].dt.to_period("M").dt.to_timestamp("M")
    s = s.groupby(["fondo", "mes"]).tail(1)[["fondo", "mes", "pct_acum", "n_instrumentos"]]
    p = panel.merge(s, on=["fondo", "mes"], how="left")
    p[["pct_acum", "n_instrumentos"]] = p.groupby("fondo")[["pct_acum", "n_instrumentos"]].ffill()
    return p
