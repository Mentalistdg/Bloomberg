"""Módulo de carga de datos y construcción del panel mensual base.

Este módulo es el punto de entrada de datos del pipeline. Se encarga de:
  1. Conectar a la base sqlite y cargar las 3 tablas crudas (historico,
     fees, subyacentes).
  2. Calcular el retorno total diario por fondo, combinando variación
     de NAV y eventos de capital (distribuciones, dividendos).
  3. Construir un total return index (TRI) y agregarlo a frecuencia
     mensual (panel fondo x fin-de-mes).
  4. Adjuntar fees y datos de concentración al panel mensual.

El retorno total que recibe el inversionista incluye dos componentes:
    ret_total_t = (precio_t / precio_{t-1} - 1) + evento_pct_t
                   ─────────────────────────────   ─────────────
                   variación de NAV                distribución de capital

El output de este módulo (panel mensual) es el input para la
ingeniería de features en features.py.

Usado por: scripts/01_build_features.py
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

    # Paso 1: winsorizar eventos de capital al percentil 99.5 para controlar
    # outliers extremos (ej. reorganizaciones corporativas, devoluciones de capital)
    cap = df["evento_pct"].replace(0, np.nan).abs().quantile(0.995)
    df["evento_pct_w"] = df["evento_pct"].clip(lower=-cap, upper=cap)

    # Paso 2: retorno por variación de precio (NAV day-over-day)
    df["ret_precio"] = df.groupby("fondo")["precio"].pct_change()

    # Paso 3: retorno total = variación de precio + evento de capital
    df["ret_total"] = df["ret_precio"].fillna(0) + df["evento_pct_w"].fillna(0)

    # Paso 4: la primera observación de cada fondo no tiene retorno calculable
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


def compute_intramonth_features(daily: pd.DataFrame) -> pd.DataFrame:
    """Calcula features derivadas de datos diarios dentro de cada mes.

    Para cada fondo y mes calcula:
      - vol_intrames: desviación estándar de ret_total dentro del mes
        (mín 2 obs, sino NaN). Captura volatilidad de alta frecuencia
        perdida al agregar a mensual.
      - autocorr_diaria: autocorrelación lag-1 de ret_total dentro del mes
        (mín 5 obs). Señal de momentum/reversión intra-mes.
      - ratio_dias_cero: fracción de días con |ret_total| < 0.0001.
        Proxy de liquidez — fondos con muchos días "planos" pueden tener
        pricing stale o baja actividad.
    """
    df = daily.dropna(subset=["ret_total"]).copy()
    df["mes"] = df["fecha"].dt.to_period("M").dt.to_timestamp("M")

    def _agg(g):
        r = g["ret_total"]
        n = len(r)
        vol = r.std() if n >= 2 else np.nan
        ac = r.autocorr(lag=1) if n >= 5 else np.nan
        cero = (r.abs() < 0.0001).mean()
        return pd.Series({"vol_intrames": vol, "autocorr_diaria": ac, "ratio_dias_cero": cero})

    result = df.groupby(["fondo", "mes"]).apply(_agg, include_groups=False).reset_index()
    return result


def build_monthly_panel(daily: pd.DataFrame) -> pd.DataFrame:
    """Construye panel mensual fondo × fin-de-mes.

    Para cada fondo y fin de mes:
      - tri_eom: total return index al cierre del mes
      - ret_mensual: retorno total del mes (compuesto desde ret_total diario)
      - n_dias_obs: cantidad de días con observación en el mes
      - vol_intrames, autocorr_diaria, ratio_dias_cero: features intra-mes
    """
    # Paso 1: construir TRI diario y asignar cada fecha a su mes
    tri = build_total_return_index(daily)
    tri["mes"] = tri["fecha"].dt.to_period("M").dt.to_timestamp("M")

    # Paso 2: para cada fondo y mes, tomar el TRI del último día hábil
    # (fin de mes efectivo) — esto da el "precio de cierre mensual"
    eom = tri.sort_values("fecha").groupby(["fondo", "mes"]).tail(1)
    eom = eom.rename(columns={"tri": "tri_eom"})[["fondo", "mes", "tri_eom"]]

    # Paso 3: retorno mensual = variación del TRI entre fin de mes actual y anterior
    eom["ret_mensual"] = eom.groupby("fondo")["tri_eom"].pct_change()

    # Paso 4: contar días con observación por mes (medida de liquidez/cobertura)
    n_obs = (
        daily.dropna(subset=["ret_total"])
        .assign(mes=lambda x: x["fecha"].dt.to_period("M").dt.to_timestamp("M"))
        .groupby(["fondo", "mes"])
        .size()
        .rename("n_dias_obs")
        .reset_index()
    )
    panel = eom.merge(n_obs, on=["fondo", "mes"], how="left")

    # Paso 5: features intra-mes derivadas de datos diarios
    intra = compute_intramonth_features(daily)
    panel = panel.merge(intra, on=["fondo", "mes"], how="left")

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
