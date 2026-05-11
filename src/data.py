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
    """Carga la tabla fees y agrupa duplicados (mismo fondo+fecha).

    Se descartan registros con fee == 0: en el dataset, las entradas con
    fee exactamente 0 corresponden a un artefacto de carga del último
    snapshot (2026-04-09) donde 9 fondos que previamente tenían fees
    no-cero (0.15–0.46%) aparecen con 0. Tratar estos ceros como NaN
    permite que el ffill/bfill en attach_fees_monthly propague el último
    fee válido conocido.
    """
    with open_db() as con:
        df = pd.read_sql(
            "SELECT fecha, fondo, fee FROM fees",
            con,
            parse_dates=["fecha"],
        )
    df = (
        df.dropna(subset=["fee"])
        .query("fee > 0")
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


def _flag_anomalous_returns(df: pd.DataFrame) -> pd.Series:
    """Detecta retornos diarios anómalos por gaps temporales o cambios de unidad.

    Regla 1 — Gap temporal: si entre dos observaciones consecutivas de un
    fondo hay > 45 días calendario, el pct_change no representa un retorno
    diario real (puede ser un fondo con reporting irregular).

    Regla 2 — Cambio estructural de nivel: si |ret| > 90% y el precio se
    mantiene en el nuevo nivel (no revierte en las siguientes 3 obs),
    indica un cambio de unidades/denominación, no un retorno real.

    Retorna una Series booleana (True = anómalo → tratar como NaN).
    """
    is_anomalous = pd.Series(False, index=df.index)

    for fondo, g in df.groupby("fondo"):
        # Regla 1: gap > 45 días entre observaciones consecutivas
        gaps = g["fecha"].diff().dt.days
        gap_mask = gaps > 45
        is_anomalous.loc[g.index[gap_mask]] = True

        # Regla 2: |ret| > 90% sin reversión (cambio de unidades)
        extreme_idx = g[g["ret_total_raw"].abs() > 0.90].index
        for idx in extreme_idx:
            pos = g.index.get_loc(idx)
            if pos + 3 < len(g):
                next_prices = g.iloc[pos + 1 : pos + 4]["precio"].values
                curr_price = g.loc[idx, "precio"]
                if curr_price > 0 and all(
                    abs(p / curr_price - 1) < 0.5 for p in next_prices
                ):
                    is_anomalous.loc[idx] = True

    return is_anomalous


def compute_daily_total_return(historico: pd.DataFrame) -> pd.DataFrame:
    """Calcula el retorno total diario por fondo.

    ret_precio_t = precio_t / precio_{t-1} - 1
    ret_total_t  = ret_precio_t + evento_pct_t

    El ret_total se winsoriza al p99.5 para neutralizar outliers extremos.
    Se winsoriza el retorno TOTAL (no solo el evento_pct) porque en eventos
    de split/reverse-split el precio cae fuertemente y el evento_pct
    compensa — winsorizar solo el evento destruye la cancelación natural
    y genera drawdowns ficticios de 70-80%.
    """
    df = historico.copy()
    df = df.sort_values(["fondo", "fecha"]).reset_index(drop=True)

    # Paso 1: retorno por variación de precio (NAV day-over-day)
    df["ret_precio"] = df.groupby("fondo")["precio"].pct_change()

    # Paso 2: retorno total crudo = variación de precio + evento de capital
    df["evento_pct_w"] = df["evento_pct"]  # mantener columna por compat
    df["ret_total_raw"] = df["ret_precio"].fillna(0) + df["evento_pct"].fillna(0)

    # Paso 2.5: detectar retornos anómalos (gaps de datos, cambios de unidad)
    anomalous = _flag_anomalous_returns(df)
    n_flagged = anomalous.sum()
    if n_flagged > 0:
        fondos_afectados = df.loc[anomalous, "fondo"].nunique()
        print(f"    data quality: {n_flagged} retornos anomalos en "
              f"{fondos_afectados} fondos -> NaN")
    df.loc[anomalous, "ret_total_raw"] = np.nan

    # Paso 3: winsorizar el retorno TOTAL al p99.5 para controlar outliers
    # genuinos sin destruir la cancelación precio/evento en splits
    cap = df["ret_total_raw"].replace(0, np.nan).abs().quantile(0.995)
    df["ret_total"] = df["ret_total_raw"].clip(lower=-cap, upper=cap)

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
      - evento_pct_mes: suma de evento_pct (winsorizado) dentro del mes.
        Captura magnitud de distribuciones de capital y dividendos en
        el mes. Insumo para construir distribution_yield_12m en features.py.
    """
    df = daily.dropna(subset=["ret_total"]).copy()
    df["mes"] = df["fecha"].dt.to_period("M").dt.to_timestamp("M")

    def _agg(g):
        r = g["ret_total"]
        n = len(r)
        vol = r.std() if n >= 2 else np.nan
        ac = r.autocorr(lag=1) if n >= 5 else np.nan
        cero = (r.abs() < 0.0001).mean()
        evt = g["evento_pct_w"].fillna(0).sum()
        return pd.Series({
            "vol_intrames": vol,
            "autocorr_diaria": ac,
            "ratio_dias_cero": cero,
            "evento_pct_mes": evt,
        })

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
    """Adjunta el fee vigente al cierre de cada mes por fondo.

    Particularidad del dataset: la tabla `fees` solo reporta valores a
    partir de 2024-01-31. Un ffill puro (mirando solo al pasado) deja
    ~88% del panel mensual en NaN, porque casi toda la historia del
    universo es pre-2024.

    Mitigación: el fee de un mutual fund USA es estructuralmente estable
    en el tiempo. Empíricamente en este dataset, dentro del subperíodo
    donde sí hay múltiples observaciones (2024-2026), la mediana de
    fondos tiene **un único valor de fee** y la desviación intra-fondo
    es esencialmente 0. Esto permite imputar hacia atrás dentro del
    mismo fondo (bfill) bajo el supuesto explícito de fee constante,
    llevando la cobertura efectiva a ~98%.

    Se preserva la bandera binaria `fee_observado`:
        - 1 si el fee de ese mes específico venía reportado en el
          dataset original (período 2024+).
        - 0 si fue rellenado por ffill o bfill desde otro mes.
    Esta flag captura indirectamente el efecto de período de reporte
    y, por extensión, parte del survivorship bias del dataset (los
    fondos con `fee_observado=1` son los que sobrevivieron hasta 2024).

    Para los ~2 fondos sin NINGÚN reporte de fee se deja NaN; la
    imputación final por mediana cross-seccional ocurre en features.py.
    """
    fees_sorted = fees.sort_values(["fondo", "fecha"])
    fees_eom = (
        fees_sorted.assign(mes=lambda x: x["fecha"].dt.to_period("M").dt.to_timestamp("M"))
        .groupby(["fondo", "mes"])
        .tail(1)[["fondo", "mes", "fee"]]
    )
    p = panel.merge(fees_eom, on=["fondo", "mes"], how="left")

    # Flag ANTES de cualquier imputación: 1 si el fee venía reportado
    # originalmente en este mes específico, 0 si no.
    p["fee_observado"] = p["fee"].notna().astype(int)

    # Paso 1 — imputación intra-fondo dentro del rango temporal del panel:
    # ffill + bfill. Justificada por estabilidad estructural del fee
    # (verificable empíricamente en el subperíodo observado).
    p["fee"] = p.groupby("fondo")["fee"].transform(lambda s: s.ffill().bfill())

    # Paso 2 — fondos que murieron antes de que empezaran los reportes
    # de fee (2024-01-31): el panel temporal del fondo y los reportes
    # de fee no se intersectan, así que el bfill no rescata nada. Bajo el
    # mismo supuesto de fee estructural, usamos el fee promedio reportado
    # del fondo (de la tabla `fees` completa) para llenar todo su panel.
    fondos_sin_fee = p.loc[p["fee"].isna(), "fondo"].unique()
    if len(fondos_sin_fee) > 0:
        fee_fondo = (
            fees.dropna(subset=["fee"])
            .groupby("fondo")["fee"]
            .mean()
            .to_dict()
        )
        mask = p["fee"].isna() & p["fondo"].isin(fondos_sin_fee)
        p.loc[mask, "fee"] = p.loc[mask, "fondo"].map(fee_fondo)

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
