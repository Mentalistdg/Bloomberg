"""Script 05/05 — Transforma artefactos del pipeline en JSONs para el dashboard.

Este script es el puente entre el pipeline de modelado (scripts 01-04)
y la aplicación web (FastAPI + React). Toma los artefactos producidos
por los scripts anteriores y los convierte en JSONs optimizados para
consumo del frontend.

No realiza ningún cálculo de modelado — solo reestructura datos.

Input:  artifacts/scores.parquet, metrics.json, drivers_elastic.csv,
        drivers_lgbm.csv, panel_features.parquet,
        assets/usa_fondos_pp.sqlite (para curvas de equity sin winsorización)
Output: app/backend/data/
          funds_summary.json  — ranking de fondos con score y métricas
          fund_detail.json    — serie temporal mensual por fondo
          drivers.json        — coeficientes ElasticNet e importancias LightGBM
          backtest.json       — Sortino por decil por mes (backtest visual)
          meta.json           — metadata global (fechas, n_fondos, métricas resumen)

Uso:
    python -m scripts.05_build_app_data
"""

from __future__ import annotations

import json
import math
import sys

import numpy as np
import pandas as pd

from dateutil.relativedelta import relativedelta
from pypfopt import expected_returns, risk_models, EfficientFrontier

from src.data import load_historico
from src.features import RF_ANNUAL, RF_MONTHLY
from src.paths import APP_DATA_DIR, ARTIFACTS_DIR

SCORE_COL = "score_elastic"  # modelo principal


def _sanitize(obj):
    """Reemplaza NaN/Inf con None recursivamente para generar JSON válido."""
    if isinstance(obj, float):
        return None if (math.isnan(obj) or math.isinf(obj)) else obj
    if isinstance(obj, dict):
        return {k: _sanitize(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_sanitize(v) for v in obj]
    return obj


def _build_display_equity(fondos: list[str]) -> dict[str, pd.DataFrame]:
    """Construye curvas de equity desde precios crudos, SIN winsorización.

    La winsorización (p99.5) se usa en el pipeline de modelado para
    controlar outliers en features, pero distorsiona las curvas de
    display: en fondos con movimientos extremos legítimos (crisis 2008,
    COVID 2020) el clipping acumula errores de hasta 30%+ en la
    trayectoria. Para visualización se usa el retorno total real.

    Returns: dict fondo → DataFrame con columnas [mes, fecha, equity].
             `fecha` es la fecha real de la última observación en el mes.
    """
    hist = load_historico()
    hist = hist[hist["fondo"].isin(fondos)].sort_values(["fondo", "fecha"])

    # Retorno total diario = variación NAV + eventos de capital (sin clip)
    hist["ret_precio"] = hist.groupby("fondo")["precio"].pct_change()
    hist["ret_total"] = hist["ret_precio"].fillna(0) + hist["evento_pct"].fillna(0)
    hist.loc[hist.groupby("fondo").head(1).index, "ret_total"] = 0

    # Neutralizar retornos por gaps de datos (> 45 días entre observaciones)
    hist["_gap_days"] = hist.groupby("fondo")["fecha"].diff().dt.days
    hist.loc[hist["_gap_days"] > 45, "ret_total"] = 0
    hist = hist.drop(columns=["_gap_days"])

    # TRI diario
    hist["tri"] = hist.groupby("fondo")["ret_total"].transform(
        lambda s: (1 + s).cumprod()
    )

    # EOM: último día hábil de cada mes
    hist["mes"] = hist["fecha"].dt.to_period("M").dt.to_timestamp("M")
    eom = hist.sort_values("fecha").groupby(["fondo", "mes"]).tail(1)
    eom = eom[["fondo", "mes", "fecha", "tri"]].copy()

    # Normalizar cada fondo a base 100
    eom["equity"] = eom.groupby("fondo")["tri"].transform(
        lambda s: 100.0 * s / s.iloc[0]
    )

    result = {}
    for fid, g in eom.groupby("fondo"):
        result[fid] = g[["mes", "fecha", "equity"]].reset_index(drop=True)
    return result


def _compute_perf_metrics(returns: np.ndarray) -> dict:
    """Métricas de performance a partir de serie de retornos mensuales."""
    n = len(returns)
    if n == 0:
        return {}
    mean_ret = np.mean(returns)
    annual_return = (1 + mean_ret) ** 12 - 1
    annual_vol = np.std(returns, ddof=1) * np.sqrt(12)
    downside = np.minimum(returns - RF_MONTHLY, 0)
    downside_dev = np.sqrt(np.mean(downside ** 2)) * np.sqrt(12)
    sortino = (annual_return - RF_ANNUAL) / downside_dev if downside_dev > 0 else None
    # Max drawdown
    equity = np.cumprod(1 + returns)
    peak = np.maximum.accumulate(equity)
    dd = (equity - peak) / peak
    max_dd = float(np.min(dd))
    hit_rate = float(np.mean(returns > 0))
    return {
        "annual_return": round(float(annual_return), 4),
        "annual_vol": round(float(annual_vol), 4),
        "sortino": round(float(sortino), 2) if sortino is not None else None,
        "max_drawdown": round(float(max_dd), 4),
        "hit_rate": round(float(hit_rate), 3),
    }


def _build_raw_monthly_returns() -> pd.DataFrame:
    """Retornos mensuales desde precios crudos, SIN winsorización.

    Consistente con _build_display_equity: neutraliza gaps > 45 días
    para evitar retornos espurios de pct_change() entre fechas distantes.
    """
    hist = load_historico()
    hist = hist.sort_values(["fondo", "fecha"])
    hist["ret_precio"] = hist.groupby("fondo")["precio"].pct_change()
    hist["ret_total"] = hist["ret_precio"].fillna(0) + hist["evento_pct"].fillna(0)
    hist.loc[hist.groupby("fondo").head(1).index, "ret_total"] = 0

    # Neutralizar retornos por gaps de datos (> 45 días entre observaciones)
    hist["_gap_days"] = hist.groupby("fondo")["fecha"].diff().dt.days
    hist.loc[hist["_gap_days"] > 45, "ret_total"] = 0
    hist = hist.drop(columns=["_gap_days"])

    hist["mes"] = hist["fecha"].dt.to_period("M").dt.to_timestamp("M")
    monthly = hist.groupby(["fondo", "mes"])["ret_total"].apply(
        lambda s: (1 + s).prod() - 1
    ).reset_index(name="ret_mensual")
    return monthly


def _close_period(
    current_weights: dict,
    current_ew_weights: dict,
    current_period_fund_rets: dict,
    current_period_ew_fund_rets: dict,
    current_rebal_date: pd.Timestamp,
    period_end_date: pd.Timestamp,
    opt_method: str = "equal_weight",
) -> dict:
    """Construye dict de periodo con holdings óptimos y EW."""
    # Holdings óptimos
    active_w = {f: w for f, w in current_weights.items() if w > 0}
    holdings, period_ret_total = [], 0.0
    for fondo, w in sorted(active_w.items(), key=lambda x: -x[1]):
        fund_rets = current_period_fund_rets.get(fondo, [])
        cum_ret = float(np.prod([1 + r for r in fund_rets]) - 1) if fund_rets else 0.0
        contrib = w * cum_ret
        period_ret_total += contrib
        holdings.append({
            "fondo": fondo,
            "weight": round(w, 4),
            "period_return": round(cum_ret, 4),
            "contribution": round(contrib, 4),
        })

    # Holdings equal-weight
    ew_holdings, ew_period_ret = [], 0.0
    n_ew = len(current_period_ew_fund_rets)
    ew_w = 1.0 / n_ew if n_ew > 0 else 0.0
    for fondo in sorted(current_period_ew_fund_rets.keys()):
        fund_rets = current_period_ew_fund_rets.get(fondo, [])
        cum_ret = float(np.prod([1 + r for r in fund_rets]) - 1) if fund_rets else 0.0
        contrib = ew_w * cum_ret
        ew_period_ret += contrib
        ew_holdings.append({
            "fondo": fondo,
            "weight": round(ew_w, 4),
            "period_return": round(cum_ret, 4),
            "contribution": round(contrib, 4),
        })

    n_months = len(next(iter(current_period_fund_rets.values()), []))
    return {
        "rebal_date": str(current_rebal_date.date())[:7],
        "period_end": str(period_end_date.date())[:7],
        "n_months": n_months,
        "portfolio_return": round(period_ret_total, 4),
        "holdings": holdings,
        "ew_portfolio_return": round(ew_period_ret, 4),
        "ew_holdings": ew_holdings,
        "optimization_method": opt_method,
    }


def _build_portfolio_json(
    scores: pd.DataFrame, panel: pd.DataFrame, overview_month: pd.Timestamp
) -> None:
    """Construye portfolio.json con backtest walk-forward del portafolio D10."""
    print("    construyendo portafolio óptimo D10 (semicovarianza)...")

    # Retornos raw (sin winsorizar) para medir retorno realizado
    panel_ret_raw = _build_raw_monthly_returns()

    LOOKBACK = 36  # meses de ventana para covarianza
    MAX_WEIGHT = 0.30
    EMA_SPAN = 9   # meses (half-life ≈ 4.5m, sesgo de recencia)
    RF_RATE = RF_ANNUAL  # 0.02
    REBALANCE_EVERY = 12  # meses entre rebalanceos

    # Meses con scores OOS, ordenados
    oos_months = sorted(scores["mes"].unique())

    # Retornos raw también para lookback de optimización (consistente con realización)
    panel_ret = panel_ret_raw.copy()

    backtest_rows = []
    latest_weights = {}
    latest_n_funds = 0
    latest_d10_funds = []

    # Estado de rebalanceo: pesos congelados entre rebalanceos
    months_since_rebal = REBALANCE_EVERY  # forzar rebalanceo en el primer mes
    current_weights = {}
    current_ew_weights = {}
    opt_positions = {}   # posiciones buy-and-hold (driftan entre rebalanceos)
    ew_positions = {}

    # Tracking de historial de rebalanceo
    rebal_history = []
    current_rebal_date = None
    current_period_fund_rets = {}  # {fondo: [ret_mes1, ret_mes2, ...]}
    current_period_ew_fund_rets = {}  # {fondo: [ret_mes1, ...]} para EW
    last_rebalance_date = None
    last_next_date = None  # tracking del último mes de realización
    current_opt_method = "equal_weight"

    for i, date in enumerate(oos_months):
        need_rebal = months_since_rebal >= REBALANCE_EVERY

        if need_rebal:
            # Cerrar periodo anterior si existe
            if current_rebal_date is not None and current_weights:
                # period_end = último mes de realización del periodo anterior
                # (el mes actual 'date' triggerea rebalanceo, pero el último
                # retorno realizado fue last_next_date del loop anterior)
                period_end_date = last_next_date if last_next_date is not None else date
                rebal_history.append(_close_period(
                    current_weights, current_ew_weights,
                    current_period_fund_rets, current_period_ew_fund_rets,
                    current_rebal_date, period_end_date,
                    current_opt_method,
                ))

            # 1. Scores de este mes → D10
            month_scores = scores[scores["mes"] == date].copy()
            if len(month_scores) < 10:
                continue
            month_scores["decil"] = pd.qcut(
                month_scores[SCORE_COL].rank(method="first"),
                10, labels=False, duplicates="drop",
            )
            d10_funds = month_scores[month_scores["decil"] == 9]["fondo"].tolist()
            if len(d10_funds) < 3:
                continue

            # 2. Trailing returns → pivot matrix
            cutoff = date - relativedelta(months=LOOKBACK)
            lookback = panel_ret[
                (panel_ret["mes"] > cutoff) & (panel_ret["mes"] <= date)
            ]
            ret_matrix = lookback.pivot_table(
                index="mes", columns="fondo", values="ret_mensual"
            )
            # Filtrar a fondos D10 con datos suficientes
            available = [f for f in d10_funds if f in ret_matrix.columns]
            if len(available) < 3:
                continue
            ret_matrix = ret_matrix[available].dropna(axis=1, thresh=12)
            if ret_matrix.shape[1] < 3 or ret_matrix.shape[0] < 12:
                continue
            ret_matrix = ret_matrix.fillna(0)

            # 3. PyPortfolioOpt: semicov + cadena de fallback robusta
            opt_method = "equal_weight"
            try:
                mu = expected_returns.ema_historical_return(
                    ret_matrix, returns_data=True, frequency=12, span=EMA_SPAN
                )
                S = risk_models.semicovariance(
                    ret_matrix, returns_data=True, frequency=12
                )

                # Regularizar si la matriz está mal condicionada
                if np.linalg.cond(S) > 1e6:
                    shrinkage = 0.01 * np.trace(S) / len(S)
                    S = S + shrinkage * np.eye(len(S))

                # Intento 1: max_sharpe con RF real
                try:
                    ef = EfficientFrontier(mu, S, weight_bounds=(0, MAX_WEIGHT))
                    ef.max_sharpe(risk_free_rate=RF_RATE)
                    weights = ef.clean_weights(cutoff=0.01)
                    opt_method = "max_sharpe"
                except Exception:
                    # Intento 2: max_sharpe con RF=0 (crisis: todos mu < RF)
                    try:
                        ef = EfficientFrontier(mu, S, weight_bounds=(0, MAX_WEIGHT))
                        ef.max_sharpe(risk_free_rate=0.0)
                        weights = ef.clean_weights(cutoff=0.01)
                        opt_method = "max_sharpe_rf0"
                    except Exception:
                        # Intento 3: max_quadratic_utility (siempre factible)
                        try:
                            ef = EfficientFrontier(mu, S, weight_bounds=(0, MAX_WEIGHT))
                            ef.max_quadratic_utility(risk_aversion=1)
                            weights = ef.clean_weights(cutoff=0.01)
                            opt_method = "max_quadratic_utility"
                        except Exception:
                            weights = {f: 1.0 / len(ret_matrix.columns)
                                       for f in ret_matrix.columns}
            except Exception:
                weights = {f: 1.0 / len(ret_matrix.columns)
                           for f in ret_matrix.columns}

            n_active = len([w for w in weights.values() if w > 0])
            print(f"    [{str(date.date())[:7]}] optimización: {opt_method}, "
                  f"{n_active} fondos")

            # 4. Equal-weight benchmark (post-filtro: solo fondos con datos suficientes)
            ew_cols = list(ret_matrix.columns)
            ew_weights = {f: 1.0 / len(ew_cols) for f in ew_cols}

            # Congelar pesos y resetear contador
            current_weights = weights
            current_ew_weights = ew_weights
            months_since_rebal = 0
            latest_d10_funds = d10_funds

            # Iniciar posiciones buy-and-hold
            opt_positions = {f: w for f, w in weights.items() if w > 0}
            ew_positions = dict(ew_weights)

            # Iniciar nuevo periodo de tracking
            current_rebal_date = date
            last_rebalance_date = str(date.date())[:7]
            current_opt_method = opt_method
            current_period_fund_rets = {f: [] for f in weights if weights[f] > 0}
            current_period_ew_fund_rets = {f: [] for f in ew_weights}

        # Usar pesos congelados (del último rebalanceo)
        weights = current_weights
        ew_weights = current_ew_weights
        if not weights:
            continue

        # 5. Retorno realizado del mes siguiente
        if i + 1 < len(oos_months):
            next_date = oos_months[i + 1]
        else:
            # Buscar siguiente mes en panel
            future = panel_ret[panel_ret["mes"] > date]["mes"]
            if len(future) == 0:
                # Almacenar pesos del último mes sin retorno realizado
                latest_weights = weights
                latest_n_funds = len([f for f, w in weights.items() if w > 0])
                continue
            next_date = future.min()

        # Retornos realizados desde precios raw (sin winsorizar),
        # consistente con las curvas de equity de la ficha de fondo.
        next_month = panel_ret_raw[panel_ret_raw["mes"] == next_date][
            ["fondo", "ret_mensual"]
        ].set_index("fondo")["ret_mensual"]

        # Retorno con pesos driftados (buy-and-hold entre rebalanceos)
        opt_total_pre = sum(opt_positions.values())
        opt_total_post = sum(
            opt_positions[f] * (1 + next_month.get(f, 0))
            for f in opt_positions
        )
        opt_ret = opt_total_post / opt_total_pre - 1 if opt_total_pre > 0 else 0.0

        ew_total_pre = sum(ew_positions.values())
        ew_total_post = sum(
            ew_positions[f] * (1 + next_month.get(f, 0))
            for f in ew_positions
        )
        ew_ret = ew_total_post / ew_total_pre - 1 if ew_total_pre > 0 else 0.0

        # Drift de posiciones
        for f in opt_positions:
            opt_positions[f] *= (1 + next_month.get(f, 0))
        for f in ew_positions:
            ew_positions[f] *= (1 + next_month.get(f, 0))

        # Tracking retornos por fondo en el periodo
        for fondo in current_period_fund_rets:
            r = next_month.get(fondo, 0)
            current_period_fund_rets[fondo].append(float(r))
        for fondo in current_period_ew_fund_rets:
            r = next_month.get(fondo, 0)
            current_period_ew_fund_rets[fondo].append(float(r))

        backtest_rows.append({
            "date": str(next_date.date())[:7],  # mes de realización (no de decisión)
            "opt_ret": float(opt_ret),
            "ew_ret": float(ew_ret),
            "n_funds": len(opt_positions),
            "is_rebalance": bool(need_rebal),
        })

        # Guardar último para pesos actuales
        latest_weights = weights
        latest_n_funds = len(opt_positions)
        months_since_rebal += 1
        last_next_date = next_date

    # Cerrar último periodo abierto
    if current_rebal_date is not None and current_weights:
        period_end_date = last_next_date if last_next_date is not None else oos_months[-1]
        rebal_history.append(_close_period(
            current_weights, current_ew_weights,
            current_period_fund_rets, current_period_ew_fund_rets,
            current_rebal_date, period_end_date,
            current_opt_method,
        ))

    if not backtest_rows:
        print("    portfolio.json: sin datos de backtest, saltando")
        return

    # Equity curves base 100 (con punto inicial en la primera fecha de rebalanceo)
    opt_equity = [100.0]
    ew_equity = [100.0]
    for row in backtest_rows:
        opt_equity.append(opt_equity[-1] * (1 + row["opt_ret"]))
        ew_equity.append(ew_equity[-1] * (1 + row["ew_ret"]))
    # Conservar el punto inicial: equity=100 antes del primer retorno realizado

    # Fechas: punto inicial (decisión) + meses de realización
    first_rebal_date = (rebal_history[0]["rebal_date"]
                        if rebal_history else backtest_rows[0]["date"])
    all_dates = [first_rebal_date] + [r["date"] for r in backtest_rows]

    # Marcadores de rebalanceo: mes de decisión/baseline (rebal_date)
    # para que el usuario mida "de línea a línea" y obtenga el retorno del periodo
    rebalance_dates_chart = [period["rebal_date"] for period in rebal_history]

    # Métricas de performance
    opt_returns = np.array([r["opt_ret"] for r in backtest_rows])
    ew_returns = np.array([r["ew_ret"] for r in backtest_rows])
    opt_metrics = _compute_perf_metrics(opt_returns)
    ew_metrics = _compute_perf_metrics(ew_returns)

    # Pesos actuales con contexto
    last_panel = panel[panel["mes"] <= overview_month].sort_values("mes")
    last_panel_by_fund = last_panel.groupby("fondo").tail(1).set_index("fondo")
    weights_detail = []
    for fondo, w in sorted(latest_weights.items(), key=lambda x: -x[1]):
        if w <= 0:
            continue
        p = last_panel_by_fund.loc[fondo] if fondo in last_panel_by_fund.index else None
        # Score del último mes
        last_s = scores[(scores["mes"] == overview_month) & (scores["fondo"] == fondo)]
        sc = float(last_s[SCORE_COL].iloc[0]) if len(last_s) > 0 else None
        weights_detail.append({
            "fondo": fondo,
            "weight": round(float(w), 4),
            "score": round(sc, 3) if sc is not None else None,
            "ret_12m": (round(float(p["ret_12m"]), 4)
                        if p is not None and pd.notna(p.get("ret_12m")) else None),
            "vol_12m": (round(float(p["vol_12m"]), 4)
                        if p is not None and pd.notna(p.get("vol_12m")) else None),
            "fee": (round(float(p["fee"]), 4)
                    if p is not None and pd.notna(p.get("fee")) else None),
        })

    portfolio = {
        "as_of": str(overview_month.date()),
        "last_rebalance_date": last_rebalance_date,
        "config": {
            "lookback_months": LOOKBACK,
            "risk_model": "semicovariance",
            "returns_model": "ema",
            "ema_span": EMA_SPAN,
            "max_weight": MAX_WEIGHT,
            "rebalance": "annual",
            "rebalance_months": REBALANCE_EVERY,
            "rf": RF_RATE,
        },
        "backtest": {
            "dates": all_dates,
            "opt_ret": [0.0] + [r["opt_ret"] for r in backtest_rows],
            "ew_ret": [0.0] + [r["ew_ret"] for r in backtest_rows],
            "opt_equity": opt_equity,
            "ew_equity": ew_equity,
            "rebalance_dates": rebalance_dates_chart,
        },
        "metrics": {
            "optimal": opt_metrics,
            "equal_weight": ew_metrics,
        },
        "current_weights": weights_detail,
        "rebalance_history": rebal_history,
    }

    with open(APP_DATA_DIR / "portfolio.json", "w") as f:
        json.dump(_sanitize(portfolio), f, indent=2)
    print(f"    portfolio.json:     {len(backtest_rows)} meses, "
          f"{len(weights_detail)} fondos, "
          f"{len(rebal_history)} periodos de rebalanceo")


def main() -> None:
    APP_DATA_DIR.mkdir(parents=True, exist_ok=True)

    print(">>> cargando artefactos")
    scores = pd.read_parquet(ARTIFACTS_DIR / "scores.parquet")
    panel = pd.read_parquet(ARTIFACTS_DIR / "panel_features.parquet")
    coefs = pd.read_csv(ARTIFACTS_DIR / "drivers_elastic.csv")
    importances = pd.read_csv(ARTIFACTS_DIR / "drivers_lgbm.csv")
    with open(ARTIFACTS_DIR / "metrics.json") as f:
        metrics = json.load(f)
    fold_diag = pd.read_csv(ARTIFACTS_DIR / "fold_diagnostics.csv")

    # ---- 1. funds_summary.json
    # Solo fondos del último mes de scoring para tener un corte
    # cross-seccional consistente (mismo período para todos).
    # Fondos scorados en meses anteriores (por salir del walk-forward)
    # no son comparables con el ranking actual.
    latest_month = scores["mes"].max()
    # Para el overview, usar el último mes con buena cobertura de target
    # realizado. Evitar meses donde datos incompletos (aún no cargados)
    # generan sesgo de supervivencia en la tabla mostrada.
    # Comparar contra fondos activos en el panel (no en scores, que ya
    # filtra por require_target=True en producción).
    active_per_month = panel.groupby("mes")["fondo"].nunique()
    has_target = scores.dropna(subset=["target_sortino"])
    if len(has_target) > 0:
        target_per_month = has_target.groupby("mes")["fondo"].nunique()
        coverage = (target_per_month / active_per_month).dropna()
        good_months = coverage[coverage >= 0.90].index
        if len(good_months) > 0:
            overview_month = good_months.max()
        else:
            overview_month = has_target["mes"].max()
    else:
        overview_month = latest_month
    last_score = scores[scores["mes"] == overview_month].copy()
    print(f"    filtro: overview_month={overview_month.date()} "
          f"(latest={latest_month.date()}, "
          f"{len(last_score)} de {scores['fondo'].nunique()} totales)")
    last_score["decil"] = pd.qcut(
        last_score[SCORE_COL].rank(method="first"),
        10, labels=False, duplicates="drop",
    ) + 1

    # Usar panel del mismo mes que el score para consistencia temporal
    panel_up_to = panel[panel["mes"] <= overview_month].sort_values("mes")
    last_panel = panel_up_to.groupby("fondo").tail(1).set_index("fondo")

    funds = []
    for _, row in last_score.iterrows():
        fid = row["fondo"]
        p = last_panel.loc[fid] if fid in last_panel.index else None
        funds.append({
            "fondo": fid,
            "fecha_score": str(row["mes"].date()),
            "score": float(row[SCORE_COL]),
            "decil": int(row["decil"]) if pd.notna(row["decil"]) else None,
            "target_realizado_6m": (None if pd.isna(row["target_ret_6m"])
                                    else float(row["target_ret_6m"])),
            "sortino_realizado_6m": (None if pd.isna(row.get("target_sortino"))
                                     else float(row["target_sortino"])),
            "ret_12m_trailing": (None if p is None or pd.isna(p.get("ret_12m"))
                                 else float(p["ret_12m"])),
            "vol_12m": (None if p is None or pd.isna(p.get("vol_12m"))
                        else float(p["vol_12m"])),
            "sharpe_12m": (None if p is None or pd.isna(p.get("sharpe_12m"))
                           else float(p["sharpe_12m"])),
            "sortino_12m": (None if p is None or pd.isna(p.get("sortino_12m"))
                            else float(p["sortino_12m"])),
            "max_dd_12m": (None if p is None or pd.isna(p.get("max_dd_12m"))
                           else float(p["max_dd_12m"])),
            "fee": (None if p is None or pd.isna(p.get("fee")) else float(p["fee"])),
            "fee_observado": (None if p is None else int(p.get("fee_observado", 0))),
            "n_instrumentos": (None if p is None or pd.isna(p.get("n_instrumentos"))
                               else int(p["n_instrumentos"])),
        })
    funds.sort(key=lambda x: x["score"], reverse=True)
    with open(APP_DATA_DIR / "funds_summary.json", "w") as f:
        json.dump(_sanitize(funds), f, indent=2)
    print(f"    funds_summary.json: {len(funds)} fondos")

    # ---- 2. fund_detail.json — historia mensual por fondo
    # Curvas de equity desde precios crudos (sin winsorización del pipeline)
    fund_ids = [f["fondo"] for f in funds]
    print("    construyendo curvas de equity desde precios crudos (sin winsor)...")
    display_equity = _build_display_equity(fund_ids)

    detail = {}
    panel_by_fondo = panel.set_index("fondo")
    scores_by_fondo = scores.set_index("fondo")
    for fid in fund_ids:
        if fid not in panel_by_fondo.index:
            continue
        p = panel_by_fondo.loc[[fid]].sort_values("mes")
        s = (scores_by_fondo.loc[[fid]].sort_values("mes")
             if fid in scores_by_fondo.index else None)
        merged = p.merge(s[["mes", SCORE_COL]] if s is not None else
                          pd.DataFrame(columns=["mes", SCORE_COL]),
                          on="mes", how="left")
        # Equity desde precios crudos, alineada a las fechas del panel
        if fid in display_equity:
            eq_df = display_equity[fid]
            merged = merged.merge(eq_df, on="mes", how="left")
            equity = merged["equity"].values
        else:
            # Fallback: reconstruir desde ret_mensual del panel
            ret = merged["ret_mensual"].fillna(0).values
            equity = 100 * np.cumprod(1 + ret)
        # Usar fecha real de observación (último día con dato en el mes)
        # en vez de end-of-month ficticio para las etiquetas del chart
        if "fecha" in merged.columns:
            dates = [str(d.date()) if pd.notna(d) else str(m.date())
                     for d, m in zip(merged["fecha"], merged["mes"])]
        else:
            dates = [str(d.date()) for d in merged["mes"]]
        detail[fid] = {
            "dates": dates,
            "ret_mensual": [None if pd.isna(r) else float(r) for r in merged["ret_mensual"]],
            "equity": equity.tolist(),
            "score": [None if pd.isna(v) else float(v) for v in merged[SCORE_COL]],
            "fee": [None if pd.isna(v) else float(v) for v in merged["fee"]],
            "n_instrumentos": [None if pd.isna(v) else int(v) for v in merged["n_instrumentos"]],
        }
    with open(APP_DATA_DIR / "fund_detail.json", "w") as f:
        json.dump(_sanitize(detail), f)
    print(f"    fund_detail.json:   {len(detail)} fondos")

    # ---- 3. drivers.json
    feat_cols = [c for c in coefs.columns if c not in ("fold", "alpha", "l1_ratio")]
    drivers = {
        "elastic_net_coefs": {
            "features": feat_cols,
            "mean": coefs[feat_cols].mean().tolist(),
            "std": coefs[feat_cols].std().tolist(),
        },
        "elastic_net_hyperparams": {
            "alpha_mean": float(coefs["alpha"].mean()),
            "l1_ratio_mean": float(coefs["l1_ratio"].mean()),
        },
        "lgbm_importance": {
            "features": [c for c in importances.columns if c != "fold"],
            "mean": importances.drop(columns="fold").mean().tolist(),
        },
    }
    with open(APP_DATA_DIR / "drivers.json", "w") as f:
        json.dump(_sanitize(drivers), f, indent=2)
    print(f"    drivers.json:       {len(feat_cols)} features")

    # ---- 4. backtest.json — top-decil vs bottom-decil vs universo

    # Precomputar Sortino forward parcial (para meses sin target_sortino completo)
    panel_sorted = panel[["fondo", "mes", "ret_mensual"]].sort_values(["fondo", "mes"])
    last_month = panel_sorted["mes"].max()

    partial_list = []
    for fondo, group in panel_sorted.groupby("fondo"):
        rets = group["ret_mensual"].values
        meses = group["mes"].values
        n = len(rets)
        partial = np.full(n, np.nan)
        months_fwd = np.zeros(n, dtype=int)
        for i in range(n):
            end = min(i + 7, n)   # hasta 6 meses forward
            k = end - i - 1
            if k >= 3:  # mínimo 3 meses para que downside_dev sea significativo
                fwd = rets[i + 1 : end]
                if not np.any(np.isnan(fwd)):
                    downside = np.minimum(fwd, 0)
                    downside_dev = np.sqrt(np.mean(downside ** 2))
                    if downside_dev > 0:
                        partial[i] = ((np.mean(fwd) - RF_MONTHLY) / downside_dev) * np.sqrt(12)
                    months_fwd[i] = k
        partial_list.append(pd.DataFrame({
            "fondo": fondo, "mes": meses,
            "partial_fwd_sortino": partial, "months_forward": months_fwd,
        }))
    partial_sortinos = pd.concat(partial_list, ignore_index=True).set_index(["fondo", "mes"])

    bt_rows = []
    for date, g in scores.groupby("mes"):
        if len(g) < 20:  # mínimo 2 fondos por decil
            continue
        gg = g.copy()

        is_prod = bool(g["fold"].iloc[0] == -1) if "fold" in g.columns else False
        is_ext = bool(g["fold"].iloc[0] == -2) if "fold" in g.columns else False

        if is_ext:
            # Merge partial Sortino para meses extendidos (sin target completo)
            gg = gg.merge(partial_sortinos, left_on=["fondo", "mes"],
                          right_index=True, how="left")
            sortino_col = "partial_fwd_sortino"
            avg_months = int(gg["months_forward"].median())
        else:
            sortino_col = "target_sortino"
            avg_months = 6

        # Excluir Sortinos infinitos (downside_dev=0) para que no contaminen la mediana
        gg[sortino_col] = gg[sortino_col].replace([np.inf, -np.inf], np.nan)

        # ElasticNet (deciles) — mediana robusta a outliers near-infinity
        gg["d"] = pd.qcut(gg[SCORE_COL].rank(method="first"), 10,
                          labels=False, duplicates="drop")
        top = gg[gg["d"] == 9][sortino_col].median()
        bot = gg[gg["d"] == 0][sortino_col].median()
        eq = gg[sortino_col].median()

        bt_rows.append({
            "date": str(date.date()),
            "top_d10": float(top) if pd.notna(top) else None,
            "bot_d1": float(bot) if pd.notna(bot) else None,
            "universe": float(eq) if pd.notna(eq) else None,
            "spread": float(top - bot) if pd.notna(top) and pd.notna(bot) else None,
            "n_funds": int(len(gg)),
            "is_production": is_prod or is_ext,
            "is_partial": is_ext,
            "months_forward": avg_months,
        })
    with open(APP_DATA_DIR / "backtest.json", "w") as f:
        json.dump(_sanitize(bt_rows), f, indent=2)
    print(f"    backtest.json:      {len(bt_rows)} fechas")

    # ---- 5. meta.json — info global
    meta = {
        "as_of": str(overview_month.date()),
        "n_funds": len(last_score),           # fondos en el corte mostrado
        "n_funds_total": int(scores["fondo"].nunique()),  # fondos modelados en total
        "n_months": int(scores["mes"].nunique()),
        "n_folds": int(len(fold_diag)),  # solo folds OOS (0-28), excluye fold -1/-2
        "metrics_summary": {
            label: {
                "ic_mean": metrics[label]["ic_summary"]["mean"],
                "ic_ir": metrics[label]["ic_summary"]["ic_ir"],
                "ic_hit_meses": metrics[label]["ic_summary"]["hit"],
                "ic_ci95_low": metrics[label]["ic_bootstrap"]["ci_low"],
                "ic_ci95_high": metrics[label]["ic_bootstrap"]["ci_high"],
                "spread_d10_d1": metrics[label]["spread_d10_d1_mean"],
                "hit_top25": metrics[label]["hit_rate_top25_mean"],
            }
            for label in ["elastic", "lgbm"]
            if label in metrics
        },
        "fold_boundaries": [
            {"fold": int(row["fold"]),
             "val_start": row["val_start"],
             "val_end": row["val_end"]}
            for _, row in fold_diag.iterrows()
        ],
        "primary_model": "elastic",
        "training_target": "target_sortino_rank_6m",
        "scorers_compared": ["elastic", "lgbm"],
    }
    with open(APP_DATA_DIR / "meta.json", "w") as f:
        json.dump(_sanitize(meta), f, indent=2, default=str)
    print(f"    meta.json: as_of={meta['as_of']}, primary_model={meta['primary_model']}")

    # ---- 6. portfolio.json — portafolio óptimo D10 con backtest walk-forward
    _build_portfolio_json(scores, panel, overview_month)

    print(f"\n>>> JSONs en {APP_DATA_DIR}")


if __name__ == "__main__":
    sys.exit(main())
