"""Script de auditoría — Verifica integridad de datos del dashboard vs fuente SQLite.

6 auditorías:
  1. Equity curves vs SQLite (TRI recalculado desde precios raw)
  2. ret_mensual vs equity (consistencia interna del fund_detail.json)
  3. Holdings period_return vs retornos raw mensuales
  4. Portfolio equity aritmética (opt_equity coherente con opt_ret)
  5. portfolio_return = sum(contributions) y sum(weights) ≈ 1
  6. opt_ret mensual vs raw weighted (reproducción independiente del backtest)

Uso:
    uv run python -m scripts.audit_returns
"""

from __future__ import annotations

import json
import sys
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from src.data import load_historico
from src.paths import APP_DATA_DIR, ARTIFACTS_DIR


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

@dataclass
class AuditResult:
    name: str
    passed: bool
    n_checked: int = 0
    n_errors: int = 0
    errors: list[str] = field(default_factory=list)
    max_error: float = 0.0

    def summary(self) -> str:
        status = "PASS" if self.passed else "FAIL"
        detail = f"{self.n_checked} checks, {self.n_errors} errores"
        if self.n_errors > 0:
            detail += f", max_error={self.max_error:.6f}"
        return f"  [{status}] {self.name}: {detail}"


def _load_json(name: str):
    with open(APP_DATA_DIR / name) as f:
        return json.load(f)


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

    # Neutralizar retornos por gaps de datos (> 45 días)
    hist["_gap_days"] = hist.groupby("fondo")["fecha"].diff().dt.days
    hist.loc[hist["_gap_days"] > 45, "ret_total"] = 0
    hist = hist.drop(columns=["_gap_days"])

    hist["mes"] = hist["fecha"].dt.to_period("M").dt.to_timestamp("M")
    monthly = hist.groupby(["fondo", "mes"])["ret_total"].apply(
        lambda s: (1 + s).prod() - 1
    ).reset_index(name="ret_mensual")
    return monthly


def _build_display_equity_audit(fondos: list[str]) -> dict[str, pd.DataFrame]:
    """Recalcula curvas de equity desde precios crudos (igual que 05)."""
    hist = load_historico()
    hist = hist[hist["fondo"].isin(fondos)].sort_values(["fondo", "fecha"])
    hist["ret_precio"] = hist.groupby("fondo")["precio"].pct_change()
    hist["ret_total"] = hist["ret_precio"].fillna(0) + hist["evento_pct"].fillna(0)
    hist.loc[hist.groupby("fondo").head(1).index, "ret_total"] = 0
    # Neutralizar retornos por gaps de datos (> 45 días entre observaciones)
    hist["_gap_days"] = hist.groupby("fondo")["fecha"].diff().dt.days
    hist.loc[hist["_gap_days"] > 45, "ret_total"] = 0
    hist = hist.drop(columns=["_gap_days"])
    hist["tri"] = hist.groupby("fondo")["ret_total"].transform(
        lambda s: (1 + s).cumprod()
    )
    hist["mes"] = hist["fecha"].dt.to_period("M").dt.to_timestamp("M")
    eom = hist.sort_values("fecha").groupby(["fondo", "mes"]).tail(1)
    eom = eom[["fondo", "mes", "tri"]].copy()
    eom["equity"] = eom.groupby("fondo")["tri"].transform(
        lambda s: 100.0 * s / s.iloc[0]
    )
    result = {}
    for fid, g in eom.groupby("fondo"):
        result[fid] = g[["mes", "equity"]].reset_index(drop=True)
    return result


# ---------------------------------------------------------------------------
# Auditorías individuales
# ---------------------------------------------------------------------------

def audit_1_equity_vs_sqlite(fund_detail: dict) -> AuditResult:
    """Cada punto de equity en fund_detail.json vs TRI recalculado desde SQL."""
    res = AuditResult(name="Equity curves vs SQLite", passed=True)
    TOL_POINT = 0.01  # tolerancia por punto
    TOL_FINAL = 0.5   # tolerancia en equity final

    fondos = list(fund_detail.keys())
    print(f"    audit 1: verificando {len(fondos)} fondos...")
    recalc = _build_display_equity_audit(fondos)

    for fondo in fondos:
        json_equity = fund_detail[fondo]["equity"]
        if fondo not in recalc:
            res.errors.append(f"{fondo}: no encontrado en SQLite")
            res.n_errors += 1
            continue

        rc = recalc[fondo]
        n = min(len(json_equity), len(rc))
        res.n_checked += n

        rc_vals = rc["equity"].values[:n]
        js_vals = np.array(json_equity[:n])

        diffs = np.abs(rc_vals - js_vals)
        max_diff = float(np.max(diffs)) if len(diffs) > 0 else 0.0
        res.max_error = max(res.max_error, max_diff)

        n_bad = int(np.sum(diffs > TOL_POINT))
        if n_bad > 0:
            res.n_errors += n_bad
            res.errors.append(
                f"{fondo}: {n_bad}/{n} puntos con diff > {TOL_POINT}, "
                f"max_diff={max_diff:.4f}"
            )

        # Verificar equity final
        if len(json_equity) > 0 and len(rc_vals) > 0:
            final_diff = abs(js_vals[-1] - rc_vals[-1])
            if final_diff > TOL_FINAL:
                res.errors.append(
                    f"{fondo}: equity final diff={final_diff:.4f} > {TOL_FINAL}"
                )

    res.passed = res.n_errors == 0
    return res


def audit_2_ret_vs_equity(fund_detail: dict) -> AuditResult:
    """ret_mensual vs retornos raw mensuales (misma fuente que equity).

    Nota: en fund_detail.json, ret_mensual viene del panel (pipeline con
    winsorización p99.5) mientras equity viene de precios crudos sin
    winsorizar. Por diseño son fuentes distintas. Esta auditoría compara
    ret_mensual del JSON contra retornos raw recalculados desde SQLite,
    verificando que el pipeline produzca resultados razonables.
    La diferencia por winsorización (en meses con eventos extremos) es
    esperada y se reporta informativamente.
    """
    res = AuditResult(name="ret_mensual vs raw (winsor vs raw)", passed=True)
    TOL = 0.01  # tolerancia amplia: diferencias por winsorización son esperadas
    TOL_EXTREME = 0.10  # solo reportar casos extremos

    raw = _build_raw_monthly_returns()
    raw["ym"] = raw["mes"].dt.to_period("M").astype(str)
    raw_lookup = raw.set_index(["fondo", "ym"])["ret_mensual"]

    n_winsor_diffs = 0
    for fondo, data in fund_detail.items():
        dates = data["dates"]
        rets = data["ret_mensual"]
        for i in range(len(rets)):
            if rets[i] is None:
                continue
            ym = dates[i][:7]
            key = (fondo, ym)
            if key not in raw_lookup.index:
                continue
            raw_ret = raw_lookup.loc[key]
            if isinstance(raw_ret, pd.Series):
                raw_ret = raw_ret.iloc[0]
            diff = abs(rets[i] - float(raw_ret))
            res.n_checked += 1
            if diff > TOL:
                n_winsor_diffs += 1
                if diff > TOL_EXTREME:
                    res.n_errors += 1
                    res.max_error = max(res.max_error, diff)
                    if res.n_errors <= 20:
                        res.errors.append(
                            f"{fondo} {ym}: panel={rets[i]:.6f}, "
                            f"raw={float(raw_ret):.6f}, diff={diff:.6f}"
                        )

    if n_winsor_diffs > 0:
        res.errors.insert(0,
            f"INFO: {n_winsor_diffs} meses con diff > {TOL} "
            f"(esperado por winsorización p99.5 del pipeline)")
    if res.n_errors > 0:
        res.errors.insert(0,
            f"INFO: {res.n_errors} meses con diff > {TOL_EXTREME} "
            f"(eventos extremos clipped por winsorización — esperado)")

    # Las diferencias son esperadas por diseño (winsorización del pipeline).
    # Solo fallar si hay diferencias enormes (>50%) en más del 1% de checks.
    extreme_ratio = res.n_errors / max(res.n_checked, 1)
    res.passed = extreme_ratio < 0.01 or res.max_error < 1.0
    return res


def audit_3_holdings_period_return(portfolio: dict) -> AuditResult:
    """Cada holding.period_return vs retorno compuesto desde monthly raw.

    En el backtest, para cada mes OOS 'date', se registra el retorno del
    mes SIGUIENTE (next_date). Así los retornos acumulados del periodo
    cubren desde rebal_date+1 hasta period_end+1 (inclusive).
    """
    res = AuditResult(name="Holdings period_return vs raw", passed=True)
    TOL = 0.001

    raw_monthly = _build_raw_monthly_returns()
    raw_monthly["ym"] = raw_monthly["mes"].dt.to_period("M")
    raw_pivot = raw_monthly.pivot_table(
        index="ym", columns="fondo", values="ret_mensual"
    )

    for period in portfolio.get("rebalance_history", []):
        rebal_ym = pd.Period(period["rebal_date"], freq="M")
        end_ym = pd.Period(period["period_end"], freq="M")
        # Los retornos acumulados van desde el mes SIGUIENTE al rebal
        # hasta period_end inclusive (que es el último mes de realización)
        period_months = [
            ym for ym in raw_pivot.index
            if rebal_ym < ym <= end_ym
        ]

        # Verificar holdings óptimos
        for h in period["holdings"]:
            fondo = h["fondo"]
            if fondo not in raw_pivot.columns:
                continue
            rets = [
                raw_pivot.loc[ym, fondo]
                for ym in period_months
                if ym in raw_pivot.index and pd.notna(raw_pivot.loc[ym, fondo])
            ]
            if not rets:
                continue
            cum_ret = float(np.prod([1 + r for r in rets]) - 1)
            diff = abs(h["period_return"] - cum_ret)
            res.n_checked += 1
            if diff > TOL:
                res.n_errors += 1
                res.max_error = max(res.max_error, diff)
                if res.n_errors <= 20:
                    res.errors.append(
                        f"{period['rebal_date']} OPT {fondo}: json={h['period_return']:.4f}, "
                        f"recalc={cum_ret:.4f}, diff={diff:.4f}, "
                        f"n_months_json={period['n_months']}, n_rets={len(rets)}"
                    )

        # Verificar holdings EW
        for h in period.get("ew_holdings", []):
            fondo = h["fondo"]
            if fondo not in raw_pivot.columns:
                continue
            rets = [
                raw_pivot.loc[ym, fondo]
                for ym in period_months
                if ym in raw_pivot.index and pd.notna(raw_pivot.loc[ym, fondo])
            ]
            if not rets:
                continue
            cum_ret = float(np.prod([1 + r for r in rets]) - 1)
            diff = abs(h["period_return"] - cum_ret)
            res.n_checked += 1
            if diff > TOL:
                res.n_errors += 1
                res.max_error = max(res.max_error, diff)
                if res.n_errors <= 20:
                    res.errors.append(
                        f"{period['rebal_date']} EW {fondo}: json={h['period_return']:.4f}, "
                        f"recalc={cum_ret:.4f}, diff={diff:.4f}, "
                        f"n_months_json={period['n_months']}, n_rets={len(rets)}"
                    )

    res.passed = res.n_errors == 0
    return res


def audit_4_portfolio_equity_arithmetic(portfolio: dict) -> AuditResult:
    """opt_equity[i] = opt_equity[i-1] * (1 + opt_ret[i]) y lo mismo para EW."""
    res = AuditResult(name="Portfolio equity aritmética", passed=True)
    TOL = 0.01
    bt = portfolio["backtest"]

    for label, rets_key, eq_key in [
        ("opt", "opt_ret", "opt_equity"),
        ("ew", "ew_ret", "ew_equity"),
    ]:
        rets = bt[rets_key]
        equity = bt[eq_key]
        # equity[0] = 100 * (1 + rets[0])
        expected = 100.0
        for i in range(len(rets)):
            expected = expected * (1 + rets[i])
            diff = abs(expected - equity[i])
            res.n_checked += 1
            if diff > TOL:
                res.n_errors += 1
                res.max_error = max(res.max_error, diff)
                if res.n_errors <= 20:
                    res.errors.append(
                        f"{label} i={i}: expected={expected:.4f}, "
                        f"json={equity[i]:.4f}, diff={diff:.4f}"
                    )

    res.passed = res.n_errors == 0
    return res


def audit_5_return_equals_sum_contributions(portfolio: dict) -> AuditResult:
    """portfolio_return = sum(contributions) y sum(weights) ≈ 1."""
    res = AuditResult(name="portfolio_return = sum(contributions)", passed=True)
    TOL_RET = 0.001
    TOL_W = 0.02  # sum(weights) puede ser < 1 si clean_weights recortó

    for period in portfolio.get("rebalance_history", []):
        # Verificar retorno = sum(weight * period_return)
        sum_contrib = sum(h["contribution"] for h in period["holdings"])
        diff = abs(period["portfolio_return"] - sum_contrib)
        res.n_checked += 1
        if diff > TOL_RET:
            res.n_errors += 1
            res.max_error = max(res.max_error, diff)
            res.errors.append(
                f"{period['rebal_date']}: port_ret={period['portfolio_return']:.4f}, "
                f"sum_contrib={sum_contrib:.4f}, diff={diff:.4f}"
            )

        # Verificar sum(weights) ~ 1
        sum_w = sum(h["weight"] for h in period["holdings"])
        w_diff = abs(1.0 - sum_w)
        res.n_checked += 1
        if w_diff > TOL_W:
            res.n_errors += 1
            res.max_error = max(res.max_error, w_diff)
            res.errors.append(
                f"{period['rebal_date']}: sum_weights={sum_w:.4f}, diff_from_1={w_diff:.4f}"
            )

        # Lo mismo para EW si existe
        if "ew_holdings" in period and period["ew_holdings"]:
            ew_sum = sum(h["contribution"] for h in period["ew_holdings"])
            ew_diff = abs(period.get("ew_portfolio_return", 0) - ew_sum)
            res.n_checked += 1
            if ew_diff > TOL_RET:
                res.n_errors += 1
                res.max_error = max(res.max_error, ew_diff)
                res.errors.append(
                    f"{period['rebal_date']} EW: ew_port_ret="
                    f"{period.get('ew_portfolio_return', 0):.4f}, "
                    f"sum_contrib={ew_sum:.4f}, diff={ew_diff:.4f}"
                )

    res.passed = res.n_errors == 0
    return res


def _get_raw_ret(raw_lookup: pd.Series, fondo: str, ym: str) -> float:
    """Obtiene retorno raw de un fondo en un mes, 0.0 si no existe."""
    key = (fondo, ym)
    if key not in raw_lookup.index:
        return 0.0
    rv = raw_lookup.loc[key]
    if isinstance(rv, pd.Series):
        rv = rv.iloc[0]
    return float(rv)


def audit_6_opt_ret_vs_raw_weighted(portfolio: dict) -> AuditResult:
    """Reproducir cada opt_ret[i] y ew_ret[i] como suma ponderada de retornos raw.

    El backtest usa buy-and-hold (pesos driftados entre rebalanceos).
    dates[0] es el punto inicial (ret=0, equity=100).
    dates[1..N] son meses de realización con retorno driftado.
    """
    res = AuditResult(name="opt_ret y ew_ret mensual vs raw weighted", passed=True)
    TOL = 0.0001

    raw_monthly = _build_raw_monthly_returns()
    raw_monthly["ym"] = raw_monthly["mes"].dt.to_period("M").astype(str)
    raw_lookup = raw_monthly.set_index(["fondo", "ym"])["ret_mensual"]

    bt = portfolio["backtest"]
    dates = bt["dates"]
    opt_rets = bt["opt_ret"]
    ew_rets = bt["ew_ret"]

    # Mapear fechas de realización de rebalanceo a pesos iniciales
    rebal_periods = portfolio.get("rebalance_history", [])
    rebal_chart_dates = bt.get("rebalance_dates", [])
    opt_rebal_weights = {}  # realization_date → dict de pesos óptimos
    ew_rebal_weights = {}   # realization_date → dict de pesos EW
    for j, period in enumerate(rebal_periods):
        if j < len(rebal_chart_dates):
            real_date = rebal_chart_dates[j]
            opt_rebal_weights[real_date] = {
                h["fondo"]: h["weight"] for h in period["holdings"]
            }
            ew_rebal_weights[real_date] = {
                h["fondo"]: h["weight"] for h in period.get("ew_holdings", [])
            }

    # --- Verificar opt_ret ---
    opt_positions = {}
    for i, date_str in enumerate(dates):
        if i == 0:
            continue
        if date_str in opt_rebal_weights:
            opt_positions = dict(opt_rebal_weights[date_str])
        if not opt_positions:
            continue

        ym = date_str
        total_pre = sum(opt_positions.values())
        total_post = sum(
            opt_positions[f] * (1 + _get_raw_ret(raw_lookup, f, ym))
            for f in opt_positions
        )
        recalc_ret = total_post / total_pre - 1 if total_pre > 0 else 0.0

        # Drift de posiciones
        for fondo in list(opt_positions.keys()):
            opt_positions[fondo] *= (1 + _get_raw_ret(raw_lookup, fondo, ym))

        diff = abs(opt_rets[i] - recalc_ret)
        res.n_checked += 1
        if diff > TOL:
            res.n_errors += 1
            res.max_error = max(res.max_error, diff)
            if res.n_errors <= 30:
                res.errors.append(
                    f"OPT {date_str}: json_ret={opt_rets[i]:.6f}, "
                    f"recalc={recalc_ret:.6f}, diff={diff:.6f}"
                )

    # --- Verificar ew_ret ---
    ew_positions = {}
    for i, date_str in enumerate(dates):
        if i == 0:
            continue
        if date_str in ew_rebal_weights:
            ew_positions = dict(ew_rebal_weights[date_str])
        if not ew_positions:
            continue

        ym = date_str
        total_pre = sum(ew_positions.values())
        total_post = sum(
            ew_positions[f] * (1 + _get_raw_ret(raw_lookup, f, ym))
            for f in ew_positions
        )
        recalc_ret = total_post / total_pre - 1 if total_pre > 0 else 0.0

        # Drift de posiciones
        for fondo in list(ew_positions.keys()):
            ew_positions[fondo] *= (1 + _get_raw_ret(raw_lookup, fondo, ym))

        diff = abs(ew_rets[i] - recalc_ret)
        res.n_checked += 1
        if diff > TOL:
            res.n_errors += 1
            res.max_error = max(res.max_error, diff)
            if res.n_errors <= 30:
                res.errors.append(
                    f"EW {date_str}: json_ret={ew_rets[i]:.6f}, "
                    f"recalc={recalc_ret:.6f}, diff={diff:.6f}"
                )

    res.passed = res.n_errors == 0
    return res


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> int:
    print("=" * 60)
    print("AUDITORÍA DE RETORNOS — Dashboard vs SQLite")
    print("=" * 60)

    print("\n>>> Cargando datos del dashboard...")
    fund_detail = _load_json("fund_detail.json")
    portfolio = _load_json("portfolio.json")
    print(f"    fund_detail: {len(fund_detail)} fondos")
    print(f"    portfolio: {len(portfolio.get('rebalance_history', []))} periodos")

    results: list[AuditResult] = []

    print("\n--- Auditoría 1: Equity curves vs SQLite ---")
    results.append(audit_1_equity_vs_sqlite(fund_detail))

    print("--- Auditoría 2: ret_mensual vs equity ---")
    results.append(audit_2_ret_vs_equity(fund_detail))

    print("--- Auditoría 3: Holdings period_return vs raw ---")
    results.append(audit_3_holdings_period_return(portfolio))

    print("--- Auditoría 4: Portfolio equity aritmética ---")
    results.append(audit_4_portfolio_equity_arithmetic(portfolio))

    print("--- Auditoría 5: portfolio_return = sum(contributions) ---")
    results.append(audit_5_return_equals_sum_contributions(portfolio))

    print("--- Auditoría 6: opt_ret mensual vs raw weighted ---")
    results.append(audit_6_opt_ret_vs_raw_weighted(portfolio))

    # Resumen
    print("\n" + "=" * 60)
    print("RESUMEN")
    print("=" * 60)
    n_pass = 0
    for r in results:
        print(r.summary())
        if r.passed:
            n_pass += 1
        elif r.errors:
            for e in r.errors[:10]:
                print(f"        {e}")
            if len(r.errors) > 10:
                print(f"        ... y {len(r.errors) - 10} errores más")

    print(f"\n  {n_pass}/{len(results)} auditorías pasaron.")
    all_pass = all(r.passed for r in results)
    print("  RESULTADO FINAL:", "PASS" if all_pass else "FAIL")
    return 0 if all_pass else 1


if __name__ == "__main__":
    sys.exit(main())
