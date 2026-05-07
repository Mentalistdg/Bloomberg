"""Script 02/05 — Reporte exploratorio (EDA) y validación de datos.

Genera 5 gráficos diagnósticos que validan la calidad y consistencia
de los datos antes de pasar a la etapa de modelado. Cada gráfico se
guarda como PNG en artifacts/plots/ y se referencia en el informe.

Este script es OPCIONAL para el pipeline de modelado (el script 03
no depende de él), pero es esencial para la documentación y el
análisis de datos.

Plots producidos:
  1. cobertura_universo.png       fondos activos por mes + entradas/salidas
  2. evento_pct_dist.png          distribución de eventos de capital + yield anual
  3. fees_dist_y_cobertura.png    distribución de fees + cobertura por año
  4. subyacentes_30pct.png        verificación de la regla del 30% de AUM
  5. retornos_mensuales.png       distribución de retornos por década

Input:  usa_fondos_pp.sqlite (directo, no usa panel_raw)
Output: artifacts/plots/*.png

Uso:
    python -m scripts.02_eda_report
"""

from __future__ import annotations

import sys

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src.data import (
    compute_daily_total_return,
    load_fees,
    load_historico,
    load_subyacentes,
)
from src.paths import PLOTS_DIR

plt.rcParams.update(
    {
        "figure.dpi": 110,
        "savefig.dpi": 130,
        "savefig.bbox": "tight",
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.grid": True,
        "grid.alpha": 0.25,
        "font.size": 10,
    }
)


def plot_cobertura_universo(historico: pd.DataFrame) -> None:
    daily_count = (
        historico.assign(mes=lambda x: x["fecha"].dt.to_period("M").dt.to_timestamp("M"))
        .groupby("mes")["fondo"]
        .nunique()
    )

    primer = historico.groupby("fondo")["fecha"].min().dt.to_period("M").value_counts().sort_index()
    ultimo = historico.groupby("fondo")["fecha"].max().dt.to_period("M").value_counts().sort_index()

    fig, axes = plt.subplots(2, 1, figsize=(9, 5), sharex=True)
    axes[0].plot(daily_count.index, daily_count.values, color="#1f4e79")
    axes[0].set_title("Fondos activos por mes")
    axes[0].set_ylabel("# fondos")

    axes[1].bar(primer.index.to_timestamp(), primer.values, width=20, color="#2e7d32",
                label="Inicio de reporte", alpha=0.7)
    axes[1].bar(ultimo.index.to_timestamp(), -ultimo.values, width=20, color="#c62828",
                label="Fin de reporte", alpha=0.7)
    axes[1].axhline(0, color="black", lw=0.5)
    axes[1].set_title("Entradas (verde) y salidas (rojo) del universo, por mes")
    axes[1].set_ylabel("# fondos")
    axes[1].legend(loc="lower right")

    fig.tight_layout()
    fig.savefig(PLOTS_DIR / "cobertura_universo.png")
    plt.close(fig)


def plot_evento_pct(historico: pd.DataFrame) -> None:
    eventos = historico[historico["evento_pct"] != 0].copy()
    eventos["anio"] = eventos["fecha"].dt.year

    yld_anual = eventos.groupby(["fondo", "anio"])["evento_pct"].sum().reset_index()
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))

    cap = eventos["evento_pct"].quantile(0.99)
    axes[0].hist(eventos["evento_pct"].clip(upper=cap, lower=-cap), bins=80, color="#1f4e79")
    axes[0].set_title(f"Distribución de evento_pct (no-cero)\nN={len(eventos):,}, "
                      f"truncado a [{-cap:.3f}, {cap:.3f}]")
    axes[0].set_xlabel("evento_pct")
    axes[0].set_ylabel("frecuencia")

    cap2 = yld_anual["evento_pct"].quantile(0.99)
    axes[1].hist(yld_anual["evento_pct"].clip(upper=cap2), bins=60, color="#2e7d32")
    axes[1].set_title("Suma anual de evento_pct por fondo\n(rendimiento implícito por distribuciones)")
    axes[1].set_xlabel("yield anual implícito")
    axes[1].set_ylabel("frecuencia (fondo-año)")

    fig.tight_layout()
    fig.savefig(PLOTS_DIR / "evento_pct_dist.png")
    plt.close(fig)


def plot_fees(fees: pd.DataFrame) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))

    axes[0].hist(fees["fee"], bins=60, color="#1f4e79")
    axes[0].set_title(f"Distribución de fee (% anual, no-NULL)\nN={len(fees):,}")
    axes[0].set_xlabel("fee (% anual)")
    axes[0].set_ylabel("frecuencia")

    fees_year = fees.assign(anio=lambda x: x["fecha"].dt.year)
    fondos_con_fee = fees_year.groupby("anio")["fondo"].nunique()
    axes[1].bar(fondos_con_fee.index, fondos_con_fee.values, color="#2e7d32")
    axes[1].set_title("Fondos con fee reportado, por año")
    axes[1].set_xlabel("año")
    axes[1].set_ylabel("# fondos")

    fig.tight_layout()
    fig.savefig(PLOTS_DIR / "fees_dist_y_cobertura.png")
    plt.close(fig)


def plot_subyacentes_30pct(sub: pd.DataFrame) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))

    axes[0].scatter(sub["n_instrumentos"], sub["pct_acum"] * 100,
                    alpha=0.6, color="#1f4e79", s=18)
    axes[0].axhline(30, color="red", lw=1, ls="--", label="umbral 30%")
    axes[0].set_xscale("log")
    axes[0].set_xlabel("n_instrumentos (escala log)")
    axes[0].set_ylabel("pct_acum (%)")
    axes[0].set_title("Concentración: piso duro en 30% del AUM")
    axes[0].legend()

    bins = pd.cut(sub["n_instrumentos"], bins=[0, 2, 5, 10, 30, 100, 200],
                  labels=["1-2", "3-5", "6-10", "11-30", "31-100", "101+"])
    bucket_stats = sub.groupby(bins, observed=True)["pct_acum"].agg(["mean", "min", "max"]) * 100

    x = np.arange(len(bucket_stats))
    axes[1].bar(x, bucket_stats["mean"], yerr=[bucket_stats["mean"] - bucket_stats["min"],
                                                bucket_stats["max"] - bucket_stats["mean"]],
                capsize=4, color="#2e7d32", alpha=0.85)
    axes[1].axhline(30, color="red", lw=1, ls="--")
    axes[1].set_xticks(x)
    axes[1].set_xticklabels(bucket_stats.index)
    axes[1].set_xlabel("bucket de n_instrumentos")
    axes[1].set_ylabel("pct_acum (%)")
    axes[1].set_title("Overshoot sobre 30% según concentración")

    fig.tight_layout()
    fig.savefig(PLOTS_DIR / "subyacentes_30pct.png")
    plt.close(fig)


def plot_retornos_mensuales(daily: pd.DataFrame) -> None:
    daily = daily.dropna(subset=["ret_total"]).copy()
    daily["mes"] = daily["fecha"].dt.to_period("M").dt.to_timestamp("M")

    monthly = (
        daily.groupby(["fondo", "mes"])["ret_total"]
        .apply(lambda r: float((1 + r).prod() - 1))
        .reset_index()
    )
    monthly["decada"] = (monthly["mes"].dt.year // 10 * 10).astype(int).astype(str) + "s"

    decadas = sorted(monthly["decada"].unique())
    fig, ax = plt.subplots(figsize=(9, 4))
    for d in decadas:
        sub = monthly[monthly["decada"] == d]["ret_total"]
        sub = sub.clip(sub.quantile(0.005), sub.quantile(0.995))
        ax.hist(sub, bins=60, alpha=0.45, label=f"{d} (N={len(sub):,})")
    ax.axvline(0, color="black", lw=0.5)
    ax.set_xlabel("retorno mensual total")
    ax.set_ylabel("frecuencia (fondo-mes)")
    ax.set_title("Distribución de retornos mensuales por década")
    ax.legend()
    fig.tight_layout()
    fig.savefig(PLOTS_DIR / "retornos_mensuales.png")
    plt.close(fig)


def main() -> None:
    PLOTS_DIR.mkdir(parents=True, exist_ok=True)

    print(">>> cargando datos")
    historico = load_historico()
    fees = load_fees()
    sub = load_subyacentes()
    daily = compute_daily_total_return(historico)

    print(">>> 1/5 cobertura del universo")
    plot_cobertura_universo(historico)
    print(">>> 2/5 evento_pct")
    plot_evento_pct(historico)
    print(">>> 3/5 fees")
    plot_fees(fees)
    print(">>> 4/5 subyacentes (regla del 30%)")
    plot_subyacentes_30pct(sub)
    print(">>> 5/5 retornos mensuales")
    plot_retornos_mensuales(daily)

    print(f"\n>>> 5 plots guardados en {PLOTS_DIR.relative_to(PLOTS_DIR.parent.parent)}")


if __name__ == "__main__":
    sys.exit(main())
