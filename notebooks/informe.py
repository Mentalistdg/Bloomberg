# ---
# jupyter:
#   jupytext:
#     formats: py:percent,ipynb
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.16.0
#   kernelspec:
#     display_name: Python 3
#     language: python
#     name: python3
# ---

# %% [markdown]
# # Scoring de fondos mutuos — informe técnico
#
# **Caso técnico — Analista de Inversiones, Renta Variable Indirecta**
#
# El objetivo es construir un modelo cuantitativo que asigne un puntaje a
# cada fondo del universo de manera de informar la decisión de inversión,
# distinguiendo fondos atractivos de poco atractivos. Más que la precisión
# absoluta del modelo, el énfasis está en la **definición del problema**,
# la **construcción de features**, la **disciplina anti-leakage** del
# esquema de validación, y la **honestidad estadística** de los resultados.
#
# Este notebook es la capa de reporting del pipeline. La lógica vive en
# `src/` y se ejecuta vía scripts en `scripts/`. Reproducibilidad end-to-end:
#
# ```
# uv sync
# uv run python -m scripts.01_build_features
# uv run python -m scripts.02_eda_report
# uv run python -m scripts.03_build_features_full
# uv run python -m scripts.04_train_and_evaluate
# uv run python -m scripts.05_build_app_data
# ```

# %%
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from IPython.display import Image, display

ROOT = Path.cwd().parent if Path.cwd().name == "notebooks" else Path.cwd()
ARTIFACTS = ROOT / "artifacts"
PLOTS = ARTIFACTS / "plots"

with open(ARTIFACTS / "metrics.json") as f:
    metrics = json.load(f)
panel = pd.read_parquet(ARTIFACTS / "panel_features.parquet")
scores = pd.read_parquet(ARTIFACTS / "scores.parquet")
coefs = pd.read_csv(ARTIFACTS / "drivers_elastic.csv")

# %% [markdown]
# ## 1. Definición del problema
#
# **Variable a predecir:** **percentil cross-seccional del Sharpe ratio
# forward 12m** (`target_sharpe_rank_12m`). Para cada (fondo, mes T) se
# calcula el Sharpe anualizado de los retornos mensuales de T+1 a T+12 y
# se rankea cross-seccionalmente entre los fondos del universo activo en T.
#
# **Por qué Sharpe forward y no retorno forward:**
#
# 1. Combina simultáneamente "retorno alto" + "riesgo bajo" en una métrica
#    única, sin debate sobre pesos relativos del composite.
# 2. Es la unidad de calidad que un comité de inversiones de AFP entiende
#    por defecto — métrica universal de la industria.
# 3. Una AFP no maximiza retorno bruto; selecciona fondos para sostener
#    afiliados a largo plazo, donde la consistencia (Sharpe alto) importa
#    más que un retorno excepcional volátil.
#
# El retorno forward 12m simple (`target_ret_12m`) se mantiene como
# **métrica reportada en paralelo** (no de entrenamiento) para discusión
# interpretable del Q5-Q1 spread en %.
#
# **Enfoque elegido: explicativo con validación predictiva.** ElasticNet
# como primario (coeficientes interpretables, defendibles ante un comité)
# + LightGBM como sanity check de no-linealidad + **AxiomaticScorer**
# (fórmula con pesos teóricos, sin entrenar) como benchmark estructural.
# Razones:
#
# - Audiencia del modelo (comité de inversiones) entiende factores y
#   exposiciones, no SHAP de modelos black-box. La interpretabilidad pesa.
# - La literatura sugiere que la persistencia de retornos netos de fondos
#   mutuos es débil (Carhart 1997, Berk & Green 2004, Fama-French 2010);
#   un modelo explicativo bien calibrado es defensable, un black-box que
#   afirma alta capacidad predictiva no.
# - Comparar 4 scoreadores (ElasticNet vs LightGBM vs Axiomático vs naive)
#   permite contrastar empírico vs teórico vs trivial.

# %% [markdown]
# ## 2. Validación de datos contra el brief
#
# Antes de ingenierizar features verifico cada uno de los 4 ítems descritos
# en el brief. Los plots referenciados están en `artifacts/plots/`.

# %%
display(Image(filename=str(PLOTS / "cobertura_universo.png")))

# %% [markdown]
# **Hallazgo 1 — cobertura del universo.** El dataset trae 277 fondos, con
# entradas distribuidas a lo largo de las décadas y salidas materiales
# durante la GFC (2008-2009) y nuevamente en 2020. Esta heterogeneidad
# temporal exige (a) restringir la ventana modelable a un período donde el
# universo activo sea significativo y (b) tratar explícitamente el sesgo
# de supervivencia en la interpretación de los resultados.

# %%
display(Image(filename=str(PLOTS / "evento_pct_dist.png")))

# %% [markdown]
# **Hallazgo 2 — `evento_pct` corresponde a distribuciones de capital
# (típicamente income).** La firma de los datos es inequívoca: 99% de los
# eventos son positivos, con cadencia aproximadamente mensual y suma anual
# por fondo en el rango de 1-3% — consistente con yield de distribución de
# fondos balanceados. Existe un outlier puntual de 85% que se interpreta
# como devolución de capital y se neutraliza por winsorización al p99.5.
#
# Implicación operativa: el retorno total que recibe el inversionista
# combina cambio de NAV con distribución, según
# `ret_total_t = (precio_t/precio_{t-1} - 1) + evento_pct_t`. Calcular
# retornos solo desde precio sub-estima sistemáticamente el rendimiento en
# fondos con distribuciones recurrentes.

# %%
display(Image(filename=str(PLOTS / "fees_dist_y_cobertura.png")))

# %% [markdown]
# **Hallazgo 3 — fees solo se reportan desde 2024, pero son estructurales.**
# La columna `fee` tiene 89% de NULL en el dataset original; **todos los
# reportes son posteriores a 2024-01-31**. Sin embargo, el fee de un fondo
# es estructuralmente estable: la mediana de fondos tiene **un único valor
# de fee en toda su historia** y la desviación intra-fondo es ≈ 0. Bajo
# este supuesto explícito se aplica `ffill+bfill` dentro de cada fondo
# para imputar todo el panel desde el valor reportado en 2024+. Cobertura
# efectiva: **99.3%** (vs 12% con solo `ffill`).
#
# Una bandera binaria `fee_observado` (1 = valor original del mes,
# 0 = imputado) permite al modelo distinguir y captura indirectamente el
# efecto de período de reporte. Para los 2 fondos sin ningún reporte se
# imputa con la mediana cross-seccional del mes (fallback mínimo).

# %%
display(Image(filename=str(PLOTS / "subyacentes_30pct.png")))

# %% [markdown]
# **Hallazgo 4 — la métrica de concentración mide "número mínimo de
# instrumentos para alcanzar el 30% del AUM".** El campo `pct_acum` tiene
# un piso duro empírico en 30.001% (mínimo absoluto) y converge a 30% en
# fondos con muchos holdings. Esa firma matemática es exclusiva de un
# algoritmo greedy que se detiene al cruzar 30% — confirmado por la
# descripción oficial del dataset.
#
# Implicación: la señal de concentración la lleva `n_instrumentos`
# (bajo = concentrado, alto = diversificado), no `pct_acum` (casi
# constante en 30%). Cobertura: 129 de 277 fondos (47%) tienen este dato,
# casi todo en snapshot reciente — para los 148 sin dato se imputa la
# mediana cross-seccional + flag.

# %%
display(Image(filename=str(PLOTS / "retornos_mensuales.png")))

# %% [markdown]
# **Hallazgo 5 — distribución de retornos mensuales.** Las décadas posteriores
# a 1990 muestran distribuciones razonablemente similares con masas
# alrededor de 0.5-1.0% mensual y colas pesadas. La GFC (década 2000s)
# presenta una cola izquierda más pronunciada. Estos retornos están ya
# netos de fees (NAV de mutual fund se publica post-fee).

# %% [markdown]
# ## 3. Construcción de features
#
# **31 features totales** divididas en tres grupos:
#
# **CORE (14)** — derivadas de retornos, no-NaN obligatorio:
#
# | Feature | Intuición financiera |
# |---|---|
# | `ret_1m`, `ret_3m`, `ret_6m`, `ret_12m` | Persistencia de momentum a múltiples horizontes. |
# | `vol_12m`, `max_dd_12m`, `sharpe_12m` | Riesgo realizado, peor caída, risk-adjusted trailing. |
# | `vol_intrames`, `autocorr_diaria`, `ratio_dias_cero` | Microestructura intra-mes y proxies de iliquidez (NAV stale). |
# | `skewness_12m` | Asimetría de la distribución de retornos: positiva = sorpresas al alza, negativa = "vereda y hueco". |
# | `hit_rate_12m` | Fracción de meses positivos en último año: consistencia (no nivel). |
# | `distribution_yield_12m` | Suma de eventos de capital 12m: estilo income/value vs growth. |
# | `persistencia_rank_12m` | 1 − \|rank_t − rank_{t-12}\|: estabilidad de la posición relativa. |
#
# **EXTENDED (5)** — features estructurales del fondo:
#
# | Feature | Intuición |
# |---|---|
# | `fee` | Fee es predictor más robusto en literatura (Carhart 1997). Imputado bajo supuesto de fee estructural (cobertura 99.3%). |
# | `log_n_instrumentos`, `pct_acum` | Concentración del portafolio (insumo "concentración del primer decil" del enunciado). |
# | `fee_observado`, `concentracion_disponible` | Flags binarias: distinguen valores originales de imputados. |
#
# **RANK (12)** — percentiles cross-seccionales dentro del mes para todas
# las features anteriores con señal de nivel/escala. Robustos a regímenes
# y reducen escala-sensibilidad.
#
# **Anti-leakage (crítico):** todas las features usan rolling windows
# estrictamente hacia atrás. Los targets — únicos elementos forward del
# sistema — se construyen con `shift(-i)` de `ret_mensual` para
# i=1..12. La separación temporal permite walk-forward CV sin
# contaminación si se respeta un embargo igual al horizonte (12m).

# %% [markdown]
# ## 4. Esquema de validación: walk-forward expanding window con embargo
#
# Dado que el target es a 12 meses forward, las ventanas de target del
# set de entrenamiento se extienden 12 meses más allá del último mes de
# features de entrenamiento. Para que el set de validación no contenga
# observaciones cuya target window se solape con la del entrenamiento,
# se introduce un **embargo de 12 meses** entre el último mes de
# entrenamiento y el primer mes de validación.

# %%
diag = pd.read_csv(ARTIFACTS / "fold_diagnostics.csv")
print(diag[["fold", "train_start", "train_end", "val_start", "val_end",
            "n_train", "n_val", "elastic_alpha", "elastic_l1_ratio"]].to_string(index=False))

# %% [markdown]
# Cada fold expande el set de entrenamiento en 12 meses adicionales y
# evalúa el siguiente bloque de 12 meses tras un gap de 12 meses. El
# resultado son 9 folds que cubren 2016-2024 como períodos de validación
# completamente fuera de muestra.

# %% [markdown]
# ## 5. Modelos y resultados out-of-sample

# %%
sm = metrics  # alias para legibilidad
labels = ["elastic", "lgbm", "benchmark", "axiomatic"]
rows = []
for label in labels:
    if label not in sm:
        continue
    m = sm[label]
    rows.append({
        "modelo": label,
        "IC mean": m["ic_summary"]["mean"],
        "IC IR": m["ic_summary"]["ic_ir"],
        "% meses IC>0": m["ic_summary"]["hit"],
        "Q5-Q1 (ret%)": m["spread_q5_q1_mean"],
        "Hit-Top25": m["hit_rate_top25_mean"],
        "CI95 IC low": m["ic_bootstrap"]["ci_low"],
        "CI95 IC high": m["ic_bootstrap"]["ci_high"],
    })
results_df = pd.DataFrame(rows).set_index("modelo")
results_df.style.format({c: "{:+.4f}" for c in results_df.columns})

# %% [markdown]
# ### Lectura crítica de los resultados
#
# - El **ElasticNet** alcanza IC mensual de +0.19, IR de +0.85, y CI95
#   bootstrap de [+0.15, +0.23] que **NO incluye al cero** — se rechaza la
#   hipótesis nula de ausencia de poder predictivo al 95%. Esta es una
#   mejora material respecto al ejercicio inicial con target retorno simple.
# - El **AxiomaticScorer** (sin entrenamiento, fórmula con pesos teóricos)
#   alcanza IC = +0.15, comparable al ML. Implicación: la mayor parte de la
#   señal en este universo se puede expresar con teoría financiera básica
#   (Sharpe alto, fee bajo, drawdown limitado, baja concentración, baja
#   iliquidez). El ML aporta refinamiento marginal pero significativo.
# - El **LightGBM** alcanza IC = +0.15, ligeramente inferior al ElasticNet
#   pero positivo. No hay evidencia fuerte de no-linealidades aprovechables.
# - El **benchmark naive** (`ret_12m_rank − fee_rank`) alcanza IC = +0.12
#   — mucho mejor que cero, lo que confirma que momentum + fee son señales
#   genuinas en este universo. Pero queda detrás de los demás scoreadores.
# - **Diebold-Mariano** ElasticNet vs benchmark: estadístico = −1.68,
#   p = 0.093 (significativo al 10%, marginal al 5%). ElasticNet vs
#   AxiomaticScorer: p = 0.40 (no significativo — ambos comparables).
#
# **Validación multi-lente OOS** — el mismo score evaluado contra
# 4 targets forward realizados (Q5-Q1 spread, horizonte 12m):
#
# | Modelo | Retorno % | Sharpe | Sortino | Max DD |
# |---|---|---|---|---|
# | **ElasticNet** | −1.3% | **+0.37** | **+0.75** | **+0.04** |
# | AxiomaticScorer | −1.0% | +0.26 | +0.70 | +0.03 |
# | LightGBM | −1.6% | +0.27 | +0.21 | +0.02 |
# | Benchmark naive | +0.5% | +0.19 | −0.26 | +0.02 |
#
# El ElasticNet entrenado con target Sharpe gana fuertemente en lentes
# risk-adjusted (Sharpe Q5-Q1 +0.37, Sortino +0.75) y reduce drawdowns
# realizados (+0.04 en max DD), pero **pierde en retorno bruto** (−1.3%).
# Esto es **comportamiento esperado y consistente con el target elegido**:
# el modelo aprendió a identificar fondos de mejor calidad ajustada por
# riesgo, no a maximizar retorno absoluto.

# %%
display(Image(filename=str(PLOTS / "signal_elastic.png")))
display(Image(filename=str(PLOTS / "signal_benchmark.png")))

# %% [markdown]
# ## 6. Drivers del score (interpretabilidad)
#
# Promedio de los coeficientes ElasticNet a través de los 9 folds, sobre
# variables estandarizadas (cada variable centrada en 0 y con desv. estándar 1):

# %%
display(Image(filename=str(PLOTS / "drivers_elastic.png")))

# %% [markdown]
# Los signos de los coeficientes promedio son consistentes con intuición
# financiera estándar:
#
# - **Positivos (premia el modelo):** `sharpe_12m_rank`, `hit_rate_12m_rank`,
#   `persistencia_rank_12m`, `max_dd_12m_rank` (rank alto = drawdown menos
#   profundo). Coherente con calidad ajustada por riesgo + consistencia +
#   estabilidad de la posición relativa.
# - **Negativos (penaliza el modelo):** `vol_12m`, `vol_12m_rank`,
#   `fee_rank`, `pct_acum_rank` (alta concentración), `autocorr_diaria_rank`
#   (proxy de iliquidez). Coherente con evitar volatilidad, fees altos,
#   concentración excesiva y subyacentes ilíquidos.
#
# La magnitud de los coeficientes es moderada — el problema es difícil
# pero la señal existe y es coherente con teoría.

# %% [markdown]
# ## 7. Limitaciones y extensiones
#
# **Limitaciones reconocidas:**
#
# 1. *Reporte de fee solo desde 2024-01-31.* El dataset reporta fees en
#    ~12% del panel originalmente. Mitigado bajo el supuesto explícito de
#    fee estructural (verificable empíricamente: std intra-fondo ≈ 0 en el
#    subperíodo observado) — `ffill+bfill` intra-fondo lleva cobertura a
#    99.3%. Limitación inevitable de la fuente; el supuesto es razonable
#    pero no verificable directamente para fechas pre-2024.
# 2. *Survivorship bias parcial.* 87.7% de fondos llegan vivos al final
#    del dataset (mayo 2026). Los 34 "muertos" tienen mediana de cobertura
#    de 2 meses → sugiere snapshot del universo vigente, no historia con
#    quiebras reales. El filtro de cobertura mínima ≥36 meses excluye a la
#    mayoría. Detalle en `artifacts/survivorship_stats.json`.
# 3. *Anonimización del universo.* Sin clasificación por estilo, región
#    o asset class, no es posible incorporar features de "fondo vs benchmark
#    de estilo". Mitigación parcial: features stylistic capturan propiedades
#    correlacionadas con estilo.
# 4. *Liquidez por proxies indirectos.* No hay variable directa; se usan
#    `autocorr_diaria`, `ratio_dias_cero`, `vol_intrames` como proxies
#    derivados de la serie diaria.
# 5. *Sin información de flujos / AUM.* Imposibilita detectar diseconomies
#    of scale (Berk-Green) y diferenciar fondos chicos vs grandes.
#
# **Extensiones con más tiempo o más datos:**
#
# - Incorporar **datos macro / de mercado** como features comunes a todos
#   los fondos (régimen de volatilidad, term spread, dollar index) para
#   condicionar el score por régimen.
# - **Datos de holdings agregados** (style box, exposición sectorial,
#   net flows) que sustituyan el proxy actual de concentración.
# - **Backtest con costos** y rebalanceo realista — seleccionar top-quintil
#   mensualmente, computar Sharpe de la cartera resultante neta de turnover.
# - **Combinación con due diligence cualitativa.** El score cuantitativo
#   debiera ser un input dentro de un proceso multi-factor, no la
#   decisión por sí sola — especialmente dado que su CI bootstrap
#   incluye al cero.

# %% [markdown]
# ## 8. Uso de IA (LLMs) en el proceso
#
# Se utilizó Claude (Anthropic) en tres etapas del proyecto:
#
# 1. **Inspección y reconciliación del dataset** — particularmente para
#    validar la interpretación del campo `pct_acum` de `subyacentes`. La
#    descripción inicial del brief sugería "primer decil" (10%), pero la
#    firma matemática del campo (piso duro en 30.001%, convergencia a 30%
#    en fondos diversificados) era inconsistente. La interacción con el
#    LLM permitió formular hipótesis competidoras y testearlas con queries
#    SQL específicas y datos reales de mercado de Bloomberg (cálculo de
#    pesos cap-weighted vs equal-weighted del S&P 500 en vivo). La
#    descripción oficial posterior confirmó el umbral del 30%.
# 2. **Code review del esquema de validación** — particularmente la
#    construcción del walk-forward con embargo, para verificar que el
#    target de 12 meses no contamina entre folds.
# 3. **Estructuración del informe** — borrador de la narrativa de cada
#    sección y los argumentos para defender la elección del target y el
#    enfoque explicativo.
#
# **Validación de las salidas del modelo:** todo código sugerido se
# ejecutó contra los datos reales y se contrastó contra literatura citada
# (Carhart 1997, Berk-Green 2004, López de Prado para anti-leakage).
# Conclusiones cualitativas se contrastaron también con la firma empírica
# del dataset.
