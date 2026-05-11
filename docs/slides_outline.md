# Outline de slides — exposición 15-20 min, máximo 10 láminas

> Estructura sugerida. Cada slide: 1.5-2 min de exposición. Markdown listo
> para pegar en PowerPoint / Keynote / Google Slides.

---

## Slide 1 — Portada + mensaje principal

**Título:** Scoring de fondos mutuos — Renta Variable Indirecta

**Subtítulo (1 línea):** Modelo cuantitativo para informar la decisión
de inversión en un universo de 277 fondos USA, con énfasis en disciplina
anti-leakage, supuestos declarados y validación robusta.

**Datos clave de portada:**
- 250 fondos modelables (≥36m de historia) · 49,147 obs panel · 9 folds walk-forward
- Modelo principal: **ElasticNet** · IC mensual = **+0.19**, CI95 bootstrap **[+0.15, +0.23]** (rechaza H₀ al 95%)
- Q5-Q1 en Sharpe realizado: **+0.37** · Q5-Q1 en Sortino: **+0.75** · Q5-Q1 en max DD: **+0.04**
- 4 scoreadores comparados: ElasticNet vs LightGBM vs AxiomaticScorer (teoría) vs Naive

---

## Slide 2 — Definición del problema

**Variable objetivo:** **percentil cross-seccional del Sharpe forward 12m**
(`target_sharpe_rank_12m`).

**Por qué Sharpe y no retorno:**
- Combina retorno + riesgo en una métrica única (sin debate sobre pesos).
- Métrica universal de la industria — el comité la entiende sin traducción.
- Una AFP no maximiza retorno bruto: selecciona fondos para sostener
  afiliados a largo plazo. Consistencia (Sharpe alto) > retorno volátil.

**Por qué cross-seccional:** la decisión de selección es relativa al
universo disponible en el mes T. Robusto a regímenes (no se contamina
con beta del universo).

**Enfoque: explicativo con validación predictiva.** Trade-off explícito:
- Audiencia (comité de inversiones) entiende factores → ElasticNet primary.
- LightGBM como sanity check no-lineal.
- AxiomaticScorer (sin entrenar, pesos teóricos) como benchmark estructural.
- Benchmark naive (`ret_12m_rank − fee_rank`) como línea base mínima.

---

## Slide 3 — Validación de datos contra el brief

> Antes de modelar, verificar que entendimos los 4 insumos del brief.

| Item del brief | Validación realizada |
|---|---|
| **Serie histórica de retornos** | 277 fondos × 1988-2026, mediana 20 años por fondo. Retorno total = NAV.pct_change() + evento_pct (winsorizado p99.5). |
| **Eventos de capital** (`evento_pct`) | 99% positivos, cadencia mensual, suma anual 1-3% → distribuciones de capital. Usado en retorno total Y como feature derivada `distribution_yield_12m`. |
| **Concentración del primer decil de subyacentes** | `pct_acum` con piso duro empírico en 30.001% (algoritmo greedy del 30% del AUM); `n_instrumentos` mide diversificación. Ambos entran al modelo (no solo proxy). |
| **Fees** | 89% NULL en raw — todos los reportes son **posteriores a 2024-01-31**. Verificación clave: el fee es estructuralmente constante intra-fondo (mediana 1 valor único, std ≈ 0). Bajo supuesto explícito de fee estructural se aplica `ffill+bfill` → cobertura efectiva **99.3%**. Bandera `fee_observado` distingue valores originales de imputados. |

**Survivorship bias detectado:** 87.7% de fondos vivos al final, "muertos"
con mediana 2m de cobertura → snapshot del universo vigente, no historia
con quiebras. Filtro mínimo ≥36m excluye los muertos cortos. Plot:
`artifacts/plots/survivorship.png`.

---

## Slide 4 — Construcción de features (intuición financiera)

**31 features totales** divididas en 3 grupos. Anti-leakage estricto.

| Grupo | Features | Intuición |
|---|---|---|
| Momentum | `ret_1m`, `ret_3m`, `ret_6m`, `ret_12m` | Persistencia (Carhart 1997) |
| Riesgo | `vol_12m`, `max_dd_12m`, `sharpe_12m` | Volatilidad, peor caso, risk-adj |
| Microestructura | `vol_intrames`, `autocorr_diaria`, `ratio_dias_cero` | Proxies de iliquidez del subyacente |
| Estilísticas | `skewness_12m`, `hit_rate_12m`, `distribution_yield_12m` | Asimetría, consistencia, income vs growth |
| Persistencia | `persistencia_rank_12m` | Estabilidad de la posición relativa del fondo |
| Costo | `fee` + flag `fee_observado` | Predictor causal directo (Carhart 1997) |
| Concentración | `log_n_instrumentos`, `pct_acum` + flag | Convicción vs diversificación |
| Ranks (12) | percentil cross-seccional de las anteriores | Robustez a régimen + escala |

**Diseño anti-leakage:** todas las rolling windows hacia atrás. Targets
forward (`shift(-i)`) son la única columna que mira adelante.

---

## Slide 5 — Esquema de validación: walk-forward + embargo

**Por qué embargo igual al horizonte:** el target spans `h` meses forward,
por lo que sin embargo el target del set de train se solapa con el de val.
Para horizonte 12m → embargo 12m.

**Layout temporal:**

```
[--- TRAIN (expanding desde 2010) ---][embargo 12m][--- VAL (12m) ---]
```

**9 folds para horizonte 12m** cubriendo 2016-2024 como períodos
completamente fuera de muestra. Cada fold expande train en 12m e incluye
nuevos datos.

**Multi-horizonte:** pipeline corre también horizontes 3m y 6m
(`artifacts/horizon_comparison.csv`) para análisis de sensibilidad.

**Cross-validación interna de hiperparámetros:** ElasticNetCV con 5-fold
interno selecciona α y l1_ratio óptimos en cada fold del walk-forward.

---

## Slide 6 — Resultados out-of-sample (horizonte 12m)

**Tabla principal (32 features):**

| Modelo | IC mensual | IR | Hit (% meses + ) | CI95 (bootstrap) |
|---|---|---|---|---|
| **ElasticNet** | **+0.190** | **+0.85** | 79.6% | **[+0.147, +0.232]** |
| LightGBM | +0.150 | +0.59 | 75.9% | [+0.100, +0.199] |
| AxiomaticScorer | +0.153 | +0.51 | 71.3% | [+0.093, +0.208] |
| Benchmark naive | +0.117 | +0.50 | 71.3% | [+0.072, +0.160] |

**Diebold-Mariano** (Newey-West, h=12):
- ElasticNet vs benchmark: estadístico = −1.68, **p = 0.093** (sig al 10%).
- ElasticNet vs Axiomático: p = 0.40 (no significativo — son comparables).

**Lectura:**
- **El IC del ElasticNet rechaza H₀ al 95%** (CI no incluye cero) —
  hay señal real.
- AxiomaticScorer (sin entrenar) compite parejo: la mayor parte de la
  señal viene de teoría financiera básica. El ML aporta refinamiento.
- LightGBM no supera al lineal — sin evidencia de no-linealidad explotable.

---

## Slide 7 — Validación multi-lente

> Un score robusto debe discriminar bien en VARIAS métricas, no solo en
> aquella con la que fue entrenado.

**Q5-Q1 spread sobre 4 targets forward realizados:**

| Modelo | Retorno % | **Sharpe** | **Sortino** | **Max DD** |
|---|---|---|---|---|
| **ElasticNet** | −1.3% | **+0.37** | **+0.75** | **+0.04** |
| AxiomaticScorer | −1.0% | +0.26 | +0.70 | +0.03 |
| LightGBM | −1.6% | +0.27 | +0.21 | +0.02 |
| Benchmark naive | +0.5% | +0.19 | −0.26 | +0.02 |

**Lectura crítica honesta:**
- ElasticNet entrenado con target Sharpe **gana fuerte en métricas
  risk-adjusted** (Sharpe Q5-Q1 +0.37, Sortino +0.75, max DD +4 pp menos).
- **Pierde en retorno bruto** (Q5-Q1 = −1.3%). Esto es **comportamiento
  esperado y consistente**: el modelo aprendió a identificar fondos de
  mejor calidad ajustada por riesgo, no a maximizar retorno absoluto.
- Para una AFP: tradeoff deseable. Para un trader de momentum: no.

---

## Slide 8 — Drivers del score (interpretabilidad)

→ Plot horizontal de coeficientes ElasticNet promedio entre folds
(features estandarizadas).

**Signos consistentes con intuición financiera:**
- **Positivos:** `sharpe_12m_rank`, `hit_rate_12m_rank`,
  `persistencia_rank_12m`, `max_dd_12m_rank` (rank alto = drawdown menos
  profundo). Calidad ajustada por riesgo + consistencia + estabilidad.
- **Negativos:** `vol_12m`, `vol_12m_rank`, `fee_rank`, `pct_acum_rank`,
  `autocorr_diaria_rank`. Penaliza volatilidad, fees, concentración,
  iliquidez.

**Lo que esto le dice al comité:** el modelo no es una caja negra. Cada
coeficiente tiene signo y magnitud defendibles desde teoría. El ranking
de un fondo se puede descomponer en sus drivers individuales (endpoint
`/api/drivers` del dashboard).

---

## Slide 9 — Supuestos declarados, limitaciones y alternativas descartadas

**Supuestos que sostienen el modelo:**
1. **Fee estructural en el tiempo** — verificable empíricamente en
   2024-2026 (std intra-fondo ≈ 0); habilita cobertura del fee 99.3%.
2. **Estabilidad de las relaciones aprendidas** — supuesto inevitable
   de cualquier modelo predictivo, mitigado por walk-forward expanding.

**Limitaciones reconocidas:**
1. Survivorship bias parcial (mitigado con filtro ≥36 meses).
2. Sin clasificación por estilo/región/asset class → ranking vs universo
   completo, no vs peers.
3. Liquidez por proxies (no variable directa).
4. Sin AUM ni flujos (no se detecta diseconomies of scale).

**Alternativas que descarté (entregable #3):**

| Alternativa | Por qué se descartó |
|---|---|
| Predecir retorno simple a 1m | A 1m es ~99% beta de mercado, casi cero alfa de fondo. Horizonte equivocado. |
| Clasificación binaria top-quintil | Pierde información ordinal. |
| RandomForest / RNN | Riesgo de overfitting con ~50K obs y ~30 features; menor interpretabilidad. |
| Modelado a frecuencia diaria | Observaciones autocorrelacionadas; más filas ≠ más información. |
| AT clásico (RSI, MACD) | Diseñado para timing intra-día, no selección de fondos a 12m. Sí mantengo features stylistic. |

**Con más tiempo / más datos:** features macro de régimen, AUM por fondo,
clasificación por estilo, backtest con costos de turnover.

---

## Slide 10 — Uso de IA y reflexión metodológica

**Herramientas usadas:** Claude (Anthropic, Opus 4.7) integrado vía CLI.

**Etapas con asistencia LLM:**
1. **Discusión metodológica iterativa** — particularmente la definición
   de variable objetivo. Conversación inicial proponía retorno forward
   12m; tras descomponer "qué hace bueno a un fondo para una AFP" se
   separó target (calidad realizada futura) de features (observables hoy).
   Resultado: cambio a Sharpe forward 12m con justificación documentada.
2. **Validación empírica de supuestos** — la decisión de imputar fee
   con `bfill` requirió consultar al SQLite directamente para verificar
   que el fee es constante intra-fondo. Solo se aceptó tras evidencia.
3. **Code review** del walk-forward, embargo e imputación.
4. **Generación de docstrings y narrativa.**

**Ejemplo concreto:** ante "¿se puede imputar fee con el último valor
del fondo?", el LLM cuestionó la premisa solicitando datos antes de
implementar. Las queries SQL revelaron que el fee se reporta solo desde
2024 (no era simplemente "11% NaN aleatorio"), lo que cambió la
estrategia de imputación de "mediana cross-seccional" a "ffill+bfill
intra-fondo bajo supuesto estructural" + flag binaria.

**Validación:** todo cambio se ejecutó contra el dataset real; los
hallazgos se reportan tal como salen del pipeline (no se suavizaron); las
decisiones se documentaron explícitamente con su tradeoff en `README.md`.

**Reflexión final:** el resultado más importante de este caso no es solo
el IC. Es el **proceso**: definición del problema con tradeoffs explícitos,
supuestos declarados y verificados con datos, anti-leakage disciplinado,
validación multi-lente, alternativas descartadas con justificación. Un
modelo cuyo intervalo bootstrap excluye al cero ES señal real — pero un
proceso que el comité puede inspeccionar línea por línea es lo que
permite tomar la decisión.
