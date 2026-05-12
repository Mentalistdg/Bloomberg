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
- 250 fondos modelables (≥36m de historia) · 50,641 obs panel · 29 folds walk-forward rolling
- Modelo principal: **ElasticNet** · IC mensual = **+0.074**, CI95 bootstrap **[+0.042, +0.107]** (rechaza H₀ al 95%)
- D10-D1 en Sharpe realizado: **+0.29** · D10-D1 en Sortino: **+0.61** · D10-D1 en max DD: **+0.019**
- 3 modelos comparados: ElasticNet vs LightGBM vs Benchmark naive

---

## Slide 2 — Definición del problema y variable objetivo

**Variable objetivo:** **percentil cross-seccional del Sortino forward 6m**
(`target_sortino_rank_6m`).

**Fórmula del Sortino:**

$$\text{Sortino} = \frac{\bar{r} - r_f}{\text{downside\_dev}} \cdot \sqrt{12}, \quad \text{downside\_dev} = \sqrt{\frac{1}{n}\sum_{i=1}^{n} \min(r_i, 0)^2}$$

**Por qué Sortino y no Sharpe o retorno:**
- El Sortino penaliza solo la volatilidad bajista — la volatilidad alcista
  es deseable en un fondo, no penalizable.
- El Sharpe trata toda volatilidad por igual, lo cual castiga fondos que
  suben mucho (exactamente lo que una AFP quiere).

**Percentil cross-seccional — cómo se construye el target:**
Para cada mes T, se calcula el Sortino forward 6m (meses T+1 a T+6) de
**todos** los fondos activos del universo. Luego se rankea de 0 (peor)
a 1 (mejor). Este percentil es el target de entrenamiento.

**Por qué percentil y no valor absoluto:**
Un Sortino de 2.0 en mercado alcista (2021) no es comparable a un Sortino
de 2.0 en mercado bajista (2008). El percentil normaliza por régimen —
un rank de 0.9 significa "top 10% de su cohorte temporal". También evita
que el modelo aprenda a predecir el mercado (fácil en-muestra, imposible
fuera de muestra) en vez de características intrínsecas del fondo.

**Horizonte 6 meses — justificación:**
- **3m** demasiado ruidoso: solo 3 obs para estimar downside deviation,
  rankings inestables.
- **12m** demasiado lento: el embargo de 12m reduce folds de validación
  (~15 vs 29), debilitando la significancia estadística.
- **6m** equilibra reactividad y estabilidad (29 folds, 6 obs para
  downside risk).

**Enfoque: explicativo con validación predictiva.** Trade-off explícito:
- Audiencia (comité de inversiones) entiende factores → ElasticNet primary.
- LightGBM como sanity check no-lineal.
- Benchmark naive (`ret_12m_rank − fee_rank`) como línea base mínima.

---

## Slide 3 — Validación de datos contra el brief

> Antes de modelar, verificar que entendimos los 4 insumos del brief.

| Item del brief | Validación realizada |
|---|---|
| **Serie histórica de retornos** | 277 fondos × 1988-2026, mediana 20 años por fondo. Retorno total = NAV.pct_change() + evento_pct (winsorizado p99.5). |
| **Eventos de capital** (`evento_pct`) | 99% positivos, cadencia mensual, suma anual 1-3% → distribuciones de capital. Usado en retorno total Y como feature derivada `distribution_yield_12m`. |
| **Concentración del 30% del AUM** | `pct_acum` con piso duro empírico en 30.001% (algoritmo greedy del 30% del AUM); `n_instrumentos` mide diversificación. Ambos entran al modelo (no solo proxy). |
| **Fees** | 89% NULL en raw — todos los reportes son **posteriores a 2024-01-31**. El fee se interpreta como **tasa anual** (expense ratio) directamente, sin anualizar. Verificación clave: el fee es estructuralmente constante intra-fondo (mediana 1 valor único, std ≈ 0). Bajo supuesto explícito de fee estructural se aplica `ffill+bfill` → cobertura efectiva **99.3%**. Bandera `fee_observado` distingue valores originales de imputados. |

**Survivorship bias detectado:** 87.7% de fondos vivos al final, "muertos"
con mediana 2m de cobertura → snapshot del universo vigente, no historia
con quiebras. Filtro mínimo ≥36m excluye los muertos cortos. Plot:
`artifacts/plots/survivorship.png`.

---

## Slide 4 — Construcción de features (intuición financiera)

**32 features totales** divididas en 3 grupos. Anti-leakage estricto.

| Grupo | Features | Intuición |
|---|---|---|
| Momentum | `ret_1m`, `ret_3m`, `ret_6m`, `ret_12m` | Persistencia de retornos |
| Riesgo | `vol_12m`, `max_dd_12m`, `sharpe_12m` | Volatilidad, peor caso, risk-adj |
| Microestructura | `vol_intrames`, `autocorr_diaria`, `ratio_dias_cero` | Proxies de iliquidez del subyacente |
| Estilísticas | `skewness_12m`, `hit_rate_12m`, `distribution_yield_12m` | Asimetría, consistencia, income vs growth |
| Persistencia | `persistencia_rank_12m` | Estabilidad de la posición relativa del fondo |
| Costo | `fee` + flag `fee_observado` | Costo garantizado, predictor de underperformance |
| Concentración | `log_n_instrumentos`, `pct_acum` + flag | Convicción vs diversificación |
| Ranks (13) | percentil cross-seccional de las anteriores | Robustez a régimen + escala |

**Diseño anti-leakage:** todas las rolling windows hacia atrás. Targets
forward (`shift(-i)`) son la única columna que mira adelante.

---

## Slide 5 — Esquema de validación: walk-forward + embargo

**Por qué embargo igual al horizonte:** el target spans `h` meses forward,
por lo que sin embargo el target del set de train se solapa con el de val.
Para horizonte 6m → embargo 6m.

**Layout temporal:**

```
[--- TRAIN (rolling 120m) ---][embargo 6m][--- VAL (12m) ---]
```

**29 folds** cubriendo 1996-2025 como períodos completamente fuera de
muestra. **Rolling window** (máximo 10 años de training) en vez de
expanding: el modelo se adapta a cambios de régimen de mercado
(pre/post-2008, era de tasa cero, inflación 2022+).

**Cross-validación interna de hiperparámetros:** ElasticNetCV con 5-fold
interno selecciona α y l1_ratio óptimos en cada fold del walk-forward.

---

## Slide 6 — Resultados out-of-sample (horizonte 6m)

**Tabla principal (32 features, target Sortino forward 6m):**

| Modelo | IC mensual | IR | Hit (% meses +) | CI95 (bootstrap) |
|---|---|---|---|---|
| **ElasticNet** | **+0.074** | **+0.26** | 60.8% | **[+0.042, +0.107]** |
| Benchmark naive | +0.074 | +0.31 | 62.1% | [+0.049, +0.100] |
| LightGBM | +0.040 | +0.15 | 57.2% | [+0.012, +0.067] |

**Diebold-Mariano** (Newey-West, h=6):
- ElasticNet vs benchmark: estadístico = −0.45, **p = 0.654** (no significativo).

**Lectura:**
- **El IC del ElasticNet rechaza H₀ al 95%** (CI no incluye cero) —
  hay señal real.
- El benchmark naive (`ret_12m_rank − fee_rank`) empata en IC: la mayor
  parte de la señal viene de momentum + fee. **Hallazgo
  honesto y esperado** — el ElasticNet ofrece ventajas operativas
  (adaptación a régimen, features de riesgo, interpretabilidad).
- LightGBM no supera al lineal — sin evidencia de no-linealidad explotable.

---

## Slide 7 — Validación multi-lente

> Un score robusto debe discriminar bien en VARIAS métricas, no solo en
> aquella con la que fue entrenado.

**D10-D1 spread sobre 4 targets forward realizados:**

| Modelo | Retorno % | **Sharpe** | **Sortino** | **Max DD** |
|---|---|---|---|---|
| **ElasticNet** | −0.6% | **+0.29** | **+0.61** | **+0.019** |
| LightGBM | +0.1% | +0.19 | +0.67 | +0.009 |
| Benchmark naive | +1.0% | +0.18 | −0.16 | +0.009 |

**Lectura crítica honesta:**
- ElasticNet entrenado con target Sortino **gana en métricas
  risk-adjusted** (Sharpe D10-D1 +0.29, Sortino +0.61, max DD +1.9 pp menos).
- **Neutro en retorno bruto** (D10-D1 ≈ 0%). Esto es **comportamiento
  esperado y consistente**: el modelo aprendió a identificar fondos de
  mejor calidad ajustada por riesgo downside, no a maximizar retorno absoluto.
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
   de cualquier modelo predictivo, mitigado por walk-forward rolling
   window (10 años) que descarta datos antiguos.

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
| RandomForest / RNN | Riesgo de overfitting con ~50K obs y ~32 features; menor interpretabilidad. |
| Modelado a frecuencia diaria | Observaciones autocorrelacionadas; más filas ≠ más información. |
| AT clásico (RSI, MACD) | Diseñado para timing intra-día, no selección de fondos a 6m. Sí mantengo features stylistic. |

**Con más tiempo / más datos:** features macro de régimen, AUM por fondo,
clasificación por estilo, capacity constraints, backtest con costos de turnover.

---

## Slide 10 — Uso de IA y reflexión metodológica

**Herramientas usadas:** Claude (Anthropic, Opus 4.6) integrado vía CLI.

**Etapas con asistencia LLM:**
1. **Discusión metodológica iterativa** — particularmente la definición
   de variable objetivo. Conversación inicial proponía retorno forward
   12m; tras descomponer "qué hace bueno a un fondo para una AFP" se
   separó target (calidad realizada futura) de features (observables hoy).
   Resultado: cambio a Sortino forward 6m con justificación documentada —
   el Sortino penaliza solo downside, la volatilidad alcista es deseable.
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
