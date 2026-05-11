# Informe — Scoring de fondos mutuos

> Caso técnico Analista de Inversiones, Renta Variable Indirecta. Documento
> resumen. El análisis técnico completo, con código y outputs, está en
> `notebooks/informe.ipynb`. La defensa visual está en `slides.pdf`.
> Documentación expandida de supuestos y limitaciones en `README.md`.

## 1. Definición del problema

**Variable a predecir:** **percentil cross-seccional del Sharpe ratio
forward 12m** (`target_sharpe_rank_12m`). Para cada (fondo, mes T) se
calcula el Sharpe anualizado de los retornos mensuales de T+1 a T+12 y
se rankea cross-seccionalmente entre los fondos del universo activo en T.

**Por qué Sharpe forward y no retorno forward:**

- Combina simultáneamente "retorno alto" + "riesgo bajo" en una métrica
  única, sin debate sobre pesos relativos.
- Es la unidad de calidad que un comité de inversiones de AFP entiende
  por defecto — métrica universal de la industria.
- Una AFP no maximiza retorno bruto; selecciona fondos para sostener
  afiliados a largo plazo, donde la consistencia (Sharpe alto) importa
  más que un retorno excepcional volátil.

El retorno forward simple (`target_ret_12m`) se mantiene como métrica
reportada en paralelo (no de entrenamiento) para discusión interpretable
del Q5-Q1 spread en %.

**Enfoque elegido: explicativo con validación predictiva.** El modelo
principal es ElasticNet (lineal regularizado, coeficientes interpretables
como drivers del score). En paralelo:

- **LightGBM** como sanity check de no-linealidad — si supera al lineal,
  hay interacciones que el lineal no captura.
- **AxiomaticScorer** — fórmula con pesos teóricos derivados de teoría
  financiera de selección de fondos (sin entrenar). Permite contrastar
  "lo que el modelo aprende empíricamente" contra "lo que se sabe
  estructuralmente".
- **Benchmark naive** — `ret_12m_rank − fee_rank`, línea base mínima
  contra la que cualquier modelo debe ganar.

Tradeoff aceptado: priorizamos defensibilidad y transparencia sobre
máxima performance OOS.

## 2. Construcción de features

**31 features totales** divididas en tres grupos:

**CORE (14)** — derivadas de retornos diarios y mensuales, no-NaN
obligatorio para que la observación sea modelable:

| Feature | Intuición |
|---|---|
| `ret_1m`, `ret_3m`, `ret_6m`, `ret_12m` | Momentum a múltiples horizontes |
| `vol_12m`, `max_dd_12m`, `sharpe_12m` | Riesgo realizado y risk-adjusted |
| `vol_intrames`, `autocorr_diaria`, `ratio_dias_cero` | Microestructura intra-mes y proxy de iliquidez |
| `skewness_12m`, `hit_rate_12m` | Asimetría y consistencia de la distribución de retornos |
| `distribution_yield_12m` | Yield anual de eventos de capital (estilo income vs growth) |
| `persistencia_rank_12m` | Estabilidad de la posición relativa del fondo en el universo |

**EXTENDED (5)** — features estructurales del fondo:

| Feature | Notas |
|---|---|
| `fee` | Imputado con ffill+bfill intra-fondo; cobertura 99.3% (ver supuesto #1 abajo) |
| `log_n_instrumentos`, `pct_acum` | Concentración del portafolio (insumo "primer decil" del enunciado) |
| `fee_observado` | Flag binaria: 1 si fee venía reportado en ese mes (período 2024+), 0 si imputado |
| `concentracion_disponible` | Flag binaria de disponibilidad de concentración |

**RANK (12)** — percentiles cross-seccionales dentro del mes para todas
las features anteriores con señal de nivel/escala. Robustos a cambios de
régimen.

**Anti-leakage (crítico):** todas las features se computan al cierre del
mes T usando exclusivamente información de t ≤ T (rolling hacia atrás).
Los targets — únicos elementos forward del sistema — se construyen con
`shift(-i)` de `ret_mensual`. Walk-forward CV con embargo de 12 meses
entre fin de train e inicio de val asegura que no se solapen ventanas
de target entre folds.

## 3. Modelo y validación

**Walk-forward expanding window**: 9 folds para horizonte 12m, mínimo
60 meses de training, 12 meses de validación, embargo = 12m.

**Resultados out-of-sample (target Sharpe forward 12m, full features):**

| Modelo | IC mensual | IR | Hit % | CI95 (bootstrap) |
|---|---|---|---|---|
| **ElasticNet** (primary) | **+0.190** | **+0.85** | 79.6% | **[+0.147, +0.232]** |
| LightGBM | +0.150 | +0.59 | 75.9% | [+0.100, +0.199] |
| AxiomaticScorer | +0.153 | +0.51 | 71.3% | [+0.093, +0.208] |
| Benchmark naive | +0.117 | +0.50 | 71.3% | [+0.072, +0.160] |

**Diebold-Mariano** (con corrección Newey-West, h=12):

- ElasticNet vs benchmark naive: estadístico = −1.68, **p = 0.093**
  (significativo al 10%, marginal al 5%).
- ElasticNet vs AxiomaticScorer: p = 0.40 (no significativo — el
  axiomático ya captura buena parte de la señal con teoría financiera).

**Multi-lente OOS** — el mismo score evaluado contra 4 targets forward
realizados distintos (Q5-Q1 spread):

| Modelo | Retorno % | Sharpe | Sortino | Max DD |
|---|---|---|---|---|
| ElasticNet | −1.3% | **+0.37** | **+0.75** | **+0.04** |
| AxiomaticScorer | −1.0% | +0.26 | +0.70 | +0.03 |
| LightGBM | −1.6% | +0.27 | +0.21 | +0.02 |
| Benchmark naive | +0.5% | +0.19 | −0.26 | +0.02 |

**Lectura:** el ElasticNet entrenado con target Sharpe gana fuertemente
en lentes risk-adjusted (Sharpe Q5-Q1 +0.37, Sortino +0.75) y reduce
drawdowns realizados, pero **pierde en retorno bruto** (Q5-Q1 = −1.3%).
Esto es **comportamiento esperado y consistente con el target elegido**:
el modelo aprende a identificar fondos de mejor calidad ajustada por
riesgo, no maximiza retorno absoluto. Para una AFP cuyo mandato es
sostener afiliados a largo plazo, el tradeoff es deseable.

## 4. Drivers del score (interpretabilidad)

Coeficientes ElasticNet promedio entre folds, sobre features estandarizadas
(positivos = empujan score arriba, negativos = empujan score abajo):

- **Positivos (premia el modelo):** `sharpe_12m_rank`, `hit_rate_12m_rank`,
  `persistencia_rank_12m`, `max_dd_12m_rank`. Coherente con calidad
  ajustada por riesgo + consistencia + estabilidad.
- **Negativos (penaliza el modelo):** `vol_12m`, `vol_12m_rank`,
  `fee_rank`, `pct_acum_rank`, `autocorr_diaria_rank`. Coherente con
  evitar volatilidad, fees altos, concentración excesiva e iliquidez.

Las magnitudes son moderadas — coherentes con la señal documentada
estadísticamente. Detalle por fold en `artifacts/drivers_elastic.csv`.

## 5. Limitaciones reconocidas

1. **Reporte de fee solo desde 2024-01-31.** El dataset reporta fees en
   ~12% del panel originalmente. Mitigado bajo el supuesto explícito de
   fee estructural (verificable empíricamente: std intra-fondo ≈ 0 en el
   subperíodo observado), llevando cobertura a 99.3%. Limitación
   inevitable de la fuente; ver supuestos en README.md.

2. **Survivorship bias parcial.** 87.7% de fondos vivos al final del
   dataset (mayo 2026). Los 34 "muertos" tienen mediana de cobertura de
   2 meses → sugiere snapshot del universo vigente, no historia con
   quiebras. El filtro de cobertura mínima ≥36 meses excluye a la mayoría.
   Plot en `artifacts/plots/survivorship.png`.

3. **Anonimización del universo.** Sin clasificación por estilo, región
   o asset class — no se pueden agregar features de "fondo vs benchmark de
   estilo". Mitigación parcial: features stylistic capturan propiedades
   correlacionadas con estilo.

4. **Liquidez por proxies indirectos.** No hay variable directa; se usan
   `autocorr_diaria`, `ratio_dias_cero`, `vol_intrames` derivados de la
   serie diaria.

5. **Sin información de flujos / AUM.** Imposibilita detectar diseconomies
   of scale (Berk-Green) y diferenciar fondos chicos vs grandes.

## 6. Extensiones con más tiempo o más datos

- **Datos**: AUM y flujos por fondo; clasificación por estilo/región/
  benchmark; extender reporte de fee a períodos pre-2024 para validar el
  supuesto estructural.
- **Modelado**: features macro y de régimen (term spread, dollar index,
  VIX) para condicionar el score por contexto de mercado; ensembles
  ElasticNet+LightGBM con stacking.
- **Validación**: análisis explícito por subperíodos (bull, COVID, alza
  tasas) para verificar estabilidad de la señal entre regímenes;
  backtest con costos realistas y rebalanceo mensual; portafolio
  construido con turnover constraint.
- **Proceso**: el output cuantitativo debe ser INPUT de un proceso
  multi-factor que incluya due diligence cualitativa (manager tenure,
  ownership stability, gobierno corporativo) — un score solo nunca
  reemplaza el juicio humano del comité de inversiones.

## 7. Uso de IA / LLMs en el proceso

**Herramientas:** Claude (Anthropic, Opus 4.7).

**Etapas donde se usó:**

1. **Discusión metodológica iterativa** — particularmente para definir
   la variable objetivo. La conversación inicial proponía retorno forward
   12m; tras discutir con el modelo qué define a un "fondo bueno" para
   una AFP (alto retorno + bajo riesgo + bajo fee + diversificado +
   líquido), se separaron conceptualmente "lo que define al fondo" (target)
   de "lo que se observa hoy del fondo" (features). Resultado: cambio del
   target a Sharpe forward 12m, con justificación documentada.

2. **Validación empírica de supuestos** — antes de imputar el fee con
   `bfill` se consultó al SQLite directamente para verificar que el fee
   es estructuralmente constante intra-fondo (mediana 1 valor único, std
   ≈ 0). El supuesto se aceptó solo después de evidencia, no por
   sugerencia del LLM.

3. **Code review** del esquema de walk-forward, embargo temporal e
   imputación, para verificar ausencia de leakage.

4. **Generación de docstrings, comentarios y narrativa de informe.**

**Cómo se validó lo que el modelo entregó:**

- Cada cambio de código se ejecutó contra el dataset real y se verificaron
  los outputs (cobertura, distribuciones, métricas) antes de aceptarlo.
- Los hallazgos numéricos (cobertura del fee, survivorship, IC del modelo)
  se reportan tal como salen del pipeline — no se suavizaron.
- Las decisiones metodológicas se documentaron explícitamente en
  `README.md` con su tradeoff, no se asumieron implícitamente.

**Ejemplo concreto de prompt útil:** durante la discusión de imputación
de fee, ante la pregunta "¿se puede imputar fee con el último valor del
fondo si es esencialmente constante?", el LLM cuestionó la premisa
solicitando datos antes de implementar. Las queries SQL resultantes
revelaron que el fee se reporta solo desde 2024 (no era simplemente "11%
de NaN aleatorio" como se había asumido inicialmente), lo que cambió la
estrategia de imputación de "mediana cross-seccional" a "ffill+bfill
intra-fondo bajo supuesto estructural" + flag binaria.

Todos los commits del repo están firmados con
`Co-Authored-By: Claude Opus 4.7`.
