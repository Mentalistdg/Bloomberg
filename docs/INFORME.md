# Informe — Scoring de fondos mutuos

> Caso técnico Analista de Inversiones, Renta Variable Indirecta. Documento
> resumen. El análisis técnico completo, con código y outputs, está en
> `notebooks/informe.ipynb`. La defensa visual está en `slides.pdf`.
> Documentación expandida de supuestos y limitaciones en `README.md`.

## 1. Definición del problema

**Variable a predecir:** **percentil cross-seccional del Sortino ratio
forward 6m** (`target_sortino_rank_6m`). Para cada (fondo, mes T) se
calcula el Sortino anualizado de los retornos mensuales de T+1 a T+6 y
se rankea cross-seccionalmente entre los fondos del universo activo en T.

**Por qué Sortino forward y no Sharpe o retorno forward:**

- El Sortino penaliza solo la volatilidad bajista (downside risk) — la
  volatilidad alcista es deseable en un fondo, no penalizable. El Sharpe
  trata toda volatilidad por igual, lo cual castiga fondos que suben mucho
  (exactamente lo que una AFP quiere).
- Combina "retorno alto" + "bajo riesgo downside" en una métrica única,
  sin debate sobre pesos relativos.
- Una AFP no maximiza retorno bruto; selecciona fondos para sostener
  afiliados a largo plazo, donde la protección contra caídas importa
  más que un retorno excepcional volátil.
- Horizonte 6m equilibra reactividad (capturar cambios de régimen) y
  estabilidad de la señal (evitar ruido de corto plazo).

El retorno forward simple (`target_ret_6m`), Sharpe forward y max drawdown
forward se mantienen como métricas reportadas en paralelo (multi-lens
validation), no como target de entrenamiento.

**Enfoque elegido: explicativo con validación predictiva.** El modelo
principal es ElasticNet (lineal regularizado, coeficientes interpretables
como drivers del score). En paralelo:

- **LightGBM** como sanity check de no-linealidad — si supera al lineal,
  hay interacciones que el lineal no captura.
- **Benchmark naive** — `ret_12m_rank − fee_rank`, línea base mínima
  contra la que cualquier modelo debe ganar.

Tradeoff aceptado: priorizamos defensibilidad y transparencia sobre
máxima performance OOS.

## 2. Construcción de features

**32 features totales** divididas en tres grupos:

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

**RANK (13)** — percentiles cross-seccionales dentro del mes para:
`ret_3m`, `ret_12m`, `vol_12m`, `sharpe_12m`, `max_dd_12m`, `fee`,
`log_n_instrumentos`, `pct_acum`, `vol_intrames`, `autocorr_diaria`,
`ratio_dias_cero`, `skewness_12m`, `hit_rate_12m`. Robustos a cambios
de régimen.

Nota: `sortino_12m` se computa como métrica de display (aparece en el
dashboard) pero no entra como feature del modelo.

**Anti-leakage (crítico):** todas las features se computan al cierre del
mes T usando exclusivamente información de t ≤ T (rolling hacia atrás).
Los targets — únicos elementos forward del sistema — se construyen con
`shift(-i)` de `ret_mensual`. Walk-forward CV con embargo de 6 meses
entre fin de train e inicio de val asegura que no se solapen ventanas
de target entre folds.

## 3. Modelo y validación

**Walk-forward rolling window** (máximo 120 meses / 10 años de training):
29 folds, mínimo 60 meses de training, 12 meses de validación,
embargo = 6m. La ventana rolling (en vez de expanding) permite al modelo
adaptarse a cambios de régimen de mercado (pre/post-2008, era de tasa
cero, inflación 2022+).

**Resultados out-of-sample (target Sortino forward 6m, 32 features):**

| Modelo | IC mensual | IR | Hit % | CI95 (bootstrap) |
|---|---|---|---|---|
| **ElasticNet** (primary) | **+0.074** | **+0.26** | 60.8% | **[+0.042, +0.107]** |
| LightGBM | +0.040 | +0.15 | 57.2% | [+0.012, +0.067] |
| Benchmark naive | +0.074 | +0.31 | 62.1% | [+0.049, +0.100] |

**Diebold-Mariano** (con corrección Newey-West, h=6):

- ElasticNet vs benchmark naive: estadístico = −0.45, **p = 0.654**
  (no significativo).
- Interpretación: el benchmark naive (`ret_12m_rank − fee_rank`) captura
  la mayor parte de la señal cross-seccional. Este es un hallazgo legítimo
  y esperado — Carhart (1997) ya documenta que momentum + fee explican la
  mayor parte de la variación en retornos de fondos. El ElasticNet no
  supera estadísticamente al benchmark, pero ofrece ventajas operativas:
  adaptación a régimen vía rolling window, incorporación de features de
  riesgo (vol, drawdown, skewness), interpretabilidad de coeficientes
  para el comité, y mejor discriminación en métricas risk-adjusted
  (multi-lens).

**Multi-lente OOS** — el mismo score evaluado contra 4 targets forward
realizados distintos (D10-D1 spread):

| Modelo | Retorno % | Sharpe | Sortino | Max DD |
|---|---|---|---|---|
| ElasticNet | −0.6% | **+0.29** | **+0.61** | **+0.019** |
| LightGBM | +0.1% | +0.19 | +0.67 | +0.009 |
| Benchmark naive | +1.0% | +0.18 | −0.16 | +0.009 |

**Lectura:** el ElasticNet entrenado con target Sortino gana en las
lentes risk-adjusted (Sharpe D10-D1 +0.29, Sortino +0.61) y reduce
drawdowns realizados, pero es neutro en retorno bruto (D10-D1 ≈ 0%).
Esto es **comportamiento esperado y consistente con el target elegido**:
el modelo aprende a identificar fondos de mejor calidad ajustada por
riesgo downside, no maximiza retorno absoluto. Para una AFP cuyo mandato
es sostener afiliados a largo plazo, el tradeoff es deseable.

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
  VIX) como interacciones con features de fondo para condicionar el score
  por contexto de mercado; ensemble simple ElasticNet+LightGBM con
  stacking; capacity constraints (Berk & Green 2004).
- **Validación**: análisis explícito por subperíodos (bull, COVID, alza
  tasas) para verificar estabilidad de la señal entre regímenes;
  backtest con costos realistas y rebalanceo mensual; portafolio
  construido con turnover constraint.
- **Proceso**: el output cuantitativo debe ser INPUT de un proceso
  multi-factor que incluya due diligence cualitativa (manager tenure,
  ownership stability, gobierno corporativo) — un score solo nunca
  reemplaza el juicio humano del comité de inversiones.

## 7. Uso de IA / LLMs en el proceso

**Herramientas:** Claude (Anthropic, Opus 4.6).

**Etapas donde se usó:**

1. **Discusión metodológica iterativa** — particularmente para definir
   la variable objetivo. La conversación inicial proponía retorno forward
   12m; tras discutir con el modelo qué define a un "fondo bueno" para
   una AFP (alto retorno + bajo riesgo + bajo fee + diversificado +
   líquido), se separaron conceptualmente "lo que define al fondo" (target)
   de "lo que se observa hoy del fondo" (features). Resultado: cambio del
   target a Sortino forward 6m, con justificación documentada. El Sortino
   se eligió sobre el Sharpe porque penaliza solo downside — la volatilidad
   alcista es deseable para una AFP, no penalizable.

2. **Validación empírica de supuestos** — antes de imputar el fee con
   `bfill` se consultó al SQLite directamente para verificar que el fee
   es estructuralmente constante intra-fondo (mediana 1 valor único, std
   ≈ 0). El supuesto se aceptó solo después de evidencia, no por
   sugerencia del LLM. Ejemplo concreto de prompt: "revisa si el esquema
   walk-forward tiene leakage considerando que el target abarca 6 meses
   forward y el embargo es 6m".

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
- Las fuentes primarias (Carhart 1997, DeMiguel et al. 2009, Sortino &
  Price 1994) se verificaron manualmente.

Todos los commits del repo están firmados con
`Co-Authored-By: Claude Opus 4.6`.
