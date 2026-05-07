# Informe — Scoring de fondos mutuos

> Caso técnico Analista de Inversiones, Renta Variable Indirecta. Documento
> resumen. El análisis técnico completo, con código y outputs, está en
> `notebooks/informe.ipynb`. La defensa visual está en `slides.pdf`.

## Definición del problema

**Variable a predecir:** percentil cross-seccional del retorno total a 12
meses forward, dentro del universo activo en cada fecha. Esta elección:

- Hace al target robusto a regímenes de mercado (es ranking, no nivel).
- Captura la unidad de decisión del comité (ranking entre fondos en una
  fecha dada).
- Usa retorno total (incluye eventos de capital), por lo que mide
  rentabilidad efectiva del inversionista.

**Enfoque elegido: explicativo > predictivo.** El modelo principal es un
ElasticNet (lineal regularizado, coeficientes interpretables como drivers
del score). Como sanity check no-lineal se entrena en paralelo un LightGBM.
Razones del enfoque:

1. La audiencia del modelo (comité de inversiones) entiende factores; la
   interpretabilidad del lineal pesa más que un eventual ajuste marginal
   de un black-box.
2. El tamaño efectivo del problema (~250 fondos × 120 meses) limita la
   capacidad útil de modelos flexibles.
3. La literatura (Carhart 1997, Berk-Green 2004, Fama-French 2010) reporta
   que la persistencia de retornos netos de fondos mutuos es marginal — la
   posición prudente es modelar con humildad estadística.

## Construcción de features

Separadas en dos grupos por densidad:

- **CORE** (no-NaN obligatorio): `ret_1m`, `ret_3m`, `ret_6m`, `ret_12m`,
  `vol_12m`, `max_dd_12m`, `sharpe_12m`. Densas tras un warmup de 12 meses.
- **EXTENDED** (sparse, imputadas con flag): `fee`, `log_n_instrumentos` —
  más una flag binaria `*_disponible` que captura el efecto de información
  faltante.

Adicionalmente se agregan **rank features cross-seccionales** sobre
`ret_3m`, `ret_12m`, `vol_12m`, `sharpe_12m`, `fee`, `log_n_instrumentos`
para robustez a regímenes y reducción de escala-sensibilidad.

**Anti-leakage:** todas las features se computan al cierre de cada mes
usando exclusivamente información histórica (rolling windows hacia atrás).
El target — único elemento del sistema que mira hacia adelante — se
construye con `prod(1+ret_{t+1..t+12})-1`. Walk-forward CV con embargo de
12 meses entre fin de train e inicio de val asegura que no se solapen las
ventanas de target entre folds.

## Validación e interpretación de resultados

Walk-forward expanding window con 9 folds, cubriendo 2016-2024 fuera de
muestra. Métricas en cada fold y agregadas:

| Modelo | IC mensual (mean) | IC IR | Q5-Q1 spread | CI95 (bootstrap) |
|---|---|---|---|---|
| ElasticNet | +0.006 | +0.02 | -0.26% | [-0.063, +0.076] |
| LightGBM | -0.045 | -0.20 | -1.55% | [-0.088, -0.002] |
| Benchmark naive | +0.031 | +0.09 | -0.26% | [-0.039, +0.096] |

El benchmark naive es una combinación lineal sin entrenamiento de
`ret_12m_rank − fee_rank`. **El test de Diebold-Mariano entre ElasticNet
y benchmark arroja p > 0.65: no hay diferencia estadística entre ambos
signals.** El intervalo bootstrap del IC de ElasticNet incluye al cero —
no se rechaza la hipótesis nula de ausencia de poder predictivo a niveles
convencionales.

Esta es una **conclusión defendible y consistente con la literatura**:
los retornos post-fee de fondos mutuos exhiben persistencia marginal. El
modelo es honesto en su humildad estadística.

## Drivers del score

Coeficientes ElasticNet promedio entre folds, sobre features estandarizadas:

- Positivos: `sharpe_12m`, `ret_12m_rank`, `ret_3m_rank`.
- Negativos: `max_dd_12m`, `vol_12m_rank`, `log_n_instrumentos_rank`.

Los signos coinciden con intuición financiera: alto Sharpe y momentum
positivo, drawdowns y volatilidad penalizados, fondos más concentrados
levemente preferidos. La magnitud absoluta es pequeña — coherente con la
señal débil documentada estadísticamente.

## Limitaciones reconocidas

1. **Cobertura de features estructurales.** `subyacentes` (concentración)
   tiene snapshot reciente para 47% del universo; `fees` arranca en 2015.
2. **Sesgo de supervivencia.** Fondos discontinuados antes de 2026 no
   aparecen — cualquier promedio histórico está sesgado al alza.
3. **Anonimización del universo.** Sin clasificación por estilo, región,
   asset class no se pueden incorporar features de cobertura.
4. **Tamaño efectivo del problema.** ~120 meses × 250 fondos limita el
   alcance de modelos flexibles.
5. **Horizonte único.** Solo se exploró target a 12 meses; horizontes más
   cortos (3m, 6m) podrían exhibir distinto poder predictivo.

## Extensiones con más tiempo

- Incorporar features macro y de mercado (régimen de volatilidad, term
  spread, dollar index) para condicionar el score por régimen.
- Datos de holdings agregados (style box, exposición sectorial, net flows)
  como reemplazo del proxy actual de concentración.
- Backtest con costos realistas, rebalanceo mensual, computar Sharpe de
  carteras top-quintil neto de turnover.
- Combinación del score cuantitativo con un proceso de due diligence
  cualitativa (manager tenure, AUM stability, consistencia operacional)
  — el output del modelo debe ser un input dentro de un proceso multi-factor.

## Uso de IA en el proceso

Se utilizó Claude (Anthropic) en tres etapas:

1. **Inspección y validación del dataset**, particularmente para
   reconciliar la descripción del campo de concentración con su firma
   matemática. Las hipótesis competidoras se testearon con queries SQL
   y datos reales de mercado vía Bloomberg.
2. **Code review del esquema de validación walk-forward** para verificar
   ausencia de leakage temporal en el target a 12 meses.
3. **Estructuración del informe** y de la defensa de elecciones
   metodológicas.

Todas las salidas del modelo se ejecutaron contra los datos reales y se
contrastaron contra literatura citada (Carhart 1997, Berk-Green 2004,
López de Prado para anti-leakage, Diebold-Mariano 1995 para el test de
superioridad predictiva).
