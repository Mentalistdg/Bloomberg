# Scoring de fondos mutuos — caso técnico AFP Habitat

Sistema cuantitativo de scoring de fondos mutuos. A partir del histórico
de precios, eventos de capital, comisiones y concentración del portafolio,
asigna a cada fondo un puntaje que informa la decisión de inversión —
distinguiendo fondos atractivos de poco atractivos.

**Énfasis del proyecto:** disciplina anti-leakage, validación estadística,
y honestidad sobre los hallazgos. El resultado más importante no es el
IC del modelo, sino la robustez del proceso.

## Entregables

| Deliverable | Ubicación |
|---|---|
| Notebook reproducible | [`notebooks/informe.ipynb`](notebooks/informe.ipynb) |
| Informe narrativo | [`INFORME.md`](INFORME.md) |
| Outline de slides (10 láminas) | [`slides_outline.md`](slides_outline.md) |
| Pipeline de código | `src/` + `scripts/` |
| Dashboard | `app/backend` (FastAPI) + `app/frontend` (React) |

## Reproducibilidad end-to-end

```powershell
# 1. crear venv y deps (uv)
uv sync

# 2. correr el pipeline completo (5 pasos secuenciales)
uv run python -m scripts.run_all

# o paso a paso:
uv run python -m scripts.01_build_features        # panel mensual base
uv run python -m scripts.02_eda_report            # plots de validación
uv run python -m scripts.03_build_features_full   # features + target
uv run python -m scripts.04_train_and_evaluate    # walk-forward + métricas
uv run python -m scripts.05_build_app_data        # JSONs para el dashboard

# 3. notebook (auto-renderiza con jupytext desde notebooks/informe.py)
uv run jupyter notebook notebooks/informe.ipynb

# 4. dashboard local
uv run uvicorn app.backend.main:app --reload --port 8000
npm --prefix app/frontend install
npm --prefix app/frontend run dev    # http://localhost:3000
```

## Dataset

El archivo `assets/usa_fondos_pp.sqlite` (~80 MB) **no se distribuye con el
repo** (ver `.gitignore`). Debe colocarse en la carpeta `assets/` antes
de ejecutar el pipeline. Tres tablas:

| Tabla | Contenido | Notas |
|---|---|---|
| `historico` | fecha, fondo, NAV (precio), evento_pct (distribuciones de capital) | Serie completa diaria por fondo. 277 fondos, 1988-11 → 2026-05. |
| `fees` | fecha, fondo, fee (% anual) | 89% NULL en crudo. **Reportado solo desde 2024-01-31** — en el panel mensual se imputa con `ffill+bfill` intra-fondo bajo el supuesto de fee estructural (verificado: std intra-fondo ≈ 0). Cobertura efectiva post-imputación: 99.3%. |
| `subyacentes` | fecha, fondo, n_instrumentos, pct_acum | Snapshots de concentración; n_instrumentos = min holdings para alcanzar 30% AUM, pct_acum es el % exacto en ese punto. Solo `ffill` por fondo (la concentración SÍ varía en el tiempo, no es estructural). |

## Organización del código: `src/` vs `scripts/`

El proyecto separa la lógica en dos capas:

**`src/` — Módulos reutilizables (el "cómo").**
Contiene funciones y clases que encapsulan cada paso del pipeline.
Estos archivos **no se ejecutan directamente** — se importan desde los
scripts. Si necesitas entender o modificar la lógica de un paso
específico (por ejemplo, cómo se calculan las features), este es el
lugar.

**`scripts/` — Puntos de entrada numerados (el "qué y cuándo").**
Cada script orquesta un paso del pipeline llamando a funciones de `src/`.
Se ejecutan en orden (`01_` → `02_` → ... → `05_`), y cada uno lee los
artefactos del paso anterior y produce los del siguiente. Si necesitas
reproducir el pipeline completo, basta con ejecutarlos en secuencia (o
usar `run_all.py`).

```
scripts/04_train_and_evaluate.py          ← "entrena modelos y evalúa"
    ↓ importa
src/model.py                              ← "cómo se define y entrena un ElasticNet"
src/splits.py                             ← "cómo se genera el walk-forward"
src/metrics.py                            ← "cómo se calcula el IC y spread"
src/validation.py                         ← "cómo se hace bootstrap y DM test"
```

Esta separación permite que un cambio en la lógica (ej. modificar la
fórmula de una feature en `src/features.py`) se propague automáticamente
a todos los scripts que la usen, sin duplicar código.

## Estructura del repo

```
.
├── assets/                           datos fuente (no versionados en git)
│   └── usa_fondos_pp.sqlite          base sqlite con historico, fees, subyacentes
│
├── src/                              módulos reutilizables (se importan, no se ejecutan)
│   ├── paths.py                      rutas centralizadas del proyecto
│   ├── data.py                       carga de sqlite, retorno total diario, panel mensual
│   ├── features.py                   ingeniería de features (31 cols) + targets forward (4 lentes)
│   ├── splits.py                     walk-forward expanding window con embargo = horizonte
│   ├── model.py                      ElasticNet, LightGBM, benchmark naive, AxiomaticScorer
│   ├── metrics.py                    IC, spread Q5-Q1, hit rate, multi-lens, persistencia rank
│   └── validation.py                 bootstrap CI del IC, test de Diebold-Mariano
│
├── scripts/                          orquestación: se ejecutan en orden 01 → 05
│   ├── 01_build_features.py          sqlite → panel mensual base (panel_raw.parquet)
│   ├── 02_eda_report.py              genera 5 gráficos diagnósticos en artifacts/plots/
│   ├── 03_build_features_full.py     panel_raw → 17 features + target (panel_features.parquet)
│   ├── 04_train_and_evaluate.py      walk-forward CV → scores, métricas, drivers, bootstrap
│   ├── 05_build_app_data.py          scores + métricas → JSONs para el dashboard
│   └── run_all.py                    ejecuta los 5 scripts en secuencia
│
├── notebooks/
│   ├── informe.py                    fuente jupytext (.py)
│   └── informe.ipynb                 notebook ejecutado
│
├── artifacts/                        outputs reproducibles del pipeline
│   ├── panel_raw.parquet             panel mensual crudo (output de script 01)
│   ├── panel_features.parquet        panel con 17 features + target (output de script 03)
│   ├── scores.parquet                score por fondo×mes, todos los folds (output de script 04)
│   ├── metrics.json                  métricas globales: IC, bootstrap, DM (output de script 04)
│   ├── drivers_elastic.csv           coeficientes ElasticNet por fold
│   ├── drivers_lgbm.csv              importancia LightGBM por fold
│   ├── fold_diagnostics.csv          metadata de cada fold del walk-forward
│   └── plots/                        figuras del EDA y diagnósticos de señal
│
├── app/
│   ├── backend/main.py               FastAPI sirviendo JSONs pre-computados
│   └── frontend/                     Vite + React + TS + Tailwind
│
├── pyproject.toml                    deps gestionadas con uv
├── INFORME.md
├── slides_outline.md
└── README.md
```

## Diseño del pipeline

```
assets/usa_fondos_pp.sqlite
        │
        ▼
  01_build_features ──→ artifacts/panel_raw.parquet
        │                       │
        │                       ▼
        │               02_eda_report ──→ artifacts/plots/ (5 gráficos diagnósticos)
        │
        ▼
  03_build_features_full ──→ artifacts/panel_features.parquet (17 features + target)
        │
        ▼
  04_train_and_evaluate ──→ artifacts/scores.parquet
        │                    artifacts/metrics.json
        │                    artifacts/drivers_elastic.csv
        │                    artifacts/drivers_lgbm.csv
        │                    artifacts/plots/signal_*.png
        ▼
  05_build_app_data ──→ app/backend/data/*.json
        │
        ▼
  FastAPI ──→ React dashboard
```

Cada script lee los artefactos del paso anterior y produce los del
siguiente. Pipeline reproducible end-to-end con
`python -m scripts.run_all`.

## Definición del problema (entregable #1)

**Variable a predecir:** **percentil cross-seccional del Sharpe forward 12m**
del fondo (`target_sharpe_rank_12m`). Mide simultáneamente retorno y riesgo
en una métrica única, alineada con la decisión real de selección de fondos
en una AFP (calidad ajustada por riesgo, no retorno bruto). El retorno
forward 12m simple se mantiene como métrica reportada en paralelo, no como
target de entrenamiento.

**Enfoque:** **explicativo con validación predictiva**. Modelo lineal
regularizado (ElasticNet) como primario — coeficientes interpretables y
defendibles ante un comité — con LightGBM como sanity check de no-linealidad
y `AxiomaticScorer` (fórmula con pesos teóricos, sin entrenar) como benchmark
estructural. Tradeoff explícito: priorizamos defensibilidad sobre máxima
performance OOS.

## Modelos comparados (4 scoreadores)

| Modelo | Rol | Mecánica |
|---|---|---|
| **ElasticNet** (principal) | Scoring interpretable | Regresión L1+L2 con CV interna de α y l1_ratio (5-fold). Coeficientes son "drivers" para el comité. |
| **LightGBM** (sanity check) | Detectar no-linealidades | Gradient boosting (depth=4, num_leaves=15). Si supera al lineal, hay interacciones no capturadas. |
| **AxiomaticScorer** (benchmark teórico) | "Sin entrenar" | Combinación lineal de ranks con pesos teóricos: +0.30·Sharpe_rank − 0.20·fee_rank + 0.20·max_dd_rank − 0.15·pct_acum_rank − 0.15·autocorr_diaria_rank. |
| **Benchmark naive** | Línea base mínima | `score = ret_12m_rank − fee_rank` (momentum + fee). |

## Validación

- **Walk-forward expanding window**: 9 folds para horizonte 12m, mínimo
  60 meses de training, 12 meses de validación, **embargo igual al horizonte
  del target** (12m para target a 12m) para evitar leakage del forward.
- **Bootstrap CI**: 5,000 iteraciones sobre la serie mensual de IC.
- **Diebold-Mariano**: test de superioridad predictiva con corrección
  Newey-West por autocorrelación. ElasticNet vs benchmark naive y vs
  AxiomaticScorer.
- **Multi-lente OOS**: el mismo score se evalúa sobre **4 targets forward
  realizados** distintos (retorno, Sharpe, Sortino, max drawdown) para
  verificar que el resultado es robusto a cómo se mida la "calidad
  realizada", no solo a la métrica con la que se entrenó.
- **Persistencia de rank**: correlación entre el rank que el modelo asigna
  al fondo en T y en T+12m (turnover implícito).

## Stack

- Python 3.14, gestor `uv`.
- pandas, numpy, scikit-learn, lightgbm, scipy.
- FastAPI + uvicorn (backend).
- React 18 + TypeScript + Vite + Tailwind + Recharts (frontend).

## Resultados resumen (target Sharpe forward 12m, full features)

**IC y CI bootstrap (horizonte 12m):**

| Modelo | IC mensual | IR | Hit (% meses + ) | CI95 (bootstrap) |
|---|---|---|---|---|
| **ElasticNet (primary)** | **+0.190** | **+0.85** | 79.6% | **[+0.147, +0.232]** |
| LightGBM | +0.150 | +0.59 | 75.9% | [+0.100, +0.199] |
| AxiomaticScorer | +0.153 | +0.51 | 71.3% | [+0.093, +0.208] |
| Benchmark naive | +0.117 | +0.50 | 71.3% | [+0.072, +0.160] |

**Diebold-Mariano:** ElasticNet vs benchmark naive `p = 0.093` (significativo
al 10%, marginal al 5%). ElasticNet vs AxiomaticScorer `p = 0.40` (no
significativo — el axiomático ya captura buena parte de la señal con teoría).

**Multi-lente — Q5-Q1 spread sobre target realizado (12m):**

| Modelo | Retorno | Sharpe | Sortino | Max DD |
|---|---|---|---|---|
| ElasticNet | −1.3% | **+0.37** | **+0.75** | **+0.04** |
| AxiomaticScorer | −1.0% | +0.26 | +0.70 | +0.03 |
| LightGBM | −1.6% | +0.27 | +0.21 | +0.02 |
| Benchmark naive | +0.5% | +0.19 | −0.26 | +0.02 |

**Lectura:** el ElasticNet entrenado con target Sharpe forward gana
fuertemente en métricas risk-adjusted (+0.37 en Sharpe Q5-Q1, +0.75 en
Sortino) y reduce drawdowns realizados (+0.04 en max DD), pero **pierde
en retorno bruto** (−1.3% Q5-Q1) — comportamiento esperado y consistente
con el target elegido. Una AFP que selecciona fondos para sostener a
afiliados a largo plazo prefiere fondos de mejor calidad ajustada por
riesgo aunque sacrifique algo de retorno bruto.

## Supuestos, limitaciones y decisiones metodológicas

Esta sección documenta explícitamente las decisiones no triviales del
pipeline. La idea es que el evaluador pueda contrastar cada decisión
contra alternativas y el porqué de la elección.

### Supuestos declarados

1. **Fee estructural en el tiempo (supuesto fuerte).** El dataset solo
   reporta fees a partir de 2024-01-31, dejando ~88% de NaN en el panel
   pre-2024. Asumimos que el fee de un fondo es estable en el tiempo
   (verificable empíricamente: en el subperíodo 2024-2026 con múltiples
   observaciones, la mediana de fondos tiene **un único valor de fee**
   y la std intra-fondo es ≈ 0). Bajo este supuesto, imputamos hacia
   atrás dentro de cada fondo (`ffill+bfill`) y para 29 fondos cuya
   historia de precios terminó antes de 2024 usamos el fee promedio
   reportado del fondo. Cobertura efectiva: 99.3% (vs 12% con solo
   `ffill`). La bandera binaria `fee_observado` (1 = valor original
   en ese mes, 0 = imputado) permite al modelo distinguir y captura
   indirectamente el efecto de período de reporte.

2. **Estabilidad temporal de las relaciones aprendidas.** Asumimos que
   las asociaciones aprendidas en el pasado (digamos, pre-2018) se
   mantienen en el período de evaluación (2019+). Inevitable en
   cualquier modelo predictivo. Verificable parcialmente con el análisis
   por subperíodos de los plots OOS — si el IC se mantiene entre regímenes
   muy distintos (bull 2017, COVID 2020, alza tasas 2022), el supuesto
   es razonable.

3. **Mediana cross-seccional como fallback de imputación.** Para los 2
   fondos sin ningún reporte de fee y para los pocos casos de
   concentración faltante, se imputa con la mediana del universo en el
   mismo mes y se marca con flag. Asume que un fondo sin información
   es "promedio" — supuesto neutral.

### Limitaciones

1. **Survivorship bias parcial.** 87.7% de los fondos llegan vivos al
   final del dataset (mayo 2026). Los 34 "muertos" tienen mediana de
   cobertura de 2 meses — sugiere que el dataset es snapshot del universo
   vigente al momento de captura, no historia con quiebras reales. El
   filtro de cobertura mínima (≥36 meses) excluye a la mayoría de los
   muertos cortos. Detalle en `artifacts/survivorship_stats.json` y plot
   en `artifacts/plots/survivorship.png`. Implicaciones: el "retorno
   promedio del universo" está sesgado al alza; la AFP también selecciona
   entre fondos vivos, así que el sesgo no invalida el modelo, solo
   afecta cómo se interpretan métricas absolutas.

2. **Universo agnóstico al benchmark.** El dataset no incluye
   clasificación por categoría/benchmark del fondo (large cap blend,
   EM equity, etc.), por lo que el modelo no puede compararse contra
   "fondos del mismo estilo". El score es relativo al universo completo,
   no a peers comparables. Mitigación parcial: las features capturan
   propiedades stylistic (vol, skewness, drawdown) que correlacionan
   con estilo.

3. **Horizonte único 12m.** Aunque el pipeline soporta horizontes 3m
   y 6m (ver `horizon_comparison.csv`), el target principal es 12m. Una
   AFP con horizonte multi-año podría preferir señales más persistentes
   (24m, 36m), pero el dataset acota qué se puede medir.

4. **Sin información de flujos / AUM.** No tenemos AUM por fondo ni
   suscripciones netas. Esto impediría detectar "decreto de retornos a
   escala" (Berk-Green) y diferenciar fondos chicos vs grandes.

5. **Liquidez por proxies.** No hay variable directa de liquidez; se
   usan tres proxies derivados de la serie diaria (autocorr_diaria,
   ratio_dias_cero, vol_intrames). Funcionan razonablemente pero son
   indirectos.

### Decisiones metodológicas no triviales

1. **Target Sharpe forward 12m vs retorno forward 12m.** Cambiamos del
   target retorno al target Sharpe porque captura simultáneamente las
   dos dimensiones de "fondo bueno" para una AFP (alto retorno + bajo
   riesgo) en una sola métrica, sin debate sobre pesos. Tradeoff: el
   modelo entrenado con Sharpe optimiza calidad ajustada por riesgo,
   sacrifica algo de retorno bruto.

2. **ElasticNet primario, no LightGBM.** Decisión de "explicativo con
   validación predictiva". ElasticNet da coeficientes interpretables
   defendibles al comité; LightGBM se mantiene como sanity check de
   no-linealidad. En la evaluación, ElasticNet supera a LightGBM en IC
   y métricas multi-lente, validando la elección.

3. **AxiomaticScorer como benchmark teórico.** Permite contrastar
   "lo que se aprende empíricamente" con "lo que se sabe estructuralmente"
   sobre selección de fondos. Si el axiomático bate al ML hay sospecha
   de overfitting; si el ML bate al axiomático, está capturando matices
   no triviales. En la evaluación quedan parejos en IC pero ElasticNet
   gana en lentes risk-adjusted.

4. **Imputación de fee con bfill bajo supuesto estructural.** Discutido
   en supuesto #1. Alternativa rechazada: dejar 88% NaN — perdería casi
   toda la utilidad de la variable más causalmente directa para selección
   de fondos.

5. **Filtro de cobertura mínima ≥36 meses.** Fondos con menos de 3 años
   de historia no entran al modelo. Justificación: con warmup de 12m +
   horizonte forward de 12m, fondos de <36m generan <12 observaciones
   entrenables — evidencia insuficiente.

### Alternativas descartadas (entregable #3)

| Alternativa | Por qué se descartó |
|---|---|
| Clasificación binaria (top-quintil vs resto) | Pierde información ordinal del ranking; menos eficiente que regresión sobre el rank cross-seccional. |
| RandomForest / XGBoost / RNN | Riesgo alto de overfitting con ~50K obs y ~30 features; menor interpretabilidad. LightGBM cubre la sanity check no-lineal. |
| Predecir retorno a 1 mes (`shift(-1)`) | A 1m el retorno es ~99% beta de mercado, casi cero alfa de fondo. Horizonte equivocado para selección de fondos en una AFP. |
| Frecuencia diaria del modelado | Las observaciones diarias están muy autocorrelacionadas; más filas no implica más información. Métricas de fondos (Sharpe, drawdown) son intrínsecamente multi-período. |
| Composite axiomático multi-target en entrenamiento | Forzaría definir pesos del composite a priori; agrega subjetividad sin claro beneficio. Lo manejamos como AxiomaticScorer (sin entrenar) en paralelo. |
| Indicadores AT clásicos (RSI, MACD, Bollinger) | Diseñados para timing de instrumentos individuales en horizontes cortos; no calzan con selección de fondos. Sí mantenemos features tipo "estilísticas de la distribución de retornos del fondo": skewness, hit-rate, persistencia rank. |

## Uso de IA / LLMs (entregable #5)

Documentado en [`INFORME.md`](INFORME.md). Resumen: Claude (Anthropic)
se usó para discutir metodología iterativamente, hacer code review de
las decisiones de imputación y target, y generar drafts de docstrings.
Cada decisión fue validada con consultas reales al dataset (no se
implementó nada "porque el LLM lo dijo"). Commits firmados con
`Co-Authored-By: Claude Opus 4.7`.
