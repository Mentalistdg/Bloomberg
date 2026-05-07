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
| `historico` | fecha, fondo, NAV (precio), evento_pct (distribuciones de capital) | Serie completa diaria por fondo |
| `fees` | fecha, fondo, fee (% anual) | 89% NULL en crudo; se propaga por fondo con forward-fill |
| `subyacentes` | fecha, fondo, n_instrumentos, pct_acum | Snapshots de concentración; n_instrumentos = min holdings para alcanzar 30% AUM |

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
│   ├── features.py                   ingeniería de features (17 cols) + target forward
│   ├── splits.py                     walk-forward expanding window con embargo de 12m
│   ├── model.py                      definición de ElasticNet, LightGBM y benchmark naive
│   ├── metrics.py                    IC de Spearman, spread Q5-Q1, hit rate top-25%
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

## Modelos

| Modelo | Rol | Justificación |
|---|---|---|
| **ElasticNet** (principal) | Scoring interpretable | Coeficientes legibles como "drivers" para comité de inversión |
| **LightGBM** (sanity check) | Detectar no-linealidades | Si supera al lineal, hay interacciones no capturadas |
| **Benchmark naive** | Línea base sin ML | `score = ret_12m_rank - fee_rank` (momentum + fee) |

## Validación

- **Walk-forward expanding window**: 9 folds, mínimo 60 meses de training,
  12 meses de validación, **embargo de 12 meses** entre train y val para
  evitar leakage del target forward a 12m.
- **Bootstrap CI**: 5,000 iteraciones sobre la serie mensual de IC.
- **Diebold-Mariano**: test de superioridad predictiva ElasticNet vs benchmark,
  con corrección Newey-West por autocorrelación.

## Stack

- Python 3.14, gestor `uv`.
- pandas, numpy, scikit-learn, lightgbm, scipy.
- FastAPI + uvicorn (backend).
- React 18 + TypeScript + Vite + Tailwind + Recharts (frontend).

## Resultados resumen

| Modelo | IC mensual | CI95 (bootstrap) | DM vs benchmark |
|---|---|---|---|
| ElasticNet (primary) | +0.006 | [-0.063, +0.076] | p = 0.66 |
| LightGBM | -0.045 | [-0.088, -0.002] | — |
| Benchmark naive (fee + momentum) | +0.031 | [-0.039, +0.096] | — |

**Lectura:** la señal predictiva es débil. El intervalo bootstrap del
IC de ElasticNet incluye al cero — no se rechaza la hipótesis nula al
95%. Resultado consistente con literatura sobre persistencia post-fee
de fondos mutuos (Carhart 1997, Berk-Green 2004, Fama-French 2010).
