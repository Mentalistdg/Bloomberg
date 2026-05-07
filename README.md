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

# 2. correr el pipeline completo (5 pasos secuenciales, ~10s total)
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

El archivo `usa_fondos_pp.sqlite` (~80 MB) **no se distribuye con el
repo** (ver `.gitignore`). Tres tablas:

- `historico` — fecha, fondo, NAV, evento_pct.
- `fees` — comisiones (89% NULL en el raw, escalonadas por fondo).
- `subyacentes` — concentración: número mínimo de holdings para alcanzar
  el 30% del AUM, y el porcentaje exacto logrado.

## Estructura del repo

```
.
├── src/                          módulos del pipeline
│   ├── data.py                   carga sqlite, retorno total, panel mensual
│   ├── features.py               feature engineering + target
│   ├── splits.py                 walk-forward expanding window con embargo
│   ├── model.py                  ElasticNet + LightGBM + benchmark naive
│   ├── metrics.py                IC, spread Q5-Q1, hit rate top-25
│   └── validation.py             bootstrap del IC, Diebold-Mariano
│
├── scripts/                      orquestación end-to-end
│   ├── 01_build_features.py
│   ├── 02_eda_report.py
│   ├── 03_build_features_full.py
│   ├── 04_train_and_evaluate.py
│   ├── 05_build_app_data.py
│   └── run_all.py
│
├── notebooks/
│   ├── informe.py                fuente jupytext (.py)
│   └── informe.ipynb             notebook ejecutado
│
├── artifacts/                    outputs reproducibles del pipeline
│   ├── panel_raw.parquet
│   ├── panel_features.parquet
│   ├── scores.parquet            score por fondo×mes para todos los folds
│   ├── metrics.json              métricas globales por modelo
│   ├── drivers_elastic.csv
│   ├── drivers_lgbm.csv
│   ├── fold_diagnostics.csv
│   └── plots/                    figuras del EDA y validación
│
├── app/
│   ├── backend/main.py           FastAPI sirviendo JSONs pre-computados
│   └── frontend/                 Vite + React + TS + Tailwind
│
├── pyproject.toml                deps gestionadas con uv
├── INFORME.md
├── slides_outline.md
└── README.md
```

## Diseño del pipeline

```
sqlite → 01 → panel_raw → 03 → panel_features → 04 → scores + metrics → 05 → JSONs
                  │                                                            │
                  └──→ 02 → plots/                                              └──→ FastAPI → React
```

Cada script lee outputs del anterior y produce artefactos en `artifacts/`.
Reproducible end-to-end con `python -m scripts.run_all`.

## Stack

- Python 3.14, gestor `uv`.
- pandas, numpy, scikit-learn, statsmodels, lightgbm, scipy.
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
