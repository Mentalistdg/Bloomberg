# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Purpose

Prueba técnica para **Analista de Renta Variable Internacional — AFP Habitat** (autor: estudiante FEN UChile). Sistema cuantitativo de scoring de fondos mutuos USA: a partir de históricos de precios, eventos de capital, comisiones y concentración, asigna un puntaje que distingue fondos atractivos de poco atractivos.

Universo de referencia: **renta variable internacional** (MSCI ACWI / World / EM, S&P 500), no renta variable chilena.

## Commands

```bash
# Dependencias (Python 3.14, gestionado con uv — nunca usar pip)
uv sync

# Pipeline completo (5 pasos secuenciales, requiere assets/usa_fondos_pp.sqlite)
uv run python -m scripts.run_all

# Pasos individuales
uv run python -m scripts.01_build_features        # panel mensual base → artifacts/panel_raw.parquet
uv run python -m scripts.02_eda_report            # plots de validación → artifacts/plots/
uv run python -m scripts.03_build_features_full   # features + target → artifacts/panel_features.parquet
uv run python -m scripts.04_train_and_evaluate    # walk-forward + métricas → artifacts/scores.parquet, metrics.json
uv run python -m scripts.05_build_app_data        # JSONs para dashboard → app/backend/data/

# Dashboard
uv run uvicorn app.backend.main:app --reload --port 8000
npm --prefix app/frontend install && npm --prefix app/frontend run dev   # http://localhost:3000

# Notebook
uv run jupyter notebook notebooks/informe.ipynb
```

## Architecture

### Data flow

```
assets/usa_fondos_pp.sqlite
    → 01_build_features → artifacts/panel_raw.parquet
                              → 02_eda_report → artifacts/plots/
    → 03_build_features_full → artifacts/panel_features.parquet
    → 04_train_and_evaluate  → artifacts/scores.parquet + metrics.json + drivers CSVs
    → 05_build_app_data      → app/backend/data/*.json → FastAPI → React
```

Each script reads artifacts from the previous step and writes to `artifacts/`. The pipeline is strictly sequential and stateless — reproducible end-to-end from `run_all.py`.

### `src/` modules

- **`data.py`** — SQLite loading, daily total return (NAV + capital events), monthly panel construction, fee/concentration attachment via forward-fill
- **`features.py`** — Feature engineering. Two groups: CORE (momentum 1/3/6/12m, vol, max drawdown, Sharpe — dense after 12m warmup) and EXTENDED (fee, log_n_instrumentos — sparse, imputed with cross-sectional median + `*_disponible` binary flag). Cross-sectional ranks on 6 features. Target: percentile of 12m forward compounded return. Exports `FEATURE_COLS`, `get_modeling_frame()`
- **`splits.py`** — Walk-forward expanding window with 12-month embargo between train end and val start (anti-leakage for 12m forward target)
- **`model.py`** — `ElasticNetModel` (primary, interpretable), `LightGBMModel` (non-linear sanity check), `benchmark_naive_score()` (ret_12m_rank − fee_rank, no training)
- **`metrics.py`** — IC (Spearman per date), quintile spread Q5-Q1, hit rate top-quartile
- **`validation.py`** — Bootstrap CI of mean IC, Diebold-Mariano test
- **`paths.py`** — Centralized paths: `PROJECT_ROOT`, `DB_PATH`, `ARTIFACTS_DIR`, `PLOTS_DIR`, `APP_DATA_DIR`

### Dashboard

- **Backend** (`app/backend/main.py`): FastAPI serving pre-computed JSONs from `app/backend/data/`. No online inference. Endpoints: `/api/meta`, `/api/funds`, `/api/funds/{fondo}`, `/api/drivers`, `/api/backtest`, `/api/health`
- **Frontend** (`app/frontend/`): React 18 + TypeScript + Vite + Tailwind + Recharts. Vite proxies `/api` to `localhost:8000`. Pages: Overview, Detail, Drivers, Backtest

### Dataset

`assets/usa_fondos_pp.sqlite` (~80 MB, **not in repo** — gitignored). Three tables:
- `historico` — fecha, securities (→ fondo), precio (NAV), evento_pct
- `fees` — fecha, fondo, fee (89% NULL in raw)
- `subyacentes` — fecha, nemo_fondo (→ fondo), pct_acum, n_instrumentos

## Conventions

- Code should be **clear and didactic** (prueba técnica context), not over-abstracted.
- Anti-leakage discipline: all features use only backward-looking information. Target is the only forward-looking element, protected by 12-month embargo in walk-forward splits.