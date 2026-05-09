# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Purpose

Prueba técnica para **Analista de Renta Variable Internacional — AFP Habitat** (autor: estudiante FEN UChile). Sistema cuantitativo de scoring de fondos mutuos USA: a partir de históricos de precios, eventos de capital, comisiones y concentración, asigna un puntaje que distingue fondos atractivos de poco atractivos.

Universo de referencia: **renta variable internacional** (MSCI ACWI / World / EM, S&P 500), no renta variable chilena.

## Commands

```bash
# Dependencias (Python >=3.14, gestionado con uv — nunca usar pip)
uv sync

# Pipeline completo (5 pasos secuenciales, requiere assets/usa_fondos_pp.sqlite)
uv run python -m scripts.run_all

# Pasos individuales
uv run python -m scripts.01_build_features        # sqlite → artifacts/panel_raw.parquet
uv run python -m scripts.02_eda_report            # panel_raw → artifacts/plots/
uv run python -m scripts.03_build_features_full   # panel_raw → artifacts/panel_features.parquet
uv run python -m scripts.04_train_and_evaluate    # panel_features → scores, metrics, drivers
uv run python -m scripts.05_build_app_data        # scores + metrics → app/backend/data/*.json

# Dashboard (backend + frontend, run both)
uv run uvicorn app.backend.main:app --reload --port 8000
npm --prefix app/frontend install && npm --prefix app/frontend run dev   # http://localhost:3000

# Notebook (jupytext source: notebooks/informe.py → rendered .ipynb)
uv run jupyter notebook notebooks/informe.ipynb
```

No tests, linting, or CI configured. Validation is embedded in the pipeline (coverage checks, diagnostic plots, bootstrap CI, Diebold-Mariano tests).

## Architecture

### Data flow

```
assets/usa_fondos_pp.sqlite
    → 01_build_features      → artifacts/panel_raw.parquet
                                   → 02_eda_report → artifacts/plots/
    → 03_build_features_full → artifacts/panel_features.parquet
    → 04_train_and_evaluate  → artifacts/scores.parquet + scores_{h}m.parquet
                               artifacts/metrics.json
                               artifacts/drivers_elastic.csv, drivers_lgbm.csv
                               artifacts/fold_diagnostics.csv
                               artifacts/plots/signal_*.png
    → 05_build_app_data      → app/backend/data/*.json → FastAPI → React
```

Each script reads artifacts from the previous step and writes to `artifacts/`. The pipeline is strictly sequential and stateless — reproducible end-to-end from `run_all.py`.

### `src/` modules

- **`paths.py`** — Centralized paths: `PROJECT_ROOT`, `DB_PATH`, `ARTIFACTS_DIR`, `PLOTS_DIR`, `APP_DATA_DIR`
- **`data.py`** — SQLite loading, daily total return (NAV + capital events, winsorized p99.5), monthly panel construction, fee attachment (ffill+bfill intra-fund, mediana cross-section fallback), concentration attachment (ffill only)
- **`features.py`** — Feature engineering producing 31 features + 4 forward targets. Three groups:
  - CORE (14): momentum 1/3/6/12m, vol_12m, max_dd_12m, sharpe_12m, intra-month features, skewness, hit_rate, distribution_yield, persistencia_rank
  - EXTENDED (5): fee, log_n_instrumentos, pct_acum + availability flags
  - RANK (12): cross-sectional percentiles of selected features
  - Primary target: `target_sharpe_rank_12m` (percentile of forward 12m Sharpe). Secondary targets: ret, sortino, max_dd (for multi-lens validation)
  - Key exports: `FEATURE_COLS`, `REDUCED_FEATURES`, `get_modeling_frame()` (filters to funds with ≥36 months history)
- **`splits.py`** — Walk-forward expanding window. Embargo = target horizon (not fixed at 12m). Min train 60 months, val 12 months per fold. `Fold` frozen dataclass
- **`model.py`** — Four scoreadores:
  - `ElasticNetModel` (primary, interpretable) — ElasticNetCV, coefs as "drivers"
  - `LightGBMModel` (sanity check) — depth=4, num_leaves=15, feature importances
  - `benchmark_naive_score()` — ret_12m_rank − fee_rank, no training
  - `axiomatic_score()` — theory-weighted rank combination (Sharpe, fee, max_dd, concentration, autocorr), no training
- **`metrics.py`** — IC (Spearman per date), quintile spread Q5-Q1, hit rate top-quartile, long-short information ratio, rank persistence, `multi_lens_evaluation()` (same score vs 4 realized targets)
- **`validation.py`** — Bootstrap CI of mean IC (5,000 iterations), Diebold-Mariano test with Newey-West correction

### Multi-horizon experiment

Script 04 trains on multiple horizons (3, 6, 12 months) × feature sets (full, reduced). Embargo equals the horizon. Outputs are per-combination: `scores_{h}m.parquet`. The 12m full-features result is the primary deliverable (`scores.parquet`).

### Dashboard

- **Backend** (`app/backend/main.py`): FastAPI serving pre-computed JSONs from `app/backend/data/`. No online inference. CORS enabled (allow_origins=*). Endpoints: `/api/meta`, `/api/funds`, `/api/funds/{fondo}`, `/api/drivers`, `/api/backtest`, `/api/health`
- **Frontend** (`app/frontend/`): React 18 + TypeScript + Vite + Tailwind + Recharts. Vite dev server on port 3000 proxies `/api` to `localhost:8000`. Pages: Overview, Detail, Drivers, Backtest. Excel export via `xlsx` library

### Dataset

`assets/usa_fondos_pp.sqlite` (~80 MB, **not in repo** — gitignored). 277 fondos, 1988-11 → 2026-05. Three tables:
- `historico` — fecha, securities (→ fondo), precio (NAV), evento_pct
- `fees` — fecha, fondo, fee (89% NULL in raw, reported only from 2024-01-31; pipeline imputes via ffill+bfill under structural stability assumption)
- `subyacentes` — fecha, nemo_fondo (→ fondo), pct_acum, n_instrumentos

## Conventions

- Code should be **clear and didactic** (prueba técnica context), not over-abstracted.
- Anti-leakage discipline: all features use only backward-looking information. Target is the only forward-looking element, protected by embargo = horizon in walk-forward splits.
- `src/` contains reusable modules (never executed directly); `scripts/` contains numbered entry points that orchestrate steps by importing from `src/`.
- Fee imputation uses ffill+bfill intra-fund (structural fee assumption, verified: intra-fund std ≈ 0). The binary flag `fee_observado` lets the model distinguish original vs imputed values.
