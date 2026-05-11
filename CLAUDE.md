# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Purpose

Prueba técnica para **Analista de Renta Variable Internacional — AFP Habitat** (autor: estudiante FEN UChile). Sistema cuantitativo de scoring de fondos mutuos USA: a partir de históricos de precios, eventos de capital, comisiones y concentración, asigna un puntaje que distingue fondos atractivos de poco atractivos.

Universo de referencia: **renta variable internacional** (MSCI ACWI / World / EM, S&P 500), no renta variable chilena.

## Commands

```bash
# Dependencias (Python >=3.12, gestionado con uv — nunca usar pip)
uv sync

# Pipeline completo (5 pasos secuenciales, requiere assets/usa_fondos_pp.sqlite)
uv run python -m scripts.run_all

# Pasos individuales
uv run python -m scripts.01_build_features        # sqlite → artifacts/panel_raw.parquet
uv run python -m scripts.02_eda_report            # panel_raw → artifacts/plots/
uv run python -m scripts.03_build_features_full   # panel_raw → artifacts/panel_features.parquet
uv run python -m scripts.04_train_and_evaluate    # panel_features → scores, metrics, drivers
uv run python -m scripts.05_build_app_data        # scores + metrics → app/backend/data/*.json

# Dashboard local (two terminals required)
# Terminal 1 — backend:
uv run uvicorn app.backend.main:app --reload --port 8000
# Terminal 2 — frontend (Vite dev server proxies /api → localhost:8000):
npm --prefix app/frontend install && npm --prefix app/frontend run dev   # http://localhost:3000

# Dashboard via Docker (production build, single container on port 8080)
docker build -t fund-scoring .
docker run -d --name scoring-dashboard -p 8080:80 --restart unless-stopped fund-scoring
# verify: curl http://localhost:8080/api/health → {"status":"ok"}

# Frontend production build (outputs to app/frontend/dist/)
npm --prefix app/frontend run build

# Notebook (jupytext source: notebooks/informe.py → rendered .ipynb)
uv run jupyter notebook notebooks/informe.ipynb
```

No tests, linting, or CI configured. Validation is embedded in the pipeline (coverage checks, diagnostic plots, bootstrap CI, Diebold-Mariano tests).

**Note:** pyproject.toml requires Python `>=3.12`. The Dockerfile uses `python:3.13-slim` for production deployment.

## Architecture

### Data flow

```
assets/usa_fondos_pp.sqlite
    → 01_build_features      → artifacts/panel_raw.parquet
                                   → 02_eda_report → artifacts/plots/
    → 03_build_features_full → artifacts/panel_features.parquet
    → 04_train_and_evaluate  → artifacts/scores.parquet
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
- **`features.py`** — Feature engineering producing 32 features + 4 forward targets (per horizon). Three groups:
  - CORE (14): ret_1m/3m/6m/12m, vol_12m, max_dd_12m, sharpe_12m, vol_intrames, autocorr_diaria, ratio_dias_cero, skewness_12m, hit_rate_12m, distribution_yield_12m, persistencia_rank_12m
  - EXTENDED (5): fee, log_n_instrumentos, pct_acum, fee_observado, concentracion_disponible
  - RANK (13): cross-sectional percentiles of `RANK_COLS` (ret_3m, ret_12m, vol_12m, sharpe_12m, max_dd_12m, fee, log_n_instrumentos, pct_acum, vol_intrames, autocorr_diaria, ratio_dias_cero, skewness_12m, hit_rate_12m)
  - Primary target: `target_sortino_rank_6m` (percentile of forward 6m Sortino). Secondary lenses: ret, sharpe, max_dd (for multi-lens validation, not training). Sortino was chosen over Sharpe because it penalizes only downside volatility — upside movement is desirable for a pension fund, not penalizable.
  - Key exports: `FEATURE_COLS`, `REDUCED_FEATURES` (curated 5-feature subset, unused in production), `get_modeling_frame()` (filters to funds with ≥36 months history, defaults to `target="sortino"`)
- **`splits.py`** — Walk-forward rolling window (`max_train_months=120`, 10 years). Embargo = target horizon (not fixed at 12m). Min train 60 months, val 12 months per fold. `Fold` frozen dataclass. If `max_train_months=None`, falls back to expanding window (legacy)
- **`model.py`** — Three scoreadores:
  - `ElasticNetModel` (primary, interpretable) — ElasticNetCV, coefs as "drivers"
  - `LightGBMModel` (sanity check) — depth=4, num_leaves=15, feature importances
  - `benchmark_naive_score()` — ret_12m_rank − fee_rank, no training
- **`metrics.py`** — IC (Spearman per date), quintile spread Q5-Q1, hit rate top-quartile, long-short information ratio, rank persistence, `multi_lens_evaluation()` (same score vs 4 realized targets)
- **`validation.py`** — Bootstrap CI of mean IC (5,000 iterations), Diebold-Mariano test with Newey-West correction

### Prediction horizon

Script 04 trains on a single horizon (6 months) with full features and target `target_sortino_rank_6m`. Embargo equals the horizon (6m). Output: `scores.parquet`. Previous multi-horizon experiments (3m, 6m, 12m × full/reduced features) were consolidated into this single configuration.

### Dashboard

- **Backend** (`app/backend/main.py`): FastAPI serving pre-computed JSONs from `app/backend/data/`. No online inference. CORS enabled (allow_origins=*). Endpoints: `/api/meta`, `/api/funds`, `/api/funds/{fondo}`, `/api/drivers`, `/api/backtest`, `/api/portfolio`, `/api/health`
- **Frontend** (`app/frontend/`): React 18 + TypeScript + Vite + Tailwind + Recharts. Vite dev server on port 3000 proxies `/api` to `localhost:8000`. Excel export via `xlsx` library. Routes:
  - `/` (Overview — fund ranking table with realized metrics: ret 6m, sortino 6m)
  - `/detail/:fondo` (Detail — equity curve with trailing/forward split, KPIs, chart ruler tool)
  - `/drivers` (Drivers — ElasticNet coefficients per fold)
  - `/backtest` (Backtest — Sortino realizado D10 vs D1 por decil + métricas OOS)
  - `/portfolio` (Portfolio — mean-variance optimized portfolio of D10 funds, equity curve, weights, rebalancing history)
  - All API types and functions defined in `app/frontend/src/services/api.ts`
- **Interactive components**: Chart ruler tool (`hooks/useChartRuler.ts` + `components/RulerOverlay.tsx`) — click two points on any equity chart to measure % change between them
- **Docker** (`Dockerfile`): Multi-stage build — Node 20 builds frontend, Python 3.13 + nginx + supervisord serves both. Nginx on port 80 routes `/api` to uvicorn and serves static frontend. Production deploy via `docker-compose.yml` on port 8080

### Dataset

`assets/usa_fondos_pp.sqlite` (~80 MB, **not tracked in git** — must be placed in `assets/` before running the pipeline). 277 fondos, 1988-11 → 2026-05. Three tables:
- `historico` — fecha, securities (→ fondo), precio (NAV), evento_pct
- `fees` — fecha, fondo, fee (89% NULL in raw, reported only from 2024-01-31; pipeline imputes via ffill+bfill under structural stability assumption)
- `subyacentes` — fecha, nemo_fondo (→ fondo), pct_acum, n_instrumentos

### Deployment

Deployed on AWS EC2 (`t3.micro`, `us-east-2`) behind Cloudflare Tunnel. Domain: `scoring.davidgonzalez.cl`. Deploy branch: `master`. See `docs/DEPLOYMENT.md` for operational details (SSH, re-deploy, troubleshooting). The EC2 also hosts an unrelated app (Cronnos) on ports 80/8000 — the scoring dashboard runs in a Docker container mapped to host port 8080.

## Conventions

- Code should be **clear and didactic** (prueba técnica context), not over-abstracted.
- Anti-leakage discipline: all features use only backward-looking information. Target is the only forward-looking element, protected by embargo = horizon in walk-forward splits.
- `src/` contains reusable modules (never executed directly); `scripts/` contains numbered entry points that orchestrate steps by importing from `src/`.
- Fee imputation uses ffill+bfill intra-fund (structural fee assumption, verified: intra-fund std ≈ 0). The binary flag `fee_observado` lets the model distinguish original vs imputed values.
- The `Bloomberg/` directory at the root is an old repo clone — it is gitignored and should be ignored. The actual code lives in `src/`, `scripts/`, and `app/` at the project root.
- `notebooks/informe.py` is the jupytext source of truth. Edit the `.py` file, not the `.ipynb` directly — the notebook is rendered from it.
- Deploy scripts live in `deploy/` (not the repo root): `deploy/deploy_ec2.sh`, `deploy/deploy_cloudflare.sh`.
- Code, comments, and variable names are in **Spanish** (e.g., `fecha`, `fondo`, `precio`, `retorno`). Keep this convention when adding new code.
- Frontend uses **npm** (not yarn/pnpm). Install: `npm --prefix app/frontend install`, dev: `npm --prefix app/frontend run dev`.
- AI-assisted commits use trailer: `Co-Authored-By: Claude Opus 4.6 <noreply@anthropic.com>`.
- **Git branches:** `main` is the default branch for PRs. `master` is the deploy branch (pushed to EC2). Keep both in sync after merging.
- **Artifacts are tracked in git** (parquet, CSV, PNG, JSON in `artifacts/` and `app/backend/data/`). This is intentional — ensures the dashboard works without re-running the pipeline. Re-run the pipeline before committing if source logic changes.

## Changelog (uncommitted, post-77d4df2)

Summary of changes made since the last commit (`77d4df2 Fix Realizado 12m`). This section helps future agents understand what was modified and why. **Clear this section after committing these changes.**

### 1. Target metric: Sharpe → Sortino
- Primary training target changed from `target_sharpe_rank_6m` to `target_sortino_rank_6m`.
- Sortino penalizes only downside volatility — upside movement is desirable for a pension fund (AFP), not penalizable. Better aligned with fund selection philosophy.
- `src/features.py`: `get_modeling_frame()` default changed to `target="sortino"`. Explicit Sortino target calculation added.
- `scripts/04_train_and_evaluate.py`: Trains on `target_sortino_rank_6m` exclusively.

### 2. Pipeline consolidation: multi-horizon → single 6m
- Previously: 3 horizons (3m, 6m, 12m) × 2 feature sets (full, reduced) = 6 configurations.
- Now: single configuration — 6m Sortino, full features.
- Deleted all `*_3m.*`, `*_6m.*`, `*_12m.*`, `*_reduced.*` artifact variants (scores, drivers, fold_diagnostics, signal plots).
- Deleted `artifacts/horizon_comparison.csv` and `artifacts/survivorship_fondos.csv`.
- AxiomaticScorer (fixed-weight benchmark) removed from production pipeline.

### 3. New dashboard page: Portfolio (`/portfolio`)
- Mean-variance optimized portfolio of D10 (top-decile) funds.
- Shows: equity curve (optimized vs equal-weight), current weights, rebalancing history.
- Uses `pyportfolioopt` (new dependency in `pyproject.toml`) with EMA returns + semicovariance risk model.
- Backend: new `/api/portfolio` endpoint serving `app/backend/data/portfolio.json`.
- Frontend: `app/frontend/src/pages/PortfolioPage.tsx`.

### 4. Chart ruler tool (interactive measurement)
- Click two points on any equity chart to measure % change between them.
- Files: `app/frontend/src/hooks/useChartRuler.ts`, `app/frontend/src/components/RulerOverlay.tsx`.
- Integrated in DetailPage and PortfolioPage.

### 5. Backtest page simplification
- Removed spread chart (D10−D1 as single line) — redundant and less intuitive than the two-line D10 vs D1 comparison.
- Page now shows: header with D10/D1 explanation, Sortino realizado por decil chart (D10 green, D1 red), metrics table, footnotes.

### 6. Overview page: realized metrics columns
- Added "Ret Real. 6m" and "Sortino Real. 6m" columns to the fund ranking table.
- Users can audit score vs realized forward performance.

### 7. Detail page enhancements
- Equity curve split: red line (trailing) + blue dashed (forward 6m OOS) with shaded forward zone.
- KPI strip shows both trailing metrics (Ret 12m, Vol, Sharpe, Sortino, Max DD) and forward OOS metrics (Ret realizado 6m, Sortino realizado 6m).

### 8. File reorganization
- Deploy scripts: root → `deploy/` (deploy_ec2.sh, deploy_cloudflare.sh).
- Documentation: root → `docs/` (INFORME.md, DEPLOYMENT.md, slides_outline.md).

### 9. API type extensions (`api.ts`)
- `Fund`: added `target_realizado_6m`, `sortino_realizado_6m`, `sortino_12m`.
- `FundDetail`: added `ret_mensual` array.
- New types: `PortfolioData`, `RebalanceHolding`.
- New function: `getPortfolio()`.

### 10. Dependencies
- Added `pyportfolioopt>=1.5` to `pyproject.toml` (portfolio optimization in `05_build_app_data.py`).

### 11. Walk-forward: expanding → rolling window
- `src/splits.py`: added `max_train_months` parameter to `walk_forward_folds()`. `None` = expanding (legacy), `int` = rolling window.
- `scripts/04_train_and_evaluate.py`: `MAX_TRAIN_MONTHS = 120` (10 years). Applied to both walk-forward folds and production train sets.
- Rolling window keeps only recent data, adapting to market regime changes (pre/post-2008, zero-rate era, 2022+ inflation).
- `notebooks/informe.py`: updated section 6 narrative, ASCII diagram, Gantt chart title.

### 12. Notebook section 7: benchmark naive integration
- Added benchmark naive (`ret_12m_rank − fee_rank`) to all tables, charts, and narrative in section 7 (Resultados OOS).
- Comparative table now shows 3 models (ElasticNet, LightGBM, Benchmark naive).
- IC mensual chart and violinplot include benchmark (orange, `#e65100`).
- Added Diebold-Mariano test output cell showing stat and p-value.
- Rewrote "Lectura critica" to honestly acknowledge that the benchmark is not statistically surpassed (DM p=0.65), explaining why this is an expected finding (Carhart 1997, momentum + fee dominance) and why the ElasticNet remains preferable (regime adaptation, risk features, interpretability, multi-lens generalization).
- Multi-lens table and interpretation updated to include benchmark comparison by lens.

### 13. Notebook: cerrar gaps vs criterios de evaluacion
- **Predictivo vs explicativo** (intro): expanded from 1 line to full subsection discussing interpretability for AFP governance, OOS validation as antidote to post-hoc rationalization, complexity vs interpretability trade-off.
- **Estacionariedad** (Section 4): new markdown cell explaining why returns are stationary, rank features are stationary by construction, and rolling window handles structural breaks.
- **Hiperparametros** (new subsection between Sections 6 and 7): explains ElasticNetCV nested CV (4 alphas × 3 l1_ratios), shows table and plot of selected alpha/l1_ratio per fold, interprets temporal evolution (high regularization early → low alpha with more data).
- **Portfolio narrative** (Section 10): corrected false claim that optimization beats equal-weight. Honest analysis: optimization provides lower max drawdown (-40.3% vs -49.8%) but not higher return. References DeMiguel et al. 2009.
- **AI section** (Section 11): added concrete prompt example (walk-forward leakage review), detailed validation process (code execution, manual fold inspection, primary source verification).
- **Extensions** (Section 11): added capacity constraints (Berk & Green 2004), simple ensemble proposal, macro regime features as interactions. New "que haria distinto" subsection with reflective bullets.
