# Frontend — Fund Scoring Dashboard

## Cómo correr en desarrollo

```bash
# Desde la raíz del repo
npm --prefix app/frontend install
npm --prefix app/frontend run dev      # http://localhost:3000

# En otra terminal (backend)
uv run uvicorn app.backend.main:app --reload --port 8000
```

Vite proxea `/api/*` al backend en `:8000`.

## Build de producción

```bash
npm --prefix app/frontend run build    # genera dist/
```

`dist/` es el bundle estático listo para servir desde nginx, S3+CloudFront,
o un mismo container que también corre el FastAPI.

## Estructura

- `src/main.tsx` — entry point + router
- `src/components/Layout.tsx` — sidebar + outlet
- `src/services/api.ts` — capa axios contra el backend
- `src/pages/` — 4 páginas: Overview (ranking), Detail (ficha), Drivers,
  Backtest

## Páginas

| Ruta | Contenido |
|---|---|
| `/overview` | Tabla sortable de fondos con score, decil, métricas |
| `/detail/:fondo` | Ficha individual con equity curve y métricas |
| `/drivers` | Coeficientes ElasticNet — interpretabilidad del score |
| `/backtest` | Spread top-Q5 vs bottom-Q1 a lo largo del tiempo |
