"""Backend FastAPI — Fund Scoring Dashboard.

Sirve JSONs pre-computados desde data/. Sin inferencia online.

Endpoints:
    GET /api/meta                resumen global, métricas y as-of date
    GET /api/funds                ranking completo de fondos por score
    GET /api/funds/{fondo}        ficha histórica de un fondo
    GET /api/drivers              coeficientes ElasticNet + importance LightGBM
    GET /api/backtest             serie temporal del spread top-Q5 vs bot-Q1
    GET /api/portfolio            portafolio óptimo D10 con backtest
    GET /api/health               liveness probe
"""

from __future__ import annotations

import json
from pathlib import Path

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware

DATA_DIR = Path(__file__).parent / "data"

app = FastAPI(
    title="Fund Scoring API",
    description="Backend del dashboard de scoring de fondos mutuos.",
    version="0.1.0",
)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


def _load(fname: str):
    p = DATA_DIR / fname
    if not p.exists():
        raise HTTPException(status_code=503, detail=f"data not built yet: {fname}")
    with open(p) as f:
        return json.load(f)


@app.get("/api/health")
def health():
    return {"status": "ok"}


@app.get("/api/meta")
def get_meta():
    return _load("meta.json")


@app.get("/api/funds")
def get_funds(decil: int | None = None, limit: int | None = None):
    funds = _load("funds_summary.json")
    if decil is not None:
        funds = [f for f in funds if f.get("decil") == decil]
    if limit is not None:
        funds = funds[:limit]
    return funds


@app.get("/api/funds/{fondo}")
def get_fund_detail(fondo: str):
    detail = _load("fund_detail.json")
    if fondo not in detail:
        raise HTTPException(status_code=404, detail=f"fondo no encontrado: {fondo}")
    summary = next((f for f in _load("funds_summary.json") if f["fondo"] == fondo), None)
    return {"fondo": fondo, "summary": summary, **detail[fondo]}


@app.get("/api/drivers")
def get_drivers():
    return _load("drivers.json")


@app.get("/api/backtest")
def get_backtest():
    return _load("backtest.json")


@app.get("/api/portfolio")
def get_portfolio():
    return _load("portfolio.json")
