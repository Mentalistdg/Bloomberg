# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Purpose of this repo

Sandbox/workspace para preparar la **prueba técnica de Analista de Renta Variable Internacional en AFP Habitat** (usuario: estudiante FEN UChile). No es una librería ni un producto: es un espacio de práctica con acceso real a datos de mercado vía Bloomberg.

Artefactos de contexto en la raíz (no son código pero son insumos):
- `Presentación RVI.pdf` — material de la entrevista / del rol.
- `usa_fondos_pp (1).zip` — dataset de fondos USA usado para ejercicios.

Cuando se trabaje en ejercicios de práctica, anclar los ejemplos al universo de **renta variable internacional** (MSCI ACWI / World / EM, S&P 500), no renta variable chilena.

## Toolchain

- Python **3.14** en `.venv/` administrado por **uv** (`C:\Users\itau_lab\.local\bin\uv`).
- No hay `pip` dentro del venv — siempre usar `uv` para instalar/sincronizar dependencias.
- No es un repo git (`git init` no se ha corrido). Si se va a versionar, confirmarlo antes con el usuario.

### Comandos comunes

```powershell
uv sync                      # instala/actualiza deps desde pyproject.toml + uv.lock
uv add <paquete>             # añade dep al pyproject y la instala
uv run main.py               # ejecuta script en el venv (recomendado sobre activar)
.venv\Scripts\python.exe ... # alternativa directa al intérprete del venv
```

## Bloomberg / blpapi

`blpapi` está instalado vía el índice oficial de Bloomberg, declarado en `pyproject.toml`:

```toml
[[tool.uv.index]]
name = "bloomberg"
url = "https://blpapi.bloomberg.com/repository/releases/python/simple/"
explicit = true

[tool.uv.sources]
blpapi = { index = "bloomberg" }
```

**Hechos importantes para no perder tiempo depurando:**
- El wheel de `blpapi >=3.26` trae `blpapi3_64.dll` embebida. **No** hace falta `BLPAPI_ROOT` ni añadir `C:\blp\DAPI` al `PATH`.
- Para correr cualquier código que use `blpapi`, la **Bloomberg Terminal debe estar abierta y logueada** y `bbcomm.exe` corriendo (escucha `localhost:8194`). Esto es Desktop API (DAPI), no server-side.
- Cuotas DAPI: ~5.000 securities únicos/día y ~500k hits/mes por terminal — campos como `EQY_DVD_HIST` cuentan mucho más que `PX_LAST`.
- Datos sujetos a las habilitaciones del usuario; campos no autorizados devuelven `NOT_AUTHORIZED` aunque se vean en la Terminal.
- Al importar `blpapi` en Python 3.14 aparecen `SyntaxWarning: invalid escape sequence` desde `resolutionlist.py` y `topiclist.py`. **Son cosméticos**, ignorar.
- `Element.append("field", "value")` es válido en runtime aunque PyCharm marque "Expected `Name`, got `str`" — el stub es estricto, blpapi convierte internamente.

`main.py` actualmente contiene un test de conexión mínimo contra `//blp/refdata` (sirve como smoke test: si corre y devuelve precios de IBM/AAPL, todo está OK).

## Servicios Bloomberg disponibles

- `//blp/refdata` — `ReferenceDataRequest` (snapshot, eq. BDP), `HistoricalDataRequest` (eq. BDH), `IntradayBarRequest`, `IntradayTickRequest`.
- `//blp/mktdata` — suscripciones tiempo real.
- `//blp/instruments` — búsqueda de tickers / ISIN / CUSIP.
- `//blp/apiflds` — metadata de campos.
- `//blp/exrsvc` — datos en bloque (eq. BDS, ej. holders, dividendos históricos).

## Convenciones de trabajo en este repo

- Cuando una tarea pueda usar datos reales de mercado, preferir **Bloomberg vía `blpapi`** sobre `yfinance`/CSV — la Terminal está disponible y los datos son los "buenos".
- Usar `yfinance` solo como fallback si la Terminal no está corriendo o como ejemplo reproducible para alguien sin BBG.
- Para ejercicios de portafolio, mantener código **claro y didáctico** (es contexto de prueba técnica), no sobre-abstraer.
