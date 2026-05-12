"""src/ — Módulos reutilizables del pipeline de scoring de fondos mutuos.

Esta carpeta contiene la lógica central del proyecto, organizada en
módulos que se importan desde los scripts de orquestación (scripts/).
Ningún archivo de src/ se ejecuta directamente.

Módulos:
    paths.py       Rutas centralizadas (DB, artifacts, plots).
    data.py        Carga de sqlite, cálculo de retorno total, panel mensual.
    features.py    Ingeniería de features (32 columnas) y target forward.
    splits.py      Generación de folds walk-forward con embargo temporal.
    model.py       Definición y entrenamiento de ElasticNet, LightGBM, benchmark.
    metrics.py     Métricas de evaluación: IC, spread D10-D1, hit rate.
    validation.py  Validación estadística: bootstrap CI, Diebold-Mariano.
"""
