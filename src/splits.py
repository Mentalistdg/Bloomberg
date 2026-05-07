"""Generación de folds walk-forward con embargo temporal.

Este módulo implementa la estrategia de validación temporal del pipeline.
En lugar de un train/test split estático, usa una ventana de entrenamiento
que se expande progresivamente y múltiples bloques de validación (folds),
simulando cómo se usaría el modelo en producción (siempre entrenando con
datos pasados y evaluando en datos futuros).

El embargo de 12 meses es CRÍTICO: dado que el target es el retorno
forward a 12 meses, sin embargo las ventanas de target del set de train
y de validación se solaparían, contaminando la evaluación. El embargo
descarta los 12 meses entre el fin de train y el inicio de validación,
garantizando independencia total.

Layout temporal por fold:

    [-- TRAIN (expanding) --][-- EMBARGO (12m) --][-- VAL (12m) --]
    fechas <= train_end       gap descartado        evaluación OOS

Ejemplo con 9 folds:
    Fold 0: train 60m,  embargo 12m, val 12m (primer fold ~2016)
    Fold 1: train 72m,  embargo 12m, val 12m
    ...
    Fold 8: train 156m, embargo 12m, val 12m (último fold ~2024)

Usado por: scripts/04_train_and_evaluate.py
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterator

import pandas as pd


@dataclass(frozen=True)
class Fold:
    fold_id: int
    train_dates: list[pd.Timestamp]
    val_dates: list[pd.Timestamp]

    @property
    def train_end(self) -> pd.Timestamp:
        return self.train_dates[-1]

    @property
    def val_start(self) -> pd.Timestamp:
        return self.val_dates[0]

    @property
    def val_end(self) -> pd.Timestamp:
        return self.val_dates[-1]


def walk_forward_folds(
    dates,
    min_train_months: int = 60,
    val_months: int = 12,
    embargo_months: int = 12,
) -> Iterator[Fold]:
    """Genera folds de walk-forward expanding window.

    Parameters
    ----------
    dates :
        secuencia de fechas (pueden repetirse). Se ordenan internamente.
    min_train_months :
        mínimo de meses de entrenamiento antes del primer fold.
    val_months :
        cantidad de meses por bloque de validación.
    embargo_months :
        meses descartados entre fin de train y inicio de val.
    """
    # Extraer meses únicos ordenados cronológicamente
    unique = sorted(pd.Series(pd.to_datetime(dates)).drop_duplicates())
    n = len(unique)

    fold_id = 0
    # El primer fold necesita al menos min_train_months de historia
    train_end_idx = min_train_months - 1
    while True:
        # Saltar embargo_months después del fin de train
        val_start_idx = train_end_idx + 1 + embargo_months
        val_end_idx = val_start_idx + val_months - 1
        # Si no hay suficientes meses para completar la ventana de val, terminamos
        if val_end_idx >= n:
            return
        yield Fold(
            fold_id=fold_id,
            train_dates=list(unique[: train_end_idx + 1]),   # ventana expanding
            val_dates=list(unique[val_start_idx : val_end_idx + 1]),
        )
        # Avanzar la ventana de train por val_months (expanding window)
        train_end_idx += val_months
        fold_id += 1
