"""Generación de folds walk-forward con embargo temporal.

Este módulo implementa la estrategia de validación temporal del pipeline.
En lugar de un train/test split estático, usa una ventana de entrenamiento
rolling (con max_train_months) y múltiples bloques de validación (folds),
simulando cómo se usaría el modelo en producción (siempre entrenando con
datos pasados y evaluando en datos futuros).

La ventana rolling (max_train_months=120, 10 años) mantiene solo datos
recientes, adaptándose a cambios de régimen de mercado (pre/post-2008,
era de tasas cero, inflación 2022+). Si max_train_months=None, se usa
expanding window (comportamiento legacy).

El embargo es CRÍTICO: dado que el target es el retorno forward,
sin embargo las ventanas de target del set de train y de validación
se solaparían, contaminando la evaluación. El embargo descarta meses
entre el fin de train y el inicio de validación, garantizando
independencia total.

Layout temporal por fold:

    [--- TRAIN (rolling, 10a max) ---][--- EMBARGO (6m) ---][--- VAL (12m) ---]
    últimos max_train_months            gap descartado        evaluación OOS

Ejemplo con max_train_months=120:
    Fold 0: train 60m,   embargo 6m, val 12m (primer fold, aún no llega al tope)
    Fold 5: train 120m,  embargo 6m, val 12m (se estabiliza en 10 años)
    ...
    Fold N: train 120m,  embargo 6m, val 12m (ventana deslizante)

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
    max_train_months: int | None = None,
) -> Iterator[Fold]:
    """Genera folds de walk-forward (rolling o expanding window).

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
    max_train_months :
        máximo de meses en la ventana de train. None = expanding (sin límite),
        int = rolling window (se recorta el inicio del train).
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
        # Rolling window: recortar inicio del train si excede max_train_months
        train_start_idx = 0
        if max_train_months is not None:
            train_start_idx = max(0, train_end_idx + 1 - max_train_months)
        yield Fold(
            fold_id=fold_id,
            train_dates=list(unique[train_start_idx : train_end_idx + 1]),
            val_dates=list(unique[val_start_idx : val_end_idx + 1]),
        )
        # Avanzar la ventana de train por val_months
        train_end_idx += val_months
        fold_id += 1
