"""Walk-forward expanding window con embargo temporal.

Para un target a 12 meses forward, usamos embargo de 12 meses entre el
fin del set de entrenamiento y el inicio del set de validación, de modo
que las ventanas de target no se solapen entre los dos sets. Esto evita
contaminación de información futura.

Layout temporal por fold:

    [-- TRAIN (expanding) --][-- embargo (12m) --][-- VAL (val_months) --]
    fechas <= train_end       fechas en gap        fechas v..v+val_months
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
    unique = sorted(pd.Series(pd.to_datetime(dates)).drop_duplicates())
    n = len(unique)

    fold_id = 0
    train_end_idx = min_train_months - 1
    while True:
        val_start_idx = train_end_idx + 1 + embargo_months
        val_end_idx = val_start_idx + val_months - 1
        if val_end_idx >= n:
            return
        yield Fold(
            fold_id=fold_id,
            train_dates=list(unique[: train_end_idx + 1]),
            val_dates=list(unique[val_start_idx : val_end_idx + 1]),
        )
        train_end_idx += val_months
        fold_id += 1
