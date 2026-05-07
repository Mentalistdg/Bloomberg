"""Entrenamiento de los modelos de scoring.

Dos modelos comparados:
    - ElasticNet (con CV interna para alpha y l1_ratio): modelo
      explicativo y principal, coeficientes interpretables como
      "drivers del score".
    - LightGBM regressor: sanity check no-lineal. Si supera al lineal
      en IC out-of-sample por un margen claro hay no-linealidad o
      interacciones que el lineal no captura; en caso contrario la
      simplicidad e interpretabilidad del lineal pesa más.

Ambos se entrenan dentro de cada fold del walk-forward y predicen
sobre el set de validación correspondiente.
"""

from __future__ import annotations

from dataclasses import dataclass

import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.linear_model import ElasticNetCV
from sklearn.preprocessing import StandardScaler


@dataclass
class ElasticNetModel:
    alphas: tuple = (0.001, 0.01, 0.1, 1.0)
    l1_ratios: tuple = (0.1, 0.5, 0.9)
    cv: int = 5
    random_state: int = 42

    def fit_predict(self, X_train, y_train, X_val):
        scaler = StandardScaler()
        Xt = scaler.fit_transform(X_train)
        Xv = scaler.transform(X_val)
        model = ElasticNetCV(
            alphas=list(self.alphas),
            l1_ratio=list(self.l1_ratios),
            cv=self.cv,
            random_state=self.random_state,
            max_iter=10_000,
            n_jobs=-1,
        )
        model.fit(Xt, y_train)
        preds = model.predict(Xv)
        return preds, dict(
            alpha=float(model.alpha_),
            l1_ratio=float(model.l1_ratio_),
            coefs=dict(zip(X_train.columns, model.coef_.tolist())),
            intercept=float(model.intercept_),
            scaler_mean=scaler.mean_.tolist(),
            scaler_scale=scaler.scale_.tolist(),
        )


@dataclass
class LightGBMModel:
    n_estimators: int = 300
    learning_rate: float = 0.03
    max_depth: int = 4
    num_leaves: int = 15
    min_child_samples: int = 50
    subsample: float = 0.8
    colsample_bytree: float = 0.8
    random_state: int = 42

    def fit_predict(self, X_train, y_train, X_val):
        model = lgb.LGBMRegressor(
            n_estimators=self.n_estimators,
            learning_rate=self.learning_rate,
            max_depth=self.max_depth,
            num_leaves=self.num_leaves,
            min_child_samples=self.min_child_samples,
            subsample=self.subsample,
            colsample_bytree=self.colsample_bytree,
            random_state=self.random_state,
            n_jobs=-1,
            verbose=-1,
        )
        model.fit(X_train, y_train)
        preds = model.predict(X_val)
        return preds, dict(
            feature_importance=dict(
                zip(X_train.columns, model.feature_importances_.astype(int).tolist())
            ),
        )


def benchmark_naive_score(X_val_df: pd.DataFrame) -> np.ndarray:
    """Benchmark trivial: score = -1 * fee_rank + ret_12m_rank.
    Combina dos heurísticas conocidas (fee bajo bueno, momentum 12m bueno).
    """
    fee_rank = X_val_df.get("fee_rank", pd.Series(0.5, index=X_val_df.index))
    mom_rank = X_val_df.get("ret_12m_rank", pd.Series(0.5, index=X_val_df.index))
    return (mom_rank - fee_rank).values
