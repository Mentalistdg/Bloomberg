"""Definición y entrenamiento de los modelos de scoring.

Este módulo define los tres modelos que se comparan en el pipeline.
Cada modelo expone un método fit_predict que recibe datos de train
y validación, y devuelve predicciones + metadata (coeficientes,
importancias, hiperparámetros seleccionados).

Modelos:

    ElasticNet (PRINCIPAL):
        Regresión lineal regularizada con penalización L1+L2. Se usa
        ElasticNetCV que internamente hace 5-fold CV para seleccionar
        los mejores alpha (fuerza de regularización) y l1_ratio (balance
        entre L1 y L2). Las features se estandarizan antes de entrenar
        para que los coeficientes sean comparables entre sí — estos
        coeficientes son los "drivers" del score que se presentan al
        comité de inversión.

    LightGBM (SANITY CHECK):
        Gradient boosting no-lineal. Si supera al ElasticNet en IC
        out-of-sample, indicaría que hay interacciones o no-linealidades
        que el modelo lineal no captura. En la práctica no lo supera,
        validando la elección del modelo lineal.

    Benchmark naive (LÍNEA BASE):
        score = ret_12m_rank - fee_rank. Sin entrenamiento. Captura
        dos heurísticas básicas: momentum (retorno reciente alto es
        bueno) y fee bajo. Sirve para evaluar si los modelos ML agregan
        valor sobre la intuición más simple posible.

Los tres se entrenan y evalúan en cada fold del walk-forward
(ver splits.py), de modo que toda la evaluación es out-of-sample.

Usado por: scripts/04_train_and_evaluate.py
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
        """Entrena ElasticNet con CV interna y predice sobre validación."""
        # Estandarizar features (media=0, std=1) — fit solo sobre train
        scaler = StandardScaler()
        Xt = scaler.fit_transform(X_train)
        Xv = scaler.transform(X_val)  # aplica misma transformación a val
        # ElasticNetCV selecciona automáticamente alpha y l1_ratio óptimos
        # usando 5-fold CV interno dentro del set de train
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
        # Devolver predicciones + metadata para diagnóstico
        return preds, dict(
            alpha=float(model.alpha_),           # regularización seleccionada
            l1_ratio=float(model.l1_ratio_),     # balance L1/L2 seleccionado
            coefs=dict(zip(X_train.columns, model.coef_.tolist())),  # "drivers"
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
        """Entrena LightGBM y predice sobre validación.
        Hiperparámetros fijos (conservadores) — no se tunan por fold."""
        model = lgb.LGBMRegressor(
            n_estimators=self.n_estimators,
            learning_rate=self.learning_rate,
            max_depth=self.max_depth,
            num_leaves=self.num_leaves,
            min_child_samples=self.min_child_samples,
            subsample=self.subsample,           # bagging de filas
            colsample_bytree=self.colsample_bytree,  # bagging de features
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


# Pesos del score axiomático. Definidos a partir de teoría financiera de
# selección de fondos, NO entrenados — esa es la razón de ser del scoreador
# (transparente y defendible al comité). Los pesos suman 1.00 en valor
# absoluto; signos elegidos según la dirección esperada de cada feature.
AXIOMATIC_WEIGHTS = {
    # +: alto Sharpe pasado = mejor calidad esperada (persistencia
    # estructural del manager + estilo del fondo)
    "sharpe_12m_rank":     +0.30,
    # -: fee alto = costo garantizado, deduce retorno mes a mes
    # (Carhart 1997: predictor más estable de underperformance)
    "fee_rank":            -0.20,
    # +: max_dd_12m es valor NEGATIVO; rank alto = drawdown menos profundo
    # = menor riesgo de cola = mejor
    "max_dd_12m_rank":     +0.20,
    # -: alta concentración = mayor riesgo idiosincrático no diversificado
    # (literal: insumo "concentración del primer decil" del enunciado)
    "pct_acum_rank":       -0.15,
    # -: alta autocorr diaria = subyacentes ilíquidos / pricing stale =
    # mayor costo de salida y opacidad
    "autocorr_diaria_rank": -0.15,
}


def axiomatic_score(X_val_df: pd.DataFrame) -> np.ndarray:
    """Score axiomático sin entrenamiento. Combinación lineal de ranks
    cross-seccionales (todos en [0, 1]) con pesos derivados de teoría
    financiera de selección de fondos.

    A diferencia de ElasticNet/LightGBM (donde el modelo aprende los
    pesos de los datos), acá los pesos están fijados a priori. Esto es
    una decisión metodológica deliberada — el axiomático representa
    "así define un analista cuantitativo a un fondo bueno", y permite
    contrastar lo que se aprende empíricamente con lo que se sabe
    estructuralmente.

    Si el axiomático bate al ML, hay sospechas de overfitting o de mala
    elección de target. Si el ML bate al axiomático, el modelo está
    capturando interacciones o pesos no triviales que la teoría no
    anticipa. Ambos resultados son informativos.

    Pesos en `AXIOMATIC_WEIGHTS` (suman 1.00 en valor absoluto).
    """
    score = pd.Series(0.0, index=X_val_df.index)
    for feature, weight in AXIOMATIC_WEIGHTS.items():
        if feature in X_val_df.columns:
            score = score + weight * X_val_df[feature]
        else:
            raise KeyError(
                f"Feature {feature!r} requerida por axiomatic_score no está "
                f"en el DataFrame. Verificar que esté en RANK_COLS."
            )
    return score.values
