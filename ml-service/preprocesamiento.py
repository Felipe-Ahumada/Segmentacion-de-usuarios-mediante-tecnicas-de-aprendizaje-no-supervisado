"""Transformadores personalizados para el pipeline de preprocesamiento.

Define Winsorizer y CorrelationFilter como transformadores compatibles con
sklearn Pipeline, siguiendo el patrón BaseEstimator + TransformerMixin.
"""

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin


class Winsorizer(BaseEstimator, TransformerMixin):
    """Trata valores extremos recortándolos a percentiles específicos.

    Args:
        limits: Tupla (inferior, superior) con las proporciones a recortar.
                Por defecto (0.05, 0.05) recorta el 5% en cada extremo.
    """

    def __init__(self, limits=(0.05, 0.05)):
        self.limits = limits

    def fit(self, X, y=None):
        if isinstance(X, pd.DataFrame):
            self.columns_ = X.columns
        else:
            self.columns_ = np.arange(X.shape[1])
        return self

    def transform(self, X):
        X = pd.DataFrame(np.asarray(X), columns=self.columns_).astype("float64").copy()
        for col in self.columns_:
            lower = X[col].quantile(self.limits[0])
            upper = X[col].quantile(1 - self.limits[1])
            X[col] = np.clip(X[col], lower, upper)
        return X


class CorrelationFilter(BaseEstimator, TransformerMixin):
    """Elimina variables con correlación superior al umbral.

    Args:
        threshold: Umbral de correlación absoluta. Variables con correlación
                   superior a este valor se eliminan (por defecto 0.9).
    """

    def __init__(self, threshold=0.9):
        self.threshold = threshold
        self.columns_to_drop_ = None

    def fit(self, X, y=None):
        X_df = pd.DataFrame(X)
        corr_matrix = X_df.corr().abs()
        upper = corr_matrix.where(
            np.triu(np.ones(corr_matrix.shape), k=1).astype(bool)
        )
        self.columns_to_drop_ = [
            col for col in upper.columns if any(upper[col] > self.threshold)
        ]
        return self

    def transform(self, X):
        X_df = pd.DataFrame(X)
        X_filtered = X_df.drop(columns=self.columns_to_drop_, errors="ignore")
        return X_filtered.values
