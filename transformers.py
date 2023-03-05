import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import StandardScaler


class NormalizeSpeed(BaseEstimator, TransformerMixin):
    def __init__(self):
        return

    def fit(self, X, y=None):
        return self

    def transform(self, X, y=None):
        scaler = StandardScaler()
        X["Speed"] = scaler.fit_transform(X[["Speed"]])
        return X


class SelectSpeedDirectionCols(BaseEstimator, TransformerMixin):
    def __init__(self):
        return

    def fit(self, X, y=None):
        return self

    def transform(self, X, y=None):
        X = X[["Direction", "Speed"]]
        return X


class DropNA(BaseEstimator, TransformerMixin):
    def __init__(self):
        return

    def fit(self, X, y=None):
        return self

    def transform(self, X, y=None):
        X = X.dropna()
        return X


class DirectionToVector(BaseEstimator, TransformerMixin):
    def __init__(self):
        return

    def fit(self, X, y=None):
        return self

    def transform(self, X, y=None):
        X = self._direction_to_degrees(X)
        direction = X["Direction"]
        speed = X["Speed"]

        direction_radians = direction * np.pi / 180

        X['Wx'] = speed * np.cos(direction_radians)
        X['Wy'] = speed * np.sin(direction_radians)

        return X.drop(columns=["Direction"])

    @staticmethod
    def _direction_to_degrees(X):
        directions = ['N', 'NNE', 'NE', 'ENE', 'E', 'ESE', 'SE', 'SSE', 'S',
                      'SSW', 'SW', 'WSW', 'W', 'WNW', 'NW', 'NNW']

        direction_degrees = [i * 22.5 for i in range(len(directions))]
        direction = dict(zip(directions, direction_degrees))
        X["Direction"] = X["Direction"].map(direction)

        return X
