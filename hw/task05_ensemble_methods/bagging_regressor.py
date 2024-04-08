import numpy as np
from sklearn.base import clone
from sklearn.utils import resample

class BaggingRegressor:
    def __init__(self, base_model, n_estimators=10, random_state=None):
        self.base_model = base_model
        self.n_estimators = n_estimators
        self.random_state = random_state
        self.models = []

    def fit(self, X, y):
        for _ in range(self.n_estimators):
            X_resample, y_resample = resample(X, y, random_state=self.random_state)
            model = clone(self.base_model)
            model.fit(X_resample, y_resample)
            self.models.append(model)

    def predict(self, X):
        predictions = np.zeros((self.n_estimators, len(X)))
        for i, model in enumerate(self.models):
            predictions[i, :] = model.predict(X)
        return predictions.mean(axis=0)


    def get_params(self, deep=True):
        return {"base_model": self.base_model, "n_estimators": self.n_estimators}

    def set_params(self, **parameters):
        for parameter, value in parameters.items():
            setattr(self, parameter, value)
        return self