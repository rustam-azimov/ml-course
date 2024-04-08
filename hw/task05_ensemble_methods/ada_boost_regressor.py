import numpy as np
from sklearn.base import clone
from sklearn.tree import DecisionTreeRegressor

class AdaBoostRegressor:
    def __init__(self, base_estimator=DecisionTreeRegressor(max_depth=1), n_estimators=50):
        self.base_estimator = base_estimator
        self.n_estimators = n_estimators
        self.estimators_ = []
        self.estimator_weights_ = np.zeros(self.n_estimators)
        self.estimator_errors_ = np.ones(self.n_estimators)

    def fit(self, X, y):
        sample_weights = np.full(X.shape[0], (1 / X.shape[0]))
        
        for iboost in range(self.n_estimators):
            sample_mask = sample_weights > 0
            estimator = clone(self.base_estimator)

            if not any(sample_mask):
                break
            
            estimator.fit(X[sample_mask], y[sample_mask], sample_weight=sample_weights[sample_mask])
            y_predict = estimator.predict(X)
        
            incorrect = y_predict != y
            estimator_error = np.mean(
                np.average(incorrect, weights=sample_weights, axis=0)
            )
            
            estimator_weight = self._beta(iboost, estimator_error)

            sample_weights *= np.exp(estimator_weight * incorrect)
            sample_weights /= sample_weights.sum() 
            
            self.estimators_.append(estimator)
            self.estimator_errors_[iboost] = estimator_error
            self.estimator_weights_[iboost] = estimator_weight

    def _beta(self, iboost, estimator_error):
        value = np.log((1 - estimator_error) / (estimator_error + 1e-5)) + np.log(self.n_estimators - 1)
        if value is None or value == float('inf'):
            return 0
        return value

    def predict(self, X):
        predictions = sum(estimator.predict(X) * weight for estimator, weight in zip(self.estimators_, self.estimator_weights_))
        return predictions / sum(self.estimator_weights_)

    def get_params(self, deep=True):
        return {"base_estimator": self.base_estimator, "n_estimators": self.n_estimators}

    def set_params(self, **parameters):
        for parameter, value in parameters.items():
            setattr(self, parameter, value)
        return self
