from sklearn.tree import DecisionTreeClassifier
import numpy as np
from scipy.stats import mode
from sklearn.metrics import accuracy_score


class Bagging:
    def __init__(self, n_learners=10):
        self.n_learners = n_learners
        self.learners = []

    def fit(self, X, y):
        self.learners = []
        n_samples = X.shape[0]

        for _ in range(self.n_learners):
            sample_indices = np.random.choice(np.arange(n_samples), size=n_samples, replace=True)
            X_sample = X[sample_indices]
            y_sample = y[sample_indices]

            learner = DecisionTreeClassifier()
            learner.fit(X_sample, y_sample)
            self.learners.append(learner)

    def predict(self, X):
        predictions = np.array([learner.predict(X) for learner in self.learners])
        majority_vote = mode(predictions, axis=0)[0]
        return majority_vote.flatten()

    def score(self, X, y):
        predictions = self.predict(X)
        return accuracy_score(y, predictions)

    def get_params(self, deep=True):
        return {"n_learners": self.n_learners}

    def set_params(self, **parameters):
        for parameter, value in parameters.items():
            setattr(self, parameter, value)
        return self
