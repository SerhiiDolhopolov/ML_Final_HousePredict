from sklearn.ensemble import RandomForestClassifier

from estimator import Estimator


class ForestEstimator(Estimator):
    def __init__(self, params=None):
        super().__init__()
        if params is None:
            params = dict()
        self.model = RandomForestClassifier(**params)

    def fit(self, X, y):
        self.model.fit(X, y)
        print(self.model.score(X, y))

    def predict(self, X):
        y_train_predict = self.model.predict(X)
        print(y_train_predict)