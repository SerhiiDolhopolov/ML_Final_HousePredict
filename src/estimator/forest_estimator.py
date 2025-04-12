from sklearn.ensemble import RandomForestClassifier

from estimator import Estimator


class ForestEstimator(Estimator):
    def __init__(self, train_X, train_y, test_X, test_y, params=None):
        super().__init__(train_X, train_y, test_X, test_y)
        if params is None:
            params = dict()
        self.model = RandomForestClassifier(**params)

    def fit(self):
        self.model.fit(self.train_X, self.train_y)
        print(self.model.score(self.train_X, self.train_y))

    def predict(self):
        y_train_predict = self.model.predict(self.train_X)
        print(y_train_predict)