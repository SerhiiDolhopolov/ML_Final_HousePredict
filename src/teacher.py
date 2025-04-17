import logging

import numpy as np
import shap
from sklearn.model_selection import GridSearchCV, cross_val_score
from sklearn.metrics import mean_squared_error


class Teacher:
    def __init__(self, 
                 estimator, 
                 X_train, 
                 y_train, 
                 uniformed_features: list[str] = None):
        self.estimator = estimator
        self.X_train = X_train
        self.y_train = y_train
        if uniformed_features is None:
            uniformed_features = []
        self.X_train.drop(columns=uniformed_features, inplace=True, errors='ignore')
        self.columns = self.X_train.columns
        
    def fit(self) -> float:
        y_train = np.log1p(self.y_train)
        self.estimator.fit(self.X_train, y_train)
        return self.estimator.score(self.X_train, y_train)
    
    def predict(self, X, y) -> np.ndarray:
        X = X[self.columns]
        y_pred = self.estimator.predict(X)
        y_pred = np.expm1(y_pred)
        logging.debug('RMSE:{0}'.format(np.sqrt(mean_squared_error(y, y_pred))))
        logging.debug('RMSLE:{0}'.format(mean_squared_error(np.log1p(y), np.log1p(y_pred))))
        return y_pred
    
    def search_params(self, params: dict, scoring: str, n_jobs: int = -1):
        grid_search = GridSearchCV(estimator=self.estimator, param_grid=params, 
                                   scoring=scoring, n_jobs=n_jobs)
        grid_search.fit(self.X_train, self.y_train)
        print("Best params:", grid_search.best_params_)
    
    def show_shap(self, X, max_display: int = -1):
        X = X[self.columns]
        shap.initjs()
        explainer = shap.Explainer(self.estimator, X)
        shap_values = explainer.shap_values(X)
        shap.summary_plot(shap_values, X, plot_type="bar", max_display=max_display)

        importance = np.abs(shap_values).mean(axis=0)
        sorted_idx = np.argsort(importance)[::-1]
        sorted_features = X.columns[sorted_idx]
        logging.debug(sorted_features[:max_display])
