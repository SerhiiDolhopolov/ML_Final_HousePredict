import logging

import numpy as np
import optuna
import shap
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import mean_squared_error, mean_squared_log_error
from sklearn.model_selection import KFold


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
        
        kf = KFold(n_splits=5, shuffle=True, random_state=0)
        rmse_scores = []
        rmsle_scores = []

        for train_idx, val_idx in kf.split(self.X_train):
            X_tr, X_val = self.X_train.iloc[train_idx], self.X_train.iloc[val_idx]
            y_tr, y_val = y_train.iloc[train_idx], y_train.iloc[val_idx]
            model = self.estimator.__class__(**self.estimator.get_params())
            model.fit(X_tr, y_tr)
            y_true = np.expm1(y_val)
            y_pred = np.expm1(model.predict(X_val))
            rmse_scores.append(np.sqrt(mean_squared_error(y_true, y_pred)))
            rmsle_scores.append(np.sqrt(mean_squared_log_error(y_true, y_pred)))

        mean_rmse = np.mean(rmse_scores)
        mean_rmsle = np.mean(rmsle_scores)
        logging.debug('Valid score RMSE: {0}'.format(mean_rmse))
        logging.debug('Valid score RMSLE: {0}'.format(mean_rmsle))
        
        
        self.estimator.fit(self.X_train, y_train)
        return self.estimator.score(self.X_train, y_train)
    
    def predict(self, X, y = None) -> np.ndarray:
        X = X[self.columns]
        y_pred = self.estimator.predict(X)
        y_pred = np.expm1(y_pred)
        if y is not None:
            logging.debug('RMSE:{0}'.format(np.sqrt(mean_squared_error(y, y_pred))))
            logging.debug('RMSLE:{0}'.format(np.sqrt(mean_squared_log_error(y, y_pred))))
        return y_pred
    
    def search_params_by_grid(self, params: dict, scoring: str, n_jobs: int = -1):
        grid_search = GridSearchCV(estimator=self.estimator, param_grid=params, 
                                   scoring=scoring, n_jobs=n_jobs)
        grid_search.fit(self.X_train, np.log1p(self.y_train))
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
