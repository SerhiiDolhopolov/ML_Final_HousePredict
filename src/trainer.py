from typing import overload

from sklearn.model_selection import train_test_split
from sklearn.base import OneToOneFeatureMixin

from estimator import Estimator
from feature_transformer import FeatureTransformer


class Trainer():
    @overload
    def __init__(self, 
                 df, 
                 target_column: str, 
                 *, 
                 test_size: float = 0.2,
                 random_state: int | None = None,
                 shuffle: bool = True,
                 logging: bool = False): ...
    
    @overload
    def __init__(self, 
                 train_df, 
                 test_X, 
                 test_y, 
                 target_column: str,
                 *,
                 logging: bool = False): ...
        
    def __init__(self, *args, **kwargs):
        self.logging = kwargs.get('logging', False)
        if len(args) == 2:
            df, self.target_column = args
            X = df.drop(columns=[self.target_column])
            y = df[self.target_column]
            self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
                X, y, **self.__get_split_params(kwargs))
                        
        elif len(args) == 4:
            train_df, self.X_test, self.y_test, self.target_column = args
            self.X_train = train_df.drop(columns=[self.target_column])
            self.y_train = train_df[self.target_column]
            
    def __get_split_params(self, kwargs):
        split_params = {
            'test_size': kwargs.get('test_size', 0.2),
            'shuffle': kwargs.get('shuffle', True),
            }
        if 'random_state' in kwargs:
            split_params['random_state'] = kwargs['random_state']
        return split_params


    def train(self, 
              feature_transformer: FeatureTransformer,
              estimator: Estimator, 
              *,
              estimator_params: dict = None):
        """Trains the model using the provided feature transformer and estimator.

        Args:
            feature_transformer (FeatureTransformer): Instance of FeatureTransformer for preprocessing.
            estimator (Estimator): Instance of Estimator for model training.
            estimator_params (dict, optional): Parameters for the estimator. Defaults to an empty dictionary.
        """
        if estimator_params is None:
            estimator_params = {}

        # Apply feature transformations
        self.X_train = feature_transformer.drop_not_needed()
        self.X_train = feature_transformer.fill_null()
        self.X_train = feature_transformer.encode()
        self.X_train = feature_transformer.normalize()
        self.X_train = feature_transformer.drop_high_correlation()

        # Train and predict using the estimator
        estimator.fit(self.X_train, self.y_train, **estimator_params)
        predictions = estimator.predict(self.X_test)
        return predictions
