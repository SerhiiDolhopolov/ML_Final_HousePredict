from abc import ABC, abstractmethod
from functools import wraps

from sklearn.base import OneToOneFeatureMixin
from split_data_type import SplitDataType


class FeatureTransformer(ABC):
    def __init__(self, 
                 df,
                 split_data_type: SplitDataType,
                 normalization_scalar: OneToOneFeatureMixin = None,
                 *,
                 logging: bool = False):
        """FeatureTransformer is an abstract class for feature transformation.
        It provides a common interface for different feature transformation methods.

        Args:
            split_data_type (SplitDataType): type of data splitting. TEST, TRAIN, or VALIDATION.
            normalization_scalar (OneToOneFeatureMixin, optional): scalar for normalization. It can be MinMaxScaler or StandardScaler, etc. from sklearn.preprocessing.
            logging (bool, optional): if needs to log the process.
        """
        self.df = df
        self.split_data_type = split_data_type
        self.normalization_scalar = normalization_scalar
        self.logging = logging
        
    @abstractmethod
    def drop_not_needed(self):
        pass
        
    @abstractmethod
    def fill_null(self):
        pass
        
    @abstractmethod
    def encode(self):
        pass
        
    @abstractmethod
    def normalize(self):
        pass
    
    @abstractmethod
    def drop_high_correlation(self):
        pass
    
    def __getattribute__(self, name):
        """Використовується для логування, якщо атр колейбл, то логування"""
        attr = super().__getattribute__(name)
        if callable(attr) and name not in ['_FeatureTransformer__log']:
            return self.__log(attr)
        return attr
    
    def __log(self, func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            result = func(*args, **kwargs)
            
            if self.logging:
                print('-' * 100)
                print('{data_type}: {func}'.format(
                    data_type=self.split_data_type.name,
                    func=func.__name__))
                has_null_mask = self.df.isnull().any()
                null_columns_count = self.df.columns[has_null_mask].size
                print("Size = {0}".format(self.df.shape))
                print("Nulls columns = {0}".format(null_columns_count))
                print('-' * 100)
                
            return result
        return wrapper
    