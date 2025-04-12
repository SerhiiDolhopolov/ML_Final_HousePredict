from abc import ABC, abstractmethod
import logging
from inspect import signature

import pandas as pd

from split_data_type import SplitDataType

#реалізація патерну Template Method
class PipelineTemplate(ABC):
    def __init__(self, df, split_data_type: SplitDataType):
        self.__df = df
        self.split_data_type = split_data_type

    def build(self) -> pd.DataFrame:
        df = self.__df.copy()
        df = self._drop_not_needed(df)
        df = self._fill_null(df)
        df = self._add_new_features(df)
        df = self._encode(df)
        df = self._normalize(df)
        df = self._drop_high_correlation(df)
        return df

    @abstractmethod
    def _drop_not_needed(self, df) -> pd.DataFrame:
        pass
        
    @abstractmethod
    def _fill_null(self, df) -> pd.DataFrame:
        pass

    @abstractmethod
    def _add_new_features(self, df) -> pd.DataFrame:
        pass
        
    @abstractmethod
    def _encode(self, df) -> pd.DataFrame:
        pass
        
    @abstractmethod
    def _normalize(self, df) -> pd.DataFrame:
        pass
    
    @abstractmethod
    def _drop_high_correlation(self, df) -> pd.DataFrame:
        pass

    def __getattribute__(self, name):
        attr = super().__getattribute__(name)
        allowed_methods = ['_drop_not_needed', '_fill_null', '_encode',
                            '_normalize', '_drop_high_correlation']
        if callable(attr) and name in allowed_methods:
            return self.__log(attr)
        return attr
    
    def __log(self, func):
        def wrapper(df, *args, **kwargs):
            result = func(df, *args, **kwargs)

            null_columns = result.columns[result.isnull().any(axis=0)]
            null_columns_list = ', '.join(null_columns) if not null_columns.empty else "None"
            logging_message = (
                f"{self.split_data_type.name} - Function: {func.__name__}\n"
                f"Size: {result.shape}\n"
                f"Columns with null: {null_columns_list}\n"
            )

            logging.debug(logging_message)

            return result
        return wrapper