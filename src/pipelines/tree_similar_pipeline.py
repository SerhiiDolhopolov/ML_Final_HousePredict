from pipelines import PipelineTemplate

import pandas as pd

from split_data_type import SplitDataType
from house_price_feature_manager import HousePriceFeatureManager


class TreeSimilarPipeline(PipelineTemplate, HousePriceFeatureManager):
    def __init__(self, df, split_data_type: SplitDataType):
        super().__init__(df, split_data_type)        

    # скоріш за все для всіх моделей однакові дії
    def _drop_not_needed(self, df) -> pd.DataFrame:
        self.drop_not_needed(df)
        return df
        
    # скоріш за все для всіх моделей однакові дії
    def _fill_null(self, df) -> pd.DataFrame:
        self.fill_null(df)
        return df
    
    # для різних моделей можуть бути різні дії, але можуть використовуватись однакові шаблони перетворень
    def _preprocess_features(self, df) -> pd.DataFrame:
        df = self.features_to_rates(df)
        df = self.features_to_others(df)
        df = self.feautures_to_is(df)
        return df
    
    # для різних моделей будуть різні дії
    def _encode(self, df) -> pd.DataFrame:
        return df
        
    # для різних моделей будуть різні дії
    def _normalize(self, df) -> pd.DataFrame:
        return df
    
    # для різних моделей будуть різні дії
    def _drop_high_correlation(self, df) -> pd.DataFrame:
        # Пірсона
        df.drop(columns=['PoolArea', 'GarageArea', 'TotRmsAbvGrd',
                         '1stFlrSF', 'BsmtFinSF2'], errors='ignore', inplace=True)
        # VIF
        df.drop(columns=['TotalBsmtSF', '2ndFlrSF'], errors='ignore', inplace=True)
        return df