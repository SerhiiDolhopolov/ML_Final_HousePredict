from pipeline import PipelineTemplate

import pandas as pd
from split_data_type import SplitDataType


class TreeSimilarPipeline(PipelineTemplate):
    def __init__(self, df, split_data_type: SplitDataType):
        super().__init__(df, split_data_type)        

    def _drop_not_needed(self, df) -> pd.DataFrame:
        return df.drop(columns=['Id'], errors='ignore')
        
    def _fill_null(self, df) -> pd.DataFrame:
        return df.fillna(-1)
    
    def _add_new_features(self, df) -> pd.DataFrame:
        return df
        
    def _encode(self, df) -> pd.DataFrame:
        return df
        
    def _normalize(self, df) -> pd.DataFrame:
        return df
    
    def _drop_high_correlation(self, df) -> pd.DataFrame:
        return df