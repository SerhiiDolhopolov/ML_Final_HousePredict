from pipeline import PipelineTemplate

import pandas as pd

from split_data_type import SplitDataType
from house_price_feature_manager import HousePriceFeatureManager


class TreeSimilarPipeline(PipelineTemplate, HousePriceFeatureManager):
    def __init__(self, df, split_data_type: SplitDataType):
        super().__init__(df, split_data_type)        

    def _drop_not_needed(self, df) -> pd.DataFrame:
        return df.drop(columns=['Id', 'Utilities', 'RoofMatl', 'Condition2',
                                'Heating'
                                ], errors='ignore')
        
    def _fill_null(self, df) -> pd.DataFrame:
        return df
    
    def _preprocess_features(self, df) -> pd.DataFrame:
        def transform_with_others(column, valid_values):
            return df[column].apply(lambda x: x if x in valid_values else 'Others')
        
        exterior_list = ['VinylSd', 'HdBoard', 'MetalSd', 'Wd Sdng', 'Plywood']
        
        df['Exterior1st'] = transform_with_others('Exterior1st', exterior_list)
        df['Exterior2nd'] = transform_with_others('Exterior2nd', exterior_list)
        
        for column in ['HeatingQC', 'ExterCond', 'BsmtCond']:
            df[column] = df[column].apply(
                lambda x: 'positive' if self.is_positive_rate(x) else
                'neutral' if self.is_neutral_rate(x) else
                'negative' if self.is_negative_rate(x) else
                'Others')
            
        df['SaleType'] = df['SaleType'].apply(lambda x: 'New' if x == 'New' else
                                              'Warranty_deed' if x in ['WD', 'CWD', 'VWD'] else
                                              'Others')
            
        df['Is_typical_functional'] = df['Functional'].apply(lambda x: x == 'Typ').astype('int8')
        df.drop(columns=['Functional'], inplace=True)
        
        df['Is_standard_electrical'] = df['Electrical'].apply(lambda x: x == 'SBrkr').astype('int8')
        df.drop(columns=['Electrical'], inplace=True)
        
        df['Is_gable_roof_style'] = df['RoofStyle'].apply(lambda x: x == 'Gable').astype('int8')
        df.drop(columns=['RoofStyle'], inplace=True)
        
        df['Is_norm_condition1'] = df['Condition1'].apply(lambda x: x == 'Norm').astype('int8')
        df.drop(columns=['Condition1'], inplace=True)
        
        
        return df
    
    def _add_new_features(self, df) -> pd.DataFrame:
        return df
        
    def _encode(self, df) -> pd.DataFrame:
        return df
        
    def _normalize(self, df) -> pd.DataFrame:
        return df
    
    def _drop_high_correlation(self, df) -> pd.DataFrame:
        return df