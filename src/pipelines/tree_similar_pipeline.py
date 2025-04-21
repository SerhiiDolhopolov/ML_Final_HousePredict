from pipelines import PipelineTemplate

import pandas as pd
from sklearn.preprocessing import LabelEncoder, QuantileTransformer

from split_data_type import SplitDataType
from house_price_processor import HousePriceProcessor as HPP


class TreeSimilarPipeline(PipelineTemplate):
    def __init__(self, 
        df, 
        split_data_type: SplitDataType, 
        label_encoder: LabelEncoder,
        quantile_transformer: QuantileTransformer,
        ):
        super().__init__(df, split_data_type)
        self.label_encoder = label_encoder
        self.quantile_transformer = quantile_transformer

    def _drop_not_needed(self, df) -> pd.DataFrame:
        df.drop(columns=['Id', 'Utilities', 'RoofMatl', 
                         'Condition2', 'Heating', 'Street', 'MSSubClass'
                         ], errors='ignore', inplace=True)
        return df

    def _fill_null(self, df) -> pd.DataFrame:
        df.fillna({
            'LotFrontage': df['LotFrontage'].mean(),
            'MasVnrArea': df['MasVnrArea'].mean(),
            'GarageYrBlt': -1,
            'GarageFinish': 'NoGarage',
            'BsmtFinSF1': 0,
            'BsmtFinSF2': 0,
            'BsmtUnfSF': 0,
            'BsmtFullBath': df['BsmtFullBath'].mean(),
            'BsmtHalfBath': df['BsmtHalfBath'].mean(),
            'GarageCars': 0,
            'GarageArea': 0,
            'TotalBsmtSF': 0
        }, inplace=True)
        HPP.transform_feature_to_is_not_none(df, 'Alley', 'WithAlley')
        HPP.transform_feature_to_is_not_none(df, 'MasVnrType', 'WithMasonry')
        HPP.transform_feature_to_is_not_none(df, 'PoolQC', 'WithPool')
        HPP.transform_feature_to_is_not_none(df, 'MiscFeature', 'WithFeature')
        return df

    def _preprocess_features(self, df) -> pd.DataFrame:
        df = self._features_to_rates(df)
        df = self._features_to_others(df)
        df = self._feautures_to_is(df)
        
        action = (self.quantile_transformer.fit_transform if self.split_data_type.TRAIN 
                  else self.quantile_transformer.transform)
        for col in ['LotFrontage', 'LotArea', 'MasVnrArea', 'BsmtFinSF1',
                    'BsmtUnfSF', '1stFlrSF', '2ndFlrSF', 'WoodDeckSF', 'OpenPorchSF']:
            df[col] = action(df[[col]])
        
        for col in ['YearBuilt', 'YearRemodAdd', 'GarageYrBlt']:
            HPP.year_validation(df, col)
        return df

    def _features_to_rates(self, df) -> pd.DataFrame:
        for column in ['HeatingQC', 'ExterCond', 'BsmtCond', 'GarageCond', 
                       'GarageQual', 'ExterQual', 'BsmtQual', 'FireplaceQu',
                       'KitchenQual', 'Fence', 'BsmtExposure']:
            df[column] = df[column].apply(lambda x: 
                'positive' if HPP.is_positive_rate(x) else
                'neutral' if HPP.is_neutral_rate(x) else
                'negative' if HPP.is_negative_rate(x) else
                'Others')
        return df

    def _features_to_others(self, df) -> pd.DataFrame: 
        exterior_list = ['VinylSd', 'HdBoard', 'MetalSd', 'Wd Sdng', 'Plywood']
        
        df['Exterior1st'] = HPP.transform_feature_with_others(df, 'Exterior1st', exterior_list)
        df['Exterior2nd'] = HPP.transform_feature_with_others(df, 'Exterior2nd', exterior_list)
        df['Foundation'] = HPP.transform_feature_with_others(df, 'Foundation', ['PConc', 'CBlock'])
        df['LotConfig'] = HPP.transform_feature_with_others(df, 'LotConfig', ['Inside', 'Corner'])
        df['GarageType'] = HPP.transform_feature_with_others(df, 'GarageType', ['Attchd', 'Detchd'])
        
        def transform_house_style(x):
            if x in ['1Story', '1.5Unf', '1.5Fin']:
                return '1Story'
            elif x in ['2Story', '2.5Unf', '2.5Fin']:
                return '2Story'
            else:
                return 'Others'
        
        df['HouseStyle'] = df['HouseStyle'].apply(lambda x: transform_house_style(x))
        
        def transorm_sale_type(x):
            if x == 'New':
                return 'New'
            elif x in ['WD', 'CWD', 'VWD']:
                return 'Warranty_deed'
            else:
                return 'Others'
            
        df['SaleType'] = df['SaleType'].apply(lambda x: transorm_sale_type(x))
        return df

    def _feautures_to_is(self, df) -> pd.DataFrame:
        bsmt_fin_type_list = ['GLQ', 'ALQ', 'BLQ', 'Rec', 'LwQ']
        HPP.transform_feature_to_are(df, 'BsmtFinType1', 'Is_finished_bsmt_fintype1', bsmt_fin_type_list)
        HPP.transform_feature_to_are(df, 'BsmtFinType2', 'Is_finished_bsmt_fintype2', bsmt_fin_type_list)
        HPP.transform_feature_to_are(df, 'Functional', 'Is_typical_functional', ['Typ'])
        HPP.transform_feature_to_are(df, 'Electrical', 'Is_standard_electrical', ['SBrkr'])
        HPP.transform_feature_to_are(df, 'RoofStyle', 'Is_gable_roofstyle', ['Gable'])
        HPP.transform_feature_to_are(df, 'Condition1', 'Is_norm_condition1', ['Norm'])
        HPP.transform_feature_to_are(df, 'SaleCondition', 'Is_normal_sale_condition', ['Normal'])
        HPP.transform_feature_to_are(df, 'LotShape', 'Is_reg_lotshape', ['Reg'])
        HPP.transform_feature_to_are(df, 'MSZoning', 'Is_residential_mszoning', ['RL', 'RM', 'RP', 'RH'])
        HPP.transform_feature_to_are(df, 'LandSlope', 'Is_Gtl_landslope', ['Gtl'])
        HPP.transform_feature_to_are(df, 'PavedDrive', 'Is_paved', ['Y'])
        HPP.transform_feature_to_are(df, 'LandContour', 'Is_level_landContour', ['Lvl'])
        HPP.transform_feature_to_are(df, 'CentralAir', 'Is_central_air', ['Y'])
        
        for column in ['EnclosedPorch', '3SsnPorch', 'ScreenPorch', 'PoolArea', 'MiscVal']:
            HPP.transform_feature_to_is_not_0(df, column, 'Is_' + column.lower())
        return df

    def _encode(self, df) -> pd.DataFrame:
        action = (self.label_encoder.fit_transform if self.split_data_type.TRAIN 
                  else self.label_encoder.transform)
        for col in df.select_dtypes(include=['object']).columns:
            df[col] = action(df[col])
        return df
        
    def _normalize(self, df) -> pd.DataFrame:
        return df
    
    def _drop_high_correlation(self, df) -> pd.DataFrame:
        # Пірсона
        df.drop(columns=['Is_miscval', 'WithMasonry', 'Is_finished_bsmt_fintype1',
                         'GarageQual', 'Exterior2nd', 'GarageCars', 'FireplaceQu',
                         'GrLivArea', 'BsmtFinSF2'
                         ], errors='ignore', inplace=True)
        # VIF
        df.drop(columns=['Is_poolarea', 'GarageYrBlt', 'YearBuilt'
                         ], errors='ignore', inplace=True)
    
        return df