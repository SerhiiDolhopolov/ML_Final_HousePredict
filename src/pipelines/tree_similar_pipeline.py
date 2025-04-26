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
            'LotFrontage': df['LotFrontage'].median(),
            'MasVnrArea': 0,
            'GarageYrBlt': 0,
            'GarageFinish': 'NoGarage',
            'BsmtFinSF1': 0,
            'BsmtFinSF2': 0,
            'BsmtUnfSF': 0,
            'BsmtFullBath': 0,
            'BsmtHalfBath': 0,
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
        self.__features_to_rates(df)
        self.__features_to_others(df)
        self.__feautures_to_is(df)
        self.__quantile_features(df)
        self.__validation_features_years(df)
        return df

    def __features_to_rates(self, df):
        for column in ['HeatingQC', 'ExterCond', 'BsmtCond', 'GarageCond', 
                       'GarageQual', 'ExterQual', 'BsmtQual', 'FireplaceQu',
                       'KitchenQual', 'Fence', 'BsmtExposure']:
            df[column] = df[column].apply(lambda x: 
                'positive' if HPP.is_positive_rate(x) else
                'neutral' if HPP.is_neutral_rate(x) else
                'negative' if HPP.is_negative_rate(x) else
                'Others')

    def __features_to_others(self, df): 
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

    def __feautures_to_is(self, df):
        bsmt_fin_type_list = ['GLQ', 'ALQ', 'BLQ', 'Rec', 'LwQ']
        HPP.transform_feature_to_is(df, 'BsmtFinType1', 'Is_finished_bsmt_fintype1', bsmt_fin_type_list)
        HPP.transform_feature_to_is(df, 'BsmtFinType2', 'Is_finished_bsmt_fintype2', bsmt_fin_type_list)
        HPP.transform_feature_to_is(df, 'Functional', 'Is_typical_functional', ['Typ'])
        HPP.transform_feature_to_is(df, 'Electrical', 'Is_standard_electrical', ['SBrkr'])
        HPP.transform_feature_to_is(df, 'RoofStyle', 'Is_gable_roofstyle', ['Gable'])
        HPP.transform_feature_to_is(df, 'Condition1', 'Is_norm_condition1', ['Norm'])
        HPP.transform_feature_to_is(df, 'SaleCondition', 'Is_normal_sale_condition', ['Normal'])
        HPP.transform_feature_to_is(df, 'LotShape', 'Is_reg_lotshape', ['Reg'])
        HPP.transform_feature_to_is(df, 'MSZoning', 'Is_residential_mszoning', ['RL', 'RM', 'RP', 'RH'])
        HPP.transform_feature_to_is(df, 'LandSlope', 'Is_Gtl_landslope', ['Gtl'])
        HPP.transform_feature_to_is(df, 'PavedDrive', 'Is_paved', ['Y'])
        HPP.transform_feature_to_is(df, 'LandContour', 'Is_level_landContour', ['Lvl'])
        HPP.transform_feature_to_is(df, 'CentralAir', 'Is_central_air', ['Y'])
        
        for column in ['EnclosedPorch', '3SsnPorch', 'ScreenPorch', 'PoolArea', 'MiscVal']:
            HPP.transform_feature_to_is_not_0(df, column, 'Is_' + column.lower())
    
    def __quantile_features(self, df): 
        columns = ['LotFrontage', 'LotArea', 'MasVnrArea', 'BsmtFinSF1',
                   'BsmtUnfSF', '1stFlrSF', '2ndFlrSF', 'WoodDeckSF', 'OpenPorchSF']
        for col in columns:
            if self.split_data_type.TRAIN:
                df[col] = self.quantile_transformer.fit_transform(df[[col]])
            else:
                df[col] = self.quantile_transformer.transform(df[[col]])
                
    def __validation_features_years(self, df):
        # Валідація в випадку дерев буде як -1, в випадку лінійної регресії або KNN це було б середнє
        for col in ['YearBuilt', 'YearRemodAdd', 'GarageYrBlt']:
            validation = lambda x: -1 if x > 2016 or x < 1800 else x
            df[col] = df[col].apply(validation)

    def _encode(self, df) -> pd.DataFrame:
        for col in df.select_dtypes(include=['object']).columns:
            if self.split_data_type.TRAIN:
                df[col] = self.label_encoder.fit_transform(df[col])
            else:
                df[col] = self.label_encoder.transform(df[col])
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