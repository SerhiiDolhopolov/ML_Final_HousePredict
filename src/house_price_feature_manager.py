import pandas as pd

from feature_manager import FeatureManager


# сюди виносяться зручні практики для багатьох пайплайнів обробки
class HousePriceFeatureManager(FeatureManager):
    def is_positive_rate(self, value) -> bool:
        return value in ['Ex', 'Gd', 'GLQ', 'Fin', 'GdPrv', 'GdWo']
    
    def is_neutral_rate(self, value) -> bool:
        return value in ['TA', 'Av', 'ALQ', 'Rec', 'RFn']
    
    def is_negative_rate(self, value) -> bool:
        return value in ['Fa', 'Po', 'Mn', 'BLQ', 'LwQ', 'Unf', 'MnPrv', 'MnWw']
    
    def drop_not_needed(self, df):
        df.drop(columns=['Id', 'Utilities', 'RoofMatl', 'Condition2','Heating', 
                         'Street'], errors='ignore', inplace=True)
        
    def fill_null(self, df):
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
            'GarageCars': 0
        }, inplace=True)
        self.transform_feature_to_is_not_none(df, 'Alley', 'WithAlley')
        self.transform_feature_to_is_not_none(df, 'MasVnrType', 'WithMasonry')
        self.transform_feature_to_is_not_none(df, 'PoolQC', 'WithPool')
        self.transform_feature_to_is_not_none(df, 'MiscFeature', 'WithFeature')
        
    def features_to_rates(self, df) -> pd.DataFrame:
        for column in ['HeatingQC', 'ExterCond', 'BsmtCond', 'GarageCond', 
                       'GarageQual', 'ExterQual', 'BsmtQual', 'FireplaceQu',
                       'KitchenQual', 'Fence', 'BsmtExposure']:
            df[column] = df[column].apply(lambda x: 
                'positive' if self.is_positive_rate(x) else
                'neutral' if self.is_neutral_rate(x) else
                'negative' if self.is_negative_rate(x) else
                'Others')
        return df
    
    def features_to_others(self, df) -> pd.DataFrame: 
        exterior_list = ['VinylSd', 'HdBoard', 'MetalSd', 'Wd Sdng', 'Plywood']
        
        df['Exterior1st'] = self.transform_feature_with_others(df, 'Exterior1st', exterior_list)
        df['Exterior2nd'] = self.transform_feature_with_others(df, 'Exterior2nd', exterior_list)
        df['Foundation'] = self.transform_feature_with_others(df, 'Foundation', ['PConc', 'CBlock'])
        df['LotConfig'] = self.transform_feature_with_others(df, 'LotConfig', ['Inside', 'Corner'])
        df['GarageType'] = self.transform_feature_with_others(df, 'GarageType', ['Attchd', 'Detchd'])
        
        
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
    
    def feautures_to_is(self, df) -> pd.DataFrame:
        bsmt_fin_type_list = ['GLQ', 'ALQ', 'BLQ', 'Rec', 'LwQ']
        self.transform_feature_to_are(df, 'BsmtFinType1', 'Is_finished_bsmt_fintype1', bsmt_fin_type_list)
        self.transform_feature_to_are(df, 'BsmtFinType2', 'Is_finished_bsmt_fintype2', bsmt_fin_type_list)
        self.transform_feature_to_are(df, 'Functional', 'Is_typical_functional', ['Typ'])
        self.transform_feature_to_are(df, 'Electrical', 'Is_standard_electrical', ['SBrkr'])
        self.transform_feature_to_are(df, 'RoofStyle', 'Is_gable_roofstyle', ['Gable'])
        self.transform_feature_to_are(df, 'Condition1', 'Is_norm_condition1', ['Norm'])
        self.transform_feature_to_are(df, 'SaleCondition', 'Is_normal_sale_condition', ['Normal'])
        self.transform_feature_to_are(df, 'LotShape', 'Is_reg_lotshape', ['Reg'])
        self.transform_feature_to_are(df, 'MSZoning', 'Is_residential_mszoning', ['RL', 'RM', 'RP', 'RH'])
        self.transform_feature_to_are(df, 'LandSlope', 'Is_Gtl_landslope', ['Gtl'])
        self.transform_feature_to_are(df, 'PavedDrive', 'Is_paved', ['Y'])
        self.transform_feature_to_are(df, 'LandContour', 'Is_level_landContour', ['Lvl'])
        self.transform_feature_to_are(df, 'CentralAir', 'Is_central_air', ['Y'])
        
        return df