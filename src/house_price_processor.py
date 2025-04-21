import pandas as pd


# сюди виносяться зручні практики для багатьох пайплайнів обробки
class HousePriceProcessor():
    @staticmethod
    def is_positive_rate(value) -> bool:
        return value in ['Ex', 'Gd', 'GLQ', 'Fin', 'GdPrv', 'GdWo']
    
    @staticmethod
    def is_neutral_rate(value) -> bool:
        return value in ['TA', 'Av', 'ALQ', 'Rec', 'RFn']
    
    @staticmethod
    def is_negative_rate(value) -> bool:
        return value in ['Fa', 'Po', 'Mn', 'BLQ', 'LwQ', 'Unf', 'MnPrv', 'MnWw']
    
    @staticmethod
    def transform_feature_with_others(df, column_name: str, valid_values: list[str]) -> pd.Series:
        return df[column_name].apply(lambda x: x if x in valid_values else 'Others')
    
    @staticmethod
    def transform_feature_to_are(df, column_name: str, new_column_name: str, true_values: list[str]):
        df[new_column_name] = df[column_name].apply(lambda x: x in true_values).astype('int8')
        df.drop(columns=[column_name], inplace=True)
    
    @staticmethod
    def transform_feature_to_is_not_none(df, column_name: str, new_column_name: str):
        df[new_column_name] = df[column_name].apply(lambda x: not pd.isnull(x)).astype('int8')
        df.drop(columns=[column_name], inplace=True)
    
    @staticmethod
    def transform_feature_to_is_not_0(df, column_name: str, new_column_name: str):
        df[new_column_name] = df[column_name].apply(lambda x: x != 0).astype('int8')
        df.drop(columns=[column_name], inplace=True)
        
    @staticmethod
    def year_validation(df, column_name: str):
        validation = lambda x: mean_year if x > 2016 or x < 1800 else x
        mean_year = df[column_name].mean()
        df[column_name] = df[column_name].apply(validation)