from abc import ABC

import pandas as pd
import numpy as np
from scipy.stats import entropy
from sklearn.preprocessing import StandardScaler
from statsmodels.stats.outliers_influence import variance_inflation_factor


class FeatureManager(ABC):
    @staticmethod
    def get_smallest_category(df) -> pd.DataFrame:
        df = df.select_dtypes(include=['object', 'category'])

        result = []
        for column in df.columns:
            value_counts = df[column].value_counts(normalize=True)
            value = value_counts.idxmin()
            percent = value_counts.min() * 100

            result.append([column, value, percent])
        return pd.DataFrame(result, columns=['column', 'value', 'frequency(%)']) \
                 .set_index('column') \
                 .sort_values(by='frequency(%)', ascending=True)
    
    @staticmethod
    def get_high_entropy(df, threshold : float = 0.95) -> list[str]:
        def calculate_entropy(series):
            value_counts = series.value_counts(normalize=True)
            return entropy(value_counts, base=2)

        threshold = np.log2(df.shape[0]) * threshold
        high_entropy_features = [col for col in df.columns 
                                 if calculate_entropy(df[col]) >= threshold]
        return high_entropy_features
    
    @staticmethod
    def get_features_with_none(df) -> list[str]:
        null_columns = df.columns[df.isnull().any(axis=0)]
        null_columns_list = ', '.join(null_columns) if not null_columns.empty else "None"
        return null_columns_list
    
    @staticmethod
    def get_high_correlation_features(df, theresold: float = 0.75) -> pd.Series:
        """Return sum of high corelation (>= theresold) features only where correlation > 0 in descending order.
    
        Returns:
            pd.Series: Features with sum of high corelation, where low correlation are thrown out
        """
        df = df.select_dtypes(include=[np.number])
        corr = df.corr()
        corr_filter = corr[(abs(corr) >= theresold) & (abs(corr) != 1)]
        corr_df = corr_filter.dropna(how='all').dropna(axis=1, how='all')
        return corr_df.abs().sum().sort_values(ascending=False)

    @staticmethod
    def get_VIF_correlation_features(df) -> pd.Series:
        df = df.select_dtypes(include=[np.number])
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(df)
        vif_data = pd.DataFrame()
        vif_data["feature"] = df.columns
        vif_data["VIF"] = [variance_inflation_factor(X_scaled, i) for i in range(X_scaled.shape[1])]

        vif_data = vif_data.sort_values(by="VIF", ascending=False)
        return vif_data
