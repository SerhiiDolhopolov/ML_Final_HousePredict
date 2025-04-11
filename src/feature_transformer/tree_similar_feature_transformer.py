from feature_transformer import FeatureTransformer

from split_data_type import SplitDataType


class TreeSimilarFeatureTransformar(FeatureTransformer):
    def __init__(self, 
                 df, 
                 split_data_type: SplitDataType, 
                 *, 
                 logging: bool = False):
        super().__init__(df=df, 
                         split_data_type=split_data_type, 
                         normalization_scalar=None, 
                         logging=logging)

    def drop_not_needed(self):
        self.df.drop(columns=['Id'], inplace=True, errors='ignore')
        
    def fill_null(self):
        self.df.fillna(-1, inplace=True)
        
    def encode(self):
        pass
        
    def normalize(self):
        pass
    
    def drop_high_correlation(self):
        pass