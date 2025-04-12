class HousePriceFeatureManager:
    def is_positive_rate(self, value) -> bool:
        return value in ['Ex', 'Gd', 'GLQ', 'Fin', 'GdPrv', 'GdWo']
    
    def is_neutral_rate(self, value) -> bool:
        return value in ['TA', 'Av', 'ALQ', 'Rec', 'RFn']
    
    def is_negative_rate(self, value) -> bool:
        return value in ['Fa', 'Po', 'Mn', 'BLQ', 'LwQ', 'Unf', 'MnPrv', 'MnWw']