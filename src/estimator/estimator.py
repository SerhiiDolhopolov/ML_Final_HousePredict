from abc import ABC, abstractmethod


class Estimator(ABC):   
    def __init__(self):
        pass
        
    @abstractmethod
    def fit(self):
        pass
    
    @abstractmethod
    def predict(self):
        pass