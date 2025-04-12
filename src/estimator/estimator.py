from abc import ABC, abstractmethod


class Estimator(ABC):   
    def __init__(self, train_X, train_y, test_X, test_y):
        self.train_X = train_X
        self.train_y = train_y
        self.test_X = test_X
        self.test_y = test_y
        
    @abstractmethod
    def fit(self):
        pass
    
    @abstractmethod
    def predict(self):
        pass