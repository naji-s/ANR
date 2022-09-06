#Custom Transformer that extracts columns passed as argument to its constructor 
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from numpy import abs as np_abs
from numpy import maximum as np_maximum
from numpy import min as np_min

class MultiViewMinMaxScaler( BaseEstimator, TransformerMixin ):
    #Class Constructor 
    def __init__(self, active=False):
        self.active = active
        self.standard_scaler_dict = dict()
        self.minmax_scaler_dict = dict()
        self.max_number = dict()
        
#         pass
#         self._feature_names = feature_names 
    
    #Return self nothing else to do here    
    def fit( self, X, y = None):
        if self.active:
            if type(X).__name__ == 'dict':
                for key in X.keys():
                    self.standard_scaler_dict[key] = MinMaxScaler()
                    self.standard_scaler_dict[key].fit(X[key])
                    self.max_number[key] = np_maximum(np_abs(self.standard_scaler_dict[key].data_min_), np_abs(self.standard_scaler_dict[key].data_max_))
            else:
                self.standard_scaler_dict[None] = MinMaxScaler()
                self.standard_scaler_dict[None].fit(X)
                self.max_number[None] = np_maximum(np_abs(self.standard_scaler_dict[None].data_min_), np_abs(self.standard_scaler_dict[None].data_max_))
                
        return self 
    
    #Method that describes what we need this transformer to do
    def transform( self, X, y = None ):
        if self.active:
            if type(X).__name__ == 'dict':
                output_dict = dict()
                for key in X.keys():
                    output_dict[key] = X[key] / self.max_number[key]
                return output_dict
            else: 
                return  X / self.max_number[None]
        else:
            return X