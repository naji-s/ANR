#Custom Transformer that extracts columns passed as argument to its constructor 
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import StandardScaler, MinMaxScaler
class MultiViewStandardScaler( BaseEstimator, TransformerMixin ):
    #Class Constructor 
    def __init__(self, active=False):
        self.active = active
        self.standard_scaler_dict = dict()
        self.minmax_scaler_dict = dict()
        
#         pass
#         self._feature_names = feature_names 
    
    #Return self nothing else to do here    
    def fit( self, X, y = None):
        print ("MultiviewStandardScaler fitting is running...")
        if self.active:
            if type(X).__name__ == 'dict':
                for key in X.keys():
                    self.standard_scaler_dict[key] = StandardScaler(with_std=True)
                    self.standard_scaler_dict[key].fit(X[key])
            else:
                self.standard_scaler_dict[None] = StandardScaler(with_std=True)
                self.standard_scaler_dict[None].fit(X)
        print ("MultiviewStandardScaler fittting ENDED...")
        return self 
    
    #Method that describes what we need this transformer to do
    def transform( self, X, y = None ):
        if self.active:
            if type(X).__name__ == 'dict':
                output_dict = dict()
                for key in X.keys():
                    output_dict[key] = self.standard_scaler_dict[key].transform(X[key])
                return output_dict
            else: 
                return self.standard_scaler_dict[None].transform(X)
        else:
            return X