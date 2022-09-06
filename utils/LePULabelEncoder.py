#Custom Transformer that extracts columns passed as argument to its constructor 
from sklearn.base import BaseEstimator, TransformerMixin
class LePULabelEncoder( BaseEstimator, TransformerMixin ):
    #Class Constructor 
    def __init__(self):
        pass
#         self._feature_names = feature_names 
    
    #Return self nothing else to do here    
    def fit(self, X, y = None ):
        return self
    
    #Method that describes what we need this transformer to do
    def transform(self, X, y = None ):
        l = y[:, 0]
        y = y[:, 1]
        X = np.vstack((X, l))
        return X, y
    